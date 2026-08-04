"""Proposes new models by asking an LLM to rewrite a model's marked blocks.

The client is asynchronous: a search runs many proposers at once, and they
spend nearly all of their time waiting on the inference server.

Given an `Analysis`, a proposer may also look at the data before it writes
anything. That turns one request into a short loop -- ask, run a snippet, ask
again -- bounded by `tool_budget` rather than by a clock, and it is why a
tool-using search runs N conversations rather than one batched request.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from discovery.analysis import RUN_PYTHON, Analysis
from discovery.model import Model, parse_blocks

SYSTEM_PROMPT = """\
You are a scientist searching for a better model. Models are programs.

You will be shown a program. Regions you may rewrite are delimited by
`# @discovery:begin <name>` and `# @discovery:end` comments. Everything outside
those markers is fixed: it is the harness that runs and scores the model, and
you must not restate or alter it.

Reply with a single fenced code block containing only the rewritten regions,
each still wrapped in its own begin/end markers and keeping its original name.
Do not include any other code. Keep indentation exactly as given. Propose a
genuine change: a different functional form, a different distribution, a new
term or structure, not a cosmetic edit.\
"""

#: appended to the system prompt when the proposer has an `Analysis`
TOOL_PROMPT = """\
Before you propose, you may look at the data with the `run_python` tool.
{description}
Each call runs in a fresh namespace. Nothing at all carries over between calls
-- not the variables you assigned, not the modules you imported, not the results
you computed -- so every snippet must repeat whatever it needs and stand
entirely on its own.

Only what you `print` comes back, cut off after {max_output} characters. You
cannot see images, so a plot tells you nothing: print the numbers you would have
looked at the picture for. You are given a copy of the data, so anything you do
to it is discarded. There is no filesystem and no network, and neither jax nor
numpyro is installed there: you cannot fit or test a model in this environment,
only look at what it is fitted to.

The model you write does not run here. This environment is pandas and numpy; the
program your block lives in may use something else entirely, so take the types
from the program, not from what the tool handed you.

You may call the tool at most {budget} times. A turn that calls the tool is not
a proposal; when you are done looking, answer with the fenced code block.\
"""

#: what an unparsed tool call looks like when it arrives as ordinary text.
#: Closing tags matter as much as opening ones: observed against Qwen3.5-35B on
#: a server whose parser demonstrably works, a reply can begin as prose and end
#: `</parameter></function></tool_call>` -- the model switched call syntax
#: part-way and the configured parser did not recognise the form it switched to.
UNPARSED = re.compile(
    r"</?tool_call>|<\|tool_call|</?function[=> ]|</?parameter[=> ]|\"name\"\s*:\s*\"run_python\"",
    re.IGNORECASE,
)


class ProposalError(ValueError):
    """A reply that could not be turned into a model.

    Carries the reply itself, so a failed agent can be inspected rather than
    just counted.
    """

    def __init__(
        self,
        message: str,
        reply: str = "",
        finish_reason: str | None = None,
        exchange: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.reply = reply
        self.finish_reason = finish_reason
        #: the turns this proposal had got through, which the transcript will
        #: not keep. A single-shot agent that fails loses one reply and the
        #: reply is the whole story; an agent with tools loses every snippet it
        #: ran and everything it read, which is most of why it failed.
        self.exchange = exchange or []


class ToolCallError(ProposalError):
    """A tool call arrived as text instead of parsed.

    Its own error, because it is not a bad reply and reads exactly like a model
    that ignored its instructions. Two different causes, and the second is the
    reason this is worth distinguishing even on a server that is set up right:

    * the server was started without `--enable-auto-tool-choice` and
      `--tool-call-parser <parser>`, so nothing is ever parsed; or
    * the model emitted a call in a syntax the configured parser does not know.
      Observed against Qwen3.5-35B on a server whose parser demonstrably works
      for most calls, a minority come back as prose ending in
      `</parameter></function></tool_call>`.

    The first is fixed once, on the server. The second is a per-sample fact of
    life, and costs that agent its proposal.
    """


class BudgetError(ProposalError):
    """The agent spent its whole tool budget without proposing anything."""


def splice(model: Model, reply: str, finish_reason: str | None = None) -> Model:
    """Turn one reply into a model, or say why it cannot be.

    Shared by the single proposal and by each sample of a batch: the two differ
    in how they ask, not in how a reply is read.
    """
    bodies = parse_blocks(reply)
    if not bodies:
        # A reasoning model can spend its whole budget thinking and return
        # nothing at all, which is a decode setting to fix, not a bad reply.
        if finish_reason == "length":
            raise ProposalError(
                "reply hit max_tokens before any @discovery blocks; raise max_tokens",
                reply,
                finish_reason,
            )
        # It can also stop cleanly having said nothing, when the reasoning
        # parser took the whole completion. That is the same problem wearing a
        # different finish_reason, and it is worth saying so rather than
        # blaming a reply that does not exist.
        if not reply.strip():
            raise ProposalError(
                "the model returned an empty reply; it spent the completion on reasoning. "
                "Lower thinking_token_budget or set thinking=False",
                reply,
                finish_reason,
            )
        raise ProposalError("proposal contained no @discovery blocks", reply, finish_reason)
    return model.substitute(bodies)


def as_message(content: str, calls: list[Any]) -> dict[str, Any]:
    """An assistant turn holding tool calls, as the next request needs it.

    Built by hand rather than taken from the response object: what goes back is
    the wire format, and the harness should not depend on which client library
    produced it.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in calls
        ],
    }


@dataclass
class ModelProposer:
    """Stores the LLM url and the transcript, and creates the prompt.

    The transcript is the running conversation: it is what makes successive
    proposals aware of what has already been tried and how it scored. Only a
    search that chains exercises it -- an i.i.d. search uses each proposer for
    exactly one call, so every draw starts from the system prompt alone.

    An `analysis` gives the proposer the `run_python` tool and adds the section
    of the system prompt that describes it. The whole exchange is kept, tool
    calls and their output included: what the agent measured is why it proposed
    what it did, and a round that hid it would leave the next round guessing.
    """

    url: str
    llm_model: str
    api_key: str = "EMPTY"
    temperature: float = 1.0
    system_prompt: str = SYSTEM_PROMPT
    client: Any = None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    #: the data environment, or None for a proposer that only writes
    analysis: Analysis | None = None
    #: tool calls one proposal may make before it has to answer
    tool_budget: int = 6

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = AsyncOpenAI(base_url=self.url, api_key=self.api_key)
        if not self.transcript:
            self.transcript.append({"role": "system", "content": self.system()})

    def system(self) -> str:
        """The system prompt, with the tool section when there is a tool."""
        if self.analysis is None:
            return self.system_prompt
        return "\n\n".join(
            [
                self.system_prompt,
                TOOL_PROMPT.format(
                    description=f"\n{self.analysis.description.strip()}\n"
                    if self.analysis.description.strip()
                    else "",
                    max_output=self.analysis.max_output,
                    budget=self.tool_budget,
                ),
            ]
        )

    async def call_tool(self, call: Any) -> dict[str, Any]:
        """Run one tool call and phrase the result as a tool turn.

        A call the harness cannot make sense of -- the wrong name, arguments
        that are not JSON -- is answered rather than raised. It is a mistake the
        agent can see and correct on its next call, and it costs it one.
        """
        assert self.analysis is not None
        name = call.function.name
        if name != RUN_PYTHON["function"]["name"]:
            output = f"There is no tool called {name!r}."
        else:
            try:
                code = json.loads(call.function.arguments)["code"]
            except (json.JSONDecodeError, KeyError, TypeError):
                output = "The arguments were not an object with a 'code' string."
            else:
                output = await self.analysis.run(code)
        return {"role": "tool", "tool_call_id": call.id, "content": output}

    @staticmethod
    def remaining(left: int) -> str:
        """What to tell the agent about the budget it has left.

        Appended to the last result of every turn, because an agent cannot
        count its own calls: told only the limit up front, it keeps exploring
        until the harness cuts it off, and everything it learned is thrown away
        with the proposal it never wrote. Counted down, it has the chance to
        stop while it still has a turn to answer in.
        """
        if left <= 0:
            return "\n\n[No tool calls left. Answer now with the fenced code block.]"
        return f"\n\n[{left} tool call{'s' if left > 1 else ''} left.]"

    @staticmethod
    def prompt(model: Model, feedback: str | None = None) -> str:
        """The user turn describing the model to improve."""
        names = ", ".join(block.name for block in model.blocks) or "(none)"
        parts = [
            "Here is the current model.",
            "",
            f"```python\n{model.source.rstrip()}\n```",
            "",
            f"Rewritable regions: {names}.",
        ]
        if feedback:
            parts += ["", "Results for this model:", feedback]
        parts += ["", "Propose an improved version of the rewritable regions."]
        return "\n".join(parts)

    async def propose(self, model: Model, feedback: str | None = None, **kwargs: Any) -> Model:
        """Ask for one new model, appending the whole exchange to the transcript.

        Sampling settings given here override this proposer's defaults. Without
        an `analysis` this is one request and one reply. With one it is a loop:
        the agent may run up to `tool_budget` snippets against the data, and
        answers when it stops calling the tool.
        """
        settings: dict[str, Any] = {"temperature": self.temperature} | kwargs
        if self.analysis is not None:
            settings.setdefault("tools", [RUN_PYTHON])
        prompt = {"role": "user", "content": self.prompt(model, feedback)}
        exchange: list[dict[str, Any]] = [prompt]

        for remaining in range(self.tool_budget, -1, -1):
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[*self.transcript, *exchange],
                **settings,
            )
            choice = response.choices[0]
            reply = choice.message.content or ""
            calls = list(getattr(choice.message, "tool_calls", None) or [])

            if calls and self.analysis is not None:
                if not remaining:
                    raise BudgetError(
                        f"the agent used all {self.tool_budget} tool calls without proposing",
                        reply,
                        getattr(choice, "finish_reason", None),
                        exchange,
                    )
                exchange.append(as_message(reply, calls))
                # Sequentially, not gathered: the calls share one pool, and an
                # agent that fanned out would take capacity from other agents in
                # proportion to how many snippets it happened to ask for.
                for call in calls:
                    exchange.append(await self.call_tool(call))
                exchange[-1]["content"] += self.remaining(remaining - 1)
                continue

            try:
                proposed = splice(model, reply, getattr(choice, "finish_reason", None))
            except ProposalError as error:
                error.exchange = exchange
                if self.analysis is not None and not calls and UNPARSED.search(reply):
                    raise ToolCallError(
                        "a tool call came back as text rather than parsed: either the server "
                        "was started without --enable-auto-tool-choice and --tool-call-parser, "
                        "or the model used a call syntax that parser does not recognise",
                        reply,
                        getattr(choice, "finish_reason", None),
                        exchange,
                    ) from error
                raise
            # The exchange is committed only once it has yielded a model: an
            # agent whose call failed, was cancelled, or came back unusable
            # starts its next round from the last transcript that made sense,
            # rather than carrying a dangling prompt or a truncated reply into
            # it. What is committed is all of it -- every snippet the agent ran
            # and everything it read -- because that is the evidence behind the
            # model it went on to write.
            self.transcript += [*exchange, {"role": "assistant", "content": reply}]
            return proposed

        raise BudgetError(
            f"the agent used all {self.tool_budget} tool calls without proposing",
            exchange=exchange,
        )

    def reset(self) -> None:
        """Drop the conversation, keeping only the system prompt."""
        self.transcript[:] = [{"role": "system", "content": self.system()}]
