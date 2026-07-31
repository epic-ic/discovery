"""Proposes new models by asking an LLM to rewrite a model's marked blocks.

The client is asynchronous: a search runs many proposers at once, and they
spend nearly all of their time waiting on the inference server.
"""

from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

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


class ProposalError(ValueError):
    """A reply that could not be turned into a model.

    Carries the reply itself, so a failed agent can be inspected rather than
    just counted.
    """

    def __init__(self, message: str, reply: str = "", finish_reason: str | None = None) -> None:
        super().__init__(message)
        self.reply = reply
        self.finish_reason = finish_reason


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
        raise ProposalError("proposal contained no @discovery blocks", reply, finish_reason)
    return model.substitute(bodies)


@dataclass
class ModelProposer:
    """Stores the LLM url and the transcript, and creates the prompt.

    The transcript is the running conversation: it is what makes successive
    proposals aware of what has already been tried and how it scored. Only a
    search that chains exercises it -- an i.i.d. search uses each proposer for
    exactly one call, so every draw starts from the system prompt alone.
    """

    url: str
    llm_model: str
    api_key: str = "EMPTY"
    temperature: float = 1.0
    system_prompt: str = SYSTEM_PROMPT
    client: Any = None
    transcript: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = AsyncOpenAI(base_url=self.url, api_key=self.api_key)
        if not self.transcript:
            self.transcript.append({"role": "system", "content": self.system_prompt})

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
        """Ask for one new model, appending both turns to the transcript.

        Sampling settings given here override this proposer's defaults.
        """
        settings: dict[str, Any] = {"temperature": self.temperature} | kwargs
        prompt = {"role": "user", "content": self.prompt(model, feedback)}
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[*self.transcript, prompt],
            **settings,
        )
        choice = response.choices[0]
        reply = choice.message.content or ""
        proposed = splice(model, reply, getattr(choice, "finish_reason", None))
        # The exchange is committed only once it has yielded a model: an agent
        # whose call failed, was cancelled, or came back unusable starts its
        # next round from the last transcript that made sense, rather than
        # carrying a dangling prompt or a truncated reply into it.
        self.transcript += [prompt, {"role": "assistant", "content": reply}]
        return proposed

    def reset(self) -> None:
        """Drop the conversation, keeping only the system prompt."""
        self.transcript[:] = [{"role": "system", "content": self.system_prompt}]
