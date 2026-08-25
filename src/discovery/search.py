"""I.I.D. search: one prompt, N samples, drawn together -- and its agentic twin.

A search is one batch of draws from ``p(model | parent, feedback)`` -- no more.
It is also one request: the server is asked for `n` samples of the same prompt,
which is what the OpenAI `n` parameter and vLLM's batched sampling are for. The
prompt is prefilled once instead of N times, and the draws are as independent as
the sampler makes them.

There is no round loop here. Iterating only means something once something
*selects* among the candidates, and selecting needs a score -- the Evaluator.
The searches that carry state between rounds (SMC, islands) are different
algorithms and belong in their own modules; the conversational machinery they
will want already lives in `ModelProposer`.
"""

import asyncio
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any

from openai import AsyncOpenAI

from discovery.analysis import Analysis
from discovery.model import Model
from discovery.proposer import SYSTEM_PROMPT, ModelProposer, splice


@dataclass(frozen=True)
class Decode:
    """Sampling settings for the batch.

    The OpenAI-compatible fields are sent as themselves. `top_k`, `min_p`,
    `repetition_penalty` and `thinking_token_budget` are vLLM extensions and
    travel in `extra_body`, as does anything in `extra`.

    `thinking` and `thinking_token_budget` are the ones that matter for a
    reasoning model. Left alone it will think for as long as it likes, and
    measured against Qwen3.5-35B-A3B it often thinks past the whole token budget
    and returns nothing at all. `thinking_token_budget` caps the reasoning while
    keeping it; `thinking=False` turns it off outright.

    One `seed` governs the whole request, and vLLM gives no way to seed a sample
    within it, so the samples differ from each other but cannot be addressed
    individually. It does not buy replay either: repeating a seeded request
    against vLLM returns different completions, because continuous batching
    makes the numerics depend on what else is in the batch.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    thinking: bool | None = None
    thinking_token_budget: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    #: sent as themselves
    STANDARD = (
        "temperature", "top_p", "max_tokens", "presence_penalty",
        "frequency_penalty", "stop", "seed",
    )
    #: accepted by vLLM only under `extra_body`
    EXTENSIONS = ("top_k", "min_p", "repetition_penalty", "thinking_token_budget")

    def kwargs(self) -> dict[str, Any]:
        """Completion keyword arguments for the request."""
        out: dict[str, Any] = {
            name: value for name in self.STANDARD if (value := getattr(self, name)) is not None
        }
        body = {name: value for name in self.EXTENSIONS if (value := getattr(self, name)) is not None}
        if self.thinking is not None:
            # Thinking is a chat-template switch rather than a sampling one.
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if body := body | self.extra:
            out["extra_body"] = body
        return out


@dataclass(frozen=True)
class Proposal:
    """What one sample came back with: a model, or why it could not be read."""

    agent: int
    model: Model | None = None
    error: Exception | None = None
    #: every turn behind this proposal, system prompt first, as it was sent. A
    #: model is the answer and the exchange is the working: what the agent
    #: measured, what came back, and what it wrote off the back of it. The
    #: report keeps only the answer, so a caller that wants the working has to
    #: be handed it here -- once the batch returns, an agent's conversation is
    #: either continued or dropped, and nothing else has a reference to it.
    transcript: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.model is not None

    @property
    def reply(self) -> str | None:
        """The text behind a failure, when there was any: a truncated or
        malformed reply is worth reading, not just counting."""
        return getattr(self.error, "reply", None)


@dataclass
class IIDSearch:
    """N independent proposals against one parent model, in one request.

    Configuration only: the search holds no state between calls, so calling it
    twice draws two independent batches rather than continuing anything.
    """

    url: str
    llm_model: str
    agents: int = 8
    decode: Decode = field(default_factory=Decode)
    api_key: str = "EMPTY"
    system_prompt: str = SYSTEM_PROMPT
    #: seconds the batch may take; it is one request, so this bounds the call
    timeout: float | None = None
    client: Any = None
    _owns_client: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.agents < 1:
            raise ValueError(f"need at least one agent, got {self.agents}")
        if self.client is None:
            self.client = AsyncOpenAI(base_url=self.url, api_key=self.api_key)
            self._owns_client = True

    async def propose(
        self, model: Model, feedback: str | None = None, **kwargs: Any
    ) -> tuple[Proposal, ...]:
        """Draw one proposal per agent, returned in sample order.

        `kwargs` override the decode settings for this call. A sample that comes
        back unreadable -- malformed, or truncated before it said anything --
        becomes a `Proposal` carrying the reason, so it costs the batch nothing
        but itself. A failure of the request itself fails the whole batch, since
        there is only the one.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ModelProposer.prompt(model, feedback)},
        ]
        async with asyncio.timeout(self.timeout):
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                n=self.agents,
                **(self.decode.kwargs() | kwargs),
            )

        proposals: list[Proposal] = []
        for agent, choice in enumerate(response.choices):
            reply = choice.message.content or ""
            # Two turns and a reply: the samples share a prompt and differ only
            # in what came back, so each one's whole conversation is the common
            # prefix with its own answer on the end.
            transcript = (*messages, {"role": "assistant", "content": reply})
            try:
                proposed = splice(model, reply, getattr(choice, "finish_reason", None))
            except Exception as error:
                proposals.append(Proposal(agent=agent, error=error, transcript=transcript))
            else:
                proposals.append(Proposal(agent=agent, model=proposed, transcript=transcript))
        return tuple(proposals)

    async def aclose(self) -> None:
        """Close the client, if this search is the one that opened it."""
        if not self._owns_client:
            return
        if close := getattr(self.client, "close", None):
            result = close()
            if isawaitable(result):
                await result

    async def __aenter__(self) -> "IIDSearch":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


@dataclass
class AgentSearch:
    """N agents against one parent model, each in its own conversation.

    The same draw as `IIDSearch`, paid for differently. An agent that uses tools
    has to be asked again after every snippet it runs, and a multi-turn exchange
    cannot be one `n`-sample request, so this is N requests where the i.i.d.
    search is one: the prompt is prefilled per agent rather than once. Prefix
    caching on the server recovers most of that, since the system prompt and the
    parent model are a shared prefix, but the batched search remains the cheaper
    way to draw proposals that do not need to look at anything.

    What is bought for it is the `analysis`. There is no search-level timeout:
    the tool budget bounds how long an agent can go on looking, `max_tokens`
    bounds how long it can go on thinking, and the client's own request timeout
    bounds a server that has stopped answering.

    Each agent keeps its conversation, so calling `propose` twice continues N
    conversations rather than starting them again. `reset` starts them over.
    """

    url: str
    llm_model: str
    agents: int = 8
    decode: Decode = field(default_factory=Decode)
    api_key: str = "EMPTY"
    system_prompt: str = SYSTEM_PROMPT
    #: the data environment offered to every agent, or None for no tools
    analysis: Analysis | None = None
    tool_budget: int = 6
    client: Any = None
    proposers: tuple[ModelProposer, ...] = field(init=False, default=())
    _owns_client: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.agents < 1:
            raise ValueError(f"need at least one agent, got {self.agents}")
        if self.client is None:
            self.client = AsyncOpenAI(base_url=self.url, api_key=self.api_key)
            self._owns_client = True
        # One client, N conversations: the agents are independent in what they
        # say and share only the connection pool they say it down.
        self.proposers = tuple(
            ModelProposer(
                url=self.url,
                llm_model=self.llm_model,
                api_key=self.api_key,
                system_prompt=self.system_prompt,
                analysis=self.analysis,
                tool_budget=self.tool_budget,
                client=self.client,
            )
            for _ in range(self.agents)
        )

    async def propose(
        self, model: Model, feedback: str | None = None, **kwargs: Any
    ) -> tuple[Proposal, ...]:
        """Draw one proposal per agent, returned in agent order.

        Every way one agent can fail -- an unreadable reply, a spent tool
        budget, a request that never came back -- costs that agent and nothing
        else, so a batch is as long as its slowest agent and as good as its best.
        """
        if self.analysis is not None:
            await self.analysis.start()
        settings = self.decode.kwargs() | kwargs

        async def one(agent: int, proposer: ModelProposer) -> Proposal:
            try:
                proposed = await proposer.propose(model, feedback, **settings)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # The conversation as it stands, which for a failure is what
                # the agent had before this round: the turns it just lost are
                # never committed, and travel on the error instead.
                return Proposal(agent=agent, error=error, transcript=tuple(proposer.transcript))
            return Proposal(agent=agent, model=proposed, transcript=tuple(proposer.transcript))

        return tuple(await asyncio.gather(*(one(a, p) for a, p in enumerate(self.proposers))))

    def reset(self) -> None:
        """Start every agent's conversation again."""
        for proposer in self.proposers:
            proposer.reset()

    async def aclose(self) -> None:
        """Close the client, if this search is the one that opened it."""
        if not self._owns_client:
            return
        if close := getattr(self.client, "close", None):
            result = close()
            if isawaitable(result):
                await result

    async def __aenter__(self) -> "AgentSearch":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def models(proposals: tuple[Proposal, ...]) -> tuple[Model, ...]:
    """The models among some proposals, dropping the samples that failed."""
    return tuple(p.model for p in proposals if p.model is not None)
