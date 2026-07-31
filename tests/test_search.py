"""I.I.D. search tests.

Offline, like the proposer's. `FakeClient` stands in for the inference server,
returning `n` samples for the one request the search makes, which lets these
tests assert what matters once a batch is drawn in one go: that the request asks
for the right number of samples with the right decode settings, that each sample
is read independently, and that one unreadable sample costs the batch nothing
but itself.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from discovery.model import Model
from discovery.search import Decode, IIDSearch, models

SOURCE = """\
def model(x, y=None):
    # @discovery:begin prior
    slope = sample("slope", Normal(0.0, 1.0))
    # @discovery:end
    return sample("y", Normal(slope * x, 1.0), obs=y)
"""


def reply(body: str) -> str:
    return (
        "Try this.\n\n```python\n    # @discovery:begin prior\n"
        f"    slope = sample(\"slope\", {body})\n"
        "    # @discovery:end\n```\n"
    )


@dataclass
class FakeClient:
    """Returns `n` choices for one request.

    `handler` is called with the sample's index and the request kwargs, and
    returns that sample's reply text, so a test can make one sample of the batch
    unreadable or truncated.
    """

    handler: Any = lambda agent, kwargs: reply(f"StudentT({agent}.0, 0.0, 1.0)")
    calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    closed: bool = False

    async def close(self) -> None:
        self.closed = True

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        choices = []
        for agent in range(kwargs.get("n", 1)):
            message = type("Message", (), {"content": self.handler(agent, kwargs)})()
            choices.append(type("Choice", (), {"message": message, "finish_reason": self.finish_reason})())
        return type("Completion", (), {"choices": choices})()


@pytest.fixture
def model():
    return Model(SOURCE)


def search(**kwargs) -> IIDSearch:
    kwargs.setdefault("agents", 4)
    client = kwargs.pop("client", None) or FakeClient(
        **{k: kwargs.pop(k) for k in ("handler", "finish_reason") if k in kwargs}
    )
    return IIDSearch(
        url="http://localhost:8000/v1", llm_model="a-model", client=client, **kwargs
    )


async def test_a_batch_returns_one_proposal_per_agent(model):
    s = search(agents=5)
    proposals = await s.propose(model)
    assert len(proposals) == 5
    assert [p.agent for p in proposals] == [0, 1, 2, 3, 4]
    assert all(p.ok for p in proposals)
    assert all("StudentT" in m.source for m in models(proposals))


async def test_the_batch_is_one_request_asking_for_n_samples(model):
    """The whole point of the switch: one prompt, prefilled once, N draws."""
    s = search(agents=16)
    await s.propose(model)
    (call,) = s.client.calls
    assert call["n"] == 16
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert "Normal(0.0, 1.0)" in call["messages"][1]["content"]


async def test_every_draw_sees_only_the_parent(model):
    """I.i.d.: one prompt, and nothing carried in from an earlier call."""
    s = search(agents=3)
    await s.propose(model)
    await s.propose(model)
    first, second = s.client.calls
    assert first["messages"] == second["messages"]
    assert all("StudentT" not in m["content"] for m in second["messages"])


async def test_the_feedback_reaches_the_prompt(model):
    s = search(agents=3)
    await s.propose(model, feedback="log likelihood: -1843.2")
    assert "log likelihood: -1843.2" in s.client.calls[0]["messages"][-1]["content"]


async def test_decode_settings_reach_the_client(model):
    s = search(agents=2)
    s.decode = Decode(
        temperature=0.7, top_p=0.9, top_k=40, max_tokens=512, stop=["\n\n\n"], seed=99
    )
    await s.propose(model)
    call = s.client.calls[0]
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.9
    assert call["max_tokens"] == 512
    assert call["stop"] == ["\n\n\n"]
    assert call["seed"] == 99
    assert call["extra_body"] == {"top_k": 40}  # a vLLM extension, not a standard field


async def test_the_thinking_budget_is_a_vllm_extension(model):
    s = search(agents=2)
    s.decode = Decode(thinking_token_budget=256)
    await s.propose(model)
    assert s.client.calls[0]["extra_body"] == {"thinking_token_budget": 256}


async def test_thinking_can_be_turned_off(model):
    """Not a sampling setting: it is a switch in the chat template."""
    s = search(agents=2)
    s.decode = Decode(thinking=False)
    await s.propose(model)
    body = s.client.calls[0]["extra_body"]
    assert body == {"chat_template_kwargs": {"enable_thinking": False}}


async def test_unset_decode_settings_are_not_sent(model):
    s = search(agents=1)
    await s.propose(model)
    call = s.client.calls[0]
    assert "max_tokens" not in call
    assert "extra_body" not in call
    assert "seed" not in call


async def test_call_kwargs_override_the_decode_settings(model):
    s = search(agents=2)
    s.decode = Decode(temperature=1.0)
    await s.propose(model, temperature=0.0)
    assert s.client.calls[0]["temperature"] == 0.0


async def test_one_unreadable_sample_does_not_sink_the_batch(model):
    def handler(agent, kwargs):
        if agent == 1:
            return "I have no suggestions."  # no blocks
        return reply("StudentT(3.0, 0.0, 1.0)")

    s = search(agents=4, handler=handler)
    proposals = await s.propose(model)
    assert len(models(proposals)) == 3
    (failed,) = [p for p in proposals if not p.ok]
    assert failed.agent == 1
    assert isinstance(failed.error, ValueError)
    assert failed.reply == "I have no suggestions."  # inspectable, not just counted


async def test_a_sample_truncated_before_it_spoke_says_so(model):
    """The failure that dominates in practice: the model thinks past its budget
    and returns nothing, which is a decode setting to fix."""
    s = search(agents=2, handler=lambda agent, kwargs: "", finish_reason="length")
    proposals = await s.propose(model)
    assert not any(p.ok for p in proposals)
    assert all("max_tokens" in str(p.error) for p in proposals)


async def test_a_failed_request_fails_the_batch(model):
    """There is only the one request, so its failure is the batch's."""

    class Boom(FakeClient):
        async def create(self, **kwargs):
            raise RuntimeError("connection reset")

    s = search(agents=4, client=Boom())
    with pytest.raises(RuntimeError):
        await s.propose(model)


async def test_the_batch_times_out(model):
    class Slow(FakeClient):
        async def create(self, **kwargs):
            await asyncio.sleep(30)

    s = search(agents=4, client=Slow(), timeout=0.05)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(s.propose(model), timeout=5)


async def test_an_injected_client_is_not_closed(model):
    """The search only closes what it opened: the caller may still be using it."""
    s = search(agents=1)
    await s.aclose()
    assert not s.client.closed


def test_at_least_one_agent_is_required():
    with pytest.raises(ValueError, match="at least one agent"):
        search(agents=0)


def test_the_search_offers_only_one_operation():
    """A batch of i.i.d. draws is all this is; iterating needs an evaluator to
    select, and that belongs to a different search."""
    operations = {
        name
        for name, value in vars(IIDSearch).items()
        if callable(value) and not name.startswith("_")
    }
    assert operations == {"propose", "aclose"}
