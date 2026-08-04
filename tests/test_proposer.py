"""Proposer tests.

These are offline. The proposer takes an OpenAI-compatible async client as a
dependency, so every test here injects `FakeClient` and asserts on our own
logic: prompt assembly, splicing a reply back into a model, transcript growth
and error paths. Nothing imports `openai` or opens a socket.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from discovery.model import Model
from discovery.proposer import (
    SYSTEM_PROMPT,
    BudgetError,
    ModelProposer,
    ProposalError,
    ToolCallError,
)

SOURCE = """\
def model(x, y=None):
    # @discovery:begin prior
    slope = sample("slope", Normal(0.0, 1.0))
    # @discovery:end
    return sample("y", Normal(slope * x, 1.0), obs=y)
"""

REPLY = """\
A heavier tail should help.

```python
    # @discovery:begin prior
    slope = sample("slope", StudentT(3.0, 0.0, 1.0))
    # @discovery:end
```
"""


@dataclass
class ToolCall:
    """One tool call, shaped like the client's object rather than the wire."""

    code: str
    id: str = "call-1"
    name: str = "run_python"

    @property
    def function(self):
        return type("Function", (), {"name": self.name, "arguments": self.arguments})()

    @property
    def arguments(self) -> str:
        return json.dumps({"code": self.code})


@dataclass
class RawArguments(ToolCall):
    """A call whose arguments are not what the schema asked for."""

    raw: str = "not json"

    @property
    def arguments(self) -> str:
        return self.raw


@dataclass
class FakeClient:
    """Stands in for an OpenAI-compatible async client, recording what it was sent.

    A reply is either the text of an answer or a list of `ToolCall`, which lets
    one fake drive both the single-shot proposer and the tool loop.
    """

    replies: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    finish_reason: str = "stop"

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        tool_calls = reply if isinstance(reply, list) else None
        message = type(
            "Message",
            (),
            {"content": "" if tool_calls else reply, "tool_calls": tool_calls},
        )()
        choice = type("Choice", (), {"message": message, "finish_reason": self.finish_reason})()
        return type("Completion", (), {"choices": [choice]})()


@dataclass
class FakeAnalysis:
    """Stands in for the worker pool: records snippets, returns canned output."""

    description: str = "It holds `values`."
    max_output: int = 4000
    outputs: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)

    async def run(self, code: str) -> str:
        self.ran.append(code)
        return self.outputs.pop(0) if self.outputs else "42"

    async def start(self):
        return self


@pytest.fixture
def model():
    return Model(SOURCE)


def proposer(*replies: str) -> ModelProposer:
    return ModelProposer(
        url="http://localhost:8000/v1",
        llm_model="a-model",
        client=FakeClient(replies=list(replies)),
    )


def test_transcript_starts_with_the_system_prompt():
    assert proposer().transcript == [{"role": "system", "content": SYSTEM_PROMPT}]


def test_prompt_includes_the_source_and_block_names(model):
    prompt = proposer().prompt(model)
    assert 'slope = sample("slope", Normal(0.0, 1.0))' in prompt
    assert "Rewritable regions: prior." in prompt
    assert "Results for this model" not in prompt


def test_prompt_includes_feedback_when_given(model):
    prompt = proposer().prompt(model, feedback="log likelihood: -123.4")
    assert "Results for this model:" in prompt
    assert "log likelihood: -123.4" in prompt


async def test_propose_splices_the_reply_into_the_model(model):
    proposed = await proposer(REPLY).propose(model)
    assert '    slope = sample("slope", StudentT(3.0, 0.0, 1.0))' in proposed.source
    assert "Normal(0.0, 1.0)" not in proposed.source
    # the fixed scaffolding is preserved verbatim
    assert "def model(x, y=None):" in proposed.source
    assert '    return sample("y", Normal(slope * x, 1.0), obs=y)' in proposed.source


async def test_propose_sends_the_configured_model_and_messages(model):
    p = proposer(REPLY)
    await p.propose(model)
    (call,) = p.client.calls
    assert call["model"] == "a-model"
    assert [message["role"] for message in call["messages"]] == ["system", "user"]


async def test_successive_proposals_see_the_earlier_turns(model):
    second = REPLY.replace("StudentT(3.0, 0.0, 1.0)", "Laplace(0.0, 1.0)")
    p = proposer(REPLY, second)
    first = await p.propose(model, feedback="score: 1.0")
    await p.propose(first, feedback="score: 2.0")
    _, call = p.client.calls
    roles = [message["role"] for message in call["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "score: 1.0" in call["messages"][1]["content"]
    assert "StudentT" in call["messages"][3]["content"]


async def test_a_reply_restating_the_whole_file_is_handled(model):
    """Observed behaviour: models often echo the entire file back.

    Only the marked bodies are taken, so any scaffolding the reply invents or
    mangles outside the markers is discarded rather than spliced in.
    """
    reply = (
        "```python\n"
        "import wrong_module\n\n"
        "def model(x, y=None):\n"
        "    # @discovery:begin prior\n"
        '    slope = sample("slope", Laplace(0.0, 1.0))\n'
        "    # @discovery:end\n"
        "    return 42\n"
        "```"
    )
    proposed = await proposer(reply).propose(model)
    assert 'slope = sample("slope", Laplace(0.0, 1.0))' in proposed.source
    assert "wrong_module" not in proposed.source
    assert "return 42" not in proposed.source
    assert '    return sample("y", Normal(slope * x, 1.0), obs=y)' in proposed.source


async def test_extra_kwargs_reach_the_client(model):
    p = proposer(REPLY)
    await p.propose(model, max_tokens=256)
    assert p.client.calls[0]["max_tokens"] == 256


async def test_a_reply_without_blocks_is_an_error(model):
    with pytest.raises(ValueError, match="no @discovery blocks"):
        await proposer("I have no suggestions.").propose(model)


async def test_a_truncated_reply_says_so(model):
    """Observed against vLLM: a reasoning model can spend its whole token
    budget thinking and return empty content. That is a max_tokens problem, and
    the error has to say so rather than blame the reply."""
    p = ModelProposer(
        url="http://localhost:8000/v1",
        llm_model="a-model",
        client=FakeClient(replies=[""], finish_reason="length"),
    )
    with pytest.raises(ProposalError, match="max_tokens") as caught:
        await p.propose(model)
    assert caught.value.finish_reason == "length"


async def test_a_failed_proposal_keeps_the_reply_for_inspection(model):
    with pytest.raises(ProposalError) as caught:
        await proposer("I have no suggestions.").propose(model)
    assert caught.value.reply == "I have no suggestions."


async def test_an_unusable_reply_is_not_kept_in_the_transcript(model):
    """The next round should start from the last exchange that made sense."""
    p = proposer(REPLY, "I have no suggestions.", REPLY)
    await p.propose(model)
    with pytest.raises(ProposalError):
        await p.propose(model)
    await p.propose(model)
    assert [m["role"] for m in p.transcript] == [
        "system", "user", "assistant", "user", "assistant",
    ]
    assert all("no suggestions" not in m["content"] for m in p.transcript)


async def test_a_reply_naming_an_unknown_block_is_an_error(model):
    reply = "```python\n# @discovery:begin nonsense\nx = 1\n# @discovery:end\n```"
    with pytest.raises(KeyError):
        await proposer(reply).propose(model)


async def test_temperature_defaults_but_kwargs_win(model):
    p = proposer(REPLY, REPLY)
    await p.propose(model)
    await p.propose(model, temperature=0.2)
    assert [call["temperature"] for call in p.client.calls] == [1.0, 0.2]


async def test_a_failed_call_leaves_the_transcript_unchanged(model):
    """A cancelled or erroring agent must not orphan a user turn.

    The next round would otherwise send two user messages in a row, with the
    dead one still describing a model the agent never answered for.
    """

    class Boom(FakeClient):
        async def create(self, **kwargs):
            raise RuntimeError("connection reset")

    p = ModelProposer(url="http://localhost:8000/v1", llm_model="a-model", client=Boom(replies=[]))
    with pytest.raises(RuntimeError):
        await p.propose(model)
    assert p.transcript == [{"role": "system", "content": SYSTEM_PROMPT}]


async def test_reset_keeps_only_the_system_prompt(model):
    p = proposer(REPLY)
    await p.propose(model)
    p.reset()
    assert p.transcript == [{"role": "system", "content": SYSTEM_PROMPT}]


# --------------------------------------------------------------------------
# with an analysis environment: the proposal becomes a short tool loop
# --------------------------------------------------------------------------


def analyst(*replies: Any, analysis: FakeAnalysis | None = None, **kwargs) -> ModelProposer:
    return ModelProposer(
        url="http://localhost:8000/v1",
        llm_model="a-model",
        client=FakeClient(replies=list(replies)),
        analysis=analysis or FakeAnalysis(),
        **kwargs,
    )


def test_the_tool_section_is_added_to_the_system_prompt():
    p = analyst(tool_budget=3)
    system = p.transcript[0]["content"]
    assert system.startswith(SYSTEM_PROMPT)
    assert "run_python" in system
    assert "It holds `values`." in system
    assert "at most 3 times" in system
    # the environment cannot fit anything, and the agent has to be told
    assert "numpyro" in system


def test_a_proposer_without_an_analysis_is_unchanged():
    assert proposer().transcript[0]["content"] == SYSTEM_PROMPT


async def test_no_tools_are_offered_without_an_analysis(model):
    p = proposer(REPLY)
    await p.propose(model)
    assert "tools" not in p.client.calls[0]


async def test_the_tool_is_offered_when_there_is_an_analysis(model):
    p = analyst(REPLY)
    await p.propose(model)
    (tool,) = p.client.calls[0]["tools"]
    assert tool["function"]["name"] == "run_python"


async def test_a_snippet_runs_and_its_output_comes_back_before_the_proposal(model):
    analysis = FakeAnalysis(outputs=["mean 9412.0"])
    p = analyst([ToolCall("print(births.mean())")], REPLY, analysis=analysis)
    proposed = await p.propose(model)

    assert analysis.ran == ["print(births.mean())"]
    assert "StudentT" in proposed.source
    # the second request carried the call and its result
    messages = p.client.calls[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "run_python"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call-1"
    assert messages[3]["content"].startswith("mean 9412.0")


async def test_the_whole_exchange_is_kept_in_the_transcript(model):
    """What the agent measured is why it proposed what it did."""
    p = analyst([ToolCall("print(1)")], REPLY)
    await p.propose(model)
    assert [m["role"] for m in p.transcript] == ["system", "user", "assistant", "tool", "assistant"]
    assert any("42" in (m.get("content") or "") for m in p.transcript)


async def test_several_calls_in_one_turn_are_all_run(model):
    analysis = FakeAnalysis(outputs=["a", "b"])
    p = analyst(
        [ToolCall("first", id="c1"), ToolCall("second", id="c2")], REPLY, analysis=analysis
    )
    await p.propose(model)
    assert analysis.ran == ["first", "second"]
    ids = [m["tool_call_id"] for m in p.client.calls[1]["messages"] if m["role"] == "tool"]
    assert ids == ["c1", "c2"]


async def test_an_agent_may_look_several_times(model):
    p = analyst([ToolCall("one")], [ToolCall("two")], [ToolCall("three")], REPLY)
    proposed = await p.propose(model)
    assert p.analysis.ran == ["one", "two", "three"]
    assert "StudentT" in proposed.source


async def test_the_tool_budget_is_enforced(model):
    p = analyst(*[[ToolCall("again")] for _ in range(5)], tool_budget=2)
    with pytest.raises(BudgetError, match="all 2 tool calls"):
        await p.propose(model)
    assert p.analysis.ran == ["again", "again"]


async def test_a_spent_budget_leaves_the_transcript_unchanged(model):
    p = analyst(*[[ToolCall("again")] for _ in range(5)], tool_budget=1)
    with pytest.raises(BudgetError):
        await p.propose(model)
    assert [m["role"] for m in p.transcript] == ["system"]


async def test_arguments_that_are_not_valid_are_answered_not_raised(model):
    """A malformed call costs the agent one call, not the proposal."""
    p = analyst([RawArguments(code="", raw="{oops")], REPLY)
    proposed = await p.propose(model)
    (result,) = [m for m in p.client.calls[1]["messages"] if m["role"] == "tool"]
    assert "not an object with a 'code' string" in result["content"]
    assert "StudentT" in proposed.source


async def test_a_call_to_an_unknown_tool_is_answered_not_raised(model):
    p = analyst([ToolCall("print(1)", name="run_r")], REPLY)
    await p.propose(model)
    (result,) = [m for m in p.client.calls[1]["messages"] if m["role"] == "tool"]
    assert "no tool called 'run_r'" in result["content"]


async def test_an_unparsed_tool_call_names_the_server_flags(model):
    """vLLM only fills `tool_calls` when it was started with a tool parser.
    Without one the call arrives as text and looks like a disobedient model."""
    reply = '<tool_call>\n{"name": "run_python", "arguments": {"code": "print(1)"}}\n</tool_call>'
    with pytest.raises(ToolCallError, match="came back as text") as caught:
        await analyst(reply).propose(model)
    assert caught.value.reply == reply


async def test_an_ordinary_bad_reply_is_still_an_ordinary_error(model):
    with pytest.raises(ProposalError, match="no @discovery blocks"):
        await analyst("I have no suggestions.").propose(model)


async def test_reset_keeps_the_tool_section(model):
    p = analyst([ToolCall("one")], REPLY)
    await p.propose(model)
    p.reset()
    assert len(p.transcript) == 1
    assert "run_python" in p.transcript[0]["content"]


async def test_a_tool_call_that_only_closes_is_still_an_unparsed_call(model):
    """Observed against Qwen3.5-35B on a correctly configured server: a reply
    starts as prose and ends `</parameter></function></tool_call>`, because the
    model switched call syntax part-way to one the parser does not know."""
    reply = 'Here are the coefficients:\nprint(x)\n</parameter>\n</function>\n</tool_call>'
    with pytest.raises(ToolCallError, match="came back as text"):
        await analyst(reply).propose(model)


async def test_an_empty_reply_blames_the_reasoning_budget(model):
    """A reasoning parser can take the whole completion and still finish
    cleanly, so `length` is not the only way to get nothing back."""
    with pytest.raises(ProposalError, match="thinking_token_budget") as caught:
        await proposer("").propose(model)
    assert caught.value.finish_reason == "stop"


async def test_a_failed_proposal_carries_what_the_agent_had_run(model):
    """The transcript keeps nothing from a failed agent, so the error has to."""
    p = analyst([ToolCall("print(df.shape)")], "no blocks here")
    with pytest.raises(ProposalError) as caught:
        await p.propose(model)
    roles = [m["role"] for m in caught.value.exchange]
    assert roles == ["user", "assistant", "tool"]
    assert caught.value.exchange[2]["content"].startswith("42")


async def test_a_spent_budget_carries_the_exchange_too(model):
    p = analyst(*[[ToolCall("again")] for _ in range(5)], tool_budget=2)
    with pytest.raises(BudgetError) as caught:
        await p.propose(model)
    assert sum(m["role"] == "tool" for m in caught.value.exchange) == 2


async def test_the_agent_is_told_how_many_calls_it_has_left(model):
    """It cannot count its own, and a budget it cannot see is one it overruns."""
    p = analyst([ToolCall("one")], [ToolCall("two")], REPLY, tool_budget=3)
    await p.propose(model)
    notes = [m["content"] for m in p.transcript if m["role"] == "tool"]
    assert "[2 tool calls left.]" in notes[0]
    assert "[1 tool call left.]" in notes[1]


async def test_the_last_call_says_to_answer_now(model):
    p = analyst([ToolCall("one")], REPLY, tool_budget=1)
    await p.propose(model)
    (note,) = [m["content"] for m in p.transcript if m["role"] == "tool"]
    assert "No tool calls left" in note
    assert "fenced code block" in note


async def test_only_the_last_result_of_a_turn_carries_the_count(model):
    p = analyst([ToolCall("a", id="c1"), ToolCall("b", id="c2")], REPLY, tool_budget=4)
    await p.propose(model)
    first, second = [m["content"] for m in p.transcript if m["role"] == "tool"]
    assert "left." not in first
    assert "[3 tool calls left.]" in second
