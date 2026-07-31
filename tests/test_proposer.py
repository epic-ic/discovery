"""Proposer tests.

These are offline. The proposer takes an OpenAI-compatible client as a
dependency, so every test here injects `FakeClient` and asserts on our own
logic: prompt assembly, splicing a reply back into a model, transcript growth
and error paths. Nothing imports `openai` or opens a socket.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from discovery.model import Model
from discovery.proposer import SYSTEM_PROMPT, ModelProposer

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
class FakeClient:
    """Stands in for an OpenAI-compatible client, recording what it was sent."""

    replies: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": self.replies.pop(0)})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


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


def test_propose_splices_the_reply_into_the_model(model):
    proposed = proposer(REPLY).propose(model)
    assert '    slope = sample("slope", StudentT(3.0, 0.0, 1.0))' in proposed.source
    assert "Normal(0.0, 1.0)" not in proposed.source
    # the fixed scaffolding is preserved verbatim
    assert "def model(x, y=None):" in proposed.source
    assert '    return sample("y", Normal(slope * x, 1.0), obs=y)' in proposed.source


def test_propose_sends_the_configured_model_and_messages(model):
    p = proposer(REPLY)
    p.propose(model)
    (call,) = p.client.calls
    assert call["model"] == "a-model"
    assert [message["role"] for message in call["messages"]] == ["system", "user"]


def test_successive_proposals_see_the_earlier_turns(model):
    second = REPLY.replace("StudentT(3.0, 0.0, 1.0)", "Laplace(0.0, 1.0)")
    p = proposer(REPLY, second)
    first = p.propose(model, feedback="score: 1.0")
    p.propose(first, feedback="score: 2.0")
    _, call = p.client.calls
    roles = [message["role"] for message in call["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "score: 1.0" in call["messages"][1]["content"]
    assert "StudentT" in call["messages"][3]["content"]


def test_a_reply_restating_the_whole_file_is_handled(model):
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
    proposed = proposer(reply).propose(model)
    assert 'slope = sample("slope", Laplace(0.0, 1.0))' in proposed.source
    assert "wrong_module" not in proposed.source
    assert "return 42" not in proposed.source
    assert '    return sample("y", Normal(slope * x, 1.0), obs=y)' in proposed.source


def test_extra_kwargs_reach_the_client(model):
    p = proposer(REPLY)
    p.propose(model, max_tokens=256)
    assert p.client.calls[0]["max_tokens"] == 256


def test_a_reply_without_blocks_is_an_error(model):
    with pytest.raises(ValueError, match="no @discovery blocks"):
        proposer("I have no suggestions.").propose(model)


def test_a_reply_naming_an_unknown_block_is_an_error(model):
    reply = "```python\n# @discovery:begin nonsense\nx = 1\n# @discovery:end\n```"
    with pytest.raises(KeyError):
        proposer(reply).propose(model)


def test_reset_keeps_only_the_system_prompt(model):
    p = proposer(REPLY)
    p.propose(model)
    p.reset()
    assert p.transcript == [{"role": "system", "content": SYSTEM_PROMPT}]
