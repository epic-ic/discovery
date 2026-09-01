"""The agentic arm: the parts that hold it comparable to the i.i.d. one.

Nothing here runs a coding agent. What is worth testing is not that `claude`
answers -- it does, and it costs money to prove -- but the three properties that
make its answer usable as an arm of this experiment: the brief both arms are
given is the same brief, the sandbox cannot contain a scored row or a way to fit,
and nothing an agent writes outside the markers survives into a candidate.
"""

import argparse
import json

import pytest

from discovery.model import Model

from examples.malaria import agentic
from examples.malaria import __main__ as driver
from examples.malaria.initial import BRIEF, STARTS


@pytest.fixture
def settings(tmp_path):
    """The defaults, pointed at a temporary sandbox."""
    parsed = agentic.options(["--sandbox", str(tmp_path / "sandbox"), "--out", str(tmp_path / "out")])
    return parsed


@pytest.fixture
def parent():
    return Model(STARTS[("curves", "griffin")])


def test_the_brief_is_the_one_the_iid_arm_is_given(settings):
    """Both arms are told the same thing, or the comparison is between briefs.

    `agentic.brief` and the driver's own call are two `BRIEF.format(...)` calls,
    and the only way they can disagree is by passing different numbers -- so the
    test is that they do not, for the same settings.
    """
    text = agentic.brief(settings, "a table")
    expected = BRIEF.format(
        table="a table",
        study=settings.n_studies,
        chains=settings.chains,
        warmup=settings.warmup,
        samples=settings.samples,
        minutes=driver.FIT_MINUTES,
        inc_max=agentic.Budget.inc_max,
    )
    assert text == expected
    # every field the template asks for was supplied: an unformatted `{...}`
    # would leave a placeholder where a number belongs.
    assert "{" not in text


def test_the_role_forbids_the_tools_that_leak(settings, tmp_path):
    """Web access and subagents are what the toolset is chosen to exclude."""
    argv = " ".join(agentic.command(settings, _role(tmp_path, settings)))
    assert "WebSearch" not in argv and "WebFetch" not in argv
    assert "Task" not in agentic.TOOLS
    assert "--no-session-persistence" in argv
    assert '--setting-sources' in argv


def _role(tmp_path, settings):
    path = tmp_path / "a00"
    path.mkdir(exist_ok=True)
    (path / "ROLE.md").write_text(agentic.role(settings))
    return path


def test_the_bench_can_run_the_model_but_cannot_fit():
    """`dmeq` so the mechanism runs as written; nothing that could fit it.

    The first half stops an agent reimplementing the incumbent in order to look
    at it, and then reasoning about its replica. The second half is what keeps
    the draw a draw: `numpyro` and `msinf` need jax, and jax is what a sandbox
    would need to run a chain.
    """
    assert set(agentic.BENCH_PACKAGES) == {"numpy", "pandas"}
    assert set(agentic.BENCH_MODULES) == {"dmeq"}
    assert not {"numpyro", "msinf", "jax"} & set(agentic.BENCH_PACKAGES + agentic.BENCH_MODULES)


def test_the_agent_is_told_the_deadline_that_will_kill_it(settings):
    """The stated budget is the enforced one, or the number is a lie."""
    settings.session_timeout = 1800.0
    text = agentic.role(settings)
    assert "30 minutes" in text
    # the sentinel is spent, and the worked example's braces survive it
    assert "@@" not in text
    assert 'ns = {"__name__": "candidate"}' in text


def test_a_proposal_splices_onto_the_parent_and_nothing_else_does(tmp_path, parent):
    """What the agent wrote outside the markers does not reach the candidate."""
    path = tmp_path / "a00"
    path.mkdir()
    (path / "model.py").write_text(parent.source + "\nSABOTAGE = 1\n")
    (path / "proposal.py").write_text(
        "# @discovery:begin parameters\n"
        "params = dict(default_parameters())\n"
        "params.update(b0=sample('b0', dist.Beta(1.3, 1.3)))\n"
        "return params\n"
        "# @discovery:end\n"
    )
    candidate, note = agentic.validate(parent, path)
    assert note["tampered"] is True
    assert "SABOTAGE" not in candidate.source
    assert note["blocks"] == ["parameters"]
    # the untouched block came from the parent, verbatim
    blocks = {block.name: block.body for block in candidate.blocks}
    assert blocks["curves"] == {b.name: b.body for b in parent.blocks}["curves"]


@pytest.mark.parametrize(
    "written, message",
    [
        (None, "no proposal.py"),
        ("nothing marked at all\n", "no @discovery blocks"),
        ("# @discovery:begin nonsense\nx = 1\n# @discovery:end\n", "unknown block"),
        ("# @discovery:begin parameters\nreturn (\n# @discovery:end\n", "does not parse"),
    ],
)
def test_a_session_that_produced_nothing_usable_says_why(tmp_path, parent, written, message):
    """Every way an agent can fail costs that agent and names itself."""
    path = tmp_path / "a00"
    path.mkdir()
    (path / "model.py").write_text(parent.source)
    if written is not None:
        (path / "proposal.py").write_text(written)
    with pytest.raises(agentic.ProposalError, match=message):
        agentic.validate(parent, path)


def test_a_proposal_that_changes_nothing_is_not_a_proposal(tmp_path, parent):
    """`splice` would accept it; an arm measuring proposals should not.

    The i.i.d. system prompt asks for a genuine change and a sample that ignored
    it comes back as a duplicate of its parent. Here it would be scored as a
    candidate, and the parent is already row zero.
    """
    path = tmp_path / "a00"
    path.mkdir()
    (path / "model.py").write_text(parent.source)
    bodies = {block.name: block.body for block in parent.blocks}
    (path / "proposal.py").write_text(
        "\n".join(block.render() for block in parent.blocks)
    )
    assert bodies  # the parent has blocks to restate
    with pytest.raises(agentic.ProposalError, match="identical"):
        agentic.validate(parent, path)


def test_a_round_reads_back_through_the_drivers_own_loader(tmp_path, settings, parent):
    """`collect` writes what `__main__ --proposals` reads, or the arm is unfittable."""
    paths = []
    for agent in range(2):
        path = tmp_path / "sandbox" / f"a{agent:02d}"
        path.mkdir(parents=True)
        (path / "model.py").write_text(parent.source)
        if agent == 0:
            (path / "proposal.py").write_text(
                "# @discovery:begin parameters\n"
                "params = dict(default_parameters())\n"
                "params.update(b0=sample('b0', dist.Beta(1.3, 1.3)))\n"
                "return params\n"
                "# @discovery:end\n"
            )
        paths.append(path)

    out = tmp_path / "out"
    kept = agentic.collect(settings, paths, out, sessions=[])
    assert kept == 1

    candidates, origin, meta = driver.load(out / "proposals")
    assert len(candidates) == 2
    assert candidates[0].source == parent.source, "candidate 0 is the parent, unmodified"
    assert origin == {0: "griffin", 1: "griffin"}
    assert meta["target"] == "curves" and meta["start"] == "griffin"

    records = [json.loads(line) for line in (out / "transcripts.jsonl").read_text().splitlines()]
    assert [r["ok"] for r in records] == [True, False]
    assert [r["candidate"] for r in records] == [1, None]
    failures = [json.loads(line) for line in (out / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 1 and "no proposal.py" in failures[0]["error"]


def test_usage_is_read_off_the_session_stream(tmp_path):
    """What a draw cost, which is the comparison the i.i.d. arm cannot yet make."""
    log = tmp_path / "a00.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"type": "system", "subtype": "init"},
                {
                    "type": "assistant",
                    "message": {"role": "assistant", "model": "claude-opus-5-20260514",
                                "content": [{"type": "text", "text": "hello"}]},
                },
                {"type": "result", "subtype": "success", "num_turns": 2, "duration_ms": 10,
                 "total_cost_usd": 0.5,
                 "usage": {"input_tokens": 3, "output_tokens": 4}},
            ]
        )
    )
    turns, usage = agentic.conversation(log)
    assert turns == [{"role": "assistant", "content": "hello"}]
    assert usage["input_tokens"] == 3 and usage["output_tokens"] == 4
    # the id the request resolved to, not the alias it was asked for: an alias
    # moves and a round has to record what actually drew it.
    assert usage["model"] == "claude-opus-5-20260514"
