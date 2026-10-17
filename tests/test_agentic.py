"""The agentic arm: what it is given, and what it must never be given.

Nothing here runs a coding agent. What is worth testing is not that `claude`
answers -- it does, and it costs money to prove -- but the properties that make
its answer usable: the brief is the brief the fit is run under, the sandbox
holds no scored row and no msinf, its gate check is the runner's gate check, and
nothing an agent writes outside the markers survives into a candidate.
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


def test_the_bench_runs_the_model_and_the_gates_but_holds_no_msinf():
    """`dmeq` so the mechanism runs as written, jax and numpyro so the gates and an
    agent's own fits run; never msinf, which would point at the repository and its
    whole table, held-out rows included."""
    assert {"numpy", "pandas", "jax", "numpyro"} <= set(agentic.BENCH_PACKAGES)
    assert set(agentic.BENCH_MODULES) == {"dmeq"}
    assert set(agentic.BENCH_FILES) == {"gates.py", "discovery/__init__.py", "discovery/model.py"}
    shipped = set(agentic.BENCH_PACKAGES) | set(agentic.BENCH_MODULES) | set(agentic.BENCH_FILES)
    assert not any("msinf" in name for name in shipped)


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


@pytest.fixture
def local(tmp_path):
    """An OpenCode round against a local server, without asking the server anything."""
    parsed = agentic.options([
        "--cli", "opencode", "--model", "glm-5.3-flash", "--url", "http://127.0.0.1:8001/v1",
        "--context-limit", "131072", "--sandbox", str(tmp_path / "sandbox"), "--out", str(tmp_path / "out"),
    ])
    return parsed


def test_opencode_is_configured_in_the_v2_shape_and_denies_what_claude_is_not_given(local):
    """The keys V2 reads -- `providers`, `package`, `settings`, `permissions` -- and
    the rules that stand in for `--tools`, since `--auto` approves whatever only asks."""
    config = agentic.opencode_config(local)
    assert config["model"] == "local/glm-5.3-flash"
    provider = config["providers"]["local"]
    assert provider["package"] == "@opencode/ai/providers/openai-compatible"
    assert provider["settings"]["baseURL"] == "http://127.0.0.1:8001/v1"
    assert provider["env"] == ["LLM_API_KEY"]
    model = provider["models"]["glm-5.3-flash"]
    assert model["limit"] == {"context": 131072, "output": 16384}
    assert model["capabilities"]["tools"] is True
    denied = {rule["action"] for rule in config["permissions"] if rule["effect"] == "deny"}
    assert {"external_directory", "webfetch", "websearch", "subagent", "skill", "question"} <= denied
    assert not [rule for rule in config["permissions"] if rule["effect"] != "deny"]


def test_opencode_reads_nothing_of_this_machine(local, tmp_path, monkeypatch):
    """Its own home, config and data beside the sandbox, the role as its only
    AGENTS.md, and no project instructions from above the sandbox."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/somewhere")
    monkeypatch.setenv("DISCOVERY_URL", "http://elsewhere/v1")
    sandbox = tmp_path / "sandbox" / "a00"
    sandbox.mkdir(parents=True)
    env = agentic.environment(local, sandbox)
    home = tmp_path / "sandbox" / "cfg-a00"
    assert env["HOME"] == str(home)
    assert env["XDG_CONFIG_HOME"] == str(home / ".config")
    assert env["XDG_DATA_HOME"] == str(home / ".local" / "share")
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    # OpenCode takes its workspace from PWD, not from the process's cwd
    assert env["PWD"] == str(sandbox) and "OLDPWD" not in env
    assert not any(key.startswith(("CLAUDE_", "DISCOVERY_")) for key in env)
    written = json.loads((home / ".config" / "opencode" / "opencode.json").read_text())
    assert written == agentic.opencode_config(local)
    assert (home / ".config" / "opencode" / "AGENTS.md").read_text() == agentic.role(local) + "\n"
    assert not list(sandbox.iterdir())


def test_opencode_runs_standalone_with_the_prompt_as_its_message(local, tmp_path):
    sandbox = tmp_path / "a00"
    sandbox.mkdir()
    (sandbox / "PROMPT.md").write_text("the task\n")
    argv = agentic.command(local, sandbox)
    assert argv[:2] == ["opencode", "run"]
    assert {"--standalone", "--auto", "--thinking"} <= set(argv)
    assert argv[argv.index("--format") + 1] == "json"
    assert local.variant, "a reasoning variant by default"
    assert argv[argv.index("--model") + 1] == f"local/glm-5.3-flash#{local.variant}", "the variant rides on the model"
    assert argv[-1] == "the task\n"
    local.variant = ""
    assert agentic.command(local, sandbox)[argv.index("--model") + 1] == "local/glm-5.3-flash"


def test_opencode_stream_becomes_turns_and_a_floor_on_usage(tmp_path):
    """Event shapes as `opencode run --format json` 2.0.27 emits them."""
    session = "ses_test"

    def event(kind, part):
        return {"type": kind, "timestamp": 1, "sessionID": session,
                "part": {"id": "prt", "sessionID": session, "messageID": "msg", **part}}

    log = tmp_path / "a00.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in [
        event("step_start", {"type": "step-start"}),
        event("reasoning", {"type": "reasoning", "text": "think",
                            "metadata": {"reasoningField": "reasoning_content"}}),
        event("tool_use", {"type": "tool", "tool": "shell", "state": {
            "status": "completed", "input": {"command": "ls"}, "output": "model.py"}}),
        event("tool_use", {"type": "tool", "tool": "read", "state": {
            "status": "error", "input": {"path": "/etc/hostname"},
            "error": "Permission denied: external_directory"}}),
        event("step_finish", {"type": "step-finish", "reason": "tool-calls", "cost": 0,
                              "tokens": {"input": 33, "output": 63, "reasoning": 0,
                                         "cache": {"read": 3355, "write": 0}}}),
        event("text", {"type": "text", "text": "DONE"}),
    ]))
    turns, usage = agentic.conversation(log)
    assert [t["content"].split("\n")[0] for t in turns] == [
        "[thinking]", "[tool_use shell]", "[tool_result completed]",
        "[tool_use read]", "[tool_result error]", "DONE",
    ]
    assert "Permission denied" in turns[4]["content"]
    assert usage["session_id"] == session
    assert usage["input_tokens"] == 33 and usage["cache_read_input_tokens"] == 3355
    assert usage["num_turns"] == 1
    assert agentic.stepped(log)


def test_opencode_startup_error_is_not_a_step(tmp_path):
    """The failure a retry is for: the model refused before anything ran."""
    log = tmp_path / "a00.jsonl"
    log.write_text(json.dumps({"type": "error", "error": {
        "type": "provider.no-route", "message": "Model unavailable: local/glm-5.3-flash"}}) + "\n")
    turns, _ = agentic.conversation(log)
    assert "provider.no-route" in turns[0]["content"]
    assert not agentic.stepped(log)


def test_opencode_is_built_on_the_repositorys_model_entry(local, tmp_path):
    """Name, capabilities and reasoning variants from `opencode.json`; the limits are
    the ones this server and these flags give, not the file's."""
    config = tmp_path / "opencode.json"
    config.write_text(json.dumps({
        "model": "remote/glm-5.3-flash",
        "providers": {"remote": {"settings": {"baseURL": "http://127.0.0.1:8001/v1"}, "models": {"glm-5.3-flash": {
            "name": "GLM on Slurm", "limit": {"context": 196608, "output": 32768},
            "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
            "variants": [{"id": "high", "settings": {"reasoningEffort": "high"}}],
        }}}},
    }))
    local.opencode_config = str(config)
    model = agentic.opencode_config(local)["providers"]["local"]["models"]["glm-5.3-flash"]
    assert model["name"] == "GLM on Slurm"
    assert model["variants"] == [{"id": "high", "settings": {"reasoningEffort": "high"}}]
    assert model["limit"] == {"context": 131072, "output": 16384}
    assert agentic.repository_model(local)[0] == "http://127.0.0.1:8001/v1"
    local.opencode_config = ""
    assert agentic.repository_model(local) == (None, {})


def test_codex_draws_on_luna_at_high_effort_for_hours(tmp_path):
    settings = agentic.options(["--cli", "codex", "--sandbox", str(tmp_path / "s"), "--out", str(tmp_path / "o")])
    argv = agentic.command(settings, tmp_path)
    assert argv[argv.index("--model") + 1] == "gpt-6-luna"
    assert argv[argv.index("-c") + 1] == 'model_reasoning_effort="high"'
    assert settings.session_timeout == agentic.SESSION_TIMEOUTS["codex"] >= 4 * 3600
    assert "240 minutes" in agentic.role(settings)


def _sandbox_with_check(tmp_path, settings, proposal: str):
    """A sandbox holding what `check.py` reads, with the bench files importable."""
    import numpy as np

    path = tmp_path / "a00"
    site = tmp_path / "site"
    (path / "data").mkdir(parents=True)
    (site / "discovery").mkdir(parents=True)
    for relative, source in agentic.BENCH_FILES.items():
        (site / relative).write_text(source.read_text() if source else "")
    np.savez(path / "data" / "arrays.npz", **agentic.arrays(settings)["npz"])
    (path / "model.py").write_text(STARTS[("curves", "griffin")])
    (path / "proposal.py").write_text(proposal)
    (path / "check.py").write_text(agentic.check_script())
    return path, site


@pytest.mark.parametrize(
    "body, passes",
    [
        ("{curves}", True),
        ("def mechanism(params, log_eir, ft, ages, prop, mu):\n    return -ages, ages * 0.0", False),
    ],
)
def test_the_sandbox_gate_check_is_the_runners(tmp_path, settings, parent, body, passes):
    """The incumbent passes it and a negative prevalence does not, through the same
    `gates.check` -- run as an agent would run it, in its own interpreter."""
    import os
    import subprocess
    import sys

    curves = {block.name: block.body for block in parent.blocks}["curves"]
    proposal = f"# @discovery:begin curves\n{body.format(curves=curves)}\n# @discovery:end\n"
    path, site = _sandbox_with_check(tmp_path, settings, proposal)
    done = subprocess.run(
        [sys.executable, "check.py"], cwd=path, capture_output=True, text=True, timeout=600,
        env=os.environ | {"PYTHONPATH": str(site)},
    )
    assert (done.returncode == 0) is passes, done.stdout + done.stderr
    assert ("passes every gate" in done.stdout) is passes


def test_the_sandbox_demography_is_the_runners(settings):
    """`check.py` rebuilds the gate site's `prop` and `mu` from the sandbox arrays;
    they are the runner's, from msinf, built at f64 as the runner builds them."""
    import jax
    import numpy as np
    from msinf.config import NumpyroConfig
    from msinf.model import build_model_kwargs

    built = agentic.arrays(settings)["npz"]
    cfg = NumpyroConfig(data_csv=settings.table, eir_priors_csv=settings.eir_priors, mortality_csv=None)
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        kwargs, _, _ = build_model_kwargs(cfg)
    finally:
        jax.config.update("jax_enable_x64", was)
    site = agentic.Budget().gate_site
    np.testing.assert_allclose(built["prop"][site], np.asarray(kwargs["prop_site"][site]), rtol=1e-12)
    np.testing.assert_allclose(
        built["re"][site] - built["grid_ageing"], np.asarray(kwargs["mu_site"][site]), rtol=1e-9, atol=1e-15
    )


def test_history_puts_the_feedback_pack_in_every_sandbox(tmp_path, monkeypatch):
    """`--history` writes the pack once, beside the round, and every agent gets a copy
    and is told to read it."""
    from tests.test_feedback import INCUMBENT, variant, write_round

    write_round(tmp_path / "runs" / "beat" / "r01", [(INCUMBENT, -2.0, True), (variant("better"), -1.0, True)])
    monkeypatch.setattr(agentic, "venv", lambda path: None)
    out = tmp_path / "runs" / "beat" / "r02"
    settings = agentic.options([
        "--agents", "2", "--sandbox", str(tmp_path / "sandbox"), "--out", str(out), "--history", str(tmp_path / "runs"),
    ])
    paths = agentic.build(settings)
    pack = (out / "feedback" / "FEEDBACK.md").read_text()
    for path in paths:
        assert (path / "feedback" / "FEEDBACK.md").read_text() == pack
        assert "feedback/FEEDBACK.md" in (path / "PROMPT.md").read_text()
        assert (path / "check.py").is_file()


def test_no_history_means_no_feedback_and_no_mention_of_it(tmp_path, monkeypatch):
    monkeypatch.setattr(agentic, "venv", lambda path: None)
    settings = agentic.options(["--agents", "1", "--sandbox", str(tmp_path / "sandbox"), "--out", str(tmp_path / "out")])
    (path,) = agentic.build(settings)
    assert not (path / "feedback").exists()
    assert "feedback" not in (path / "PROMPT.md").read_text()


def test_codex_is_sent_its_role_ahead_of_the_task(tmp_path):
    """Codex has no system-prompt flag, so the role rides on stdin; the others get it elsewhere."""
    path = tmp_path / "a00"
    path.mkdir()
    (path / "ROLE.md").write_text("the role\n")
    (path / "PROMPT.md").write_text("the task\n")

    def sent(*cli):
        settings = agentic.options([*cli, "--sandbox", str(tmp_path / "s"), "--out", str(tmp_path / "o")])
        return agentic.stdin(settings, path).decode()

    codex = sent("--cli", "codex")
    assert codex.index("the role") < codex.index("the task")
    assert sent() == "the task\n"
    assert sent("--cli", "opencode") == ""


def test_the_task_does_not_tell_an_agent_with_hours_to_stop_at_its_first_draft():
    """'When proposal.py is written, you are done' ended sessions in under a minute."""
    assert "you are done" not in agentic.TASK
    assert "check.py" in agentic.TASK and "keep improving" in agentic.TASK
