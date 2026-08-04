"""Analysis pool tests.

These start real subprocesses -- the isolation being tested *is* the process
boundary, and a mocked worker would test nothing that matters. They use a setup
module written into a temporary directory rather than the example's, so nothing
here needs numpy or pandas.
"""

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from discovery.analysis import Analysis, truncate

SETUP = '''\
"""A tiny stand-in for a real analysis environment."""

DESCRIPTION = "It holds `values`, a list of three numbers."

CALLS = []


def namespace():
    CALLS.append(1)
    return {"values": [1, 2, 3], "booted": len(CALLS)}
'''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A setup module on the path, and a cwd the worker inherits it from."""
    (tmp_path / "tiny_setup.py").write_text(SETUP)
    monkeypatch.chdir(tmp_path)
    return "tiny_setup"


def pool(setup: str, **kwargs) -> Analysis:
    return Analysis(setup=setup, workers=kwargs.pop("workers", 1), **kwargs)


def test_truncate_says_where_it_cut():
    assert truncate("abc", 10) == "abc"
    cut = truncate("abcdefghij", 4)
    assert cut.startswith("abcd")
    assert "6 more characters" in cut


async def test_a_snippet_gets_the_namespace_and_its_output_comes_back(setup):
    async with pool(setup) as analysis:
        assert "6" in await analysis.run("print(sum(values))")


async def test_the_description_comes_from_the_setup_module(setup):
    async with pool(setup) as analysis:
        assert analysis.description == "It holds `values`, a list of three numbers."


async def test_an_explicit_description_wins(setup):
    async with pool(setup, description="mine") as analysis:
        assert analysis.description == "mine"


async def test_a_snippet_that_raises_returns_its_traceback(setup):
    """A bad snippet is a result, not a failure: the agent reads it and retries."""
    async with pool(setup) as analysis:
        output = await analysis.run("print('before')\nraise ValueError('nope')")
    assert "before" in output
    assert "ValueError: nope" in output
    # the worker's own frames are not the agent's business
    assert "discovery/analysis.py" not in output


async def test_a_snippet_that_prints_nothing_is_told_so(setup):
    async with pool(setup) as analysis:
        assert "printed nothing" in await analysis.run("x = 1")


async def test_nothing_carries_over_between_calls(setup):
    async with pool(setup) as analysis:
        await analysis.run("values.append(4)\nleaked = 'yes'")
        assert "[1, 2, 3]" in await analysis.run("print(values)")
        assert "NameError" in await analysis.run("print(leaked)")


async def test_the_namespace_is_rebuilt_for_every_execution(setup):
    async with pool(setup) as analysis:
        first = await analysis.run("print(booted)")
        second = await analysis.run("print(booted)")
    assert int(first.strip()) < int(second.strip())


async def test_a_snippet_cannot_create_a_file(setup, tmp_path):
    """Not even an empty one: `open(path, 'w')` truncates, and an rlimit alone
    would let it, since creating and truncating write no bytes."""
    target = tmp_path / "written.txt"
    async with pool(setup) as analysis:
        output = await analysis.run(f"open({str(target)!r}, 'w').write('hello')")
    assert "PermissionError" in output
    assert not target.exists()


async def test_a_snippet_cannot_truncate_a_file_that_already_exists(setup, tmp_path):
    existing = tmp_path / "precious.txt"
    existing.write_text("keep me")
    async with pool(setup) as analysis:
        await analysis.run(f"open({str(existing)!r}, 'w')")
        await analysis.run(f"import os; os.remove({str(existing)!r})")
    assert existing.read_text() == "keep me"


async def test_a_snippet_can_still_read(setup, tmp_path):
    readable = tmp_path / "readable.txt"
    readable.write_text("data")
    async with pool(setup) as analysis:
        assert "data" in await analysis.run(f"print(open({str(readable)!r}).read())")


async def test_a_snippet_cannot_open_a_socket_or_a_process(setup):
    async with pool(setup) as analysis:
        assert "PermissionError" in await analysis.run("import socket; socket.socket()")
        assert "PermissionError" in await analysis.run("import subprocess; subprocess.run(['ls'])")


async def test_a_refused_write_does_not_take_the_worker_down(setup):
    async with pool(setup) as analysis:
        await analysis.run("open('x.txt', 'w').write('hello')")
        assert "6" in await analysis.run("print(sum(values))")


async def test_a_runaway_snippet_is_stopped_and_the_pool_recovers(setup):
    async with pool(setup, timeout=1.0) as analysis:
        output = await analysis.run("while True:\n    pass")
        assert "was stopped" in output
        # the killed worker is replaced, so the pool still serves
        assert "6" in await analysis.run("print(sum(values))")


async def test_workers_are_recycled_after_their_run_count(setup):
    """A fresh namespace does not reset sys.modules; recycling bounds how far a
    snippet that patched something can follow later agents."""
    async with pool(setup, recycle=2) as analysis:
        await analysis.run("import sys; sys.modules['tiny_setup'].DESCRIPTION = 'patched'")
        await analysis.run("print(1)")
        for _ in range(20):  # the replacement boots in the background
            if "patched" not in await analysis.run(
                "import sys; print(sys.modules['tiny_setup'].DESCRIPTION)"
            ):
                return
            await asyncio.sleep(0.1)
        pytest.fail("the patched worker was never recycled")


async def test_the_pool_bounds_how_many_snippets_run_at_once(setup):
    async with pool(setup, workers=2) as analysis:
        outputs = await asyncio.gather(*(analysis.run("print('ok')") for _ in range(6)))
    assert all("ok" in output for output in outputs)


async def test_a_long_wait_for_a_worker_is_logged(setup, caplog):
    async with pool(setup, workers=1, warn_after=0.0) as analysis:
        await asyncio.gather(*(analysis.run("print(1)") for _ in range(3)))
    assert any("analysis workers" in record.message for record in caplog.records)


async def test_a_cancelled_snippet_does_not_return_its_worker(setup):
    """Otherwise the next agent to check that worker out reads this one's output."""
    async with pool(setup, workers=1, timeout=30.0) as analysis:
        slow = asyncio.create_task(analysis.run("import time; time.sleep(30); print('stale')"))
        await asyncio.sleep(0.5)
        slow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await slow
        assert "stale" not in await analysis.run("print('fresh')")


async def test_a_setup_module_that_cannot_import_says_why(tmp_path, monkeypatch):
    (tmp_path / "broken_setup.py").write_text("raise RuntimeError('no data here')")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="no data here"):
        await Analysis(setup="broken_setup", workers=1).start()


async def test_setup_env_reaches_the_worker(tmp_path, monkeypatch):
    (tmp_path / "env_setup.py").write_text(
        textwrap.dedent(
            """\
            import os

            def namespace():
                return {"where": os.environ.get("WHERE", "unset")}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    async with Analysis(setup="env_setup", workers=1, setup_env={"WHERE": "here"}) as analysis:
        assert "here" in await analysis.run("print(where)")


@pytest.mark.skipif(sys.platform != "linux", reason="rlimits are checked on linux")
async def test_the_address_space_is_capped(setup):
    async with pool(setup, memory_mb=256) as analysis:
        output = await analysis.run("x = bytearray(2 * 1024 * 1024 * 1024)")
    assert "MemoryError" in output


async def test_closing_kills_the_workers(setup):
    analysis = pool(setup, workers=2)
    await analysis.start()
    workers = [analysis._idle.get_nowait() for _ in range(2)]
    for worker in workers:
        analysis._idle.put_nowait(worker)
    assert all(worker.alive for worker in workers)
    await analysis.aclose()
    assert not any(worker.alive for worker in workers)


async def test_the_worker_runs_outside_the_callers_directory(setup):
    async with pool(setup) as analysis:
        output = await analysis.run("import os; print(os.getcwd())")
    assert "discovery-analysis-" in output
    assert Path(output.strip()).name.startswith("discovery-analysis-")


async def test_a_proposer_looks_at_the_data_before_it_writes(setup):
    """The one test that runs the real pool through the real tool loop.

    Everything else here mocks the model or mocks the pool; this asserts that
    the two halves fit -- a tool call becomes an execution, and its output comes
    back as the turn the next request is built from.
    """
    from discovery.model import Model
    from discovery.proposer import ModelProposer

    from tests.test_proposer import REPLY, SOURCE, FakeClient, ToolCall

    async with pool(setup) as analysis:
        proposer = ModelProposer(
            url="http://localhost:8000/v1",
            llm_model="a-model",
            client=FakeClient(replies=[[ToolCall("print(sum(values))")], REPLY]),
            analysis=analysis,
        )
        proposed = await proposer.propose(Model(SOURCE))

    assert "StudentT" in proposed.source
    (result,) = [m for m in proposer.transcript if m["role"] == "tool"]
    assert result["content"].startswith("6")
