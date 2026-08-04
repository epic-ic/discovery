"""A pool of worker processes that run analysis snippets for proposers.

A proposer that can look at the data before it writes a model writes a better
model. This is the environment it looks with: numpy, pandas, the data being
fitted, and nothing else.

It is not a sandbox, and it is not trying to be. Three things make it safe
enough for code a language model wrote to explore data with:

* **A fresh namespace per call.** `setup` names a module exposing `namespace()`,
  which the worker calls before every execution. The snippet is handed a copy,
  so mutation is contained by construction and no two agents can see each
  other's bindings. Nothing carries over between calls, which is also why any
  worker can take any call and why the pool is a queue rather than an
  assignment.
* **No writing anything, anywhere.** Once the imports are done the worker
  installs an audit hook that refuses to open a file for writing, to rename or
  unlink one, to open a socket, or to start a process. Audit hooks are stdlib,
  they see the operation rather than the library that reached for it, and they
  cannot be uninstalled once added. `RLIMIT_FSIZE` is clamped to zero behind it
  and the worker runs in a temporary directory, but the hook is what does the
  work: an rlimit alone still lets `open(path, "w")` create and *truncate* a
  file, since neither of those writes a byte.
* **Disposable workers.** A snippet that runs too long has its worker killed and
  replaced, and every worker is recycled after `recycle` executions. A fresh
  namespace resets the bindings but not `sys.modules`: a snippet that
  monkeypatches numpy would otherwise follow every later agent through that
  worker. Recycling bounds how far that can travel.

None of this survives a determined adversary with a Python prompt -- the hook
covers the operations CPython audits, and the point of `exec` is that arbitrary
code runs. The failure being defended against is the pandas snippet that
overwrites the data file it was reading, not the attacker.

Run as a module -- `python -m discovery.analysis <setup>` -- this file is the
worker: one JSON object per line in, one per line out.
"""

import asyncio
import contextlib
import importlib
import io
import json
import logging
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: what a setup module must expose: a callable building a fresh namespace
NAMESPACE = "namespace"
#: what it may expose to describe that namespace to the proposer
DESCRIPTION = "DESCRIPTION"

#: the tool as the model is offered it
RUN_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run a self-contained Python snippet against the data being modelled, "
            "with numpy and pandas available. Only what you print comes back. "
            "Nothing carries over between calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute. print() what you want to see.",
                }
            },
            "required": ["code"],
        },
    },
}


def truncate(text: str, limit: int) -> str:
    """Cut output to `limit` characters, saying so where it was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... truncated, {len(text) - limit} more characters"


# --------------------------------------------------------------------------
# the worker: everything below `main` runs in the child process
# --------------------------------------------------------------------------


#: audited operations a snippet has no business performing
FORBIDDEN = frozenset(
    {
        # changing the filesystem
        "os.mkdir", "os.rmdir", "os.remove", "os.rename", "os.link", "os.symlink",
        "os.truncate", "os.chmod", "os.chown", "os.putenv", "os.unsetenv",
        "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.move",
        "shutil.rmtree", "shutil.unpack_archive",
        # leaving the machine
        "socket.__new__", "socket.bind", "socket.connect", "socket.getaddrinfo",
        "socket.sendto", "urllib.Request",
        # leaving the process
        "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork",
        "os.forkpty", "pty.spawn",
        # going under all of the above
        "ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function",
    }
)

#: `open` flags that mean the caller intends to change something
WRITING = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def guard(event: str, args: tuple) -> None:
    """Refuse anything that would leave a trace outside this process.

    A `PermissionError` rather than something of our own: it is an `OSError`,
    which is what every caller between the snippet and the operation is already
    written to expect. Import machinery swallowing a failed bytecode cache write
    keeps working, and the snippet sees an ordinary filesystem denial.
    """
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if (mode and any(character in mode for character in "wxa+")) or ((flags or 0) & WRITING):
            raise PermissionError(f"the analysis environment cannot write files: {path!r}")
    elif event in FORBIDDEN:
        raise PermissionError(f"the analysis environment does not allow {event}")


def clamp(memory_mb: int) -> None:
    """Give up everything a snippet should not be able to do.

    Called after the setup module is imported, not before: an import may
    legitimately write a bytecode cache or open a data file, and it is the
    snippets being restrained rather than the environment they run in.

    There is deliberately no `RLIMIT_CPU`. It counts the process's whole life,
    and a worker that serves fifty executions would hit it for reasons that have
    nothing to do with the snippet running when it does; a runaway loop is the
    parent's wall clock to catch, and it catches it by killing the worker.
    """
    import resource
    import signal

    # A write past RLIMIT_FSIZE raises SIGXFSZ, whose default action is to kill
    # the process. Ignored, the write fails with EFBIG instead, which reaches
    # the snippet as an ordinary exception and the agent as a traceback.
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if memory_mb:
        size = memory_mb * 1024 * 1024
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (size, size))
    # Last, and never undone: an audit hook cannot be removed once installed,
    # which is exactly the property wanted here.
    sys.addaudithook(guard)


def execute(code: str, build: Any, limit: int) -> str:
    """Run one snippet in a fresh namespace and return what it printed.

    A snippet that raises is not a failure of the tool: the traceback is the
    result, and the agent can read it and try again.
    """
    stream = io.StringIO()
    namespace = build()
    namespace["__name__"] = "analysis"
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(code, "<analysis>", "exec"), namespace)  # noqa: S102 -- the point of the worker
    except (Exception, SystemExit) as error:
        # `.tb_next` drops this frame: the snippet's traceback is about the
        # snippet, and the worker's own line numbers only confuse it.
        printed = stream.getvalue()
        report = "".join(traceback.format_exception(type(error), error, error.__traceback__.tb_next))
        return truncate(f"{printed}{report}", limit)
    printed = stream.getvalue()
    return truncate(printed, limit) if printed.strip() else "(the snippet printed nothing)"


def reply(payload: dict[str, Any]) -> None:
    """One JSON object on one line, flushed."""
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Serve executions on stdin until it closes."""
    argv = sys.argv[1:] if argv is None else argv
    setup, memory_mb = argv[0], int(argv[1])
    try:
        module = importlib.import_module(setup)
        build = getattr(module, NAMESPACE)
        build()  # fail now, while there is somebody listening, not on first use
    except Exception:
        reply({"ready": False, "error": traceback.format_exc()})
        return 1
    clamp(memory_mb)
    reply({"ready": True, "description": getattr(module, DESCRIPTION, "")})

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        reply({"output": execute(request["code"], build, request.get("limit", 4000))})
    return 0


# --------------------------------------------------------------------------
# the pool: everything below runs in the parent
# --------------------------------------------------------------------------


class Worker:
    """One child process, and the executions it has served."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.runs = 0
        self.description = ""

    @property
    def alive(self) -> bool:
        return self.process.returncode is None

    async def handshake(self, timeout: float) -> None:
        """Wait for the setup module to import, or say why it did not."""
        async with asyncio.timeout(timeout):
            line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("the analysis worker died before it was ready")
        hello = json.loads(line)
        if not hello.get("ready"):
            raise RuntimeError(f"the analysis setup module failed to import:\n{hello.get('error', '')}")
        self.description = hello.get("description", "")

    async def execute(self, code: str, limit: int, timeout: float) -> str:
        """Run one snippet, or raise `TimeoutError` and leave the worker to be killed."""
        self.process.stdin.write(json.dumps({"code": code, "limit": limit}).encode() + b"\n")
        await self.process.stdin.drain()
        async with asyncio.timeout(timeout):
            line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("the analysis worker died while running the snippet")
        self.runs += 1
        return json.loads(line)["output"]

    async def kill(self) -> None:
        if self.alive:
            self.process.kill()
            await self.process.wait()


@dataclass
class Analysis:
    """A fixed number of worker processes, handed out one execution at a time.

    `setup` is the importable module that builds the namespace; `description`
    is what the proposer is told is in it, and defaults to the module's
    `DESCRIPTION`.

    The pool is the scarce thing and the queue is what bounds it: there are
    `workers` of them however many agents are asking. Waiting for one is
    ordinary and unlogged; waiting longer than `warn_after` is not, and says how
    many other agents were in the queue, since that is the number that decides
    whether `workers` is set right.
    """

    setup: str
    description: str = ""
    workers: int = 4
    #: seconds one snippet may run before its worker is killed
    timeout: float = 20.0
    #: characters of output carried back
    max_output: int = 4000
    #: executions a worker serves before it is replaced
    recycle: int = 50
    memory_mb: int = 2048
    #: environment for the setup module, for whatever it needs to build from
    setup_env: dict[str, str] = field(default_factory=dict)
    #: seconds in the queue past which the wait is worth knowing about
    warn_after: float = 5.0
    #: seconds a worker gets to import its setup module
    boot_timeout: float = 120.0

    _idle: asyncio.Queue = field(init=False, default_factory=asyncio.Queue)
    _started: bool = field(init=False, default=False)
    _waiting: int = field(init=False, default=0)
    _replacing: set = field(init=False, default_factory=set)
    _home: Any = field(init=False, default=None)

    async def _spawn(self) -> Worker:
        """Boot one worker: a temporary home, the repository importable, one thread."""
        if self._home is None:
            self._home = tempfile.TemporaryDirectory(prefix="discovery-analysis-")
        path = os.pathsep.join(p for p in (str(Path.cwd()), os.environ.get("PYTHONPATH", "")) if p)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "discovery.analysis",
            self.setup,
            str(self.memory_mb),
            cwd=self._home.name,
            env=os.environ
            | {
                "PYTHONPATH": path,
                # A describe() does not need a thread pool the size of the
                # machine, and four workers each taking one is how a pool that
                # is meant to be cheap stops being cheap.
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            | self.setup_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=1 << 20,
        )
        worker = Worker(process)
        try:
            await worker.handshake(self.boot_timeout)
        except BaseException:
            await worker.kill()
            raise
        return worker

    async def start(self) -> "Analysis":
        """Boot every worker, so the first agent to ask does not pay for it."""
        if self._started:
            return self
        self._started = True
        try:
            workers = await asyncio.gather(*(self._spawn() for _ in range(self.workers)))
        except BaseException:
            self._started = False
            raise
        for worker in workers:
            self._idle.put_nowait(worker)
        self.description = self.description or workers[0].description
        return self

    def _replace(self) -> None:
        """Boot a replacement, without making the agent who lost a worker wait.

        The pool runs one short of `workers` until it arrives. That is the right
        trade in both cases it happens: a killed worker means somebody's snippet
        is already over time, and a recycled one means the pool is busy.
        """

        async def boot() -> None:
            try:
                self._idle.put_nowait(await self._spawn())
            except Exception:
                log.exception("could not replace an analysis worker; the pool is one smaller")

        task = asyncio.create_task(boot())
        self._replacing.add(task)
        task.add_done_callback(self._replacing.discard)

    async def _acquire(self) -> Worker:
        """The next idle worker, noting when the queue made somebody wait."""
        if not self._started:
            await self.start()
        if not self._idle.empty():
            return self._idle.get_nowait()
        loop = asyncio.get_running_loop()
        started = loop.time()
        self._waiting += 1
        try:
            worker = await self._idle.get()
        finally:
            self._waiting -= 1
        if (waited := loop.time() - started) >= self.warn_after:
            log.warning(
                "waited %.1fs for one of %d analysis workers, %d agents still queued",
                waited,
                self.workers,
                self._waiting,
            )
        return worker

    def _release(self, worker: Worker) -> None:
        """Put a worker back, or retire it and boot its replacement."""
        if worker.alive and worker.runs < self.recycle:
            self._idle.put_nowait(worker)
        else:
            self._replace()

    async def run(self, code: str) -> str:
        """Execute one snippet and return what the agent should be told.

        Everything that can go wrong with a snippet -- an exception, a runaway
        loop, a worker that died under it -- comes back as text rather than as a
        raised error. The agent has a budget of calls and can spend the next one
        on the traceback; a proposal is not lost because one snippet was bad.
        """
        worker = await self._acquire()
        try:
            output = await worker.execute(code, self.max_output, self.timeout)
        except asyncio.CancelledError:
            # The agent gave up while this worker was mid-snippet. It cannot go
            # back in the queue: the next agent to take it would read the output
            # of this one's code.
            await worker.kill()
            self._replace()
            raise
        except TimeoutError:
            await worker.kill()
            self._replace()
            return f"The snippet was still running after {self.timeout:.0f}s and was stopped."
        except Exception as error:
            await worker.kill()
            self._replace()
            return f"The analysis worker failed: {type(error).__name__}: {error}"
        self._release(worker)
        return output

    async def aclose(self) -> None:
        """Kill every worker and clean up the temporary home."""
        for task in list(self._replacing):
            task.cancel()
        self._replacing.clear()
        workers: list[Worker] = []
        while not self._idle.empty():
            workers.append(self._idle.get_nowait())
        await asyncio.gather(*(worker.kill() for worker in workers), return_exceptions=True)
        if self._home is not None:
            self._home.cleanup()
            self._home = None
        self._started = False

    async def __aenter__(self) -> "Analysis":
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


if __name__ == "__main__":
    raise SystemExit(main())
