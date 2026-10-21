"""Draw malaria proposals with OpenCode against a local OpenAI-compatible server.

The server's url, the model's limits and its reasoning variants come from the
repository's `opencode.json`, and the slots and per-slot context from the
server itself, so with `$LLM_API_KEY` set nothing else needs saying:

    uv run python -m examples.malaria.opencode --agents 8 --out runs/beat/r02 --history runs/beat

`--url`, `--variant` and `--session-timeout` override them.
"""

import asyncio
import sys

from examples.malaria import agentic


async def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--cli" not in args:
        args += ["--cli", "opencode"]
    return await agentic.main(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
