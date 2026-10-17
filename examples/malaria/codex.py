"""Draw malaria proposals with the subscription-authenticated Codex CLI.

The model and its reasoning effort default in `agentic` (`gpt-6-luna`, high);
`--model` and `--reasoning-effort` override them.
"""

import asyncio
import sys

from examples.malaria import agentic


async def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--cli" not in args:
        args += ["--cli", "codex"]
    return await agentic.main(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
