"""Draw malaria proposals with the subscription-authenticated Codex CLI."""

import asyncio
import sys

from examples.malaria import agentic


async def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--cli" not in args:
        args += ["--cli", "codex"]
    if "--model" not in args:
        args += ["--model", "gpt-5"]
    return await agentic.main(args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
