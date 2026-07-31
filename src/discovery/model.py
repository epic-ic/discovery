"""Models are code.

A model is a source file that runs on its own. Regions the search is allowed to
rewrite are delimited by marker comments::

    # @discovery:begin prior
    mu = sample("mu", Normal(0.0, 1.0))
    # @discovery:end

Everything outside the markers is fixed scaffolding: imports, the data, the
likelihood harness, whatever the experiment holds constant. Everything inside is
what a proposer may replace.
"""

import re
import textwrap
from dataclasses import dataclass, replace

BEGIN = re.compile(r"^(?P<indent>[ \t]*)#\s*@discovery:begin(?:\s+(?P<name>\S+))?\s*$")
END = re.compile(r"^[ \t]*#\s*@discovery:end\s*$")
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def reindent(body: str, indent: str) -> str:
    """Set a body's indentation to `indent`, keeping its shape inside.

    Where a block sits in the file is the scaffolding's business and not the
    proposer's: the begin marker states the indentation, and a body that arrives
    flush left -- as one written on its own, away from the function it belongs
    to, naturally does -- is put back where the block was. Relative indentation
    within the body is untouched, and a body that already matches comes back
    unchanged.
    """
    lines = textwrap.dedent(body).splitlines()
    return "\n".join(f"{indent}{line}" if line.strip() else "" for line in lines)


@dataclass(frozen=True)
class Block:
    """A rewritable region of a model."""

    name: str
    body: str
    indent: str

    def render(self) -> str:
        """The block as it appears in source, markers included."""
        return f"{self.indent}# @discovery:begin {self.name}\n{self.body}\n{self.indent}# @discovery:end"


@dataclass(frozen=True)
class Model:
    """A candidate model: executable source with rewritable blocks."""

    source: str

    @property
    def blocks(self) -> tuple[Block, ...]:
        """The rewritable blocks, in source order."""
        blocks: list[Block] = []
        open_block: tuple[str, str, int] | None = None
        lines = self.source.splitlines()
        for number, line in enumerate(lines):
            begin = BEGIN.match(line)
            if begin:
                if open_block is not None:
                    raise ValueError(f"nested @discovery:begin on line {number + 1}")
                name = begin["name"] or f"block_{len(blocks)}"
                open_block = (name, begin["indent"], number)
            elif END.match(line):
                if open_block is None:
                    raise ValueError(f"@discovery:end without begin on line {number + 1}")
                name, indent, start = open_block
                blocks.append(Block(name, "\n".join(lines[start + 1 : number]), indent))
                open_block = None
        if open_block is not None:
            raise ValueError(f"unterminated @discovery:begin '{open_block[0]}'")
        return tuple(blocks)

    def substitute(self, bodies: dict[str, str]) -> "Model":
        """Return a copy with the named blocks' bodies replaced.

        Unknown names are an error; blocks left unmentioned are kept as they
        are. Each new body is put at its block's indentation, so a proposal
        written flush left still splices into an indented block.
        """
        known = {block.name for block in self.blocks}
        if unknown := set(bodies) - known:
            raise KeyError(f"no such block(s): {sorted(unknown)}")

        out: list[str] = []
        seen = 0
        replacing = False
        for line in self.source.splitlines():
            begin = BEGIN.match(line)
            if begin:
                name = begin["name"] or f"block_{seen}"
                seen += 1
                out.append(line)
                if replacing := name in bodies:
                    out.append(reindent(bodies[name], begin["indent"]))
            elif replacing and END.match(line):
                replacing = False
                out.append(line)
            elif not replacing:
                out.append(line)
        return replace(self, source="\n".join(out) + ("\n" if self.source.endswith("\n") else ""))


def parse_blocks(text: str) -> dict[str, str]:
    """Pull marked blocks out of free text, unwrapping a code fence if present.

    Used to read a proposer's reply with exactly the grammar the model uses.
    """
    fences = FENCE.findall(text)
    source = "\n".join(fences) if fences else text
    return {block.name: block.body for block in Model(source).blocks}
