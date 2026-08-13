"""The models the search starts from, as real modules rather than string literals.

Each module in this package is a *fragment*, not a candidate: it is importable,
lintable and testable Python, and `initial.py` lifts its marked regions out and
splices them into the target's template. What lives here is only what differs
between starts -- the imports, the parameter block and the mechanism. The shared
prose (what a candidate's `curves` is handed, what it must leave behind) and the
fixed lines below the rewritable region live once, in the template.

Three regions per module, and the names are the contract:

    preamble      module-level imports and constants, spliced above the
                  functions. Scaffolding in the spliced candidate: it carries no
                  markers there, so a proposer cannot add an import and has to
                  work with what the start bound.
    parameters    the body of `parameters()` -- every `numpyro.sample` the
                  candidate wants, since the fit runs it outside the `vmap`.
    curves        the whole mechanism, for the `curves` target: a
                  `mechanism(params, log_eir, ft, ages, prop, mu)` and whatever it
                  is built out of.
    immunity      the whole `def immunity(...)`, for the `immunity` target.

Whole functions and not bodies, at both targets: there is no boundary inside
either that belongs to the scaffolding, and the fixed `curves` wrapper below them
is what holds the unit conversion.

A module carries `curves` or `immunity`, never both, and `starts/griffin.py`
carries neither -- the incumbent's mechanism is lifted out of `dmeq` at both
targets, so there is nothing about it worth writing here. `initial.STARTS` is
keyed by the `(target, start)` pair.

A fragment, and not a runnable model, and the boundary is worth being precise
about. A start module is real Python: it imports, it is typed and linted, and its
regions are extracted from its own source. What it is *not* is executable on its
own, because the template supplies names its mechanism uses -- `Immunity`,
`AgeGrid`, `age_grid`, `calculate_immunity`, `GH_NODES`, `_non_het_prev` -- all of
which are fixed scaffolding rather than anything a start should restate. Where that
leaves a name unused or undefined in one file, there is a `# ruff: noqa` at the top
of it saying which and why. `curves_scratch.py` is the one that happens to be
self-contained.

Extraction is `inspect.getsource` plus `discovery.model`'s own block parser, so
these regions are read with exactly the grammar a proposal is read with. The
incumbent's immunity block is the one that does not come from here: it is
`inspect.getsource(dmeq.griffin_immunity)`, so the mechanism a proposer is asked
to correct is the mechanism the fit runs, and cannot drift from it.
"""

from inspect import getsource
from types import ModuleType

from discovery.model import Model


def regions(module: ModuleType) -> dict[str, str]:
    """A start module's marked regions, by name, at their own indentation.

    `Model.substitute` reindents a body into whatever block it lands in, so a
    `parameters` body extracted at four spaces splices correctly into a template
    whose block sits at four spaces, and an `immunity` function extracted flush
    left splices correctly at module level.
    """
    return {block.name: block.body for block in Model(getsource(module)).blocks}


def documentation(module: ModuleType) -> str:
    """The module docstring, which becomes the candidate's.

    It describes the model, not the fragment: the note about being extracted
    rather than executed belongs to this package, and is here rather than
    repeated at the top of every start.
    """
    if not module.__doc__:
        raise ValueError(f"{module.__name__} has no docstring to hand the candidate")
    return module.__doc__.strip("\n")
