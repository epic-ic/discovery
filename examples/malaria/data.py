"""The study table, the site EIR priors, and the name of a split.

Stdlib only, and for the same reason `birthdates/data.py` is: the driver imports
this module and must not pull jax -- or msinf, or dmeq -- into the parent
process. The runner imports it too, in the child process (see `fit.py`).

**This module does not partition.** That is where the example departs from
`birthdates`, where the holdout is three lines of list slicing and belongs beside
the reader. Here, which rows can be scored apart from which is a property of
msinf's likelihood rather than of the table: incidence cells cannot be split,
because `v_jk` is integrated out at the (site, age-group) level and the marginal
is a joint density over the whole cell; and site indices renumber the moment a
study is dropped, which silently misaligns the held-out arrays against the
fitted ones. So `msinf.splits` owns the drawing and the refusing, and this module
only reads the table, names a split and says what is in it.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

#: where the example's own copies of the inputs live
HERE = Path(__file__).resolve().parent

#: the <=20y study table. Already truncated on disk -- there is no `truncate_age`
#: anywhere in msinf, and adding one here would be a second way to draw the same
#: boundary that could disagree with the CSV.
DEFAULT_TABLE = Path(os.environ.get("MALARIA_TABLE", HERE / "study_data_2014_age_le20.csv"))

#: per-site EIR priors, resolved over every site in the table including the ones
#: a study holdout removes from the fit: scoring a held-out site means drawing
#: its `log_eir` from the prior, so the prior row has to exist.
DEFAULT_EIR_PRIORS = Path(os.environ.get("MALARIA_EIR_PRIORS", HERE / "eir_site_priors_griffin2014.csv"))

#: optional per-site WPP death rates on the model age grid; msinf's `null`
#: default keeps dmeq's uniform rate, which is what every published run used.
DEFAULT_MORTALITY = os.environ.get("MALARIA_MORTALITY") or None

#: the splits, as `msinf.splits.KINDS` names them. Reported separately and never
#: averaged: the study split holds `log_eir` on its prior and is the blunter of
#: the two, the age-band split leaves `log_eir` informed and tests age structure.
SPLITS = ("study", "age_band")

#: observation type codes in the table
TYPE_INCIDENCE = 1
TYPE_PREV_MICROSCOPY = 2

STREAMS = {TYPE_INCIDENCE: "incidence", TYPE_PREV_MICROSCOPY: "prevalence"}


class MissingData(FileNotFoundError):
    """The inputs are not on this machine, and this says which.

    Its own error because it is the one failure that is not about the code: the
    example is complete without the tables, and a run without them should say so
    once, up front, rather than fail inside a child process.
    """


@dataclass(frozen=True)
class Inputs:
    """The files a fit reads, all of them absolute."""

    table: Path
    eir_priors: Path
    mortality: str | None = None

    def check(self) -> "Inputs":
        """Fail now, in the parent, naming every file that is not there."""
        missing = [p for p in (self.table, self.eir_priors) if not Path(p).is_file()]
        if self.mortality and not Path(self.mortality).is_file():
            missing.append(Path(self.mortality))
        if missing:
            raise MissingData(
                "the malaria example needs input tables that are not on this machine:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\nSet $MALARIA_TABLE and $MALARIA_EIR_PRIORS, or copy the tables in beside "
                "this module. They are confirmed redistributable and the study table is already "
                "truncated to <=20y on disk."
            )
        return self


def inputs(
    table: str | Path = DEFAULT_TABLE,
    eir_priors: str | Path = DEFAULT_EIR_PRIORS,
    mortality: str | None = DEFAULT_MORTALITY,
) -> Inputs:
    """The resolved inputs, checked."""
    return Inputs(Path(table).resolve(), Path(eir_priors).resolve(), mortality).check()


@dataclass(frozen=True)
class Summary:
    """What is in the table, for the brief and for the run's first line.

    Deliberately shallow: it counts rows, and counting rows is the one thing a
    proposer may know about the whole table without knowing the holdout. Anything
    finer belongs behind an `Analysis` that is built from the training rows only.
    """

    rows: int
    studies: int
    sites: int
    by_stream: dict[str, int]
    age_lo: float
    age_hi: float

    def __str__(self) -> str:
        streams = ", ".join(f"{n} {name}" for name, n in sorted(self.by_stream.items()))
        return (
            f"{self.rows} rows ({streams}) from {self.studies} studies, "
            f"{self.sites} sites, ages {self.age_lo:g}-{self.age_hi:g}y"
        )


def read(path: str | Path = DEFAULT_TABLE) -> Summary:
    """Count the table without loading it into anything heavier than dicts."""
    rows = 0
    studies: set[str] = set()
    sites: set[str] = set()
    by_stream: dict[str, int] = {}
    lo, hi = float("inf"), float("-inf")
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            studies.add(row["study_index"])
            # The canonical site is the country/site pair where those columns
            # exist -- incidence and prevalence rows for one physical site can
            # carry different `site_index` values, which is exactly the encoding
            # msinf.data.encode_canonical_sites exists to undo.
            sites.add(
                f"{row['country_name']}||{row['site_name']}"
                if "country_name" in row and "site_name" in row
                else row["site_index"]
            )
            name = STREAMS.get(int(row["type"]), f"type {row['type']}")
            by_stream[name] = by_stream.get(name, 0) + 1
            lo = min(lo, float(row["age0"]))
            hi = max(hi, float(row["age1"]))
    if not rows:
        raise ValueError(f"{path} has no observation rows")
    return Summary(rows, len(studies), len(sites), by_stream, lo, hi)


def label(kind: str, seed: int = 42, n_studies: int = 3, from_age: float = 5.0) -> str:
    """A split's name, spelled the way `msinf.splits.Split.label` spells it.

    Duplicated on purpose: the driver prints the label before anything has
    imported msinf, and one f-string is a cheaper coupling than importing the
    inference package into the parent process to ask.
    """
    if kind == "none":
        return "none"
    if kind == "study":
        return f"study_n{n_studies}_seed{seed}"
    if kind == "age_band":
        return f"age_band_from{from_age:g}"
    raise ValueError(f"unknown split kind {kind!r}; expected one of {('none',) + SPLITS}")
