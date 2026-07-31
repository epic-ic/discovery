"""Daily birth counts, 1969-1988, split in time.

Stdlib only: this module is imported by the driver, which must not pull jax into
the parent process, and by the runner, which builds arrays from what it returns.

The holdout is the tail of the series rather than a random sample of days. A
model that finds the generating process -- a trend, a yearly season, a weekday
effect, the holidays -- keeps its predictions on years it has never seen; a
model that only smooths its neighbours does not, and a random holdout would let
it hide.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

#: the data as shipped, beside the repository root
DEFAULT = Path(__file__).resolve().parents[2] / "birthdates-1968-1988.txt"

#: days in the year, for turning `doy` into a phase
YEAR = 365.25


@dataclass(frozen=True)
class Series:
    """One contiguous stretch of days and the counts observed on them.

    `features` is what the model is handed and the only thing it may look at:

    * `t`     -- years since 1969-01-01, continuous
    * `doy`   -- day of the year, 1 to 366, continuous
    * `dow`   -- day of the week, 0 for Monday to 6 for Sunday, whole
    * `month` -- 0 for January to 11 for December, whole
    * `year`  -- years since 1969, 0 to 19, whole

    The three whole ones are integers rather than floats because that is what
    they are for: a model reaches for a weekday effect by indexing a vector of
    seven with `features["dow"]`, and a float would only be a cast away from
    working.
    """

    features: dict[str, list[float] | list[int]]
    births: list[int]
    dates: list[str]

    def __len__(self) -> int:
        return len(self.births)

    @property
    def span(self) -> str:
        return f"{self.dates[0]} to {self.dates[-1]}"


def read(path: Path | str = DEFAULT) -> Series:
    """Read the whole series as a single `Series`."""
    features: dict[str, list] = {n: [] for n in ("t", "doy", "dow", "month", "year")}
    births: list[int] = []
    dates: list[str] = []
    with open(path, newline="") as handle:
        for day, row in enumerate(csv.DictReader(handle)):
            year, month, doy = int(row["year"]), int(row["month"]), int(row["day_of_year"])
            features["t"].append(day / YEAR)
            features["doy"].append(float(doy))
            features["dow"].append(int(row["day_of_week"]) - 1)
            features["month"].append(month - 1)
            features["year"].append(year - 1969)
            births.append(int(row["births"]))
            dates.append(f"{year:04d}-{month:02d}-{int(row['day']):02d}")
    return Series(features, births, dates)


def partition(series: Series, holdout_from: int = 1985) -> tuple[Series, Series]:
    """Split into everything before `holdout_from` and everything from it on.

    `t` keeps counting across the split: the holdout is the future, and a model
    with a trend has to extrapolate it rather than start over.
    """
    cut = next(
        (day for day, year in enumerate(series.features["year"]) if year + 1969 >= holdout_from),
        len(series),
    )
    if cut in (0, len(series)):
        raise ValueError(f"holdout_from={holdout_from} leaves one side of the split empty")
    return (
        Series({n: v[:cut] for n, v in series.features.items()}, series.births[:cut], series.dates[:cut]),
        Series({n: v[cut:] for n, v in series.features.items()}, series.births[cut:], series.dates[cut:]),
    )


def load(path: Path | str = DEFAULT, holdout_from: int = 1985) -> tuple[Series, Series]:
    """The training and holdout series, in that order."""
    return partition(read(path), holdout_from)
