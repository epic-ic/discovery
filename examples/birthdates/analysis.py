"""What a proposer may look at before it writes a model: the training days.

The holdout is not here, and that is the whole point of the module. A proposer
that could measure the years it will be scored on would be choosing its model
with the answer in front of it, and `elpd_holdout` would stop meaning what the
report says it means. `namespace()` therefore builds from `data.load(...)[0]`
and never touches the second half of that pair.

numpy and pandas are imported inside `namespace`, not at module scope: the
driver imports this module for its `DESCRIPTION` and must stay as light as
`data` is, while the analysis worker is the only process that needs the arrays.
"""

import os

from examples.birthdates import data

#: what the proposer is told is in the namespace
DESCRIPTION = """\
It holds the training days only -- the held-out years are not available there,
and no amount of looking will reveal them:

    df                  a DataFrame, one row per day, columns
                          births -- the daily count you are modelling
                          t      -- years since 1969-01-01, float
                          doy    -- day of the year, 1.0 to 366.0, float
                          dow    -- 0 for Monday to 6 for Sunday, int
                          month  -- 0 for January to 11 for December, int
                          year   -- years since 1969, int
                          date   -- 'YYYY-MM-DD'
    births              the births column again, as a plain int array
    np, pd              numpy and pandas

The columns are named for the `features` keys the model is given, so what you
find as `df["dow"]` is what the model reads as `features["dow"]`. Take a feature
out of the frame when you need it -- `dow = df["dow"].values` -- rather than
expecting a bare name to exist.
"""


def namespace() -> dict:
    """A fresh namespace over the training days.

    Called once per execution, so the copy is what makes the data read-only:
    a snippet may do anything it likes to this frame and the next snippet will
    not know about it.

    The features live in `df` and are deliberately *not* also bound as bare
    names. Observed against Qwen3.5-35B: given a `dow` array, an agent writes
    `[births[dow == dow].mean() for dow in range(7)]`, the loop variable shadows
    the array, `dow == dow` is `True`, and it gets the global mean back seven
    times over without anything raising. Short feature names are exactly what a
    data analyst reaches for as loop variables, and a silently wrong answer is
    far worse here than a `KeyError` -- the agent believed those numbers and
    proposed from them.
    """
    import numpy as np
    import pandas as pd

    train, _ = data.load(
        os.environ.get("BIRTHDATES_DATA") or data.DEFAULT,
        int(os.environ.get("BIRTHDATES_HOLDOUT_FROM", "1985")),
    )
    births = np.asarray(train.births, dtype=int)
    frame = pd.DataFrame(
        {
            "births": births,
            "t": np.asarray(train.features["t"], dtype=float),
            "doy": np.asarray(train.features["doy"], dtype=float),
            "dow": np.asarray(train.features["dow"], dtype=int),
            "month": np.asarray(train.features["month"], dtype=int),
            "year": np.asarray(train.features["year"], dtype=int),
            "date": train.dates,
        }
    )
    return {"df": frame, "births": births, "np": np, "pd": pd}
