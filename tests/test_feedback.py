"""The feedback pack, and the table and promotion built on the same rounds.

The property that matters above all: nothing about the held-out fold reaches an
agent. Every fake round here carries held-out scores, vectors and a held-out
fitted table with a sentinel in them, and the pack must not contain any of it.
"""

import json

import numpy as np
import pandas as pd
import pytest

from discovery.model import Model

from examples.malaria import feedback, rounds
from examples.malaria.fit import SCORING
from examples.malaria.initial import STARTS

#: a number that appears only in held-out metrics, so finding it anywhere in the
#: pack means a held-out number leaked
SENTINEL = -98765.4321

SPLIT = "study_n3_seed42"

INCUMBENT = STARTS[("curves", "griffin")]


def variant(text: str) -> str:
    """A candidate that differs from the incumbent in its parameters block."""
    return Model(INCUMBENT).substitute({"parameters": f"    # {text}\n    return {{}}"}).source


def fitted(sentinel: bool = False) -> pd.DataFrame:
    """A small `fitted_*` table in the runner's shape."""
    rows = []
    for stream in ("incidence", "prevalence"):
        for age0, age1 in ((0, 1), (1, 5), (5, 15)):
            rows.append(
                {
                    "reference": "Ye_etal_2007", "site_name": "Nouna", "site_i": 0, "study_index": 1, "stream": stream,
                    "age0": age0, "age1": age1, "observed": SENTINEL if sentinel else 0.4,
                    "fitted": 0.5, "pred_q05": 0.2, "pred_q50": 0.35, "pred_q95": 0.6, "pit": 0.7, "elpd": -3.0,
                }
            )
    return pd.DataFrame(rows)


def log_eir():
    """Two chains that agree on sites 1-3 and split on site 0 (training) and 4 (held out)."""
    draws = np.random.default_rng(0).normal(3.0, 0.05, size=(2, 10, 5))
    draws[1, :, [0, 4]] += 2.0
    return draws


def write_round(directory, members, split=SPLIT, scoring=SCORING):
    """A fitted round: `members` is `(source, loo, ok)` per candidate, incumbent first.

    `scoring=None` writes the `fit.json` of a round scored before scoring was recorded.
    """
    (directory / "artifacts").mkdir(parents=True)
    (directory / "proposals").mkdir()
    (directory / "proposals" / "round.json").write_text(json.dumps({"target": "curves", "start": "griffin"}))
    meta = {"split": split, "warmup": 150, "samples": 150, "chains": 4}
    (directory / "fit.json").write_text(json.dumps(meta | ({"scoring": scoring} if scoring else {})))
    with open(directory / "report.jsonl", "w") as handle:
        for index, (source, loo, ok) in enumerate(members):
            stem = f"artifacts/{index:04d}-study"
            fitted().to_csv(directory / f"{stem}.fitted_train.csv", index=False)
            fitted(sentinel=True).to_csv(directory / f"{stem}.fitted_holdout.csv", index=False)
            np.savez(
                directory / f"{stem}.samples.npz.npz",
                b0=np.linspace(0.1, 0.9, 20).reshape(2, 10),
                log_eir=log_eir(),
                __grouped_by_chain__=np.asarray(True),
            )
            metrics = {
                "study.pointwise_loo_incidence": [loo, loo, loo],
                "study.pointwise_loo_prevalence": [loo, loo + 0.1],
                "study.pointwise_train_incidence": [loo + 1] * 3,
                "study.pointwise_train_prevalence": [loo + 1] * 2,
                "study.pointwise_holdout_incidence": [SENTINEL, -1.0],
                "study.pointwise_holdout_prevalence": [SENTINEL],
                "study.elpd_holdout": SENTINEL,
                "study.elpd_holdout_se": 1.0,
                "study.elpd_holdout_mcse": SENTINEL,
                "study.coverage90_holdout_prev_m": SENTINEL,
                "study.elpd_loo_train": 5 * loo,
                "study.elpd_loo_train_mcse": 0.3,
                "study.chain_weights": [0.4, 0.6],
                "study.effective_chains": 1.9,
                "study.kl.b0": 0.4,
                "parameters": 18.0,
                "unidentified": 1.0,
                "rhat_max": 1.01,
                "ess_tail_min": 300.0,
                "divergences": 0.0,
                "mcmc_seconds": 600.0,
                "study.fitted_train": {"artifact": f"{stem}.fitted_train.csv", "media_type": "text/csv"},
                "study.fitted_holdout": {"artifact": f"{stem}.fitted_holdout.csv", "media_type": "text/csv"},
                "study.samples.npz": {"artifact": f"{stem}.samples.npz.npz", "media_type": "application/npz"},
            }
            record = {
                "index": index,
                "digest": feedback.digest(source),
                "ok": ok,
                "model": source,
                "metrics": {"posterior": metrics if ok else {}},
                "errors": {} if ok else {"posterior": "GateError: prevalence must lie in [0, 1]"},
            }
            handle.write(json.dumps(record) + "\n")
    return directory


@pytest.fixture
def history(tmp_path):
    root = tmp_path / "runs"
    write_round(
        root / "beat" / "r01",
        [(INCUMBENT, -2.0, True), (variant("better"), -1.0, True), (variant("worse"), -3.0, True),
         (variant("broken"), 0.0, False)],
    )
    # another fold: its training rows are this fold's held-out ones
    write_round(root / "beat" / "other-fold", [(INCUMBENT, -2.0, True), (variant("elsewhere"), 9.0, True)],
                split="study_n3_seed7")
    return root


def test_the_pack_holds_nothing_from_the_held_out_fold(history, tmp_path):
    pack = feedback.write([history], tmp_path / "pack", split=SPLIT, pictures=False)
    files = [path for path in pack.rglob("*") if path.is_file()]
    assert files
    for path in files:
        text = path.read_text()
        assert "holdout" not in text.lower(), path
        assert str(SENTINEL) not in text and "98765" not in text, path


def test_the_pack_skips_rounds_fitted_under_another_split(history, tmp_path):
    pack = feedback.write([history], tmp_path / "pack", split=SPLIT, pictures=False)
    text = (pack / "FEEDBACK.md").read_text()
    assert "other-fold" not in text
    assert "elsewhere" not in "".join(p.read_text() for p in pack.rglob("blocks.py"))


def test_candidates_are_ranked_by_training_loo_against_their_rounds_incumbent(history, tmp_path):
    pack = feedback.write([history], tmp_path / "pack", split=SPLIT, pictures=False)
    names = sorted(path.name for path in (pack / "candidates").iterdir())
    assert names == ["01-beat_r01-1", "02-beat_r01-2"]
    assert "# better" in (pack / "candidates" / "01-beat_r01-1" / "blocks.py").read_text()
    board = (pack / "FEEDBACK.md").read_text()
    # +1 nat on each of five units, paired: a lead of +5.0 with no spread over
    # units, so its whole error is the two fits' Monte Carlo errors, 0.3 each
    assert "+5.0+-0.4 loo" in board
    assert "GateError: prevalence must lie in [0, 1]" in board


def test_a_candidates_directory_is_what_an_agent_needs_to_build_on_it(history, tmp_path):
    pack = feedback.write([history], tmp_path / "pack", split=SPLIT, pictures=False)
    best = pack / "candidates" / "01-beat_r01-1"
    assert {p.name for p in best.iterdir()} == {"blocks.py", "posterior.csv", "fitted_train.csv", "residuals.txt", "scores.json"}
    # blocks.py is in proposal.py's format: it splices back onto the incumbent
    from discovery.model import parse_blocks

    assert set(parse_blocks((best / "blocks.py").read_text())) == {"parameters", "curves"}
    posterior = pd.read_csv(best / "posterior.csv")
    assert list(posterior["parameter"]) == ["b0"], "per-site and per-study vectors are left out"
    scores = json.loads((best / "scores.json").read_text())
    assert set(scores) <= set(feedback.TRAIN) | {"study.kl.b0"}
    assert (pack / "incumbent" / "residuals.txt").is_file()


def test_no_finished_round_means_no_pack(tmp_path):
    (tmp_path / "drawn" / "proposals").mkdir(parents=True)
    assert feedback.write([tmp_path], tmp_path / "pack", split=SPLIT, pictures=False) is None


def test_the_table_for_people_shows_the_held_out_difference(history):
    text = rounds.table(feedback.candidates([history]))
    assert "d elpd/holdout" in text
    assert "beat/r01#0*" in text, "the incumbent is marked"
    # the two folds are reported apart, never ranked against each other
    assert text.count("## ") == 2


def test_promotion_picks_by_training_loo_and_pairs_with_the_incumbent(history, tmp_path):
    rows = feedback.candidates([history], SPLIT)
    to = tmp_path / "confirm"
    rounds.promote(rows, 1, to)
    written = sorted(p.name for p in (to / "proposals").glob("*.py"))
    assert written == ["0000-griffin.py", "0001-griffin.py"]
    assert (to / "proposals" / "0000-griffin.py").read_text() == INCUMBENT
    assert "# better" in (to / "proposals" / "0001-griffin.py").read_text()
    meta = json.loads((to / "proposals" / "round.json").read_text())
    assert meta["promoted_from"][0]["index"] == 1 and meta["candidates"] == 2


def test_a_site_the_chains_split_on_is_named_only_if_it_is_a_training_site(history, tmp_path):
    """Site 0 is in `fitted_train` and is named; site 4 split just as badly but has no
    training row -- it is held out -- so it must not be."""
    pack = feedback.write([history], tmp_path / "pack", split=SPLIT, pictures=False)
    text = (pack / "incumbent" / "residuals.txt").read_text()
    assert "Nouna (Ye_etal_2007): r-hat" in text
    assert text.count("r-hat") == 1
    assert "rhat" in (pack / "incumbent" / "posterior.csv").read_text().splitlines()[0]


def test_failures_of_the_machine_are_left_out(history, tmp_path):
    report = history / "beat" / "r01" / "report.jsonl"
    with report.open("a") as handle:
        handle.write(json.dumps({
            "index": 9, "digest": feedback.digest("x"), "ok": False, "model": "x", "metrics": {},
            "errors": {"posterior": "Traceback ...\nFileNotFoundError: [Errno 2] No such file or directory: "
                                    "'~/.cache/arviz/daily_warning.tmp'"},
        }) + "\n")
    text = (feedback.write([history], tmp_path / "pack", split=SPLIT, pictures=False) / "FEEDBACK.md").read_text()
    assert "GateError" in text
    assert "FileNotFoundError" not in text and "#9" not in text


def test_a_round_scored_before_the_current_scoring_is_left_out_until_rescored(tmp_path):
    root = tmp_path / "runs"
    write_round(root / "beat" / "old", [(INCUMBENT, -2.0, True), (variant("pooled"), 9.0, True)], scoring=None)
    write_round(root / "beat" / "new", [(INCUMBENT, -2.0, True), (variant("stacked"), -1.0, True)])
    pack = feedback.write([root], tmp_path / "pack", split=SPLIT, pictures=False)
    blocks = "".join(p.read_text() for p in pack.rglob("blocks.py"))
    assert "stacked" in blocks and "pooled" not in blocks
    # the table for people keeps both, apart
    text = rounds.table(feedback.candidates([root]))
    assert "pooled chains, marginal LOO" in text and "stacked chains, conditional LOO" in text


def test_the_residuals_separate_the_mechanism_from_the_study_effect(history, tmp_path):
    pack = feedback.write([history], tmp_path / "pack", split=SPLIT, pictures=False)
    text = (pack / "incumbent" / "residuals.txt").read_text()
    assert "obs/fit" in text and "own study effect" in text
    # 0.4 observed against 0.5 fitted, in every band
    assert "0.8" in text.split("By stream and age band:")[1].split("By study:")[0]
    # each chain's weight beside the EIR it settled on
    assert "(0.40)" in text and "(0.60)" in text
