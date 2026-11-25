import hashlib

import numpy as np
import pandas as pd
import pytest

from uplift_targeting import criteo, modeling
from uplift_targeting.learners import ResponseModel


def simulate(n, seed):
    """Criteo-like data: 12 features, 85% treated. Ads raise visits for users with
    f0 > 0, and a visit converts more often when f1 > 0, so the conversion uplift
    depends on both."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    arm = (rng.random(n) < 0.85).astype(np.int64)
    visit = rng.random(n) < 0.05 + 0.25 * arm * (X[:, 0] > 0)
    conversion = visit & (rng.random(n) < 0.1 + 0.4 * (X[:, 1] > 0))
    Y = pd.DataFrame({"conversion": conversion.astype(np.int8), "visit": visit.astype(np.int8)})
    return X, arm, Y


def test_candidates_for_criteo_mirror_hillstrom():
    cands = modeling.candidates("conversion")
    assert len(cands) == 18
    assert {c["trained_on"] for c in cands} == {"conversion", "visits"}
    assert all(c["trained_on"] == "conversion" for c in cands if c["learner"] == ResponseModel.name)


def test_download_refuses_a_file_whose_checksum_changed(tmp_path):
    path = tmp_path / "data.csv.gz"
    path.write_bytes(b"not the dataset")
    good = hashlib.sha256(b"not the dataset").hexdigest()
    # The file is already there, so nothing is fetched.
    assert criteo.download(path, url="unused", expected_sha256=good) == path
    with pytest.raises(ValueError, match="SHA-256"):
        criteo.download(path, url="unused", expected_sha256="0" * 64)


def test_load_reads_the_criteo_layout_and_ignores_exposure(tmp_path):
    X, arm, Y = simulate(500, seed=0)
    raw = pd.DataFrame(X, columns=criteo.FEATURES)
    raw["treatment"], raw["conversion"], raw["visit"] = arm, Y["conversion"], Y["visit"]
    raw["exposure"] = arm * Y["visit"].to_numpy()
    path = tmp_path / "criteo.csv.gz"
    raw.to_csv(path, index=False, compression="gzip")
    X2, arm2, Y2 = criteo.load(path, expected_rows=500)
    np.testing.assert_allclose(X2, X, rtol=1e-6)
    np.testing.assert_array_equal(arm2, arm)
    assert list(Y2.columns) == ["conversion", "visit"]
    with pytest.raises(ValueError, match="rows"):
        criteo.load(path, expected_rows=501)


def test_difference_matches_the_plain_formula_and_ignores_constant_weights():
    rng = np.random.default_rng(1)
    y = rng.normal(size=10_000)
    treated = rng.random(10_000) < 0.5
    diff, se = criteo.difference(y, treated)
    assert diff == pytest.approx(y[treated].mean() - y[~treated].mean())
    assert se == pytest.approx(np.sqrt(y[treated].var() / treated.sum() + y[~treated].var() / (~treated).sum()))
    assert criteo.difference(y, treated, np.full(10_000, 3.0)) == pytest.approx((diff, se))


def test_select_applies_the_rule_to_uplift_models_only():
    table = pd.DataFrame(
        {
            "learner": ["Response model", "S-learner", "S-learner", "T-learner"],
            "trained_on": ["conversion", "conversion", "visits", "visits"],
            "area_over_random_per_100k": [50.0, 10.0, 30.0, 20.0],
        },
        index=["response", "s_conversion", "s_visits", "t_visits"],
    )
    assert criteo.select(table) == {
        "chosen": "s_visits",
        "best_visits": "s_visits",
        "best_conversion": "s_conversion",
        "best_response": "response",
    }


def test_propensity_check_sees_no_signal_when_treatment_is_random():
    X, arm, _ = simulate(20_000, seed=2)
    e_train, e_test, check = criteo.propensity(X[:15_000], arm[:15_000], X[15_000:])
    assert abs(check["auc"] - 0.5) < 4 * check["auc_null_se"]
    assert len(e_test) == 5_000 and np.all((e_train > 0) & (e_train < 1))


def test_propensity_check_detects_treatment_that_depends_on_the_features():
    X, _, _ = simulate(20_000, seed=3)
    rng = np.random.default_rng(3)
    arm = (rng.random(20_000) < np.where(X[:, 2] > 1, 0.97, 0.8)).astype(np.int64)
    _, _, check = criteo.propensity(X[:15_000], arm[:15_000], X[15_000:])
    assert check["auc"] > 0.5 + 4 * check["auc_null_se"]


def test_pipeline_runs_end_to_end_on_simulated_data(tmp_path):
    X, arm, Y = simulate(40_000, seed=4)
    cands = [
        {"learner": "Response model", "trained_on": "conversion", "reg": "conservative"},
        {"learner": "S-learner", "trained_on": "conversion", "reg": "conservative"},
        {"learner": "T-learner", "trained_on": "visits", "reg": "conservative"},
    ]
    results, figures = tmp_path / "results", tmp_path / "figures"
    summary = criteo.run_pipeline(X, arm, Y, results, figures, cv_users=15_000, n_boot=40,
                                  cands=cands, log=lambda *args: None)
    for name in ("experiment.json", "cv_comparison.csv", "selection.json", "summary.json",
                 "uplift_curves_cv.csv", "uplift_curves_test_conversion.csv",
                 "uplift_curves_test_visit.csv", "deciles.csv"):
        assert (results / name).exists()
    # The chosen model is the uplift candidate with the largest cross-validated area.
    table = pd.read_csv(results / "cv_comparison.csv", index_col="candidate")
    uplift = table[table["learner"] != ResponseModel.name]
    assert summary["selection"]["chosen"] == uplift["area_over_random_per_100k"].idxmax()
    # The simulated ads work for half the users, so targeting must beat random.
    lo, _ = summary["conversion"]["as_randomized"]["models"]["chosen"]["30pct"]["vs_random_ci"]
    assert lo > 0
    assert set(summary["power"]["as_randomized"]) == {"chosen_vs_random", "best_visits_vs_best_conversion"}
    deciles = pd.read_csv(results / "deciles.csv")
    assert deciles["users"].sum() == summary["users"]["test"]
    assert deciles["treated_share"].between(0.7, 1).all()
    counts = summary["outcome_counts"]
    assert counts["train"]["conversion"] + counts["test"]["conversion"] == int(Y["conversion"].sum())
    for name in ("deciles_conversions", "cv_vs_test", "cv_ranking"):
        assert (figures / f"{name}.png").exists() and (figures / f"{name}-dark.png").exists()
    # The test stage can be rerun from the stored selection, without cross-validating
    # again, and gives exactly the same results.
    again = criteo.run_pipeline(X, arm, Y, results, figures, n_boot=40, reuse_selection=True,
                                make_plots=False, log=lambda *args: None)
    for key in ("selection", "conversion", "visit", "power"):
        assert again[key] == summary[key]
