"""Replication on Criteo's uplift dataset: python -m uplift_targeting.criteo

In a set of advertising experiments, Criteo withheld ads at random from 15% of users
(Diemert et al. 2018; dataset v2.1: 13,979,592 users, 12 anonymized features). Two
outcomes were recorded: a visit to the advertiser's site (4.7% of users) and a
conversion (0.29%). The data has no money, so conversions play the role spend played
for Hillstrom.

Two questions, answered on held-out data:
  (a) Does an uplift model trained on visits rank users by conversion uplift better
      than one trained on conversions?
  (b) Does the uplift model chosen by the pre-set rule beat random targeting on
      conversions?

The protocol is Hillstrom's, scaled up:
1. Download and verify the data; check the randomization.
2. Split 60/40, stratified by treatment and conversion.
3. Cross-validate the same 18 candidates on a stratified 2-million-user subsample of
   the training set, and pick with SELECTION_RULE. The pick is written to disk before
   the test set is scored.
4. Refit the picked models on the full training set and evaluate them once on the test
   set, with 2,000 Poisson-bootstrap resamples for intervals.
5. Write every number to results/criteo/ and every chart to figures/criteo/.

Criteo states that treatment is independent of the features. The randomization check
tests that claim. So every test-set estimate is reported two ways: comparing treated
and control users directly, as the experiment's design intends ("as randomized"), and
reweighting them by each user's estimated chance of treatment ("reweighted").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config
from . import evaluation as ev
from .learners import ResponseModel, VisitProxy, _OutcomeModel, build, scores
from .modeling import BUDGETS, FRACTIONS, candidates, label
from .run import _jsonable

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ["conversion", "visit"]  # the outcome that matters first
ARM_LABELS = ("Control (no ads)", "Treated (ads)")
PER = 100_000  # effects are reported per 100,000 users
# Figures published on the dataset's page, to compare with what the file holds.
PUBLISHED = {
    "users": 13_979_592,
    "treated_share": 0.85,
    "visit_rate": 0.046992,
    "conversion_rate": 0.00292,
}

SELECTION_RULE = (
    "Pick the uplift model whose out-of-fold predictions give the largest area between "
    "the conversion uplift curve and the random-targeting line. Cross-validation: 5 "
    "folds on a stratified subsample of the training set (cv_users), comparing treated "
    "and control users directly, as the experiment's design intends."
)
ROLES = {
    "chosen": "Chosen uplift model",
    "best_visits": "Best uplift model trained on visits",
    "best_conversion": "Best uplift model trained on conversions",
    "best_response": "Response model (likely converters)",
}
ESTIMATORS = ("as_randomized", "reweighted")


def _log(*args) -> None:
    print(*args, flush=True)


# --- data ------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(
    path: Path = config.CRITEO_FILE,
    url: str = config.CRITEO_URL,
    expected_sha256: str = config.CRITEO_SHA256,
) -> Path:
    """Fetch the file once (311 MB) and refuse to continue if its checksum has changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _log(f"Downloading {url} (311 MB)")
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)  # only a complete download gets the final name
    digest = sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            f"{path} has SHA-256 {digest}, expected {expected_sha256}. "
            "Delete the file and rerun to download a fresh copy."
        )
    return path


def load(path: Path = config.CRITEO_FILE, expected_rows: int | None = config.CRITEO_ROWS):
    """Features (float32, to halve memory), treatment (0/1) and outcomes, one row per user.

    The file's `exposure` column (whether a treated user actually saw an ad) is recorded
    after assignment, so it is not used: the effect studied is that of being assigned
    to treatment.
    """
    dtypes = {f: np.float32 for f in FEATURES} | {c: np.int8 for c in ["treatment", *OUTCOMES]}
    raw = pd.read_csv(path, usecols=list(dtypes), dtype=dtypes)
    if expected_rows is not None and len(raw) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, found {len(raw):,}")
    for col in ["treatment", *OUTCOMES]:
        if not raw[col].isin((0, 1)).all():
            raise ValueError(f"Column {col} should hold only 0 and 1")
    X = raw[FEATURES].to_numpy()
    arm = raw["treatment"].to_numpy(dtype=np.int64)
    Y = raw[OUTCOMES].reset_index(drop=True)
    return X, arm, Y


def split(arm: np.ndarray, conversion: np.ndarray, seed: int = config.SEED):
    """Train and test indices, stratified by treatment and conversion, as for Hillstrom."""
    return train_test_split(
        np.arange(len(arm)), test_size=config.TEST_SIZE, random_state=seed,
        stratify=arm * 2 + conversion,
    )


def subsample(arm: np.ndarray, conversion: np.ndarray, size: int, seed: int = config.SEED):
    """Indices of a random subsample of `size` users, stratified the same way."""
    if size >= len(arm):
        return np.arange(len(arm))
    keep, _ = train_test_split(
        np.arange(len(arm)), train_size=size, random_state=seed, stratify=arm * 2 + conversion
    )
    return keep


# --- the experiment and its randomization -----------------------------------------------


def difference(y: np.ndarray, treated: np.ndarray, weight: np.ndarray | None = None):
    """Treated-minus-control difference in (weighted) mean outcome and its standard error.

    Each weighted mean is a ratio estimator, with variance sum(w^2 (y - mean)^2) / sum(w)^2
    by linearization. With equal weights that is the usual variance / n.
    """
    parts = []
    for group in (treated, ~treated):
        v = y[group]
        w = np.ones(len(v)) if weight is None else weight[group]
        mean = np.dot(w, v) / w.sum()
        parts.append((mean, np.dot(w**2, (v - mean) ** 2) / w.sum() ** 2))
    (mean_t, var_t), (mean_c, var_c) = parts
    return float(mean_t - mean_c), float(np.sqrt(var_t + var_c))


def experiment_summary(X: np.ndarray, arm: np.ndarray, Y: pd.DataFrame) -> dict:
    """Arm sizes, outcome rates, average effects and covariate balance, all users."""
    treated = arm == 1
    arms = {}
    for a, name in enumerate(ARM_LABELS):
        g = arm == a
        arms[name] = {"users": int(g.sum())} | {
            f"{y}_rate": float(Y[y].to_numpy()[g].mean()) for y in OUTCOMES
        }
    effects = {}
    for y in OUTCOMES:
        v = Y[y].to_numpy(dtype=float)
        diff, se = difference(v, treated)
        effects[y] = {
            "per_100k": diff * PER,
            "ci_per_100k": [(diff - 1.96 * se) * PER, (diff + 1.96 * se) * PER],
            "relative": diff / v[~treated].mean(),
        }
    smd = {}
    for j, f in enumerate(FEATURES):
        t, c = X[treated, j].astype(float), X[~treated, j].astype(float)
        smd[f] = abs(t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
    visit, conversion = Y["visit"].to_numpy(), Y["conversion"].to_numpy()
    return {
        "users": int(len(arm)),
        "treated_share": float(treated.mean()),
        "visit_rate": float(visit.mean()),
        "conversion_rate": float(conversion.mean()),
        "published": PUBLISHED,
        "arms": arms,
        "effects": effects,
        "conversions_without_a_visit": int(((conversion == 1) & (visit == 0)).sum()),
        "max_abs_standardized_mean_difference": max(smd.values()),
        "standardized_mean_difference": smd,
    }


def propensity(X_train, arm_train, X_test, seed: int = config.SEED):
    """Each user's estimated chance of treatment given the features (the propensity).

    If treatment were independent of the features, as Criteo states, no model could
    predict it better than chance: the area under the ROC curve (AUC) would be 0.5, give
    or take `auc_null_se`. The model is cross-fitted in two halves of the training set:
    each half is scored by the model fit on the other half, and test users by the
    average of the two. No outcome is used.
    """
    e_train, e_test = np.zeros(len(X_train)), np.zeros(len(X_test))
    halves = StratifiedKFold(2, shuffle=True, random_state=seed)
    for fit_idx, pred_idx in halves.split(X_train, arm_train):
        model = _OutcomeModel(True, seed, "flexible").fit(X_train[fit_idx], arm_train[fit_idx])
        e_train[pred_idx] = model.predict(X_train[pred_idx])
        e_test += model.predict(X_test) / 2
    n1 = int(arm_train.sum())
    n0 = len(arm_train) - n1
    check = {
        "auc": float(roc_auc_score(arm_train, e_train)),
        # Standard error of the AUC (a Mann-Whitney statistic) when the features carry
        # no information about treatment.
        "auc_null_se": float(np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))),
        "propensity_quantiles": {
            q: float(np.quantile(e_train, q)) for q in (0, 0.001, 0.01, 0.5, 0.99, 0.999, 1)
        },
    }
    return e_train, e_test, check


def inverse_propensity(e: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """Each user's weight within their arm: 1 / P(own arm | features)."""
    return np.where(arm == 1, 1 / e, 1 / (1 - e))


# --- model selection on the training set -------------------------------------------------


def out_of_fold_scores(X, arm, Y, cands, seed: int = config.SEED, log=_log) -> dict:
    """Score every cross-validation user with models that never saw them."""
    folds = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=seed)
    strata = arm * 2 + Y["conversion"].to_numpy()
    out = {label(c): np.zeros(len(X)) for c in cands}
    for fold, (fit_idx, pred_idx) in enumerate(folds.split(X, strata), start=1):
        X_fit, arm_fit, Y_fit, X_pred = X[fit_idx], arm[fit_idx], Y.iloc[fit_idx], X[pred_idx]
        for c in cands:
            start = time.time()
            model = build(c["learner"], c["trained_on"], c["reg"], seed, outcome="conversion")
            model.fit(X_fit, arm_fit, Y_fit)
            out[label(c)][pred_idx] = scores(model, X_pred)[:, 0]
            log(f"  fold {fold}/{config.N_FOLDS}  {label(c):<42} {time.time() - start:5.0f}s")
    return out


def _point(fraction: float) -> int:
    return int(np.argmin(np.abs(FRACTIONS - fraction)))


def cross_validate(X, arm, Y, weight, cands, log=_log) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold uplift curves of every candidate, summarized per 100,000 users."""
    oof = out_of_fold_scores(X, arm, Y, cands, log=log)
    treated = arm == 1
    conversion = Y["conversion"].to_numpy(dtype=float)
    visit = Y["visit"].to_numpy(dtype=float)
    rows = []
    for c in cands:
        s = oof[label(c)]
        curve = ev.uplift_curve(conversion, treated, s, FRACTIONS)
        reweighted = ev.cell_uplift_curve(
            ev.ranking_cells(s, treated, FRACTIONS), conversion, FRACTIONS, weight=weight
        )
        visits = ev.uplift_curve(visit, treated, s, FRACTIONS)
        row = dict(c)
        row["area_over_random_per_100k"] = ev.area_over_random(FRACTIONS, curve) * PER
        row["area_over_random_reweighted_per_100k"] = ev.area_over_random(FRACTIONS, reweighted) * PER
        row["area_over_random_visits_per_100k"] = ev.area_over_random(FRACTIONS, visits) * PER
        for b in (*BUDGETS, 1.0):
            row[f"gain_at_{int(b * 100)}pct_per_100k"] = curve[_point(b)] * PER
        rows.append(row)
    return pd.DataFrame(rows, index=[label(c) for c in cands]), oof


def select(table: pd.DataFrame) -> dict[str, str]:
    """Apply SELECTION_RULE, and pick the best candidate of each family the same way."""
    area = table["area_over_random_per_100k"]
    uplift = table["learner"] != ResponseModel.name
    families = {
        "chosen": uplift,
        "best_visits": uplift & (table["trained_on"] == "visits"),
        "best_conversion": uplift & (table["trained_on"] == "conversion"),
        "best_response": ~uplift,
    }
    return {role: area[mask].idxmax() for role, mask in families.items() if mask.any()}


def fit_by_label(lbl: str, X, arm, Y):
    learner, trained_on, reg = (part.strip() for part in lbl.split("|"))
    return build(learner, trained_on, reg, config.SEED, outcome="conversion").fit(X, arm, Y)


# --- the one-time evaluation on the test set --------------------------------------------


def evaluate(test_scores: dict, arm, Y, weight, n_boot: int, seed: int = config.SEED, log=_log):
    """Uplift curves of each ranking, for both outcomes and both estimators, plus
    Poisson-bootstrap replicates. Every curve uses the same resamples, so every
    difference between models (or estimators) is paired.

    Returns {estimator: {outcome: {"curves": {label: curve}, "boots": {label: (n_boot, F)}}}}.
    """
    treated = arm == 1
    values = Y[OUTCOMES].to_numpy(dtype=float)
    weights = {"as_randomized": None, "reweighted": weight}
    cells = {lbl: ev.ranking_cells(s, treated, FRACTIONS) for lbl, s in test_scores.items()}
    out = {est: {y: {"curves": {}, "boots": {}} for y in OUTCOMES} for est in ESTIMATORS}
    for lbl, s in test_scores.items():
        reweighted = ev.cell_uplift_curve(cells[lbl], values, FRACTIONS, weight=weight)
        for j, y in enumerate(OUTCOMES):
            out["as_randomized"][y]["curves"][lbl] = ev.uplift_curve(values[:, j], treated, s, FRACTIONS)
            out["reweighted"][y]["curves"][lbl] = reweighted[j]
            for est in ESTIMATORS:
                out[est][y]["boots"][lbl] = np.zeros((n_boot, len(FRACTIONS)))
    start = time.time()
    for b, count in enumerate(ev.poisson_counts(len(arm), n_boot, seed)):
        for lbl in test_scores:
            for est, w in weights.items():
                curves = ev.cell_uplift_curve(cells[lbl], values, FRACTIONS, count=count, weight=w)
                for j, y in enumerate(OUTCOMES):
                    out[est][y]["boots"][lbl][b] = curves[j]
        if (b + 1) % 200 == 0 or b + 1 == n_boot:
            log(f"  bootstrap {b + 1}/{n_boot}: {time.time() - start:.0f}s")
    return out


def _area(curves: np.ndarray) -> np.ndarray:
    """`ev.area_over_random` along the last axis, for many curves at once."""
    return np.trapezoid(curves - FRACTIONS * curves[..., -1:], FRACTIONS, axis=-1)


def summarize(curves: dict, boots: dict, roles: dict) -> dict:
    """Area over random targeting, gains at each budget, and paired differences between
    the roles, per 100,000 users with 95% bootstrap intervals."""

    def ci(samples):
        return [float(v) * PER for v in ev.interval(samples)]

    first = next(iter(curves))
    full, full_boot = curves[first][-1], boots[first][:, -1]  # the same for every ranking
    models = {}
    for role, lbl in roles.items():
        c, b = curves[lbl], boots[lbl]
        entry = {
            "area_over_random": ev.area_over_random(FRACTIONS, c) * PER,
            "area_ci": ci(_area(b)),
        }
        for budget in BUDGETS:
            i = _point(budget)
            entry[f"{int(budget * 100)}pct"] = {
                "gain": float(c[i]) * PER,
                "gain_ci": ci(b[:, i]),
                "share_of_full_gain": float(c[i] / full),
                "share_of_full_gain_ci": [float(v) for v in ev.interval(b[:, i] / full_boot)],
                "vs_random": float(c[i] - budget * full) * PER,
                "vs_random_ci": ci(b[:, i] - budget * full_boot),
            }
        models[role] = entry
    # "chosen" is the better of the two uplift families, so it is left out of the pairs.
    distinct = [r for r in roles if r != "chosen"]
    differences = {}
    for i, a in enumerate(distinct):
        for z in distinct[i + 1 :]:
            ca, cz, ba, bz = curves[roles[a]], curves[roles[z]], boots[roles[a]], boots[roles[z]]
            area_diff = _area(ba) - _area(bz)
            entry = {
                "area": {
                    "estimate": float(_area(ca) - _area(cz)) * PER,
                    "ci": ci(area_diff),
                    "share_of_resamples_above_zero": float((area_diff > 0).mean()),
                }
            }
            for budget in BUDGETS:
                k = _point(budget)
                d = ba[:, k] - bz[:, k]
                entry[f"{int(budget * 100)}pct"] = {
                    "estimate": float(ca[k] - cz[k]) * PER,
                    "ci": ci(d),
                    "share_of_resamples_above_zero": float((d > 0).mean()),
                }
            differences[f"{a} minus {z}"] = entry
    return {"full_gain": float(full) * PER, "full_gain_ci": ci(full_boot), "models": models,
            "differences": differences}


def power_analysis(conversion: dict, cv_table: pd.DataFrame, picks: dict, test_users: int) -> dict:
    """How small a gain could this test set detect, and is that small enough?

    As for Hillstrom: the standard error comes from the 95% bootstrap interval, and a
    gain is detectable with 80% power, in a two-sided test at the 5% level, when it is
    2.8 standard errors. All amounts are conversions per 100,000 users.
    """

    def se_of(lo, hi):
        return (hi - lo) / (2 * 1.96)

    chosen = cv_table.loc[picks["chosen"]]
    out = {"chosen_vs_random": {}, "best_visits_vs_best_conversion": {}}
    for budget in BUDGETS:
        key = f"{int(budget * 100)}pct"
        m = conversion["models"]["chosen"][key]
        se = se_of(*m["vs_random_ci"])
        cv_gain = chosen[f"gain_at_{key}_per_100k"] - budget * chosen["gain_at_100pct_per_100k"]
        times = (2.8 * se / cv_gain) ** 2 if cv_gain > 0 else None
        out["chosen_vs_random"][key] = {
            "standard_error": se,
            "detectable_gain": 2.8 * se,
            "observed_gain": m["vs_random"],
            "cv_suggested_gain": cv_gain,
            "times_current_test_set": times,
            "test_users_needed": None if times is None else times * test_users,
        }
    diff = conversion["differences"].get("best_visits minus best_conversion", {})
    for key, d in diff.items():
        se = se_of(*d["ci"])
        out["best_visits_vs_best_conversion"][key] = {
            "standard_error": se,
            "detectable_difference": 2.8 * se,
            "observed_difference": d["estimate"],
        }
    return out


def deciles(score, arm, Y, propensity_score) -> pd.DataFrame:
    """Observed effects within each tenth of test users ranked by `score`, per 100,000
    users, as randomized and reweighted. Ties keep the split's random order, as in the
    uplift curves. Also each tenth's share of treated users and mean estimated chance
    of treatment: in a randomized experiment, both would be 85% in every tenth."""
    n = len(score)
    order = np.argsort(-score, kind="stable")
    tenth = np.empty(n, dtype=int)
    tenth[order] = np.arange(n) * 10 // n + 1  # 1 = highest predicted uplift
    treated = arm == 1
    weight = inverse_propensity(propensity_score, arm)
    rows = []
    for d in range(1, 11):
        g = tenth == d
        row = {
            "decile": d,
            "users": int(g.sum()),
            "treated_share": float(treated[g].mean()),
            "mean_estimated_treatment_chance": float(propensity_score[g].mean()),
            "predicted_conversion_uplift_per_100k": float(score[g].mean()) * PER,
            "control_conversion_rate": float(Y["conversion"].to_numpy()[g & ~treated].mean()),
        }
        for y in OUTCOMES:
            v = Y[y].to_numpy(dtype=float)[g]
            for suffix, w in (("", None), ("_reweighted", weight[g])):
                diff, se = difference(v, treated[g], w)
                row[f"{y}_uplift{suffix}_per_100k"] = diff * PER
                row[f"{y}_uplift{suffix}_lo_per_100k"] = (diff - 1.96 * se) * PER
                row[f"{y}_uplift{suffix}_hi_per_100k"] = (diff + 1.96 * se) * PER
        rows.append(row)
    return pd.DataFrame(rows)


def curves_table(result: dict, outcome: str, roles: dict) -> pd.DataFrame:
    """Test-set uplift curves of every role, with 95% bands, per 100,000 users."""
    table = pd.DataFrame({"fraction_targeted": FRACTIONS})
    for est in ESTIMATORS:
        suffix = "" if est == "as_randomized" else "_reweighted"
        r = result[est][outcome]
        for role, lbl in roles.items():
            lo, hi = ev.interval(r["boots"][lbl])
            table[f"{role}{suffix}"] = r["curves"][lbl] * PER
            table[f"{role}{suffix}_lo"] = lo * PER
            table[f"{role}{suffix}_hi"] = hi * PER
    return table


# --- the pipeline ------------------------------------------------------------------------


def run_pipeline(
    X: np.ndarray,
    arm: np.ndarray,
    Y: pd.DataFrame,
    results_dir: Path = config.CRITEO_RESULTS_DIR,
    figures_dir: Path = config.CRITEO_FIGURES_DIR,
    cv_users: int = config.CRITEO_CV_USERS,
    n_boot: int = config.CRITEO_N_BOOTSTRAP,
    cands: list[dict] | None = None,
    reuse_selection: bool = False,
    make_plots: bool = True,
    log=_log,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cands = candidates("conversion") if cands is None else cands

    def write_json(name: str, payload) -> None:
        (results_dir / name).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    experiment = experiment_summary(X, arm, Y)
    log(
        f"Loaded {experiment['users']:,} users, {experiment['treated_share']:.1%} treated; "
        f"max |SMD| = {experiment['max_abs_standardized_mean_difference']:.3f}"
    )
    train_idx, test_idx = split(arm, Y["conversion"].to_numpy())
    X_train, arm_train = X[train_idx], arm[train_idx]
    Y_train = Y.iloc[train_idx].reset_index(drop=True)
    X_test, arm_test = X[test_idx], arm[test_idx]
    Y_test = Y.iloc[test_idx].reset_index(drop=True)

    # Randomization check on the training set. The same models give the test-set weights.
    e_train, e_test, check = propensity(X_train, arm_train, X_test)
    w_train, w_test = inverse_propensity(e_train, arm_train), inverse_propensity(e_test, arm_test)
    check["training_effects_per_100k"] = {
        y: {
            est: difference(Y_train[y].to_numpy(dtype=float), arm_train == 1, w)[0] * PER
            for est, w in (("as_randomized", None), ("reweighted", w_train))
        }
        for y in OUTCOMES
    }
    experiment["propensity_check"] = check
    write_json("experiment.json", experiment)
    log(f"Propensity AUC {check['auc']:.4f} (0.5 +/- {check['auc_null_se']:.4f} if independent)")

    # --- Model selection on the training set only ------------------------------------
    if reuse_selection:
        cv_table = pd.read_csv(results_dir / "cv_comparison.csv", index_col="candidate")
        stored = json.loads((results_dir / "selection.json").read_text())
        picks = {role: stored[role] for role in ROLES if role in stored}
        cv_size = stored["cv_users"]
    else:
        cv_idx = subsample(arm_train, Y_train["conversion"].to_numpy(), cv_users)
        cv_size = len(cv_idx)
        log(f"Cross-validating {len(cands)} candidates on {cv_size:,} training users")
        Y_cv = Y_train.iloc[cv_idx].reset_index(drop=True)
        cv_table, oof = cross_validate(X_train[cv_idx], arm_train[cv_idx], Y_cv, w_train[cv_idx], cands, log)
        picks = select(cv_table)
        # Later steps use the table as stored, so a rerun with reuse_selection matches.
        cv_table = cv_table.round(4)
        cv_table.to_csv(results_dir / "cv_comparison.csv", index_label="candidate")
        # The pick is on disk before any test-set outcome is looked at.
        write_json("selection.json", {"rule": SELECTION_RULE, "cv_users": cv_size, **picks})
        cv_curves = pd.DataFrame({"fraction_targeted": FRACTIONS})
        conversion_cv, treated_cv = Y_cv["conversion"].to_numpy(dtype=float), arm_train[cv_idx] == 1
        for role, lbl in picks.items():
            cv_curves[role] = ev.uplift_curve(conversion_cv, treated_cv, oof[lbl], FRACTIONS) * PER
        cv_curves.round(4).to_csv(results_dir / "uplift_curves_cv.csv", index=False)
    log("Chosen:", picks["chosen"])

    # --- One-time evaluation on the test set -----------------------------------------
    fitted = {}
    for lbl in dict.fromkeys(picks.values()):
        start = time.time()
        fitted[lbl] = fit_by_label(lbl, X_train, arm_train, Y_train)
        log(f"  fit on the full training set: {lbl} {time.time() - start:.0f}s")
    test_scores = {lbl: scores(model, X_test)[:, 0] for lbl, model in fitted.items()}
    result = evaluate(test_scores, arm_test, Y_test, w_test, n_boot, log=log)
    by_outcome = {
        y: {est: summarize(result[est][y]["curves"], result[est][y]["boots"], picks) for est in ESTIMATORS}
        for y in OUTCOMES
    }
    summary = {
        "selection": picks,
        "labels": {role: ROLES[role] for role in picks},
        "users": {"train": len(train_idx), "test": len(test_idx), "cross_validation": cv_size},
        "outcome_counts": {
            name: {y: int(part[y].sum()) for y in OUTCOMES}
            for name, part in (("train", Y_train), ("test", Y_test))
        },
        "bootstrap": {"method": "Poisson", "resamples": n_boot},
        "conversions_per_visit": {
            lbl: float(model.per_visit_[0]) for lbl, model in fitted.items() if isinstance(model, VisitProxy)
        },
        **by_outcome,
        "power": {
            est: power_analysis(by_outcome["conversion"][est], cv_table, picks, len(test_idx))
            for est in ESTIMATORS
        },
    }
    write_json("summary.json", summary)
    for y in OUTCOMES:
        curves_table(result, y, picks).round(4).to_csv(
            results_dir / f"uplift_curves_test_{y}.csv", index=False
        )
    deciles(test_scores[picks["chosen"]], arm_test, Y_test, e_test).round(4).to_csv(
        results_dir / "deciles.csv", index=False
    )
    if make_plots:
        from . import criteo_plots

        criteo_plots.make_all(results_dir, figures_dir)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reuse-selection", action="store_true",
        help="skip cross-validation and evaluate the picks already in results/criteo/selection.json",
    )
    args = parser.parse_args(argv)
    start = time.time()
    X, arm, Y = load(download())
    run_pipeline(X, arm, Y, reuse_selection=args.reuse_selection)
    _log(f"Done in {(time.time() - start) / 60:.0f} min. Results in {config.CRITEO_RESULTS_DIR}")


if __name__ == "__main__":
    main()
