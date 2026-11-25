"""Model selection: cross-validation on the training set only, with a rule fixed in
advance, so the test set is used exactly once."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config
from . import evaluation as ev
from .data import arrays
from .learners import REGULARIZATION, UPLIFT_LEARNERS, ResponseModel, build, scores

FRACTIONS = np.linspace(0, 1, 101)
BUDGETS = (0.1, 0.3, 0.5)

SELECTION_RULE = (
    "Pick the uplift model whose out-of-fold predictions give the largest area between "
    "the men's-email uplift curve (incremental spend) and the random-targeting line."
)


def candidates(outcome: str = "spend") -> list[dict]:
    """The 18 candidates: a response model, and S-, T-, X- and DR-learners trained on
    `outcome` (the outcome that matters) or on visits, each at two levels of
    regularization."""
    out = [
        {"learner": ResponseModel.name, "trained_on": outcome, "reg": reg}
        for reg in REGULARIZATION
    ]
    for learner in UPLIFT_LEARNERS:
        for trained_on in (outcome, "visits"):
            for reg in REGULARIZATION:
                out.append({"learner": learner, "trained_on": trained_on, "reg": reg})
    return out


def label(c: dict) -> str:
    return f"{c['learner']} | {c['trained_on']} | {c['reg']}"


def out_of_fold_scores(df: pd.DataFrame, cands: list[dict]) -> dict[str, np.ndarray]:
    """Score every training customer with models that never saw them."""
    X, arm, Y = arrays(df)
    folds = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.SEED)
    strata = arm * 2 + df["conversion"].to_numpy()
    out = {label(c): np.zeros((len(df), 2)) for c in cands}
    for fit_idx, pred_idx in folds.split(X, strata):
        for c in cands:
            model = build(c["learner"], c["trained_on"], c["reg"], config.SEED)
            model.fit(X[fit_idx], arm[fit_idx], Y.iloc[fit_idx])
            out[label(c)][pred_idx] = scores(model, X[pred_idx])
    return out


def mens_pair(df: pd.DataFrame) -> np.ndarray:
    """Customers in the men's-email arm or the control arm."""
    return df["arm"].isin((0, 1)).to_numpy()


def mens_curve(df: pd.DataFrame, score: np.ndarray, outcome: str = "spend") -> np.ndarray:
    """Uplift curve for sending the men's email to the top fraction by `score`."""
    pair = mens_pair(df)
    y = df[outcome].to_numpy(dtype=float)[pair]
    treated = df["arm"].to_numpy()[pair] == 1
    return ev.uplift_curve(y, treated, score[pair, 0], FRACTIONS)


def cross_validate(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cands = candidates()
    oof = out_of_fold_scores(train, cands)
    rows = []
    for c in cands:
        s = oof[label(c)]
        curve = mens_curve(train, s)
        row = dict(c)
        row["area_over_random"] = ev.area_over_random(FRACTIONS, curve)
        row["area_over_random_visits"] = ev.area_over_random(
            FRACTIONS, mens_curve(train, s, "visit")
        )
        for b in BUDGETS:
            row[f"gain_at_{int(b * 100)}pct"] = float(np.interp(b, FRACTIONS, curve))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.index = [label(c) for c in cands]
    return table, oof


def select(table: pd.DataFrame) -> dict[str, str]:
    """Apply the selection rule, plus the best of each comparison family."""
    uplift = table[table["learner"] != ResponseModel.name]
    return {
        "chosen": uplift["area_over_random"].idxmax(),
        "best_response": table[table["learner"] == ResponseModel.name]["area_over_random"].idxmax(),
        "best_spend_uplift": uplift[uplift["trained_on"] == "spend"]["area_over_random"].idxmax(),
    }


def fit_by_label(lbl: str, train: pd.DataFrame):
    learner, trained_on, reg = (part.strip() for part in lbl.split("|"))
    X, arm, Y = arrays(train)
    return build(learner, trained_on, reg, config.SEED).fit(X, arm, Y)
