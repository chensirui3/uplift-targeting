"""Uplift learners for a randomized experiment with a control arm and one or more
treatment arms: three arms in Hillstrom (no email, men's, women's), two in Criteo.

Every uplift learner answers the same question. For customer x and treatment arm a:

    tau_a(x) = E[Y | x, arm = a] - E[Y | x, control]

That is how much the treatment *changes* the outcome. A response model answers a
different question (how much will this customer spend if emailed?). It is included
because it is the baseline most marketing teams start from.

All learners share one interface:
    fit(X, arm, Y)     X: features, arm: 0 = control, 1..K = treatments, Y: outcomes table
    predict_uplift(X)  -> array (n_customers, K), predicted tau for each treatment arm
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

CONTROL = 0

# Two tree shapes, each tried for every learner in cross-validation. Weak, noisy
# signals usually need the conservative one; stronger signals can use the flexible one.
REGULARIZATION = {
    "flexible": dict(max_leaf_nodes=15, min_samples_leaf=100),
    "conservative": dict(max_leaf_nodes=7, min_samples_leaf=500),
}


def _gbm(binary: bool, seed: int, reg: str, min_leaf_floor: int = 0):
    shape = REGULARIZATION[reg]
    params = dict(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=shape["max_leaf_nodes"],
        min_samples_leaf=max(shape["min_samples_leaf"], min_leaf_floor),
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=seed,
    )
    if binary:
        return HistGradientBoostingClassifier(**params)
    return HistGradientBoostingRegressor(**params)


class _OutcomeModel:
    """Gradient-boosted estimate of E[Y | x]; a classifier when Y is 0/1."""

    def __init__(self, binary: bool, seed: int, reg: str, min_leaf_floor: int = 0):
        self.binary = binary
        self.model = _gbm(binary, seed, reg, min_leaf_floor)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_OutcomeModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.binary:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


def _is_binary(y: np.ndarray) -> bool:
    return bool(np.isin(y, (0, 1)).all())


class _Learner:
    name = "base"

    def __init__(self, target: str = "spend", seed: int = 0, reg: str = "flexible"):
        self.target = target
        self.seed = seed
        self.reg = reg

    def _y(self, Y: pd.DataFrame) -> np.ndarray:
        return Y[self.target].to_numpy(dtype=float)

    def _outcome_model(self, binary: bool, offset: int, min_leaf_floor: int = 0):
        return _OutcomeModel(binary, self.seed + offset, self.reg, min_leaf_floor)


class TLearner(_Learner):
    """One outcome model per arm; uplift is the difference between their predictions."""

    name = "T-learner"

    def fit(self, X, arm, Y):
        y = self._y(Y)
        self.arms_ = np.unique(arm)
        binary = _is_binary(y)
        self.models_ = [
            self._outcome_model(binary, int(a)).fit(X[arm == a], y[arm == a]) for a in self.arms_
        ]
        return self

    def predict_outcomes(self, X) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict_uplift(self, X) -> np.ndarray:
        mu = self.predict_outcomes(X)
        return mu[:, 1:] - mu[:, [0]]


class SLearner(_Learner):
    """A single model with the arm as an input feature; uplift is the change in its
    prediction when only the arm is switched. Trees can ignore the arm entirely, so
    this learner tends to shrink effects toward zero."""

    name = "S-learner"

    def _design(self, X, arm):
        onehot = (np.asarray(arm)[:, None] == self.arms_[None, 1:]).astype(float)
        return np.hstack([X, onehot])

    def fit(self, X, arm, Y):
        y = self._y(Y)
        self.arms_ = np.unique(arm)
        self.model_ = self._outcome_model(_is_binary(y), 0).fit(self._design(X, arm), y)
        return self

    def predict_outcomes(self, X) -> np.ndarray:
        n = len(X)
        return np.column_stack(
            [self.model_.predict(self._design(X, np.full(n, a))) for a in self.arms_]
        )

    def predict_uplift(self, X) -> np.ndarray:
        mu = self.predict_outcomes(X)
        return mu[:, 1:] - mu[:, [0]]


class XLearner(_Learner):
    """Kunzel et al. (2019). Impute each customer's individual effect using the other
    group's outcome model, then model those imputed effects directly."""

    name = "X-learner"

    def fit(self, X, arm, Y):
        y = self._y(Y)
        self.first_ = TLearner(self.target, self.seed, self.reg).fit(X, arm, Y)
        self.arms_ = self.first_.arms_
        mu = self.first_.predict_outcomes(X)
        self.second_ = []
        for j, a in enumerate(self.arms_[1:], start=1):
            treated, control = arm == a, arm == CONTROL
            # Treated customers: what they did minus what they'd have done without email.
            effect_treated = y[treated] - mu[treated, 0]
            # Control customers: what they'd have done with the email minus what they did.
            effect_control = mu[control, j] - y[control]
            tau_t = self._outcome_model(False, 10 + int(a)).fit(X[treated], effect_treated)
            tau_c = self._outcome_model(False, 20 + int(a)).fit(X[control], effect_control)
            # Weight by the propensity score. Assignment was random, so it is constant.
            g = treated.sum() / (treated.sum() + control.sum())
            self.second_.append((tau_t, tau_c, g))
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return np.column_stack(
            [g * tau_c.predict(X) + (1 - g) * tau_t.predict(X) for tau_t, tau_c, g in self.second_]
        )


class DRLearner(_Learner):
    """Kennedy (2023) doubly robust learner. Build a per-customer pseudo-outcome whose
    expectation is the true effect (using cross-fitted outcome models and the known
    assignment shares), then regress it on the features."""

    name = "DR-learner"

    def __init__(self, target="spend", seed=0, reg="flexible", n_folds: int = 5):
        super().__init__(target, seed, reg)
        self.n_folds = n_folds

    def fit(self, X, arm, Y):
        y = self._y(Y)
        self.arms_ = np.unique(arm)
        mu = np.zeros((len(y), len(self.arms_)))
        folds = KFold(self.n_folds, shuffle=True, random_state=self.seed)
        for fit_idx, pred_idx in folds.split(X):
            t = TLearner(self.target, self.seed, self.reg)
            mu[pred_idx] = t.fit(X[fit_idx], arm[fit_idx], Y.iloc[fit_idx]).predict_outcomes(
                X[pred_idx]
            )
        share = np.array([np.mean(arm == a) for a in self.arms_])
        self.models_ = []
        for j, a in enumerate(self.arms_[1:], start=1):
            pseudo = (
                mu[:, j]
                - mu[:, 0]
                + (arm == a) * (y - mu[:, j]) / share[j]
                - (arm == CONTROL) * (y - mu[:, 0]) / share[0]
            )
            # Pseudo-outcomes are very noisy, so the final model gets larger leaves.
            model = self._outcome_model(False, 30 + int(a), min_leaf_floor=300)
            self.models_.append(model.fit(X, pseudo))
        return self

    def predict_uplift(self, X) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])


class VisitProxy:
    """Train an uplift learner on site visits, then express its effects in units of the
    outcome that matters (`outcome`): dollars of spend for Hillstrom, conversions for
    Criteo.

    Purchases are rare (about 1 Hillstrom customer in 110, 3 Criteo users in 1,000), so
    the outcome that matters is very noisy. Visits are about 16 times more common in
    both. The wrapped learner models the visit effect, which is multiplied by the
    incremental outcome per incremental visit (estimated from the training data,
    separately for each treatment arm). The assumption: an extra visit is worth the
    same, on average, whichever customer makes it. Rankings, and therefore the targeting
    decisions in this project, do not depend on that assumption; only the scale of the
    predictions does.
    """

    def __init__(self, inner: _Learner, outcome: str = "spend"):
        if inner.target != "visit":
            raise ValueError(f"{type(self).__name__} expects a learner trained on 'visit'")
        self.inner = inner
        self.outcome = outcome
        self.name = f"{inner.name} on visits"

    def fit(self, X, arm, Y):
        self.inner.fit(X, arm, Y)
        value = Y[self.outcome].to_numpy(dtype=float)
        visit = Y["visit"].to_numpy(dtype=float)
        control = arm == CONTROL
        self.per_visit_ = np.array(
            [
                (value[arm == a].mean() - value[control].mean())
                / (visit[arm == a].mean() - visit[control].mean())
                for a in np.unique(arm)[1:]
            ]
        )
        return self

    @property
    def dollars_per_visit_(self) -> np.ndarray:
        """Hillstrom's name for `per_visit_`: incremental spend per incremental visit."""
        return self.per_visit_

    def predict_uplift(self, X) -> np.ndarray:
        return self.inner.predict_uplift(X) * self.per_visit_


# Hillstrom's name for the wrapper: there, the outcome that matters is spend.
VisitsToDollars = VisitProxy


class ResponseModel(_Learner):
    """"Target the likely buyers": predicts the outcome that matters (spend, or
    conversion for Criteo) *if treated*, one model per treatment arm.

    It never looks at the control group, so it cannot tell a customer the email
    persuades from one who would have bought anyway. Its scores are not uplift, so
    they are used only to rank customers.
    """

    name = "Response model"

    def fit(self, X, arm, Y):
        y = self._y(Y)
        self.arms_ = np.unique(arm)
        binary = _is_binary(y)
        self.models_ = [
            self._outcome_model(binary, int(a)).fit(X[arm == a], y[arm == a])
            for a in self.arms_[1:]
        ]
        return self

    def predict_response(self, X) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])


UPLIFT_LEARNERS = {
    "S-learner": SLearner,
    "T-learner": TLearner,
    "X-learner": XLearner,
    "DR-learner": DRLearner,
}


def build(learner: str, trained_on: str, reg: str, seed: int, outcome: str = "spend"):
    """Construct a candidate model, e.g. build("X-learner", "visits", "flexible", 7).

    `outcome` is the outcome that matters: spend for Hillstrom, conversion for Criteo.
    `trained_on` is either that outcome or "visits"."""
    if trained_on not in (outcome, "visits"):
        raise ValueError(f"trained_on must be {outcome!r} or 'visits', not {trained_on!r}")
    if learner == ResponseModel.name:
        return ResponseModel(outcome, seed, reg)
    if trained_on == "visits":
        return VisitProxy(UPLIFT_LEARNERS[learner]("visit", seed, reg), outcome)
    return UPLIFT_LEARNERS[learner](outcome, seed, reg)


def scores(model, X: np.ndarray) -> np.ndarray:
    """Ranking scores, one column per treatment arm: predicted uplift for uplift
    learners, the predicted outcome if treated for the response model."""
    if isinstance(model, ResponseModel):
        return model.predict_response(X)
    return model.predict_uplift(X)
