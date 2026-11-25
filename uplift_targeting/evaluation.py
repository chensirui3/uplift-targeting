"""Model-free evaluation that relies on the randomization.

Each customer was assigned an arm at random (about one third each). So the customers
who happened to get arm `a` are a random sample of everyone, and the average outcome
that *any* targeting policy would produce can be estimated without trusting a model:

    V(policy) = sum over arms a of  mean over customers in arm a of  y * 1[policy picks a]

For each arm, only the customers who received what the policy would have sent them
count toward the estimate. This is the stratified (Hajek) inverse-propensity estimator.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

CONTROL = 0


def _as_matrix(assign: np.ndarray) -> np.ndarray:
    assign = np.asarray(assign)
    return assign[:, None] if assign.ndim == 1 else assign


def policy_value(assign: np.ndarray, arm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimated mean outcome per customer if every customer got `assign`.

    `assign` has shape (n,) for one policy or (n, P) for P policies at once.
    """
    A = _as_matrix(assign)
    value = np.zeros(A.shape[1])
    for a in np.unique(arm):
        in_arm = arm == a
        value += (y[in_arm, None] * (A[in_arm] == a)).mean(axis=0)
    return value if np.ndim(assign) == 2 else value[0]


def incremental_profit(assign, arm, y, margin: float, cost: float):
    """Profit per customer relative to emailing nobody:
    margin * (incremental revenue) - cost * (share of customers emailed)."""
    A = _as_matrix(assign)
    baseline = y[arm == CONTROL].mean()
    emailed = (A != CONTROL).mean(axis=0)
    profit = margin * (policy_value(A, arm, y) - baseline) - cost * emailed
    return profit if np.ndim(assign) == 2 else profit[0]


def uplift_curve(y: np.ndarray, treated: np.ndarray, score: np.ndarray, fractions: np.ndarray):
    """Incremental outcome per customer in the whole population from treating only the
    top `fraction` of customers by `score` (binary-treatment uplift curve).

    At fraction 1 this equals the average treatment effect; random targeting traces the
    straight line from 0 to that point.
    """
    order = np.argsort(-score, kind="stable")
    y, t = y[order], treated[order].astype(float)
    n = len(y)
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    k = np.clip(np.round(np.asarray(fractions) * n).astype(int), 1, n) - 1
    ok = (n_t[k] > 0) & (n_c[k] > 0)
    lift = np.where(ok, y_t[k] / np.maximum(n_t[k], 1) - y_c[k] / np.maximum(n_c[k], 1), 0.0)
    gain = lift * (k + 1) / n
    return np.where(np.asarray(fractions) == 0, 0.0, gain)


def area_over_random(fractions: np.ndarray, gain: np.ndarray) -> float:
    """Area between an uplift curve and the random-targeting line (higher is better)."""
    random_line = fractions * gain[-1]
    return float(np.trapezoid(gain - random_line, fractions))


def bootstrap_counts(arm: np.ndarray, n_boot: int, seed: int, batch: int = 250) -> Iterator[np.ndarray]:
    """Yield (batch, n) matrices of resampling counts. Customers are resampled within
    their arm, which keeps each arm's size fixed, mirroring the experiment's design."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(arm == a) for a in np.unique(arm)]
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        counts = np.zeros((b, len(arm)))
        for idx in members:
            counts[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=b)
        yield counts
        done += b


def bootstrap_profit(assign, arm, y, margin: float, cost: float, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, P) bootstrap replicates of `incremental_profit` for P policies."""
    A = _as_matrix(assign)
    arm_size = np.bincount(arm)[arm]  # size of each customer's arm
    match_y = y[:, None] * (A == arm[:, None])
    emailed = (A != CONTROL).astype(float)
    is_control_y = y * (arm == CONTROL)
    out = []
    for counts in bootstrap_counts(arm, n_boot, seed):
        w = counts / arm_size  # weights that give per-arm means
        value = w @ match_y
        baseline = w @ is_control_y
        share_emailed = (counts / len(y)) @ emailed
        out.append(margin * (value - baseline[:, None]) - cost * share_emailed)
    return np.vstack(out)


def bootstrap_uplift_curve(y, treated, score, fractions, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, len(fractions)) bootstrap replicates of `uplift_curve`.

    Resamples are drawn in the original customer order, so calls with the same data
    and seed reuse the same resamples and curves from different scores are paired.
    """
    order = np.argsort(-score, kind="stable")
    y, t = y[order], treated[order].astype(float)
    n = len(y)
    k = np.clip(np.round(np.asarray(fractions) * n).astype(int), 1, n) - 1
    out = []
    for counts in bootstrap_counts(np.asarray(treated).astype(int), n_boot, seed):
        counts = counts[:, order]
        n_t = np.cumsum(counts * t, axis=1)[:, k]
        n_c = np.cumsum(counts * (1 - t), axis=1)[:, k]
        y_t = np.cumsum(counts * y * t, axis=1)[:, k]
        y_c = np.cumsum(counts * y * (1 - t), axis=1)[:, k]
        lift = y_t / np.maximum(n_t, 1) - y_c / np.maximum(n_c, 1)
        share = (n_t + n_c) / n
        gain = np.where((n_t > 0) & (n_c > 0), lift * share, 0.0)
        out.append(np.where(np.asarray(fractions) == 0, 0.0, gain))
    return np.vstack(out)


# --- Large data (Criteo): one sort per ranking, Poisson bootstrap, optional weights -----


def ranking_cells(score: np.ndarray, treated: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Group customers once per ranking, so that every resample can reuse one sort.

    A customer's cell is 2 * j + treated, where j is the first point of the uplift curve
    (the top `fractions[j]` of customers by `score`, cut exactly as in `uplift_curve`)
    whose group includes them. `fractions` must be in ascending order.
    """
    fractions = np.asarray(fractions)
    if np.any(np.diff(fractions) < 0):
        raise ValueError("fractions must be in ascending order")
    n = len(score)
    k = np.clip(np.round(fractions * n).astype(int), 1, n) - 1
    order = np.argsort(-score, kind="stable")
    first_point = np.empty(n, dtype=np.intp)
    first_point[order] = np.searchsorted(k, np.arange(n), side="left")
    return 2 * first_point + np.asarray(treated).astype(np.intp)


def cell_uplift_curve(cells, y, fractions, count=None, weight=None) -> np.ndarray:
    """`uplift_curve` computed from `ranking_cells`, for one outcome (y of shape (n,)) or
    for several at once (the columns of y, one curve per row of the result).

    count:  how many times each customer is drawn, e.g. from `poisson_counts`. The
            default, once each, gives the plain estimate.
    weight: each customer's weight within their arm. The default, equal weights, is
            right when everyone had the same chance of treatment. Inverse propensity
            weights, 1 / P(own arm | features), correct for that chance varying with the
            features. The share of customers targeted is counted without weights.
    """
    fractions = np.asarray(fractions)
    n_points = len(fractions)
    Y = np.asarray(y, dtype=float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    count = np.ones(len(cells)) if count is None else np.asarray(count, dtype=float)
    w = count if weight is None else count * weight

    def cumulative(values):
        # Sums per (curve point, arm), accumulated down the ranking: one row per point,
        # columns control and treated. Customers past the last point fall in the
        # dropped final pair of cells.
        sums = np.bincount(cells, weights=values, minlength=2 * (n_points + 1))
        return sums[: 2 * n_points].reshape(n_points, 2).cumsum(axis=0)

    people = cumulative(count)
    in_arm = people if weight is None else cumulative(w)
    share = people.sum(axis=1) / count.sum()
    both_arms = (in_arm[:, 0] > 0) & (in_arm[:, 1] > 0)
    out = np.zeros((Y.shape[1], n_points))
    for j in range(Y.shape[1]):
        total = cumulative(w * Y[:, j])
        mean = np.divide(total, in_arm, out=np.zeros_like(total), where=in_arm > 0)
        out[j] = np.where(both_arms, (mean[:, 1] - mean[:, 0]) * share, 0.0)
    out[:, fractions == 0] = 0.0
    return out[0] if np.ndim(y) == 1 else out


def poisson_counts(n: int, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Yield `n_boot` vectors of resampling counts for a Poisson bootstrap.

    Each customer is drawn an independent Poisson(1) number of times, which approximates
    resampling n customers with replacement (Hanley and MacGibbon 2006). Unlike
    `bootstrap_counts`, one resample needs memory for only n numbers, so millions of
    customers can be resampled one resample at a time. Arm sizes vary slightly between
    resamples; `cell_uplift_curve` divides by each arm's resampled total.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.poisson(1.0, n).astype(float)


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval along axis 0."""
    lo = (1 - level) / 2
    return np.quantile(samples, lo, axis=0), np.quantile(samples, 1 - lo, axis=0)
