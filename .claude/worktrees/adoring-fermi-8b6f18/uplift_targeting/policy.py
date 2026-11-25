"""Turn model scores into decisions: which email, if any, each customer receives.

Arm codes: 0 = no email, 1 = men's email, 2 = women's email. Score matrices have one
column per email arm (column 0 is arm 1).
"""

from __future__ import annotations

import numpy as np


def best_arm(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each customer's highest-scoring email arm and that score."""
    return scores.argmax(axis=1) + 1, scores.max(axis=1)


def threshold_policy(uplift: np.ndarray, breakeven: float) -> np.ndarray:
    """Send the email with the largest predicted incremental revenue, but only if it
    is expected to pay for itself. Needs scores in dollars of incremental revenue."""
    arm, value = best_arm(uplift)
    return np.where(value > breakeven, arm, 0)


def top_fraction_policy(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Email the top `fraction` of customers by their best score, each with their
    best-scoring arm. Works for any score, including a response model's."""
    arm, value = best_arm(scores)
    n = len(value)
    k = int(round(fraction * n))
    order = np.argsort(-value, kind="stable")
    assign = np.zeros(n, dtype=int)
    assign[order[:k]] = arm[order[:k]]
    return assign


def blanket_policy(n: int, arm: int) -> np.ndarray:
    """Everyone gets the same arm."""
    return np.full(n, arm, dtype=int)
