import numpy as np
import pytest

from uplift_targeting import evaluation as ev


def simulate_experiment(n, seed):
    """Three-arm randomized experiment with known potential outcomes."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    arm = rng.integers(0, 3, size=n)
    y0 = 1 + x + rng.normal(size=n)
    potential = np.column_stack([y0, y0 + 2 * (x > 0), y0 + 1 * (x <= 0)])
    y = potential[np.arange(n), arm]
    return x, arm, y, potential


def test_policy_value_is_unbiased_for_a_targeted_policy():
    x, arm, y, potential = simulate_experiment(300_000, seed=0)
    policy = np.where(x > 0, 1, 2)
    truth = potential[np.arange(len(x)), policy].mean()
    estimate = ev.policy_value(policy, arm, y)
    assert estimate == pytest.approx(truth, abs=0.02)


def test_policy_value_accepts_many_policies_at_once():
    x, arm, y, _ = simulate_experiment(5_000, seed=1)
    policies = np.column_stack([np.zeros_like(arm), np.ones_like(arm), np.where(x > 0, 1, 2)])
    together = ev.policy_value(policies, arm, y)
    one_by_one = [ev.policy_value(policies[:, p], arm, y) for p in range(3)]
    np.testing.assert_allclose(together, one_by_one)


def test_profit_of_emailing_nobody_is_zero():
    _, arm, y, _ = simulate_experiment(5_000, seed=2)
    nobody = np.zeros_like(arm)
    assert ev.incremental_profit(nobody, arm, y, margin=0.3, cost=0.1) == pytest.approx(0.0)


def test_uplift_curve_ends_at_the_average_effect():
    rng = np.random.default_rng(3)
    y = rng.normal(size=2_000)
    treated = rng.integers(0, 2, size=2_000)
    gain = ev.uplift_curve(y, treated, rng.normal(size=2_000), np.array([0.0, 0.5, 1.0]))
    assert gain[0] == 0.0
    assert gain[-1] == pytest.approx(y[treated == 1].mean() - y[treated == 0].mean())


def test_uplift_curve_matches_scikit_uplift():
    sklift_metrics = pytest.importorskip("sklift.metrics")
    rng = np.random.default_rng(4)
    n = 3_000
    y = rng.binomial(1, 0.2, size=n).astype(float)
    treated = rng.integers(0, 2, size=n)
    score = rng.normal(size=n)  # continuous, so there are no ties
    num_all, theirs = sklift_metrics.uplift_curve(y, score, treated)
    ours = ev.uplift_curve(y, treated, score, num_all / n) * n
    # Skip the first few cutoffs, where one group can still be empty.
    np.testing.assert_allclose(ours[20:], theirs[20:], rtol=1e-9, atol=1e-9)


def test_area_over_random_is_zero_for_random_targeting():
    fractions = np.linspace(0, 1, 11)
    assert ev.area_over_random(fractions, 0.8 * fractions) == pytest.approx(0.0)
    assert ev.area_over_random(fractions, np.sqrt(fractions)) > 0


def test_bootstrap_profit_is_centered_on_the_estimate():
    x, arm, y, _ = simulate_experiment(20_000, seed=5)
    policies = np.column_stack([np.ones_like(arm), np.where(x > 0, 1, 2)])
    point = ev.incremental_profit(policies, arm, y, margin=0.3, cost=0.1)
    reps = ev.bootstrap_profit(policies, arm, y, 0.3, 0.1, n_boot=400, seed=0)
    assert reps.shape == (400, 2)
    np.testing.assert_allclose(reps.mean(axis=0), point, atol=0.01)
    lo, hi = ev.interval(reps)
    assert np.all(lo < point) and np.all(point < hi)


def test_bootstrap_keeps_arm_sizes_fixed():
    arm = np.repeat([0, 1, 2], [10, 20, 30])
    for counts in ev.bootstrap_counts(arm, n_boot=5, seed=0, batch=5):
        for a, size in zip([0, 1, 2], [10, 20, 30]):
            assert np.all(counts[:, arm == a].sum(axis=1) == size)


# --- large-data versions (Criteo) ---------------------------------------------------------


def test_cell_uplift_curve_matches_uplift_curve():
    rng = np.random.default_rng(6)
    n = 5_000
    y = rng.binomial(1, 0.1, size=n).astype(float)
    treated = rng.random(n) < 0.85
    score = np.round(rng.normal(size=n), 1)  # many ties
    fractions = np.linspace(0, 1, 101)
    cells = ev.ranking_cells(score, treated, fractions)
    np.testing.assert_allclose(
        ev.cell_uplift_curve(cells, y, fractions),
        ev.uplift_curve(y, treated, score, fractions),
        rtol=1e-9, atol=1e-12,
    )


def test_cell_uplift_curve_takes_several_outcomes_at_once():
    rng = np.random.default_rng(7)
    n = 3_000
    y = rng.binomial(1, [0.05, 0.3], size=(n, 2)).astype(float)
    treated = rng.integers(0, 2, size=n)
    fractions = np.linspace(0, 1, 11)
    cells = ev.ranking_cells(rng.normal(size=n), treated, fractions)
    both = ev.cell_uplift_curve(cells, y, fractions)
    assert both.shape == (2, 11)
    for j in range(2):
        np.testing.assert_allclose(both[j], ev.cell_uplift_curve(cells, y[:, j], fractions))


def test_reweighting_corrects_a_chance_of_treatment_that_varies_with_the_features():
    """Customers with x > 1 are treated 95% of the time, the rest 80%, and they buy more
    anyway. Comparing the arms directly then overstates the effect (+0.05 for everyone);
    inverse propensity weights recover it."""
    rng = np.random.default_rng(8)
    n = 400_000
    x = rng.normal(size=n)
    e = np.where(x > 1, 0.95, 0.80)
    treated = rng.random(n) < e
    y = (rng.random(n) < 0.05 + 0.30 * (x > 1) + 0.05 * treated).astype(float)
    fractions = np.array([0.0, 0.3, 1.0])
    cells = ev.ranking_cells(x, treated, fractions)
    truth = 0.05 * fractions
    direct = ev.cell_uplift_curve(cells, y, fractions)
    weight = np.where(treated, 1 / e, 1 / (1 - e))
    reweighted = ev.cell_uplift_curve(cells, y, fractions, weight=weight)
    assert direct[1] > truth[1] + 0.02
    np.testing.assert_allclose(reweighted, truth, atol=0.006)


def test_poisson_bootstrap_agrees_with_the_multinomial_bootstrap():
    rng = np.random.default_rng(9)
    n = 20_000
    y = rng.binomial(1, 0.1, size=n).astype(float)
    treated = rng.integers(0, 2, size=n)
    score = rng.normal(size=n)
    fractions = np.linspace(0, 1, 11)
    cells = ev.ranking_cells(score, treated, fractions)
    poisson = np.array(
        [ev.cell_uplift_curve(cells, y, fractions, count=c) for c in ev.poisson_counts(n, 300, seed=0)]
    )
    multinomial = ev.bootstrap_uplift_curve(y, treated, score, fractions, n_boot=300, seed=0)
    ratio = poisson[:, 1:].std(axis=0) / multinomial[:, 1:].std(axis=0)
    assert np.all((ratio > 0.8) & (ratio < 1.25))
    np.testing.assert_allclose(poisson.mean(axis=0), ev.uplift_curve(y, treated, score, fractions), atol=0.002)


def test_poisson_counts_are_reproducible():
    first = list(ev.poisson_counts(1_000, 3, seed=1))
    again = list(ev.poisson_counts(1_000, 3, seed=1))
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    assert abs(np.mean(first) - 1) < 0.05


def test_ranking_cells_rejects_unsorted_fractions():
    with pytest.raises(ValueError):
        ev.ranking_cells(np.arange(5.0), np.ones(5), np.array([0.5, 0.1]))
