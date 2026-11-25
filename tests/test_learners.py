import numpy as np
import pandas as pd
import pytest

from uplift_targeting import learners as lr
from uplift_targeting.policy import threshold_policy


def synthetic(n=30_000, seed=0):
    """Men's email adds $2 when x1 > 0; women's email adds $1 when x1 <= 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    arm = rng.integers(0, 3, size=n)
    tau = np.column_stack([2.0 * (X[:, 1] > 0), 1.0 * (X[:, 1] <= 0)])
    y0 = 1 + X[:, 0] + rng.normal(scale=0.5, size=n)
    effect = np.column_stack([np.zeros(n), tau])[np.arange(n), arm]
    Y = pd.DataFrame({"spend": y0 + effect})
    return X, arm, Y, tau


@pytest.mark.parametrize("name", ["S-learner", "T-learner", "X-learner", "DR-learner"])
def test_learners_recover_a_known_effect(name):
    X, arm, Y, tau = synthetic()
    model = lr.build(name, "spend", "flexible", 0).fit(X, arm, Y)
    X_new, _, _, tau_new = synthetic(n=5_000, seed=1)
    pred = model.predict_uplift(X_new)
    assert pred.shape == (5_000, 2)
    assert np.abs(pred - tau_new).mean() < 0.35
    # The decision is what matters: right email for each customer.
    right = np.where(X_new[:, 1] > 0, 1, 2)
    assert (threshold_policy(pred, breakeven=0.2) == right).mean() > 0.9


def test_visit_proxy_converts_visit_effects_to_dollars():
    rng = np.random.default_rng(2)
    n = 60_000
    X = rng.normal(size=(n, 2))
    arm = rng.integers(0, 3, size=n)
    p = 0.10 + 0.15 * (arm == 1) * (X[:, 0] > 0) + 0.10 * (arm == 2) * (X[:, 0] <= 0)
    visit = rng.binomial(1, p)
    Y = pd.DataFrame({"visit": visit, "spend": 10.0 * visit})  # every visit is worth $10
    model = lr.build("T-learner", "visits", "flexible", 0).fit(X, arm, Y)
    np.testing.assert_allclose(model.dollars_per_visit_, [10.0, 10.0])
    pred = model.predict_uplift(np.array([[1.0, 0.0], [-1.0, 0.0]]))
    np.testing.assert_allclose(pred, [[1.5, 0.0], [0.0, 1.0]], atol=0.25)


def test_response_model_favours_sure_things_and_uplift_does_not():
    """Half the customers buy whether or not they are emailed ("sure things"); the
    other half buy rarely but respond to the email ("persuadables")."""
    rng = np.random.default_rng(3)
    n = 40_000
    sure_thing = rng.integers(0, 2, size=n)
    X = np.column_stack([sure_thing, rng.normal(size=n)])
    arm = rng.integers(0, 3, size=n)
    emailed = arm > 0
    spend = np.where(sure_thing == 1, 5.0, 1.0 + 2.0 * emailed) + rng.normal(scale=0.5, size=n)
    Y = pd.DataFrame({"spend": spend})

    probe = np.array([[1.0, 0.0], [0.0, 0.0]])  # a sure thing, then a persuadable
    response = lr.build("Response model", "spend", "flexible", 0).fit(X, arm, Y)
    uplift = lr.build("T-learner", "spend", "flexible", 0).fit(X, arm, Y)
    response_scores, uplift_scores = lr.scores(response, probe), lr.scores(uplift, probe)
    assert response_scores[0].max() > response_scores[1].max()  # prefers the sure thing
    assert uplift_scores[1].max() > uplift_scores[0].max()  # prefers the persuadable


def test_visits_wrapper_rejects_a_learner_trained_on_spend():
    with pytest.raises(ValueError):
        lr.VisitsToDollars(lr.TLearner("spend", 0))


def test_paired_bootstrap_curves_share_resamples():
    from uplift_targeting import evaluation as ev

    rng = np.random.default_rng(5)
    y = rng.normal(size=500)
    treated = rng.integers(0, 2, size=500)
    fr = np.linspace(0, 1, 11)
    a = ev.bootstrap_uplift_curve(y, treated, rng.normal(size=500), fr, n_boot=50, seed=1)
    b = ev.bootstrap_uplift_curve(y, treated, rng.normal(size=500), fr, n_boot=50, seed=1)
    # Different rankings, same resamples: the end points (everyone treated) coincide.
    np.testing.assert_allclose(a[:, -1], b[:, -1])


def test_build_trains_on_the_outcome_that_matters():
    direct = lr.build("S-learner", "conversion", "flexible", 0, outcome="conversion")
    assert direct.target == "conversion"
    proxy = lr.build("S-learner", "visits", "flexible", 0, outcome="conversion")
    assert proxy.inner.target == "visit" and proxy.outcome == "conversion"
    assert lr.build("Response model", "conversion", "flexible", 0, outcome="conversion").target == "conversion"
    with pytest.raises(ValueError):
        lr.build("S-learner", "spend", "flexible", 0, outcome="conversion")


def test_visit_proxy_converts_visit_effects_to_conversions():
    rng = np.random.default_rng(10)
    n = 40_000
    X = rng.normal(size=(n, 2))
    arm = (rng.random(n) < 0.85).astype(int)  # two arms, as in Criteo
    visit = rng.binomial(1, 0.05 + 0.10 * arm * (X[:, 0] > 0))
    conversion = visit * rng.binomial(1, 0.2, size=n)
    Y = pd.DataFrame({"visit": visit, "conversion": conversion})
    model = lr.build("T-learner", "visits", "flexible", 0, outcome="conversion").fit(X, arm, Y)
    t, c = arm == 1, arm == 0
    expected = (conversion[t].mean() - conversion[c].mean()) / (visit[t].mean() - visit[c].mean())
    np.testing.assert_allclose(model.per_visit_, [expected])
    probe = np.array([[1.0, 0.0], [-1.0, 0.0]])
    pred = model.predict_uplift(probe)
    assert pred.shape == (2, 1)
    np.testing.assert_allclose(pred, model.inner.predict_uplift(probe) * expected)
