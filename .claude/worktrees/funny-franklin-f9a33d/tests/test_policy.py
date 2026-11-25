import numpy as np

from uplift_targeting.policy import blanket_policy, threshold_policy, top_fraction_policy


def test_threshold_policy_picks_the_best_email_only_when_it_pays():
    uplift = np.array([[0.9, 0.2], [0.1, 0.5], [0.2, 0.3]])
    np.testing.assert_array_equal(threshold_policy(uplift, breakeven=0.4), [1, 2, 0])


def test_top_fraction_policy_emails_the_highest_scores():
    scores = np.array([[0.1, 0.0], [0.9, 0.2], [0.3, 0.8], [0.5, 0.4]])
    np.testing.assert_array_equal(top_fraction_policy(scores, 0.5), [0, 1, 2, 0])
    assert (top_fraction_policy(scores, 0.0) == 0).all()
    assert (top_fraction_policy(scores, 1.0) > 0).all()


def test_blanket_policy():
    np.testing.assert_array_equal(blanket_policy(3, 1), [1, 1, 1])
