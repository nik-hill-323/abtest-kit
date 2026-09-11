import numpy as np
import pytest

from abtest import bayes_means, bayes_proportions


def test_clear_winner_has_high_probability():
    r = bayes_proportions(100, 1000, 150, 1000)
    assert r.prob_treatment_better > 0.99
    assert r.expected_loss_choosing_treatment < 0.001
    assert r.lift_ci_low > 0


def test_symmetric_arms_are_a_coin_flip():
    r = bayes_proportions(100, 1000, 100, 1000)
    assert r.prob_treatment_better == pytest.approx(0.5, abs=0.01)
    assert r.expected_loss_choosing_treatment == pytest.approx(r.expected_loss_choosing_control, rel=0.05)


def test_posterior_mean_matches_beta_formula():
    r = bayes_proportions(30, 100, 40, 100, prior_alpha=1, prior_beta=1)
    assert r.control_estimate == pytest.approx(31 / 102, abs=0.003)
    assert r.treatment_estimate == pytest.approx(41 / 102, abs=0.003)


def test_bayes_means_detects_shift():
    rng = np.random.default_rng(0)
    r = bayes_means(rng.normal(0, 1, 2000), rng.normal(0.2, 1, 2000))
    assert r.prob_treatment_better > 0.99


def test_deterministic_with_seed():
    a = bayes_proportions(10, 100, 12, 100, seed=7)
    b = bayes_proportions(10, 100, 12, 100, seed=7)
    assert a == b
