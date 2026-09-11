import numpy as np
import pytest

from abtest import cuped_adjust, ttest_means


def test_cuped_reduces_variance_and_keeps_lift():
    rng = np.random.default_rng(5)
    n = 4000
    pre_c = rng.normal(50, 10, n)
    pre_t = rng.normal(50, 10, n)
    post_c = pre_c * 0.8 + rng.normal(0, 5, n)
    post_t = pre_t * 0.8 + rng.normal(0, 5, n) + 1.0  # true lift of 1.0

    raw = ttest_means(post_c, post_t)
    adj = cuped_adjust(post_c, pre_c, post_t, pre_t)
    cuped = ttest_means(adj.control_adjusted, adj.treatment_adjusted)

    assert adj.variance_reduction > 0.5
    assert cuped.difference == pytest.approx(raw.difference, abs=0.3)
    assert (cuped.ci_high - cuped.ci_low) < 0.75 * (raw.ci_high - raw.ci_low)


def test_uncorrelated_covariate_does_nothing_much():
    rng = np.random.default_rng(6)
    y_c, y_t = rng.normal(size=500), rng.normal(size=500)
    x_c, x_t = rng.normal(size=500), rng.normal(size=500)
    adj = cuped_adjust(y_c, x_c, y_t, x_t)
    assert abs(adj.theta) < 0.15
    assert adj.variance_reduction < 0.05


def test_constant_covariate_is_safe():
    adj = cuped_adjust([1, 2, 3], [5, 5, 5], [2, 3, 4], [5, 5, 5])
    assert adj.theta == 0.0
    assert adj.variance_reduction == 0.0
