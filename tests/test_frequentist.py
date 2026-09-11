import numpy as np
import pytest
from scipy import stats

from abtest import ttest_means, ztest_proportions


def test_ztest_agrees_with_chi_square():
    r = ztest_proportions(480, 1000, 530, 1000)
    table = np.array([[480, 520], [530, 470]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    assert r.p_value == pytest.approx(p, rel=1e-6)
    assert r.z_stat**2 == pytest.approx(chi2, rel=1e-6)


def test_ztest_lift_and_ci():
    r = ztest_proportions(100, 1000, 130, 1000)
    assert r.absolute_lift == pytest.approx(0.03)
    assert r.relative_lift == pytest.approx(0.3)
    assert r.ci_low < 0.03 < r.ci_high
    assert r.significant


def test_ztest_no_effect():
    r = ztest_proportions(100, 1000, 100, 1000)
    assert r.p_value == pytest.approx(1.0)
    assert not r.significant


def test_ztest_one_sided_is_half_two_sided_when_positive():
    two = ztest_proportions(100, 1000, 120, 1000, two_sided=True)
    one = ztest_proportions(100, 1000, 120, 1000, two_sided=False)
    assert one.p_value == pytest.approx(two.p_value / 2)


def test_ztest_rejects_bad_counts():
    with pytest.raises(ValueError):
        ztest_proportions(1001, 1000, 10, 1000)


def test_ttest_recovers_known_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 5000)
    b = rng.normal(10.5, 2, 5000)
    r = ttest_means(a, b)
    assert r.difference == pytest.approx(0.5, abs=0.15)
    assert r.ci_low < 0.5 < r.ci_high
    assert r.significant


def test_ttest_matches_scipy():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=50), rng.normal(size=60)
    r = ttest_means(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_ttest_false_positive_rate_is_controlled():
    rng = np.random.default_rng(3)
    hits = sum(
        ttest_means(rng.normal(size=100), rng.normal(size=100)).significant for _ in range(400)
    )
    assert hits / 400 < 0.09
