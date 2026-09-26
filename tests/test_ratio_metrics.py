import math

import numpy as np
import pytest
from scipy import stats

from abtest import RatioResult, ratio_estimate, ratio_metric


def simulate_arm(rng, n_users, revenue_per_session=1.0, session_mean=3.0):
    """Users with a varying number of sessions and a user-level revenue effect.

    The user effect is what makes sessions from the same user correlated, which
    is exactly the structure a session-level t-test gets wrong.
    """
    sessions = 1 + rng.poisson(session_mean - 1, size=n_users)
    user_effect = rng.lognormal(mean=-0.125, sigma=0.5, size=n_users)  # mean 1
    revenue = sessions * revenue_per_session * user_effect * rng.gamma(4, 0.25, size=n_users)
    return revenue, sessions


def test_ratio_is_the_pooled_total_not_the_average_of_user_ratios():
    revenue = np.array([10.0, 2.0])
    sessions = np.array([10.0, 1.0])
    # Per-user ratios are 1.0 and 2.0, so their average is 1.5.
    assert ratio_estimate(revenue, sessions).ratio == pytest.approx(12 / 11)


def test_reduces_to_the_mean_when_every_denominator_is_one():
    rng = np.random.default_rng(0)
    y = rng.normal(5, 2, size=400)
    x = np.ones_like(y)
    est = ratio_estimate(y, x)
    assert est.ratio == pytest.approx(y.mean())
    # Var(R) collapses to Var(y)/n, the textbook variance of a sample mean.
    assert est.standard_error == pytest.approx(y.std(ddof=1) / math.sqrt(y.size))


def test_matches_a_z_test_on_means_when_denominators_are_one():
    rng = np.random.default_rng(1)
    a = rng.normal(5.0, 2.0, size=2000)
    b = rng.normal(5.3, 2.0, size=2000)
    r = ratio_metric(a, np.ones_like(a), b, np.ones_like(b))

    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    z = (b.mean() - a.mean()) / se
    assert r.z_stat == pytest.approx(z)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(abs(z)))


def test_delta_method_standard_error_matches_the_bootstrap():
    """The analytic variance should agree with resampling users."""
    rng = np.random.default_rng(42)
    revenue, sessions = simulate_arm(rng, 3000)
    analytic = ratio_estimate(revenue, sessions).standard_error

    idx = rng.integers(0, revenue.size, size=(2000, revenue.size))
    boot = revenue[idx].sum(axis=1) / sessions[idx].sum(axis=1)
    assert analytic == pytest.approx(boot.std(ddof=1), rel=0.06)


def test_naive_session_level_standard_error_is_too_small():
    """The reason the module exists: pooling sessions ignores within-user correlation."""
    rng = np.random.default_rng(7)
    revenue, sessions = simulate_arm(rng, 3000)
    delta_se = ratio_estimate(revenue, sessions).standard_error

    # Treat every session as an independent observation of revenue per session.
    per_session = np.repeat(revenue / sessions, sessions.astype(int))
    naive_se = per_session.std(ddof=1) / math.sqrt(per_session.size)
    assert naive_se < delta_se


def test_confidence_interval_covers_zero_under_the_null():
    """A/A experiments: the 95% interval should miss zero about 5% of the time."""
    rng = np.random.default_rng(2025)
    trials = 400
    misses = 0
    for _ in range(trials):
        rc, sc = simulate_arm(rng, 1500)
        rt, st = simulate_arm(rng, 1500)
        r = ratio_metric(rc, sc, rt, st)
        if not (r.ci_low <= 0 <= r.ci_high):
            misses += 1
    # Three Monte Carlo standard errors at 400 trials is about 0.033.
    assert 0.02 <= misses / trials <= 0.085


def test_naive_session_level_test_over_rejects_where_the_delta_method_does_not():
    rng = np.random.default_rng(303)
    trials = 300
    naive_rejects = delta_rejects = 0
    for _ in range(trials):
        rc, sc = simulate_arm(rng, 1200)
        rt, st = simulate_arm(rng, 1200)
        if ratio_metric(rc, sc, rt, st).significant:
            delta_rejects += 1
        a = np.repeat(rc / sc, sc.astype(int))
        b = np.repeat(rt / st, st.astype(int))
        if stats.ttest_ind(b, a, equal_var=False).pvalue < 0.05:
            naive_rejects += 1
    assert delta_rejects / trials <= 0.09
    assert naive_rejects > delta_rejects


def test_detects_a_real_lift():
    rng = np.random.default_rng(11)
    rc, sc = simulate_arm(rng, 6000, revenue_per_session=1.0)
    rt, st = simulate_arm(rng, 6000, revenue_per_session=1.10)
    r = ratio_metric(rc, sc, rt, st)
    assert r.significant
    assert r.relative_lift == pytest.approx(0.10, abs=0.04)
    assert r.relative_ci_low <= 0.10 <= r.relative_ci_high


def test_relative_interval_brackets_the_absolute_one():
    rng = np.random.default_rng(5)
    rc, sc = simulate_arm(rng, 2000)
    rt, st = simulate_arm(rng, 2000, revenue_per_session=1.05)
    r = ratio_metric(rc, sc, rt, st)
    assert r.relative_ci_low < r.relative_lift < r.relative_ci_high
    assert r.ci_low < r.absolute_lift < r.ci_high
    assert r.relative_ci_low == pytest.approx(r.ci_low / r.control_ratio, rel=0.05)


def test_users_with_no_sessions_stay_in_the_denominator_of_n():
    active_revenue = np.array([4.0, 6.0, 5.0])
    active_sessions = np.array([2.0, 3.0, 2.0])
    with_zeros_revenue = np.concatenate([active_revenue, np.zeros(3)])
    with_zeros_sessions = np.concatenate([active_sessions, np.zeros(3)])

    active = ratio_estimate(active_revenue, active_sessions)
    padded = ratio_estimate(with_zeros_revenue, with_zeros_sessions)
    assert padded.ratio == pytest.approx(active.ratio)
    assert padded.n_users == 6
    # More randomised users, same totals: the estimate is less certain per user
    # but the dilution is real information, not noise to be dropped.
    assert padded.variance != active.variance


def test_one_sided_test_reports_an_open_upper_bound():
    rng = np.random.default_rng(3)
    rc, sc = simulate_arm(rng, 800)
    rt, st = simulate_arm(rng, 800, revenue_per_session=1.2)
    r = ratio_metric(rc, sc, rt, st, two_sided=False)
    assert r.ci_high == math.inf
    assert r.relative_ci_high == math.inf
    assert r.p_value < 0.05


def test_to_dict_is_json_ready():
    rng = np.random.default_rng(4)
    rc, sc = simulate_arm(rng, 300)
    rt, st = simulate_arm(rng, 300)
    d = ratio_metric(rc, sc, rt, st).to_dict()
    assert d["significant"] in (True, False)
    assert set(RatioResult.__dataclass_fields__) <= set(d)
    assert all(isinstance(v, (int, float, bool)) for v in d.values())


def test_relative_figures_are_nan_when_a_ratio_is_not_positive():
    zero = np.zeros(50)
    denom = np.ones(50) * 2
    r = ratio_metric(zero, denom, np.ones(50), denom)
    assert math.isnan(r.relative_lift)
    assert math.isnan(r.relative_ci_low)
    assert r.absolute_lift == pytest.approx(0.5)


@pytest.mark.parametrize(
    "numerator, denominator",
    [
        ([1.0, 2.0], [1.0]),              # mismatched lengths
        ([1.0], [1.0]),                   # a single user
        ([1.0, 2.0], [0.0, 0.0]),         # denominator sums to zero
        ([1.0, 2.0], [1.0, -1.0]),        # negative denominator
        ([1.0, float("nan")], [1.0, 1.0]),
    ],
)
def test_rejects_invalid_input(numerator, denominator):
    with pytest.raises(ValueError):
        ratio_estimate(numerator, denominator)


def test_error_message_names_the_arm():
    good_y, good_x = [1.0, 2.0], [1.0, 1.0]
    with pytest.raises(ValueError, match="treatment"):
        ratio_metric(good_y, good_x, [1.0, 2.0], [0.0, 0.0])
