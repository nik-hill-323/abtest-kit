import numpy as np
import pytest

from abtest import adjust, benjamini_hochberg, bonferroni
from abtest.frequentist import ztest_proportions

# The 15 p-values from Benjamini and Hochberg (1995), section 5, reanalysing the
# Needleman et al. neurobehavioural study. The paper reports that at q = 0.05 the
# step-up procedure rejects 4 hypotheses where Bonferroni rejects 3.
BH1995 = [
    0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
]


def test_bonferroni_matches_published_example():
    r = bonferroni(BH1995, alpha=0.05)
    assert r.n_significant == 3
    assert r.rejected[:4] == [True, True, True, False]
    # 0.0019 * 15 = 0.0285, the largest adjusted value still under 0.05
    assert r.adjusted_p_values[2] == pytest.approx(0.0285)


def test_benjamini_hochberg_matches_published_example():
    r = benjamini_hochberg(BH1995, alpha=0.05)
    assert r.n_significant == 4
    assert r.rejected[:5] == [True, True, True, True, False]
    # q_(4) = 15/4 * 0.0095 = 0.035625, and no larger p-value pulls it down
    assert r.adjusted_p_values[3] == pytest.approx(0.035625)


def test_bonferroni_caps_at_one():
    r = bonferroni([0.4, 0.6, 0.9], alpha=0.05)
    assert r.adjusted_p_values == [pytest.approx(1.0)] * 3
    assert r.n_significant == 0


def test_bonferroni_equals_alpha_over_m_threshold():
    p = [0.001, 0.02, 0.03, 0.2]
    r = bonferroni(p, alpha=0.05)
    expected = [v <= 0.05 / 4 for v in p]
    assert r.rejected == expected


def test_bh_adjusted_values_are_monotone_in_p():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, size=40)
    r = benjamini_hochberg(p)
    adjusted = np.asarray(r.adjusted_p_values)[np.argsort(p)]
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_bh_is_never_more_conservative_than_bonferroni():
    rng = np.random.default_rng(11)
    p = rng.beta(0.5, 5, size=25)
    bh = np.asarray(benjamini_hochberg(p).adjusted_p_values)
    bonf = np.asarray(bonferroni(p).adjusted_p_values)
    assert np.all(bh <= bonf + 1e-12)


def test_tied_p_values_share_an_adjusted_value():
    r = benjamini_hochberg([0.01, 0.01, 0.01, 0.9])
    assert r.adjusted_p_values[0] == r.adjusted_p_values[1] == r.adjusted_p_values[2]


def test_adjustment_follows_input_order():
    shuffled = [0.6, 0.0001, 0.3, 0.0004]
    r = benjamini_hochberg(shuffled, alpha=0.05)
    assert r.rejected == [False, True, False, True]
    assert r.p_values == shuffled


def test_metric_names_are_carried_through():
    r = adjust([0.001, 0.2, 0.04], method="bonferroni", metrics=["revenue", "ctr", "retention"])
    assert r.significant_metrics == ["revenue"]
    assert r.to_dict()["n_significant"] == 1


def test_default_metric_names():
    r = bonferroni([0.001, 0.5])
    assert r.metrics == ["metric_1", "metric_2"]


def test_adjust_dispatches_by_name():
    p = BH1995
    assert adjust(p, method="bh").method == "benjamini-hochberg"
    assert adjust(p, method="FDR").n_significant == benjamini_hochberg(p).n_significant
    assert adjust(p, method="bonferroni").n_significant == 3
    with pytest.raises(ValueError):
        adjust(p, method="sidak")


@pytest.mark.parametrize("bad", [[], [0.1, 1.2], [-0.01], [float("nan")]])
def test_rejects_invalid_p_values(bad):
    with pytest.raises(ValueError):
        bonferroni(bad)


def test_rejects_mismatched_metric_names():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, 0.2], metrics=["only_one"])


def test_bonferroni_controls_family_wise_error_under_the_null():
    """5 null metrics per experiment: uncorrected leaks, Bonferroni does not."""
    rng = np.random.default_rng(2024)
    n, rate, m_metrics, trials = 4000, 0.1, 5, 3000
    uncorrected = corrected = 0
    for _ in range(trials):
        p_values = []
        for _ in range(m_metrics):
            c = rng.binomial(n, rate)
            t = rng.binomial(n, rate)
            p_values.append(ztest_proportions(c, n, t, n).p_value)
        if min(p_values) < 0.05:
            uncorrected += 1
        if bonferroni(p_values, alpha=0.05).n_significant > 0:
            corrected += 1
    assert uncorrected / trials > 0.18  # nominal 1 - 0.95**5 = 0.226
    # Bonferroni holds the family-wise rate at alpha; the slack is three Monte
    # Carlo standard errors, which is 0.004 at this number of trials.
    assert corrected / trials <= 0.062


def test_bh_controls_false_discovery_rate():
    """8 nulls and 2 real effects: the share of false positives stays under alpha."""
    rng = np.random.default_rng(99)
    trials, n_null, n_real = 600, 8, 2
    false_discovery_proportions = []
    for _ in range(trials):
        nulls = rng.uniform(0, 1, size=n_null)
        reals = rng.uniform(0, 1, size=n_real) * 1e-4
        p_values = np.concatenate([nulls, reals])
        r = benjamini_hochberg(p_values, alpha=0.05)
        rejected = np.asarray(r.rejected)
        if rejected.any():
            false_discovery_proportions.append(rejected[:n_null].sum() / rejected.sum())
        else:
            false_discovery_proportions.append(0.0)
    assert np.mean(false_discovery_proportions) <= 0.05
