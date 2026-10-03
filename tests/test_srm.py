import numpy as np
import pytest

from abtest import srm_check
from abtest.cli import main


def test_matches_kohavi_textbook_example():
    # Kohavi, Tang, Xu (2020), ch. 3: 821,588 vs 815,482 users on a 50/50 split
    # has a p-value of about 1.8e-6, an SRM despite a ratio of 0.993.
    r = srm_check([821588, 815482])
    assert r.p_value == pytest.approx(1.8e-6, rel=0.02)
    assert r.mismatch
    assert r.df == 1


def test_two_arm_chi2_equals_squared_z_for_one_proportion():
    a, b = 50_400, 49_600
    n = a + b
    z = (a - n / 2) / np.sqrt(n * 0.25)
    r = srm_check([a, b])
    assert r.chi2 == pytest.approx(z**2)


def test_exact_split_has_no_mismatch():
    r = srm_check([5000, 5000])
    assert r.chi2 == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    assert not r.mismatch


def test_unequal_allocation_uses_configured_ratios():
    # 10/90 split: 1,000 vs 9,000 is exactly on target, but would be a huge SRM
    # if the default 50/50 split were assumed.
    assert not srm_check([1000, 9000], expected_ratios=[0.1, 0.9]).mismatch
    assert srm_check([1000, 9000]).mismatch


def test_ratios_are_normalised():
    r1 = srm_check([3100, 2950, 6100], expected_ratios=[1, 1, 2])
    r2 = srm_check([3100, 2950, 6100], expected_ratios=[0.25, 0.25, 0.5])
    assert r1.p_value == pytest.approx(r2.p_value)
    assert r1.expected == pytest.approx([3037.5, 3037.5, 6075.0])
    assert r1.df == 2
    assert r1.arms == ["control", "treatment_1", "treatment_2"]


def test_false_alarm_rate_under_correct_assignment_is_near_threshold():
    rng = np.random.default_rng(0)
    threshold = 0.05
    alarms = [
        srm_check(rng.multinomial(20_000, [0.5, 0.5]), threshold=threshold).mismatch
        for _ in range(2000)
    ]
    assert np.mean(alarms) == pytest.approx(threshold, abs=0.015)


def test_detects_one_percent_loss_in_treatment():
    rng = np.random.default_rng(1)
    control, treatment = rng.multinomial(1_000_000, [0.5, 0.5])
    treatment = int(treatment * 0.99)  # 1% of treatment users never logged
    assert srm_check([control, treatment]).mismatch


def test_observed_ratios_and_to_dict():
    r = srm_check([600, 400], arms=["A", "B"])
    assert r.observed_ratios == pytest.approx([0.6, 0.4])
    d = r.to_dict()
    assert d["arms"] == ["A", "B"]
    assert d["observed_ratios"] == pytest.approx([0.6, 0.4])
    assert d["mismatch"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observed": [100]},
        {"observed": [100, -1]},
        {"observed": [100.5, 99]},
        {"observed": [0, 0]},
        {"observed": [100, 100], "expected_ratios": [0.5]},
        {"observed": [100, 100], "expected_ratios": [0.5, 0.0]},
        {"observed": [100, 100], "threshold": 1.5},
        {"observed": [100, 100], "arms": ["only_one"]},
    ],
)
def test_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        srm_check(**kwargs)


def test_cli_srm(capsys):
    main(["srm", "--counts", "821588", "815482"])
    out = capsys.readouterr().out
    assert '"mismatch": true' in out


def test_cli_srm_with_ratios(capsys):
    main(["srm", "--counts", "1000", "9000", "--ratios", "0.1", "0.9"])
    out = capsys.readouterr().out
    assert '"mismatch": false' in out
