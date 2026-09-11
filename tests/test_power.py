import pytest

from abtest import mde_proportions, sample_size_means, sample_size_proportions


def test_sample_size_matches_textbook_value():
    # Classic example: 10% baseline, detect 12%, alpha 0.05 two-sided, power 0.8.
    # Standard tables give about 3,800 per arm.
    n = sample_size_proportions(0.10, 0.02)
    assert 3700 <= n <= 3900


def test_relative_mde_equals_absolute_equivalent():
    assert sample_size_proportions(0.10, 0.2, relative=True) == sample_size_proportions(0.10, 0.02)


def test_smaller_effects_need_more_users():
    assert sample_size_proportions(0.05, 0.005) > sample_size_proportions(0.05, 0.01)


def test_higher_power_needs_more_users():
    assert sample_size_proportions(0.05, 0.01, power=0.9) > sample_size_proportions(0.05, 0.01, power=0.8)


def test_means_sample_size():
    # (1.96 + 0.84)^2 * 2 * (10/2)^2 = 392 per arm
    assert sample_size_means(std=10, mde=2) == 393 or sample_size_means(std=10, mde=2) == 392


def test_mde_roundtrips_with_sample_size():
    n = sample_size_proportions(0.10, 0.02)
    mde = mde_proportions(0.10, n)
    assert mde == pytest.approx(0.02, rel=0.08)


@pytest.mark.parametrize("baseline", [0, 1, -0.1, 1.5])
def test_invalid_baseline(baseline):
    with pytest.raises(ValueError):
        sample_size_proportions(baseline, 0.01)
