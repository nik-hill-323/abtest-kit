import pytest

from abtest import mde_means, mde_proportions, sample_size_means, sample_size_proportions


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
    # (1.96 + 0.8416)^2 * 2 * (10/2)^2 = 392.4, rounded up to 393 per arm
    assert sample_size_means(std=10, mde=2) == 393


# Cohen, Statistical Power Analysis for the Behavioral Sciences (2nd ed., 1988),
# table 2.4.1: per-group n for a two-sample t-test at alpha = 0.05 two-tailed
# and power = 0.80, indexed by effect size d = mde / std.
COHEN_TABLE = [(0.2, 393), (0.3, 175), (0.4, 99), (0.5, 64), (0.8, 26), (1.0, 17)]


@pytest.mark.parametrize("d,published_n", COHEN_TABLE)
def test_means_sample_size_matches_cohen_table(d, published_n):
    n = sample_size_means(std=1.0, mde=d)
    # The normal approximation ignores the t-distribution's heavier tails, so it
    # can come in one user per arm below the published figure, never above it.
    assert published_n - 1 <= n <= published_n


def test_means_sample_size_only_depends_on_the_ratio():
    assert sample_size_means(std=20, mde=4) == sample_size_means(std=5, mde=1)


def test_mde_means_inverts_sample_size_means():
    n = sample_size_means(std=10, mde=2)
    mde = mde_means(std=10, n_per_arm=n)
    # n was rounded up, so the detectable effect is a hair below the target.
    assert mde <= 2
    assert mde == pytest.approx(2, rel=1e-3)


def test_mde_means_scales_with_std():
    assert mde_means(std=20, n_per_arm=500) == pytest.approx(2 * mde_means(std=10, n_per_arm=500))


def test_mde_means_shrinks_with_more_users():
    assert mde_means(std=10, n_per_arm=4000) < mde_means(std=10, n_per_arm=1000)


def test_mde_means_grows_with_higher_power():
    assert mde_means(std=10, n_per_arm=1000, power=0.9) > mde_means(std=10, n_per_arm=1000)


@pytest.mark.parametrize("kwargs", [{"std": 0}, {"std": -1}, {"n_per_arm": 0}, {"n_per_arm": -5}])
def test_mde_means_rejects_invalid_inputs(kwargs):
    args = {"std": 10, "n_per_arm": 1000, **kwargs}
    with pytest.raises(ValueError):
        mde_means(**args)


def test_mde_roundtrips_with_sample_size():
    n = sample_size_proportions(0.10, 0.02)
    mde = mde_proportions(0.10, n)
    assert mde == pytest.approx(0.02, rel=0.08)


@pytest.mark.parametrize("baseline", [0, 1, -0.1, 1.5])
def test_invalid_baseline(baseline):
    with pytest.raises(ValueError):
        sample_size_proportions(baseline, 0.01)
