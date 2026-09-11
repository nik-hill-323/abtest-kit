import numpy as np
import pytest

from abtest.sequential import msprt_from_observations, msprt_proportions


def test_msprt_false_positive_rate_under_continuous_peeking():
    """Peeking after every 20 users, a fixed-horizon test would reject far more than
    5% of null experiments. The mSPRT must stay at or below alpha."""
    rng = np.random.default_rng(10)
    rejections = 0
    trials = 300
    for _ in range(trials):
        a = rng.normal(size=2000)
        b = rng.normal(size=2000)
        r = msprt_from_observations(a, b, variance=1.0, look_every=20)
        rejections += r.stop
    assert rejections / trials <= 0.06


def test_msprt_detects_real_effect():
    rng = np.random.default_rng(11)
    a = rng.normal(0, 1, 3000)
    b = rng.normal(0.3, 1, 3000)
    r = msprt_from_observations(a, b, variance=1.0, look_every=50)
    assert r.stop
    assert r.observed_lift == pytest.approx(0.3, abs=0.1)


def test_p_value_path_is_monotone_non_increasing():
    rng = np.random.default_rng(12)
    r = msprt_from_observations(rng.normal(size=500), rng.normal(0.2, size=500), look_every=10)
    assert all(x >= y for x, y in zip(r.p_value_path, r.p_value_path[1:]))


def test_msprt_proportions_shapes_and_lift():
    n = np.array([100, 200, 300, 400])
    r = msprt_proportions(
        control_conversions=np.array([10, 20, 30, 40]),
        treatment_conversions=np.array([15, 32, 48, 64]),
        control_n=n,
        treatment_n=n,
    )
    assert r.looks == 4
    assert r.observed_lift == pytest.approx(0.06)
    assert r.p_value_path[-1] == r.p_value


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        msprt_proportions(np.array([1, 2]), np.array([1]), np.array([10, 20]), np.array([10, 20]))
