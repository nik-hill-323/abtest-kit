"""Power analysis for two-arm experiments."""

from __future__ import annotations

import math

from scipy import stats


def _z(alpha: float, power: float, two_sided: bool) -> tuple[float, float]:
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    return z_alpha, z_beta


def sample_size_proportions(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    relative: bool = False,
) -> int:
    """Per-arm sample size to detect a change in conversion rate.

    Parameters
    ----------
    baseline : conversion rate of the control arm, in (0, 1)
    mde      : minimum detectable effect. Absolute (p2 - p1) unless ``relative``
               is True, in which case it is a fraction of ``baseline``.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be in (0, 1)")
    p1 = baseline
    p2 = baseline * (1 + mde) if relative else baseline + mde
    if not 0 < p2 < 1:
        raise ValueError("baseline + mde must be in (0, 1)")
    z_alpha, z_beta = _z(alpha, power, two_sided)
    p_bar = (p1 + p2) / 2
    num = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(num / (p2 - p1) ** 2)


def sample_size_means(
    std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """Per-arm sample size to detect a difference ``mde`` in means, given a common
    standard deviation ``std``."""
    if std <= 0:
        raise ValueError("std must be positive")
    if mde == 0:
        raise ValueError("mde must be non-zero")
    z_alpha, z_beta = _z(alpha, power, two_sided)
    return math.ceil(2 * ((z_alpha + z_beta) * std / mde) ** 2)


def mde_proportions(
    baseline: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> float:
    """Smallest absolute lift detectable with ``n_per_arm`` users per arm.

    Uses the conservative approximation that both arms share the baseline
    variance, which is accurate for small effects.
    """
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    z_alpha, z_beta = _z(alpha, power, two_sided)
    se = math.sqrt(2 * baseline * (1 - baseline) / n_per_arm)
    return (z_alpha + z_beta) * se
