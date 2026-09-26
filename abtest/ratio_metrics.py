"""Ratio metrics: per-user numerators over per-user denominators.

Revenue per session, clicks per impression and items per order are all ratios
whose denominator varies by user. The metric an experiment reports is the
*pooled* ratio - total revenue over total sessions - not the average of each
user's own ratio, because a user with twenty sessions should carry more weight
than a user with one.

That makes the estimator a ratio of two sample means, which is not a mean, so
its standard error is not ``std / sqrt(n)``. Running a t-test over the session
table instead treats sessions from the same user as independent, understates
the variance, and produces confidence intervals that are too narrow: on
correlated data a nominal 95% interval can cover well under 95% of the time.

The delta method gives the right answer. Linearising ``R = mean(y) / mean(x)``
around the two means,

    Var(R) ~= (1 / (n * mean(x)^2)) * (Var(y) - 2*R*Cov(y, x) + R^2 * Var(x))

where ``n`` is the number of *users* and the variances are taken over users.
The covariance term is what the naive session-level test throws away.

References
----------
Deng, Knoblich, Lu. *Applying the Delta Method in Metric Analytics.* KDD 2018.
Kohavi, Tang, Xu. *Trustworthy Online Controlled Experiments*, chapter 18.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class RatioEstimate:
    """A pooled ratio for one arm, with its delta-method standard error."""

    ratio: float
    variance: float
    n_users: int

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["standard_error"] = self.standard_error
        return d


@dataclass
class RatioResult:
    control_ratio: float
    treatment_ratio: float
    absolute_lift: float
    relative_lift: float
    standard_error: float
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    relative_ci_low: float
    relative_ci_high: float
    control_n_users: int
    treatment_n_users: int
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


def _as_pair(numerator, denominator, arm: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(numerator, dtype=float)
    x = np.asarray(denominator, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError(f"{arm}: numerator and denominator must be one-dimensional")
    if y.shape != x.shape:
        raise ValueError(f"{arm}: numerator and denominator must have one value per user")
    if y.size < 2:
        raise ValueError(f"{arm}: needs at least two users")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError(f"{arm}: numerator and denominator must be finite")
    if np.any(x < 0):
        raise ValueError(f"{arm}: denominator cannot be negative")
    if x.sum() <= 0:
        raise ValueError(f"{arm}: denominator sums to zero, the ratio is undefined")
    return y, x


def ratio_estimate(numerator, denominator, arm: str = "arm") -> RatioEstimate:
    """Pooled ratio ``sum(numerator) / sum(denominator)`` and its delta-method variance.

    Both arrays carry one value per user: ``numerator[i]`` is that user's total
    revenue and ``denominator[i]`` their number of sessions. Users with a zero
    denominator are kept - they contribute nothing to the numerator and nothing
    to the denominator, but they are part of the randomised population and
    dropping them would bias the estimate.
    """
    y, x = _as_pair(numerator, denominator, arm)
    n = y.size
    x_bar = x.mean()
    ratio = float(y.sum() / x.sum())

    cov = np.cov(y, x, ddof=1)
    var_y, cov_yx, var_x = float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])
    variance = (var_y - 2 * ratio * cov_yx + ratio**2 * var_x) / (n * x_bar**2)
    # Rounding can push a genuinely zero variance a hair below zero.
    variance = max(variance, 0.0)
    return RatioEstimate(ratio=ratio, variance=float(variance), n_users=n)


def ratio_metric(
    control_numerator,
    control_denominator,
    treatment_numerator,
    treatment_denominator,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> RatioResult:
    """Compare a pooled ratio metric across two arms using the delta method.

    The absolute interval is a Wald interval on the difference of the two
    ratios. The relative interval is built on the log of the ratio of ratios,
    which keeps it from crossing -100% and is more accurate in small samples
    than a symmetric interval on the percentage itself. Both are reported as
    fractions, so 0.031 means a 3.1% lift.

    The relative figures are ``nan`` when either arm's ratio is not strictly
    positive, where a percentage change has no useful sign.
    """
    c = ratio_estimate(control_numerator, control_denominator, arm="control")
    t = ratio_estimate(treatment_numerator, treatment_denominator, arm="treatment")

    diff = t.ratio - c.ratio
    se = math.sqrt(c.variance + t.variance)
    z = diff / se if se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z)) if two_sided else stats.norm.sf(z)

    z_crit = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    ci_low = diff - z_crit * se
    ci_high = diff + z_crit * se if two_sided else float("inf")

    if c.ratio > 0 and t.ratio > 0:
        relative = t.ratio / c.ratio - 1
        se_log = math.sqrt(c.variance / c.ratio**2 + t.variance / t.ratio**2)
        rel_low = math.exp(math.log(t.ratio / c.ratio) - z_crit * se_log) - 1
        rel_high = (
            math.exp(math.log(t.ratio / c.ratio) + z_crit * se_log) - 1
            if two_sided
            else float("inf")
        )
    else:
        relative = rel_low = rel_high = float("nan")

    return RatioResult(
        control_ratio=c.ratio,
        treatment_ratio=t.ratio,
        absolute_lift=float(diff),
        relative_lift=float(relative),
        standard_error=float(se),
        z_stat=float(z),
        p_value=float(p),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        relative_ci_low=float(rel_low),
        relative_ci_high=float(rel_high),
        control_n_users=c.n_users,
        treatment_n_users=t.n_users,
        alpha=alpha,
    )
