"""Classical two-sample tests with effect-size confidence intervals."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class ProportionResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    z_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


@dataclass
class MeansResult:
    control_mean: float
    treatment_mean: float
    difference: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    df: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


def ztest_proportions(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> ProportionResult:
    """Two-proportion z-test.

    The p-value uses the pooled standard error (the null hypothesis of equal rates);
    the confidence interval on the lift uses the unpooled standard error, which is
    the standard Wald interval.
    """
    for name, value in (("control_n", control_n), ("treatment_n", treatment_n)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= control_conversions <= control_n or not 0 <= treatment_conversions <= treatment_n:
        raise ValueError("conversions must be between 0 and n")

    p1 = control_conversions / control_n
    p2 = treatment_conversions / treatment_n
    diff = p2 - p1

    pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treatment_n))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    if two_sided:
        p = 2 * stats.norm.sf(abs(z))
    else:
        p = stats.norm.sf(z)

    se_unpooled = math.sqrt(p1 * (1 - p1) / control_n + p2 * (1 - p2) / treatment_n)
    z_crit = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled if two_sided else float("inf")

    return ProportionResult(
        control_rate=p1,
        treatment_rate=p2,
        absolute_lift=diff,
        relative_lift=diff / p1 if p1 > 0 else float("nan"),
        z_stat=z,
        p_value=float(p),
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
    )


def ttest_means(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> MeansResult:
    """Welch's t-test for a difference in means, which does not assume equal variances."""
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each arm needs at least two observations")

    alternative = "two-sided" if two_sided else "greater"
    res = stats.ttest_ind(b, a, equal_var=False, alternative=alternative)

    diff = b.mean() - a.mean()
    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    # Welch-Satterthwaite degrees of freedom
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df) if two_sided else stats.t.ppf(1 - alpha, df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se if two_sided else float("inf")

    return MeansResult(
        control_mean=float(a.mean()),
        treatment_mean=float(b.mean()),
        difference=float(diff),
        t_stat=float(res.statistic),
        p_value=float(res.pvalue),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        df=float(df),
        alpha=alpha,
    )
