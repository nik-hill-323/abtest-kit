"""Bayesian inference for two-arm experiments.

Conversion rates use a Beta-Binomial model; continuous metrics use a
Normal-Normal model with a known-variance approximation (the sample variance),
which is accurate for the sample sizes typical of online experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class BayesResult:
    control_estimate: float
    treatment_estimate: float
    prob_treatment_better: float
    expected_loss_choosing_treatment: float
    expected_loss_choosing_control: float
    lift_ci_low: float
    lift_ci_high: float
    credible_level: float

    def to_dict(self) -> dict:
        return asdict(self)


def bayes_proportions(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95,
    n_samples: int = 200_000,
    seed: int | None = 0,
) -> BayesResult:
    """Beta-Binomial comparison of two conversion rates.

    ``expected_loss_choosing_treatment`` is the expected absolute drop in
    conversion rate if treatment is shipped but control was actually better.
    A common decision rule is to ship when this falls below a threshold such as
    0.1 percentage points.
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("sample sizes must be positive")
    rng = np.random.default_rng(seed)
    a = rng.beta(prior_alpha + control_conversions, prior_beta + control_n - control_conversions, n_samples)
    b = rng.beta(
        prior_alpha + treatment_conversions, prior_beta + treatment_n - treatment_conversions, n_samples
    )
    return _summarise(a, b, credible_level)


def bayes_means(
    control: np.ndarray,
    treatment: np.ndarray,
    credible_level: float = 0.95,
    n_samples: int = 200_000,
    seed: int | None = 0,
) -> BayesResult:
    """Normal posterior on each arm's mean with a flat prior, using the sample
    variance as if known. Equivalent to a Bayesian reading of Welch's test."""
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("each arm needs at least two observations")
    rng = np.random.default_rng(seed)
    sa = rng.normal(a.mean(), a.std(ddof=1) / np.sqrt(a.size), n_samples)
    sb = rng.normal(b.mean(), b.std(ddof=1) / np.sqrt(b.size), n_samples)
    return _summarise(sa, sb, credible_level)


def _summarise(a: np.ndarray, b: np.ndarray, credible_level: float) -> BayesResult:
    lift = b - a
    lo, hi = np.quantile(lift, [(1 - credible_level) / 2, 1 - (1 - credible_level) / 2])
    return BayesResult(
        control_estimate=float(a.mean()),
        treatment_estimate=float(b.mean()),
        prob_treatment_better=float((b > a).mean()),
        expected_loss_choosing_treatment=float(np.maximum(a - b, 0).mean()),
        expected_loss_choosing_control=float(np.maximum(b - a, 0).mean()),
        lift_ci_low=float(lo),
        lift_ci_high=float(hi),
        credible_level=credible_level,
    )


def beta_posterior(conversions: int, n: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
    """The Beta posterior for one arm, useful for plotting."""
    return stats.beta(prior_alpha + conversions, prior_beta + n - conversions)
