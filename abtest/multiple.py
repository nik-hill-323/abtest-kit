"""Corrections for testing several metrics in one experiment.

An experiment that reads out five metrics at alpha = 0.05 has roughly a 23%
chance of at least one false positive, not 5%. The two standard answers are to
control the family-wise error rate (the probability of *any* false positive) or
the false discovery rate (the expected share of false positives among the
metrics you declare significant).

Both functions here return adjusted p-values, so a metric is significant when
its adjusted p-value is at or below the original alpha and the reported number
carries the correction with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class MultipleTestResult:
    metrics: list[str]
    p_values: list[float]
    adjusted_p_values: list[float]
    rejected: list[bool]
    method: str
    alpha: float

    @property
    def n_significant(self) -> int:
        return sum(self.rejected)

    @property
    def significant_metrics(self) -> list[str]:
        return [m for m, r in zip(self.metrics, self.rejected, strict=True) if r]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_significant"] = self.n_significant
        d["significant_metrics"] = self.significant_metrics
        return d


def _prepare(
    p_values: Sequence[float], metrics: Sequence[str] | None
) -> tuple[np.ndarray, list[str]]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p_values must all be in [0, 1]")
    if metrics is None:
        names = [f"metric_{i + 1}" for i in range(p.size)]
    else:
        names = list(metrics)
        if len(names) != p.size:
            raise ValueError("metrics and p_values must be the same length")
    return p, names


def bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
    metrics: Sequence[str] | None = None,
) -> MultipleTestResult:
    """Bonferroni correction: control the family-wise error rate at ``alpha``.

    Each p-value is multiplied by the number of metrics (capped at 1), which is
    the same decision as comparing the raw p-value to ``alpha / m``. Valid
    whatever the dependence between metrics, and conservative when they are
    correlated - which metrics from the same experiment usually are.
    """
    p, names = _prepare(p_values, metrics)
    adjusted = np.minimum(p * p.size, 1.0)
    return MultipleTestResult(
        metrics=names,
        p_values=[float(v) for v in p],
        adjusted_p_values=[float(v) for v in adjusted],
        rejected=[bool(v) for v in adjusted <= alpha],
        method="bonferroni",
        alpha=alpha,
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
    metrics: Sequence[str] | None = None,
) -> MultipleTestResult:
    """Benjamini-Hochberg step-up: control the false discovery rate at ``alpha``.

    Sort the m p-values and reject the k smallest, where k is the largest i with
    ``p_(i) <= i * alpha / m``. The adjusted values returned here are the
    equivalent q-values (the smallest FDR level at which each metric would be
    declared significant), so ``adjusted <= alpha`` reproduces that rejection
    set. Assumes independent or positively dependent tests; it is uniformly less
    conservative than Bonferroni.
    """
    p, names = _prepare(p_values, metrics)
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order]

    q = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity: a metric can never be more significant than a larger one.
    q = np.minimum.accumulate(q[::-1])[::-1]
    # Tied p-values must share an adjusted value; q is now non-decreasing, so the
    # first member of each tie group holds that group's minimum.
    uniq, first_of_group = np.unique(ranked, return_index=True)
    q = q[first_of_group][np.searchsorted(uniq, ranked)]
    q = np.clip(q, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = q
    return MultipleTestResult(
        metrics=names,
        p_values=[float(v) for v in p],
        adjusted_p_values=[float(v) for v in adjusted],
        rejected=[bool(v) for v in adjusted <= alpha],
        method="benjamini-hochberg",
        alpha=alpha,
    )


_METHODS = {
    "bonferroni": bonferroni,
    "bh": benjamini_hochberg,
    "benjamini-hochberg": benjamini_hochberg,
    "fdr": benjamini_hochberg,
}


def adjust(
    p_values: Sequence[float],
    method: str = "bh",
    alpha: float = 0.05,
    metrics: Sequence[str] | None = None,
) -> MultipleTestResult:
    """Apply a correction by name: ``bonferroni`` for FWER, ``bh`` for FDR."""
    try:
        fn = _METHODS[method.lower()]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_METHODS)}"
        ) from None
    return fn(p_values, alpha=alpha, metrics=metrics)
