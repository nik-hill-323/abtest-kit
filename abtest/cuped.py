"""CUPED: Controlled-experiment Using Pre-Experiment Data (Deng et al., 2013).

Subtracting ``theta * (x - mean(x))`` from each user's metric, where ``x`` is
the same metric measured before the experiment, removes the variance explained
by the covariate and tightens confidence intervals without biasing the lift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CupedResult:
    theta: float
    variance_reduction: float
    control_adjusted: np.ndarray
    treatment_adjusted: np.ndarray

    def to_dict(self) -> dict:
        d = asdict(self)
        d["control_adjusted"] = self.control_adjusted.tolist()
        d["treatment_adjusted"] = self.treatment_adjusted.tolist()
        return d


def cuped_adjust(
    control_metric: np.ndarray,
    control_covariate: np.ndarray,
    treatment_metric: np.ndarray,
    treatment_covariate: np.ndarray,
) -> CupedResult:
    """Return CUPED-adjusted metrics for both arms.

    ``theta`` is estimated on the pooled data as cov(y, x) / var(x), and the
    covariate is centred on its pooled mean so that the adjustment shifts both
    arms by the same amount in expectation.
    """
    y = np.concatenate([np.asarray(control_metric, float), np.asarray(treatment_metric, float)])
    x = np.concatenate([np.asarray(control_covariate, float), np.asarray(treatment_covariate, float)])
    if y.shape != x.shape:
        raise ValueError("metric and covariate must align")
    if y.size < 3:
        raise ValueError("need at least three observations")
    var_x = x.var(ddof=1)
    if var_x == 0:
        theta = 0.0
    else:
        theta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    x_mean = x.mean()
    adjusted = y - theta * (x - x_mean)
    var_before = y.var(ddof=1)
    var_after = adjusted.var(ddof=1)
    reduction = 1 - var_after / var_before if var_before > 0 else 0.0
    n_c = len(control_metric)
    return CupedResult(
        theta=theta,
        variance_reduction=float(reduction),
        control_adjusted=adjusted[:n_c],
        treatment_adjusted=adjusted[n_c:],
    )
