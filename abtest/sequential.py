"""Always-valid inference with the mixture sequential probability ratio test.

Peeking at a classical test inflates the false positive rate. The mSPRT
(Johari, Pekelis and Walsh, 2017; the method behind Optimizely's Stats Engine)
produces a p-value that stays valid no matter how often you look, at the cost
of some power relative to a fixed-horizon test of the same final sample size.

The test statistic for a difference in means with known variance is

    Lambda_n = sqrt(2 s^2 / (2 s^2 + n tau^2))
               * exp( n^2 tau^2 (xbar_B - xbar_A)^2 / (4 s^2 (2 s^2 + n tau^2)) )

where ``tau^2`` is the variance of the Normal mixing distribution over the
alternative and ``n`` is the per-arm sample size. The always-valid p-value is
``min(1, 1 / max_k Lambda_k)`` over all looks so far.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class SequentialResult:
    looks: int
    n_per_arm: int
    observed_lift: float
    lambda_max: float
    p_value: float
    alpha: float
    p_value_path: list[float] = field(default_factory=list)

    @property
    def stop(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stop"] = self.stop
        return d


def _msprt_lambda(n: int, diff: float, var: float, tau_sq: float) -> float:
    if n <= 0 or var <= 0:
        return 1.0
    denom = 2 * var + n * tau_sq
    return math.sqrt(2 * var / denom) * math.exp(n * n * tau_sq * diff * diff / (4 * var * denom))


def msprt_means(
    control_cumulative: np.ndarray,
    treatment_cumulative: np.ndarray,
    variance: float,
    tau: float | None = None,
    alpha: float = 0.05,
) -> SequentialResult:
    """Run the mSPRT over a sequence of looks.

    Parameters
    ----------
    control_cumulative, treatment_cumulative :
        arrays of the same length giving the running mean of each arm at each
        look; look ``k`` is assumed to have ``k+1`` observations per arm scaled
        by ``per_look`` (see below). For per-look sample counts pass the running
        means at those counts and supply ``n`` implicitly via array length. Use
        :func:`msprt_from_observations` for raw observation streams.
    variance :
        the (assumed known) variance of a single observation
    tau :
        mixing standard deviation over the alternative. Defaults to
        ``sqrt(variance) / 2``, a common practical choice.
    """
    a = np.asarray(control_cumulative, dtype=float)
    b = np.asarray(treatment_cumulative, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("cumulative arrays must be one-dimensional and the same length")
    tau_sq = (tau if tau is not None else math.sqrt(variance) / 2) ** 2
    lambdas = [_msprt_lambda(k + 1, b[k] - a[k], variance, tau_sq) for k in range(a.size)]
    running_max = np.maximum.accumulate(lambdas)
    p_path = [min(1.0, 1.0 / lm) for lm in running_max]
    return SequentialResult(
        looks=a.size,
        n_per_arm=a.size,
        observed_lift=float(b[-1] - a[-1]),
        lambda_max=float(running_max[-1]),
        p_value=p_path[-1],
        alpha=alpha,
        p_value_path=p_path,
    )


def msprt_from_observations(
    control: np.ndarray,
    treatment: np.ndarray,
    variance: float | None = None,
    tau: float | None = None,
    alpha: float = 0.05,
    look_every: int = 1,
) -> SequentialResult:
    """Convenience wrapper: feed raw per-user observations, in arrival order,
    and evaluate the mSPRT at every ``look_every`` users per arm."""
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    n = min(a.size, b.size)
    if n < 2:
        raise ValueError("need at least two observations per arm")
    a, b = a[:n], b[:n]
    var = variance if variance is not None else float(np.concatenate([a, b]).var(ddof=1))
    tau_sq = (tau if tau is not None else math.sqrt(var) / 2) ** 2
    idx = np.arange(look_every, n + 1, look_every)
    if idx[-1] != n:
        idx = np.append(idx, n)
    ca = np.cumsum(a)[idx - 1] / idx
    cb = np.cumsum(b)[idx - 1] / idx
    lambdas = [_msprt_lambda(int(k), float(cb[i] - ca[i]), var, tau_sq) for i, k in enumerate(idx)]
    running_max = np.maximum.accumulate(lambdas)
    p_path = [min(1.0, 1.0 / lm) for lm in running_max]
    return SequentialResult(
        looks=len(idx),
        n_per_arm=int(idx[-1]),
        observed_lift=float(cb[-1] - ca[-1]),
        lambda_max=float(running_max[-1]),
        p_value=p_path[-1],
        alpha=alpha,
        p_value_path=p_path,
    )


def msprt_proportions(
    control_conversions: np.ndarray,
    treatment_conversions: np.ndarray,
    control_n: np.ndarray,
    treatment_n: np.ndarray,
    tau: float | None = None,
    alpha: float = 0.05,
) -> SequentialResult:
    """mSPRT for conversion rates given cumulative counts at each look.

    The Bernoulli variance is estimated from the pooled rate at the latest look
    and treated as known, which is standard practice for this test.
    """
    cc = np.asarray(control_conversions, dtype=float)
    tc = np.asarray(treatment_conversions, dtype=float)
    cn = np.asarray(control_n, dtype=float)
    tn = np.asarray(treatment_n, dtype=float)
    if not (cc.shape == tc.shape == cn.shape == tn.shape):
        raise ValueError("all inputs must have the same shape")
    if np.any(cn <= 0) or np.any(tn <= 0):
        raise ValueError("sample sizes must be positive at every look")
    pooled = (cc[-1] + tc[-1]) / (cn[-1] + tn[-1])
    var = max(pooled * (1 - pooled), 1e-12)
    tau_sq = (tau if tau is not None else math.sqrt(var) / 2) ** 2
    lambdas = []
    for k in range(cc.size):
        # harmonic mean of the two arm sizes plays the role of per-arm n
        n_eff = 2 * cn[k] * tn[k] / (cn[k] + tn[k])
        lambdas.append(_msprt_lambda(n_eff, tc[k] / tn[k] - cc[k] / cn[k], var, tau_sq))
    running_max = np.maximum.accumulate(lambdas)
    p_path = [min(1.0, 1.0 / lm) for lm in running_max]
    return SequentialResult(
        looks=cc.size,
        n_per_arm=int(min(cn[-1], tn[-1])),
        observed_lift=float(tc[-1] / tn[-1] - cc[-1] / cn[-1]),
        lambda_max=float(running_max[-1]),
        p_value=p_path[-1],
        alpha=alpha,
        p_value_path=p_path,
    )
