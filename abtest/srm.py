"""Sample ratio mismatch (SRM) check.

If an experiment was configured to split traffic 50/50 and the arms came back
with 821,588 and 815,482 users, the gap looks small but the chance of a split
that uneven under true 50/50 assignment is about 2 in a million. Something
between assignment and logging is dropping or duplicating users in one arm
(a redirect that loses slow clients, a bot filter that reacts to the
treatment, a crash in the new code path), and every metric comparison from that
experiment is suspect.

The check is a chi-square goodness-of-fit test of the observed user counts
against the configured allocation. It runs on every experiment, usually before
anyone looks at a metric, so a strict threshold is used to keep false alarms
rare: 0.001 by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    arms: list[str]
    observed: list[int]
    expected: list[float]
    expected_ratios: list[float]
    chi2: float
    df: int
    p_value: float
    threshold: float
    mismatch: bool

    @property
    def observed_ratios(self) -> list[float]:
        total = sum(self.observed)
        return [n / total for n in self.observed]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observed_ratios"] = self.observed_ratios
        return d


def srm_check(
    observed: Sequence[int],
    expected_ratios: Sequence[float] | None = None,
    threshold: float = 0.001,
    arms: Sequence[str] | None = None,
) -> SRMResult:
    """Test whether observed arm sizes are consistent with the configured split.

    ``observed`` holds the user count for each arm. ``expected_ratios`` is the
    allocation the experiment was configured with (any positive weights; they are
    normalised), defaulting to an equal split. ``mismatch`` is True when the
    chi-square p-value falls below ``threshold``, in which case the experiment's
    metric results should not be trusted until the cause is found.
    """
    n = np.asarray(observed, dtype=float)
    if n.ndim != 1 or n.size < 2:
        raise ValueError("observed must list the user count of at least two arms")
    if np.any(~np.isfinite(n)) or np.any(n < 0) or np.any(n != np.round(n)):
        raise ValueError("observed counts must be non-negative integers")
    if n.sum() == 0:
        raise ValueError("observed counts sum to zero")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")

    if expected_ratios is None:
        w = np.ones(n.size)
    else:
        w = np.asarray(expected_ratios, dtype=float)
        if w.shape != n.shape:
            raise ValueError("expected_ratios and observed must be the same length")
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("expected_ratios must all be positive")
    ratios = w / w.sum()

    if arms is None:
        if n.size == 2:
            names = ["control", "treatment"]
        else:
            names = ["control"] + [f"treatment_{i}" for i in range(1, n.size)]
    else:
        names = list(arms)
        if len(names) != n.size:
            raise ValueError("arms and observed must be the same length")

    expected = ratios * n.sum()
    chi2, p_value = stats.chisquare(n, f_exp=expected)
    return SRMResult(
        arms=names,
        observed=[int(v) for v in n],
        expected=[float(v) for v in expected],
        expected_ratios=[float(v) for v in ratios],
        chi2=float(chi2),
        df=int(n.size - 1),
        p_value=float(p_value),
        threshold=threshold,
        mismatch=bool(p_value < threshold),
    )
