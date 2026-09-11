"""abtest-kit: design and analyse A/B tests.

Modules
-------
power       sample size and minimum detectable effect for proportions and means
frequentist two-proportion z-test, Welch t-test, confidence intervals
bayesian    Beta-Binomial and Normal-Normal posteriors, P(B > A), expected loss
sequential  mixture sequential probability ratio test (mSPRT) with always-valid p-values
cuped       CUPED variance reduction using a pre-experiment covariate
"""

from .bayesian import BayesResult, bayes_means, bayes_proportions
from .cuped import cuped_adjust
from .frequentist import MeansResult, ProportionResult, ttest_means, ztest_proportions
from .power import mde_proportions, sample_size_means, sample_size_proportions
from .sequential import SequentialResult, msprt_means, msprt_proportions

__all__ = [
    "BayesResult",
    "MeansResult",
    "ProportionResult",
    "SequentialResult",
    "bayes_means",
    "bayes_proportions",
    "cuped_adjust",
    "mde_proportions",
    "msprt_means",
    "msprt_proportions",
    "sample_size_means",
    "sample_size_proportions",
    "ttest_means",
    "ztest_proportions",
]

__version__ = "0.1.0"
