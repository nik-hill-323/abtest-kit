"""abtest-kit: design and analyse A/B tests.

Modules
-------
power         sample size and minimum detectable effect for proportions and means
frequentist   two-proportion z-test, Welch t-test, confidence intervals
bayesian      Beta-Binomial and Normal-Normal posteriors, P(B > A), expected loss
sequential    mixture sequential probability ratio test (mSPRT) with always-valid p-values
cuped         CUPED variance reduction using a pre-experiment covariate
multiple      Bonferroni and Benjamini-Hochberg corrections for several metrics
ratio_metrics delta-method inference for per-user ratio metrics
"""

from .bayesian import BayesResult, bayes_means, bayes_proportions
from .cuped import cuped_adjust
from .frequentist import MeansResult, ProportionResult, ttest_means, ztest_proportions
from .multiple import MultipleTestResult, adjust, benjamini_hochberg, bonferroni
from .power import mde_means, mde_proportions, sample_size_means, sample_size_proportions
from .ratio_metrics import RatioEstimate, RatioResult, ratio_estimate, ratio_metric
from .sequential import SequentialResult, msprt_means, msprt_proportions

__all__ = [
    "BayesResult",
    "MeansResult",
    "MultipleTestResult",
    "ProportionResult",
    "RatioEstimate",
    "RatioResult",
    "SequentialResult",
    "adjust",
    "bayes_means",
    "bayes_proportions",
    "benjamini_hochberg",
    "bonferroni",
    "cuped_adjust",
    "mde_means",
    "mde_proportions",
    "msprt_means",
    "msprt_proportions",
    "ratio_estimate",
    "ratio_metric",
    "sample_size_means",
    "sample_size_proportions",
    "ttest_means",
    "ztest_proportions",
]

__version__ = "0.1.0"
