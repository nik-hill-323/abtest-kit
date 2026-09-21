# abtest-kit

A small, tested Python library for the statistics that come up in every online
experiment: how many users you need, whether the result is real, how likely
treatment is to be better, whether you are allowed to stop early, and how to
shrink the confidence interval with pre-experiment data.

```bash
pip install -e ".[dev]"
abtest ztest --control 480 1000 --treatment 530 1000
```

```json
{
  "control_rate": 0.48,
  "treatment_rate": 0.53,
  "absolute_lift": 0.05,
  "relative_lift": 0.104,
  "z_stat": 2.24,
  "p_value": 0.025,
  "ci_low": 0.006,
  "ci_high": 0.094,
  "alpha": 0.05,
  "significant": true
}
```

## What is in it

| Module | Functions | Method |
|---|---|---|
| `abtest.power` | `sample_size_proportions`, `sample_size_means`, `mde_proportions`, `mde_means` | Closed-form normal approximation |
| `abtest.frequentist` | `ztest_proportions`, `ttest_means` | Pooled-SE z-test, Welch's t-test, Wald intervals |
| `abtest.bayesian` | `bayes_proportions`, `bayes_means` | Beta-Binomial and Normal posteriors, P(B > A), expected loss |
| `abtest.sequential` | `msprt_from_observations`, `msprt_proportions` | Mixture SPRT with always-valid p-values (Johari et al., 2017) |
| `abtest.cuped` | `cuped_adjust` | CUPED variance reduction (Deng et al., 2013) |

Every function returns a dataclass with a `to_dict()` method so results drop
straight into JSON, a DataFrame, or a dashboard.

## Examples

**Plan the test.** A 5% baseline conversion rate and you want to detect a
0.5 percentage point lift at 80% power:

```python
from abtest import sample_size_proportions
sample_size_proportions(baseline=0.05, mde=0.005)   # 31,234 users per arm
```

Or the other way round, when the traffic is fixed and you want to know what the
test can actually see. Two weeks of traffic is 8,000 users per arm and revenue
per user has a standard deviation of $12:

```python
from abtest import mde_means
mde_means(std=12, n_per_arm=8000)                   # $0.53 per user
```

Both are on the CLI too:

```bash
abtest mde-means --std 12 --n-per-arm 8000          # {"mde": 0.5315634208977957}
```

**Read the result two ways.** Frequentist and Bayesian answers from the same counts:

```python
from abtest import ztest_proportions, bayes_proportions

z = ztest_proportions(1480, 30000, 1620, 30000)
z.p_value, (z.ci_low, z.ci_high)          # 0.010, (0.0011, 0.0082)

b = bayes_proportions(1480, 30000, 1620, 30000)
b.prob_treatment_better                    # 0.995
b.expected_loss_choosing_treatment         # 0.000003  (risk of shipping B if A is truly better)
```

**Stop early without lying to yourself.** Feed observations in arrival order and
check the always-valid p-value at every look:

```python
from abtest.sequential import msprt_from_observations
r = msprt_from_observations(control_revenue, treatment_revenue, look_every=100)
r.stop, r.p_value, r.p_value_path[-5:]
```

`examples/peeking_simulation.py` shows the difference: peeking at a t-test every
25 users on 500 A/A experiments produces a false positive rate around 30%; the
mSPRT stays below 5%.

**Tighten the interval.** If you have the same metric from before the experiment:

```python
from abtest import cuped_adjust, ttest_means
adj = cuped_adjust(post_c, pre_c, post_t, pre_t)
adj.variance_reduction                     # e.g. 0.62
ttest_means(adj.control_adjusted, adj.treatment_adjusted)
```

## Tests

```bash
pytest
```

The suite checks each method against an independent reference: the z-test
against a chi-square test, Welch's test against SciPy, sample sizes against
Cohen's published power tables, and the sequential test against a simulated false positive
rate under continuous peeking.

## References

- Johari, Pekelis, Walsh. *Always Valid Inference: Bringing Sequential Analysis to A/B Testing.* 2017.
- Deng, Xu, Kohavi, Walker. *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.* WSDM 2013.
- Kohavi, Tang, Xu. *Trustworthy Online Controlled Experiments.* Cambridge University Press, 2020.
