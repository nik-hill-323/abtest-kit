"""Why you cannot peek at a fixed-horizon test, and what the mSPRT does about it.

Runs 500 A/A experiments (no real effect) and checks for significance after
every 25 users per arm. The naive test "finds" an effect far more often than
its nominal 5%; the mSPRT stays under alpha.

    python examples/peeking_simulation.py
"""

import numpy as np

from abtest import ttest_means
from abtest.sequential import msprt_from_observations

rng = np.random.default_rng(0)
TRIALS, N, LOOK = 500, 2000, 25

naive_hits = seq_hits = 0
for _ in range(TRIALS):
    a, b = rng.normal(size=N), rng.normal(size=N)
    naive_hits += any(ttest_means(a[:k], b[:k]).significant for k in range(LOOK, N + 1, LOOK))
    seq_hits += msprt_from_observations(a, b, variance=1.0, look_every=LOOK).stop

print(f"false positive rate with peeking, naive t-test: {naive_hits / TRIALS:.1%}")
print(f"false positive rate with peeking, mSPRT:         {seq_hits / TRIALS:.1%}")
