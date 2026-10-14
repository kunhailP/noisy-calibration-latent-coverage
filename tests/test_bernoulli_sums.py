"""Lemmas A and B of notes/heterogeneous_latent.md on sums of independent Bernoulli variables."""
import numpy as np
from scipy import stats


def _pmf(p):
    a = np.array([1.0])
    for x in p:
        a = np.convolve(a, [1 - x, x])
    return a


def _vectors(rng, count=3000):
    for _ in range(count):
        n = int(rng.integers(1, 30))
        kind = rng.integers(3)
        if kind == 0:
            yield rng.random(n)
        elif kind == 1:
            yield rng.beta(.3, .3, n)
        else:                                      # near the extremal form {0, a, 1}
            yield np.clip(rng.choice([0., 1., rng.random()], n) + rng.normal(0, .05, n), 0, 1)


def test_mode_beyond_mean():
    for p in _vectors(np.random.default_rng(1)):
        a, mu = _pmf(p), p.sum()
        for j in range(int(np.ceil(mu)), len(p)):
            assert a[j + 1] <= a[j] + 1e-14


def test_binomial_comparison():
    for p in _vectors(np.random.default_rng(2)):
        n, mu = len(p), p.sum()
        F = np.cumsum(_pmf(p))
        Fb = stats.binom.cdf(np.arange(n + 1), n, mu / n)
        for c in range(n + 1):
            if c >= mu + 1:
                assert F[c] >= Fb[c] - 1e-12
