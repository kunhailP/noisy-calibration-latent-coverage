"""Latent and noise laws of the simulations, with their class memberships.

Latent laws have mean 0 and variance 1 and carry an exact distribution function, so the latent
coverage given the data, pr(|W| <= T), needs no Monte Carlo. Noise laws have mean 0 and are
scaled to a given variance per unit.
"""
import math

import numpy as np
from scipy import special, stats


class Latent:
    def __init__(self, name, cdf, rvs, classes):
        self.name, self.cdf, self.rvs, self.classes = name, cdf, rvs, frozenset(classes)

    def coverage(self, t):
        return self.cdf(t) - self.cdf(-t)

    def oracle_radius(self, q):
        from scipy.optimize import brentq
        return brentq(lambda r: self.coverage(r) - q, 1e-9, 50, xtol=1e-13)


# truncated exponential: density proportional to e^{4y} on [0, 1], standardized
_TE_B = 4.0
_TE_M = (math.exp(_TE_B) * (_TE_B - 1) + 1) / (_TE_B * (math.exp(_TE_B) - 1))
_TE_S = math.sqrt((math.exp(_TE_B) * (_TE_B ** 2 - 2 * _TE_B + 2) - 2)
                  / (_TE_B ** 2 * (math.exp(_TE_B) - 1)) - _TE_M ** 2)


def _te_cdf(w):
    y = np.clip(np.asarray(w) * _TE_S + _TE_M, 0, 1)
    return np.expm1(_TE_B * y) / math.expm1(_TE_B)


def _te_rvs(n, rng):
    y = np.log1p(rng.random(n) * math.expm1(_TE_B)) / _TE_B
    return (y - _TE_M) / _TE_S


def bimodal_params(p=0.4, ratio=2.5):
    """p N(m1, s^2) + (1 - p) N(m2, s^2), mean 0, variance 1, (m2 - m1)/s = ratio. At p = 0.4,
    ratio = 2.5 it is bimodal and bi-log-concave but not log-concave (Supplementary Lemma S3 of
    the Biometrika version; certified by `bimodal_blc_certificate` in ball arithmetic)."""
    d = ratio / math.sqrt(1 + p * (1 - p) * ratio ** 2)
    return p, -(1 - p) * d, p * d, d / ratio


_BP = bimodal_params()


def _bm_cdf(w):
    p, m1, m2, s = _BP
    w = np.asarray(w)
    return p * special.ndtr((w - m1) / s) + (1 - p) * special.ndtr((w - m2) / s)


def _bm_rvs(n, rng):
    p, m1, m2, s = _BP
    return np.where(rng.random(n) < p, m1, m2) + s * rng.standard_normal(n)


_GAMMA = stats.gamma(2, loc=-math.sqrt(2), scale=1 / math.sqrt(2))

LC, BLC = 'log_concave', 'bi_log_concave'

LATENT = {
    'normal': Latent('normal', special.ndtr, lambda n, rng: rng.standard_normal(n), (LC, BLC)),
    'Laplace': Latent('Laplace', lambda w: stats.laplace.cdf(np.asarray(w) * math.sqrt(2)),
                      lambda n, rng: rng.laplace(size=n) / math.sqrt(2), (LC, BLC)),
    'centred gamma': Latent('centred gamma', _GAMMA.cdf,
                            lambda n, rng: _GAMMA.rvs(size=n, random_state=rng), (LC, BLC)),
    'truncated exponential': Latent('truncated exponential', _te_cdf, _te_rvs, (LC, BLC)),
    'bimodal': Latent('bimodal', _bm_cdf, _bm_rvs, (BLC,)),
    # outside the class: polynomial tails make log(1 - F) convex far out
    't3': Latent('t3', lambda w: stats.t.cdf(np.asarray(w) * math.sqrt(3), 3),
                 lambda n, rng: rng.standard_t(3, n) / math.sqrt(3), ()),
}

# noise laws: sampler of unit-variance draws and the classes they belong to
GAUSS, SU, LCN, MEDZ = 'gaussian', 'symmetric_unimodal', 'mean_zero_log_concave', 'median_zero'


def _unit(name, n, rng):
    if name == 'Gaussian':
        return rng.standard_normal(n)
    if name == 'uniform':
        return math.sqrt(3) * rng.uniform(-1, 1, n)
    if name == 'Laplace':
        return rng.laplace(size=n) / math.sqrt(2)
    if name == 't3':
        return rng.standard_t(3, n) / math.sqrt(3)
    if name == 'centred exp':
        return rng.exponential(size=n) - 1
    if name == '-centred exp':
        return 1 - rng.exponential(size=n)
    raise ValueError(name)


NOISE_CLASSES = {
    'Gaussian': {GAUSS, SU, LCN, MEDZ},
    'uniform': {SU, LCN, MEDZ},
    'Laplace': {SU, LCN, MEDZ},
    't3': {SU, MEDZ},
    'centred exp': {LCN},
    '-centred exp': {LCN},
}

# noise designs: each unit draws its law independently and uniformly from the list
DESIGNS = {
    'Gaussian': ('Gaussian',),
    'uniform': ('uniform',),
    'Laplace': ('Laplace',),
    't3': ('t3',),
    'centred exp': ('centred exp',),
    'mixed symmetric unimodal': ('Gaussian', 'uniform', 'Laplace', 't3'),
    'mixed log-concave': ('Gaussian', 'uniform', 'Laplace', 'centred exp', '-centred exp'),
    'mixed across classes': ('Gaussian', 'uniform', 'Laplace', 't3', 'centred exp',
                             '-centred exp'),
}


def design_classes(design):
    """Noise classes that contain every law of the design."""
    return set.intersection(*(NOISE_CLASSES[n] for n in DESIGNS[design]))


def draw_noise(design, D, rng):
    """One noise draw per unit with variance D_i, the law chosen per unit at random."""
    laws = DESIGNS[design]
    n = len(D)
    pick = rng.integers(len(laws), size=n) if len(laws) > 1 else np.zeros(n, dtype=int)
    out = np.empty(n)
    for j, name in enumerate(laws):
        m = pick == j
        out[m] = _unit(name, int(m.sum()), rng)
    return np.sqrt(D) * out
