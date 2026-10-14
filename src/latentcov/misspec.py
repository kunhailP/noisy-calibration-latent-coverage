"""One-end levels for single noise laws, to measure misspecification of the noise class.

For one standardized noise law eps (all its scales), the one-end level is

    Psi_eps(q) = sup_{s > 0} E{min(1, q exp(-s eps))},

attained along exponential-tail latent laws (Proposition 2 of the paper, with the class {s eps}).
The two ends of [-t, t] see eps and -eps, so max(Psi_eps, Psi_{-eps}) is a lower bound on any noisy
level valid for bi-log-concave latent laws under this noise; if it exceeds the level of the assumed
class, the rank computed for that class loses its guarantee as K grows.
"""
import math

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


class Noise:
    """A standardized noise law: mean zero, variance one, given by a frozen scipy law or atoms."""

    def __init__(self, name, law=None, atoms=None, cls=''):
        self.name, self.law, self.atoms, self.cls = name, law, atoms, cls

    def negated(self):
        if self.atoms is not None:
            x, w = self.atoms
            return Noise('-' + self.name, atoms=(-np.asarray(x), np.asarray(w)), cls=self.cls)
        law = self.law
        neg = _Negated(law)
        return Noise('-' + self.name, law=neg, cls=self.cls)

    def expect(self, f, kinks=()):
        if self.atoms is not None:
            x, w = self.atoms
            return float(np.sum(np.asarray(w) * np.array([f(v) for v in x])))
        lo, hi = self.law.ppf(1e-13), self.law.ppf(1 - 1e-13)
        pts = sorted(p for p in (self.law.median(), 0.0, *kinks) if lo < p < hi)
        return quad(lambda v: f(v) * self.law.pdf(v), lo, hi, points=pts or None, limit=500,
                    epsabs=1e-13, epsrel=1e-12)[0]


class _Negated:
    def __init__(self, law):
        self.base = law

    def pdf(self, v):
        return self.base.pdf(-v)

    def ppf(self, u):
        return -self.base.ppf(1 - u)

    def median(self):
        return -self.base.median()


def gamma_s(q, s, noise):
    """E min(1, q exp(-s eps)) for the noise law at scale s."""
    a = math.log(1 / q)
    return noise.expect(lambda v: 1.0 if s * v <= -a else q * math.exp(-s * v), kinks=(-a / s,))


def psi_noise(q, noise):
    """One-end level of a single noise law over its scales; returns (level, maximizing scale)."""
    grid = np.exp(np.linspace(math.log(1e-3), math.log(50), 120))
    vals = [gamma_s(q, s, noise) for s in grid]
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(lambda ls: -gamma_s(q, math.exp(ls), noise), bounds=(math.log(lo), math.log(hi)),
                        method='bounded', options={'xatol': 1e-10})
    best = max((-r.fun, math.exp(r.x)), (vals[i], grid[i]))
    if vals[-1] >= best[0] - 1e-12:      # increasing to the end of the grid: report the limit as s grows
        best = (max(best[0], vals[-1]), float('inf'))
    return best


def two_end_lower_bound(q, noise):
    """max over the two ends of the one-end level: a lower bound on any valid noisy level."""
    a, sa = psi_noise(q, noise)
    b, sb = psi_noise(q, noise.negated())
    return (a, sa, '+') if a >= b else (b, sb, '-')


def _centred(law):
    m, v = law.mean(), law.var()
    return _Std(law, m, math.sqrt(v))


class _Std:
    """(X - mean)/sd of a frozen scipy law."""

    def __init__(self, law, m, sd):
        self.law, self.m, self.sd = law, m, sd

    def pdf(self, v):
        return self.sd * self.law.pdf(self.m + self.sd * v)

    def ppf(self, u):
        return (self.law.ppf(u) - self.m) / self.sd

    def median(self):
        return (self.law.median() - self.m) / self.sd


def standard_noises():
    """Noise laws used in R09, standardized to mean zero and variance one, with their classes."""
    sym = lambda law: _Std(law, 0.0, math.sqrt(law.var()))
    out = [
        Noise('Gaussian', law=sym(stats.norm()), cls='Gaussian'),
        Noise('uniform', law=sym(stats.uniform(loc=-1, scale=2)), cls='symmetric unimodal'),
        Noise('Laplace', law=sym(stats.laplace()), cls='symmetric unimodal'),
        Noise('logistic', law=sym(stats.logistic()), cls='symmetric unimodal'),
        Noise('t5', law=sym(stats.t(5)), cls='symmetric unimodal'),
        Noise('centred exponential', law=_centred(stats.expon()), cls='mean-zero log-concave'),
        Noise('centred gamma(2)', law=_centred(stats.gamma(2)), cls='mean-zero log-concave'),
        Noise('centred gamma(5)', law=_centred(stats.gamma(5)), cls='mean-zero log-concave'),
        Noise('skew-normal(4)', law=_centred(stats.skewnorm(4)), cls='mean-zero log-concave'),
        Noise('centred lognormal(0.5)', law=_centred(stats.lognorm(0.5)), cls='mean zero'),
        Noise('centred lognormal(1)', law=_centred(stats.lognorm(1.0)), cls='mean zero'),
        Noise('centred chi-square(1)', law=_centred(stats.chi2(1)), cls='mean zero'),
        Noise('two-point symmetric', atoms=([-1.0, 1.0], [0.5, 0.5]), cls='median zero'),
        Noise('two-point, p = 0.2', atoms=([-0.5, 2.0], [0.8, 0.2]), cls='mean zero'),
    ]
    return out
