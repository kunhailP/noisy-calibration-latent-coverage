"""R06: check of the heterogeneous-latent corollary (notes/heterogeneous_latent.md).

Calibration unit i has its own latent law G_i and Gaussian noise sigma_i Z; the new unit has law
G_0 with r0 = Q_q(|W_new|) = 1. With pi_i = pr(|V_i| < r0), the latent reliability of rank k is
pr{sum_i Bernoulli(pi_i) <= k - 1}. The corollary says pi_i <= Psi_N(q) whenever G_i is
bi-log-concave and Q_q(|W_i|) >= r0.

Latent laws are exponential tails W = b - E/u (rate u), the extremal laws of Proposition 1, with
b chosen so that Q_q(|W|) = r_i = 1 + tau_i/u_i (tau in units of the tail scale 1/u); sigma is
adversarial per unit (pi_i maximal). Each pi_i is computed exactly (closed form, maximised in
sigma numerically).

Part 1, unit parameters i.i.d. from a design: the indicators are i.i.d. with H = E pi, and the
reliability is pr{Bin(K, H) <= k - 1}. H is exact for the homogeneous and the point-mass designs
and a Monte Carlo mean over 4000 draws for the heterogeneous ones (its standard error is
reported; the reliabilities of those rows carry that error).
  homogeneous extremal              tau = 0, u = 200 (Theorem 2 tight)
  homogeneous, slightly easier      r_i = 0.999995 (tau = -0.001), u = 200 and 2000
  heterogeneous, condition holds    u_i = 200 L_i, L_i lognormal(0, 0.7), tau_i = |N(0, 0.02^2)|
  10% easier units                  as above, 10% of units with tau_i = -0.02 (condition fails)
  random unit (mixture)             point masses at 0 and 1 + eta, the new unit a random unit

Part 2, fixed heterogeneous units (independent, not identically distributed): one draw of the
parameters of K units is held fixed and the reliability is the Poisson-binomial probability,
computed exactly by recursion, for K = 1000 and 10000.

  python experiments/r06_heterogeneous_latent.py
Writes results/heterogeneous_latent.csv and results/heterogeneous_latent_fixed.csv.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr

from latentcov.levels import level
from latentcov.rank import rank_for_level

RESULTS = Path(__file__).resolve().parents[1] / 'results'
DELTA = 0.05
KS = [110, 1000, 10_000, 100_000, 1_000_000]


def cdf_v(x, b, u, sigma):
    """pr(b - E/u + sigma Z <= x)."""
    d = (x - b) / sigma
    return float(ndtr(d) + math.exp(u * (x - b) + (u * sigma) ** 2 / 2 + log_ndtr(-d - u * sigma)))


def b_for(r, u, q):
    """b with Q_q(|b - E/u|) = r: e^{-u(b - r)} (1 - e^{-2ur}) = q, needs 1 - e^{-2ur} > q."""
    return r + math.log(-math.expm1(-2 * u * r) / q) / u


def pi(r, u, sigma, q, r0=1.0):
    b = b_for(r, u, q)
    return cdf_v(r0, b, u, sigma) - cdf_v(-r0, b, u, sigma)


def pi_adv(r, u, q):
    """max over sigma of pr(|V| < 1) for the exponential tail with Q_q(|W|) = r."""
    res = minimize_scalar(lambda ls: -pi(r, u, math.exp(ls), q), bounds=(-12, 3),
                          method='bounded', options={'xatol': 1e-12})
    return -res.fun


def _mean_se(x):
    x = np.asarray(x)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def heterogeneous_units(q, rng, n, easier_share=0.0):
    """pi_i for n units with u_i = 200 L_i and tau_i = |N(0, 0.02^2)|, a share set to -0.02."""
    u = 200 * rng.lognormal(0, .7, n)
    tau = np.abs(rng.normal(0, .02, n))
    if easier_share:
        tau = np.where(rng.random(n) < easier_share, -.02, tau)
    return np.array([pi_adv(1 + t / ui, ui, q) for t, ui in zip(tau, u)])


def designs(q, rng, n=4000):
    """(design, H, standard error of H); the standard error is 0 where H is exact."""
    yield 'homogeneous extremal', pi_adv(1.0, 200.0, q), 0.0
    for u in (200.0, 2000.0):
        yield f'homogeneous, slightly easier (r_i = 0.999995, u = {u:g})', \
            pi_adv(0.999995, u, q), 0.0
    yield ('heterogeneous, condition holds', *_mean_se(heterogeneous_units(q, rng, n)))
    yield ('10% easier units (tau = -0.02)', *_mean_se(heterogeneous_units(q, rng, n, .1)))
    # random unit: a share w of point masses at 1 + eta, the rest at 0, w just above 1 - q, so
    # Q_q of the mixture is 1 + eta; small Gaussian noise; pr(|V| < 1 + eta) at the two atoms
    w, eta, s = (1 - q) * 1.001, 1e-3, 1e-4
    p_out = ndtr(0.0) - ndtr(-2 * (1 + eta) / s)
    p_in = ndtr((1 + eta) / s) - ndtr(-(1 + eta) / s)
    yield 'random unit (mixture of point masses, sigma = 1e-4)', w * p_out + (1 - w) * p_in, 0.0


def poisson_binomial_cdf(p):
    """pr{sum Bernoulli(p_i) <= k} for k = 0..n, by the exact recursion."""
    a = np.zeros(len(p) + 1)
    a[0] = 1.0
    for j, x in enumerate(p, start=1):
        a[1:j + 1] = a[1:j + 1] * (1 - x) + a[:j] * x
        a[0] *= 1 - x
    return np.cumsum(a)


def main():
    rng = np.random.default_rng(20261007)
    rows, fixed = [], []
    for q in (0.8, 0.9):
        psi = level('gaussian', q).sufficient
        ranks = {'usual': lambda K: rank_for_level(K, q, DELTA),
                 'Gaussian class (Theorem 2)': lambda K: rank_for_level(K, psi, DELTA),
                 'no latent shape': lambda K: rank_for_level(K, 1 - (1 - q) / 2, DELTA)}
        for design, H, se in designs(q, rng):
            for K in KS:
                for name, f in ranks.items():
                    k = f(K)
                    rel = float('nan') if k is None else float(stats.binom.cdf(k - 1, K, H))
                    rows.append(dict(q=q, design=design, H=H, H_se=se, psi=psi,
                                     H_minus_psi=H - psi, K=K, rule=name, k=k, reliability=rel))
        for K in (1000, 10_000):
            for design, share in (('heterogeneous, condition holds', 0.0),
                                  ('10% easier units (tau = -0.02)', .1)):
                p = heterogeneous_units(q, rng, K, share)
                F = poisson_binomial_cdf(p)
                for name, f in ranks.items():
                    k = f(K)
                    fixed.append(dict(q=q, design=design, K=K, max_pi=p.max(), mean_pi=p.mean(),
                                      psi=psi, rule=name, k=k,
                                      reliability=float('nan') if k is None else F[k - 1]))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / 'heterogeneous_latent.csv', index=False)
    fx = pd.DataFrame(fixed)
    fx.to_csv(RESULTS / 'heterogeneous_latent_fixed.csv', index=False)
    pd.set_option('display.width', 250)
    hs = d.drop_duplicates(['q', 'design'])[['q', 'design', 'H', 'H_se', 'H_minus_psi']]
    print(hs.to_string(index=False))
    wide = d.pivot_table(index=['q', 'design', 'rule'], columns='K', values='reliability')
    print(wide.round(4).to_string())
    print(fx.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
