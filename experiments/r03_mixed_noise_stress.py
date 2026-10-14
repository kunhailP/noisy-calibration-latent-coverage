"""R03: exact latent reliability under mixed noise laws at the extremal latent law.

Latent law: the exponential tail with pr(W <= t) = q (log-concave, hence bi-log-concave), in the
small-noise limit where the left end does not matter. In units of the exponential's scale, a unit
whose noise is c * eps, eps of unit variance, has noisy coverage at the latent q-quantile

    h(eps, c) = E min(1, q e^{-c eps})                         (Proposition 1).

Each unit draws its noise law from a mixture and its scale c = c0 L, L lognormal with log-scale
0.7 and mean 1 (or L = 1), independently. The |V_i| are then i.i.d. and

    H = E h(eps, c0 L),   latent reliability of rank k = pr{Bin(K, H) <= k - 1}

exactly. c0 is chosen adversarially (H maximal). Theorem 2 says H <= Psi_E(q) for every mixture
within the class, so the class rank keeps reliability >= 1 - delta, while the usual rank fails as
K grows whenever H > q.

  python experiments/r03_mixed_noise_stress.py
Writes results/mixed_noise_stress.csv and results/mixed_noise_stress_summary.csv.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr

from latentcov.levels import level
from latentcov.rank import rank_for_level

RESULTS = Path(__file__).resolve().parents[1] / 'results'
DELTA = 0.05
KS = [110, 1000, 10_000, 100_000, 1_000_000]

# unit-variance densities (support) of the noise laws
DENS = {
    'uniform': (lambda e: np.full_like(e, 1 / (2 * math.sqrt(3))), -math.sqrt(3), math.sqrt(3)),
    'Laplace': (lambda e: math.sqrt(2) / 2 * np.exp(-math.sqrt(2) * np.abs(e)), -np.inf, np.inf),
    't3': (lambda e: math.sqrt(3) * stats.t.pdf(math.sqrt(3) * e, 3), -np.inf, np.inf),
    'centred exp': (lambda e: np.exp(-(e + 1)), -1.0, np.inf),
    '-centred exp': (lambda e: np.exp(-(1 - e)), -np.inf, 1.0),
}


def h(law, c, q):
    """E min(1, q e^{-c eps}) for unit-variance eps of the given law, c > 0."""
    e0 = math.log(q) / c                       # q e^{-c eps} >= 1 iff eps <= e0
    if law == 'Gaussian':
        return float(ndtr(e0) + math.exp(math.log(q) + c * c / 2 + log_ndtr(-e0 - c)))
    f, lo, hi = DENS[law]
    g = lambda e: float(f(np.array(e)))
    quad = lambda fn, x0, x1: integrate.quad(fn, x0, x1, epsabs=1e-14, epsrel=1e-12,
                                             limit=400)[0]
    # the mass below e0 and the weighted mass above it, each split at finite cut points so
    # that quad sees the bulk of the integrand on a finite interval
    if e0 <= lo:
        below = 0.0
    elif math.isfinite(lo):
        below = quad(g, lo, e0)
    else:
        below = quad(g, -np.inf, min(e0, -40.0)) + (quad(g, -40.0, e0) if e0 > -40 else 0.0)
    a = max(lo, e0)
    if a >= hi:
        return below
    wt = lambda e: math.exp(math.log(q) - c * e) * g(e)
    b = min(hi, a + 60 / c, max(a, 0.0) + 60.0)
    above = quad(wt, a, b) + (quad(wt, b, hi) if b < hi else 0.0)
    return below + above


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)       # E f(N(0, 1)) = sum w f(x)/sqrt(2pi)


def H(mixture, c0, q, hetero):
    """Average noisy coverage at the latent q-quantile over the noise mixture and the scales."""
    if hetero:
        L = np.exp(0.7 * _GH_X - 0.7 ** 2 / 2)
        w = _GH_W / math.sqrt(2 * math.pi)
    else:
        L, w = np.array([1.0]), np.array([1.0])
    total = 0.0
    for law, p in mixture.items():
        total += p * sum(wi * h(law, c0 * Li, q) for Li, wi in zip(L, w))
    return total


def worst_H(mixture, q, hetero):
    r = minimize_scalar(lambda lc: -H(mixture, math.exp(lc), q, hetero),
                        bounds=(math.log(1e-3), math.log(20)), method='bounded',
                        options={'xatol': 1e-9})
    grid = np.geomspace(1e-3, 20, 120)
    vals = [H(mixture, c, q, hetero) for c in grid]
    i = int(np.argmax(vals))
    if vals[i] > -r.fun:                       # the bounded search found a local maximum only
        r = minimize_scalar(lambda lc: -H(mixture, math.exp(lc), q, hetero),
                            bounds=(math.log(grid[max(i - 1, 0)]),
                                    math.log(grid[min(i + 1, len(grid) - 1)])),
                            method='bounded', options={'xatol': 1e-10})
    return -r.fun, math.exp(r.x)


def mix(extremal, w, others):
    m = {extremal: w} if isinstance(extremal, str) else {e: w / len(extremal) for e in extremal}
    for o in others:
        m[o] = m.get(o, 0) + (1 - w) / len(others)
    return m


# (class, q, label, mixture); every law of a mixture lies in the class
CASES = []
for q in (0.8, 0.9):
    for w in (1.0, 0.5, 0.25):
        CASES.append(('symmetric_unimodal', q, f'uniform {w:.2f} + rest Gaussian/Laplace/t3',
                      mix('uniform', w, ('Gaussian', 'Laplace', 't3'))))
for w in (1.0, 0.5, 0.25):
    CASES.append(('mean_zero_log_concave', 0.9,
                  f'centred exp {w:.2f} + rest Gaussian/uniform/Laplace/-centred exp',
                  mix('centred exp', w, ('Gaussian', 'uniform', 'Laplace', '-centred exp'))))
CASES.append(('mean_zero_log_concave', 0.9, 'centred exp both signs 0.5/0.5',
              mix(('centred exp', '-centred exp'), 1.0, ())))


def _worst(args):
    i, hetero = args
    _, q, _, mixture = CASES[i]
    return worst_H(mixture, q, hetero)


def main():
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)
    jobs = [(i, hetero) for i in range(len(CASES)) for hetero in (False, True)]
    with Pool(len(jobs)) as pool:
        worst = dict(zip(jobs, pool.map(_worst, jobs, chunksize=1)))
    rows, summ = [], []
    for i, (cls, q, label, mixture) in enumerate(CASES):
        lv = level(cls, q)
        for hetero in (False, True):
            Hq, c0 = worst[i, hetero]
            assert Hq <= lv.sufficient + 1e-9, (cls, q, label, Hq)   # Theorem 2 bound
            s = dict(noise_class=cls, q=q, mixture=label, hetero_scales=hetero, c0=c0, H=Hq,
                     excess=Hq - q, class_level=lv.sufficient)
            for K in KS:
                for rule, p in (('usual', q), ('class rank', lv.sufficient)):
                    k = rank_for_level(K, p, DELTA)
                    rel = float(stats.binom.cdf(k - 1, K, Hq)) if k is not None else np.nan
                    rows.append(dict(s, K=K, rule=rule, rank=k, reliability=rel))
                    s[f'{rule} K={K}'] = rel
            summ.append(s)
            print(f'{cls:22s} q={q} hetero={hetero!s:5s} H={Hq:.7f} c0={c0:.4f}  {label}',
                  flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / 'mixed_noise_stress.csv', index=False)
    s = pd.DataFrame(summ)
    s.to_csv(RESULTS / 'mixed_noise_stress_summary.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['noise_class', 'q', 'mixture', 'hetero_scales', 'H'] + \
        [f'{r} K={K}' for r in ('usual', 'class rank') for K in (1000, 10_000, 100_000)]
    print(s[cols].round(4).to_string(index=False))


def figure():
    """Reliability against K for the usual and the class rank, every mixture, q = 0.9."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = pd.read_csv(RESULTS / 'mixed_noise_stress_summary.csv')
    d = d[d.q == 0.9]
    Ks = np.unique(np.round(np.geomspace(30, 1e7, 120)).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8), sharey=True)
    for ax, cls in zip(axes, ('symmetric_unimodal', 'mean_zero_log_concave')):
        lv = level(cls, 0.9).sufficient
        for _, r in d[d.noise_class == cls].iterrows():
            for rule, p, ls in (('usual', 0.9, '--'), ('class rank', lv, '-')):
                rel = [stats.binom.cdf(k - 1, K, r.H) if (k := rank_for_level(K, p, DELTA))
                       else np.nan for K in Ks]
                ax.plot(Ks, rel, 'k', ls=ls, lw=0.8 + 0.6 * r.hetero_scales,
                        alpha=0.35 + 0.65 * (r.H == d[d.noise_class == cls].H.max()))
        ax.axhline(1 - DELTA, color='0.6', lw=0.8)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_xlabel('$K$')
        ax.set_title(cls.replace('_', ' ') + ' noise, mixed')
    axes[0].set_ylabel('Latent reliability')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_mixed_noise.pdf')
    fig.savefig(RESULTS / 'fig_mixed_noise.png', dpi=130)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['figure']:
        main()
    figure()
