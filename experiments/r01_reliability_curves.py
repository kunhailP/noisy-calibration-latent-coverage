"""R01: exact latent reliability of the usual rank at the Gaussian-extremal law, q = 0.8, 0.9, 0.95.

Law (threshold scaled to 1, Gaussian noise N(0, s^2)): the exponential tail
    W = 1 + (s/u)(a - E),  E ~ Exp(1),  a = log(1/q),
with u the maximiser of E min(1, q e^{-uZ}) (Proposition 1). Then pr(W <= 1) = q and, as s -> 0,
the noisy coverage at the latent q-quantile r_q of |W| tends to Psi_N(q). With H the distribution
function of |W + e|, the latent reliability of rank k is pr{Bin(K, H(r_q)) <= k - 1} exactly,
as in the proof of Proposition 3, so no Monte Carlo error enters.

The usual rank fails once sqrt(K) (H(r_q) - q) / sqrt(q (1 - q)) exceeds the normal quantile of the
rank's own slack; the reliability is about 1/2 near K_half = z^2 q (1 - q) / (H(r_q) - q)^2.

  python experiments/r01_reliability_curves.py
Writes results/reliability_curves.csv, results/reliability_summary.json, results/fig_reliability_q.pdf.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from latentcov.levels import LEGACY_GAUSSIAN, level, psi_gaussian_float
from latentcov.rank import marginal_rank, rank, rank_for_level

RESULTS = Path(__file__).resolve().parents[1] / 'results'
RESULTS.mkdir(exist_ok=True)
S = 1e-3                     # noise sd relative to the threshold; the left end is then negligible
DELTA = 0.05
QS = (0.8, 0.9, 0.95)
QUOTED = [110, 1000, 10_000, 100_000, 1_000_000, 10_000_000]


class ExtremalLaw:
    def __init__(self, q, s=S):
        self.q, self.s = q, s
        self.psi, self.u = psi_gaussian_float(q)
        self.lam = self.u / s                         # rate of (s/u) E
        self.A = 1 + s * math.log(1 / q) / self.u     # right end of the support of W

    def latent_cdf(self, w):
        return math.exp(min(0.0, self.lam * (w - self.A)))

    def noisy_cdf(self, c):
        """pr(W + e <= c), e ~ N(0, s^2)."""
        z, s, lam = c - self.A, self.s, self.lam
        return ndtr(z / s) + math.exp(lam * z + (lam * s) ** 2 / 2 + log_ndtr(-z / s - lam * s))

    def r_q(self):
        cov = lambda r: self.latent_cdf(r) - self.latent_cdf(-r) - self.q
        return brentq(cov, 0.5, 1.5, xtol=1e-15)

    def H(self, r):
        return self.noisy_cdf(r) - self.noisy_cdf(-r)


def reliability(K, k, H):
    return float(stats.binom.cdf(k - 1, K, H)) if k is not None else float('nan')


_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)
_GL_U, _GL_W = (_GL_X + 1) / 2, _GL_W / 2


def mean_latent_coverage(law, K, k):
    """E pr(|W| <= T | D) with T = |V|_(k): F(T) ~ Beta(k, K + 1 - k) for F the distribution
    function of |V| (continuous), so the mean is the integral over u of G(F^{-1}(Beta^{-1}(u))),
    G the distribution function of |W|; Gauss-Legendre in u."""
    if k is None or k > K:
        return float('nan')
    b = stats.beta.ppf(_GL_U, k, K + 1 - k)
    G = lambda t: law.latent_cdf(t) - law.latent_cdf(-t)
    vals = []
    for bi in b:
        t = brentq(lambda x: law.H(x) - bi, 1e-9, 3.0, xtol=1e-14)
        vals.append(G(t))
    return float(np.dot(_GL_W, vals))


def main():
    Ks = np.unique(np.r_[np.round(np.geomspace(30, 1e7, 90)).astype(int), QUOTED])
    rows, summary = [], {}
    for q in QS:
        law = ExtremalLaw(q)
        r = law.r_q()
        H = law.H(r)
        g = level('gaussian', q)
        su = level('symmetric_unimodal', q)
        z = stats.norm.ppf(1 - DELTA)
        summary[str(q)] = dict(u=law.u, psi_gaussian_float=law.psi, r_q=r, H_at_r_q=H,
                               excess=H - q, gaussian_level=[g.necessary, g.sufficient],
                               gaussian_level_biometrika=LEGACY_GAUSSIAN[q],
                               K_half_normal_approx=z ** 2 * q * (1 - q) / (H - q) ** 2)
        print(f'q = {q}: u = {law.u:.5f}, r_q = {r:.9f}, H(r_q) = {H:.8f}, '
              f'Psi_N = {law.psi:.8f}, excess = {H - q:.3e}', flush=True)
        for K in Ks:
            K = int(K)
            gr = rank(K, q, DELTA, 'gaussian')
            ranks = {'marginal': marginal_rank(K, q),
                     'usual': rank_for_level(K, q, DELTA),
                     'gaussian_certified': gr.k,                    # exact: p_N(q) = Psi_N(q)
                     'gaussian_biometrika': rank_for_level(K, LEGACY_GAUSSIAN[q][1], DELTA),
                     'symmetric_unimodal': rank_for_level(K, su.sufficient, DELTA)}
            for rule, k in ranks.items():
                if k is not None and k <= K:
                    cov = (mean_latent_coverage(law, K, k)
                           if rule in ('usual', 'gaussian_certified') else float('nan'))
                    rows.append(dict(q=q, K=K, rule=rule, rank=k,
                                     reliability=reliability(K, k, H), mean_latent_coverage=cov))
    df = pd.DataFrame(rows)
    for q in QS:
        u = df[(df.q == q) & (df.rule == 'usual')].sort_values('K')
        for tau in (0.95, 0.9, 0.5):
            below = u[u.reliability < tau]
            summary[str(q)][f'first_grid_K_usual_below_{tau}'] = (
                int(below.K.iloc[0]) if len(below) else None)
        c = df[(df.q == q) & (df.rule == 'gaussian_certified')]
        summary[str(q)]['min_reliability_gaussian_certified'] = float(c.reliability.min())
    df.to_csv(RESULTS / 'reliability_curves.csv', index=False)
    json.dump(summary, open(RESULTS / 'reliability_summary.json', 'w'), indent=1)
    show = df[df.K.isin(QUOTED)].pivot_table(index=['q', 'K'], columns='rule', values='reliability')
    print(show.round(4).to_string())
    print(json.dumps(summary, indent=1))
    figure(df)


def figure(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0), sharex=True)
    for j, q in enumerate(QS):
        ax, bx = axes[0, j], axes[1, j]
        for rule, ls, lab in (('gaussian_certified', '-', 'rank of Theorem 3 (Gaussian)'),
                              ('usual', '--', 'usual rank'), ('marginal', ':', 'marginal rank')):
            g = df[(df.q == q) & (df.rule == rule)].sort_values('K')
            ax.plot(g.K, g.reliability, 'k', ls=ls, lw=1.3, label=lab)
            if rule != 'marginal':
                bx.plot(g.K, g.mean_latent_coverage, 'k', ls=ls, lw=1.3, label=lab)
        ax.axhline(1 - DELTA, color='0.6', lw=0.8)
        bx.axhline(q, color='0.6', lw=0.8)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_title(f'$q = {q}$')
        bx.set_xlabel('$K$')
        lo, hi = {0.8: (0.006, 0.014), 0.9: (0.0012, 0.0048), 0.95: (0.0004, 0.0016)}[q]
        bx.set_ylim(q - lo, q + hi)
    axes[0, 0].set_ylabel('Latent reliability')
    axes[1, 0].set_ylabel('Mean latent coverage')
    axes[0, 0].legend(frameon=False, loc='lower left', fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_reliability_q.pdf')
    fig.savefig(RESULTS / 'fig_reliability_q.png', dpi=130)


if __name__ == '__main__':
    main()
