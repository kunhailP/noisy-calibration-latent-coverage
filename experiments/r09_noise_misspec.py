"""R09: misspecified noise class. For single noise laws inside and outside the classes of the paper,
the lower bound max(Psi_eps, Psi_{-eps}) on the noisy level (src/latentcov/misspec.py) and the exact
latent reliability, at that level, of the ranks computed for the Gaussian, symmetric unimodal and
(at q = 0.9) mean-zero log-concave classes. At the extremal latent law for the noise law, the count
N = #{i : |V_i| < r} is Bin(K, H) with H the level, so the reliability of rank k is pr{Bin(K, H) <= k - 1}.

  python experiments/r09_noise_misspec.py [procs]
Writes results/noise_misspec.csv.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from latentcov.contamination import psi
from latentcov.misspec import standard_noises, two_end_lower_bound
from latentcov.rank import rank_for_level

RESULTS = Path(__file__).resolve().parents[1] / 'results'
QS = (0.8, 0.9, 0.95)
KS = (110, 1000, 10_000, 100_000, 1_000_000)
DELTA = 0.05
LC_LEVEL_09 = 0.906          # certified sufficient level, mean-zero log-concave noise, q = 0.9


def first_below(level, H, thr):
    """Smallest K (29 <= K <= 1e7, all K to 3000, then a log grid refined by bisection) at which the
    rank for `level` has reliability below thr when the true noisy coverage is H."""
    def rel(K):
        k = rank_for_level(K, level, DELTA)
        return np.nan if k is None else float(stats.binom.cdf(k - 1, K, H))
    for K in range(29, 3001):
        r = rel(K)
        if r == r and r < thr:
            return K
    grid = np.unique(np.geomspace(3000, 1e7, 300).astype(int))
    prev = 3000
    for K in grid:
        r = rel(int(K))
        if r == r and r < thr:
            lo, hi = prev, int(K)
            while hi - lo > max(1, lo // 2000):
                mid = (lo + hi) // 2
                rm = rel(mid)
                if rm == rm and rm < thr:
                    hi = mid
                else:
                    lo = mid
            return hi
        prev = int(K)
    return np.nan


def job(args):
    i, q = args
    noise = standard_noises()[i]
    H, s, side = two_end_lower_bound(q, noise)
    rows = []
    rules = [('Gaussian', psi(q, 'gaussian')), ('symmetric unimodal', psi(q, 'symmetric_unimodal'))]
    if q == 0.9:
        rules.append(('mean-zero log-concave', LC_LEVEL_09))
    for rule, level in rules:
        row = dict(noise=noise.name, noise_class=noise.cls, q=q, level_lower_bound=H, scale=s, end=side,
                   rule=rule, rule_level=level, excess=H - level)
        for K in KS:
            k = rank_for_level(K, level, DELTA)
            row[f'rel_{K}'] = np.nan if k is None else float(stats.binom.cdf(k - 1, K, H))
        row['first_K_below_0.95'] = first_below(level, H, 0.95) if H > level else np.nan
        row['first_K_below_0.5'] = first_below(level, H, 0.5) if H > level else np.nan
        rows.append(row)
    return rows


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    jobs = [(i, q) for i in range(len(standard_noises())) for q in QS]
    with Pool(procs) as pool:
        out = [r for rows in pool.map(job, jobs, chunksize=1) for r in rows]
    d = pd.DataFrame(out)
    d.to_csv(RESULTS / 'noise_misspec.csv', index=False)
    pd.set_option('display.width', 250)
    x = d[d.q == 0.9]
    print(x[['noise', 'noise_class', 'rule', 'level_lower_bound', 'excess', 'rel_10000', 'rel_100000',
             'first_K_below_0.95', 'first_K_below_0.5']].to_string(index=False))


if __name__ == '__main__':
    main()
