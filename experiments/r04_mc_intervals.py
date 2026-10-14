"""Monte Carlo uncertainty for the comparison with existing methods (R04), from results/competitors_v2_hetldc.csv.

Per setting (latent law, noise scenario, K) and rule:
  - reliability = share of data sets with latent coverage >= q, with a Wilson 95% interval;
  - mean latent coverage with its standard error over data sets;
  - width relative to the usual rank, mean(width) / mean(width_usual), with a paired bootstrap 95%
    interval that resamples data sets jointly for both rules.
Reliabilities of exactly specified laws elsewhere in the paper are exact and need no interval.

  python experiments/r04_mc_intervals.py
Writes results/competitors_mc_intervals.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / 'results'
Q, B, SEED = 0.9, 2000, 20261014
RULES = ['ours: Gaussian', 'ours: symmetric unimodal', 'shape_free', 'fay_herriot', 'latentcp',
         'cohen_stable']


def wilson(x, n, level=0.95):
    z = stats.norm.ppf(0.5 + level / 2)
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    d = pd.read_csv(RESULTS / 'competitors_v2_hetldc.csv')
    d = d[d.method.isin(RULES + ['usual'])]
    rng = np.random.default_rng(SEED)
    rows = []
    for (latent, noise, K), g in d.groupby(['latent', 'noise', 'K']):
        w = g.pivot_table(index='seed', columns='method', values='width')
        c = g.pivot_table(index='seed', columns='method', values='cov')
        n = len(w)
        idx = rng.integers(0, n, size=(B, n))
        wu = w['usual'].to_numpy()
        for m in RULES:
            if m not in c:
                continue
            cov = c[m].to_numpy()
            x = int(np.sum(cov >= Q - 1e-12))
            lo, hi = wilson(x, n)
            wm = w[m].to_numpy()
            ratio = wm.mean() / wu.mean()
            boot = wm[idx].mean(axis=1) / wu[idx].mean(axis=1)
            rows.append(dict(latent=latent, noise=noise, K=K, rule=m, n=n, reliability=x / n,
                             rel_lo=lo, rel_hi=hi, mean_cov=cov.mean(),
                             mean_cov_se=cov.std(ddof=1) / np.sqrt(n), width_ratio=ratio,
                             width_lo=np.quantile(boot, 0.025), width_hi=np.quantile(boot, 0.975)))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'competitors_mc_intervals.csv', index=False)
    pd.set_option('display.width', 220)
    print(out[out.rule.isin(['fay_herriot', 'ours: Gaussian'])].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
