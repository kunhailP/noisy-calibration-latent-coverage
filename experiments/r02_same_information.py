"""R02: rules with the same information, extended (Table 3 of the Biometrika version, E43).

No rule is told the noise law or the variances. Each rule is the noisy threshold T = |V|_(k) at
the rank of Theorem 2 for a stated noise class; the rules differ only through the noisy level.
Extensions over E43:
  - latent laws: the four log-concave laws of E43, the bimodal law that is bi-log-concave but not
    log-concave, and t3 (outside the class) as a control;
  - noise designs: one law per setting as in E43, and laws drawn per unit within the symmetric
    unimodal class, within the mean-zero log-concave class (both signs of the centred
    exponential), and across the two classes (covered by no class rule);
  - q = 0.8, 0.9, 0.95 (the log-concave rule is certified only at q = 0.9).
Noise variances D_i = 0.577 L_i / E(L_i), L_i lognormal with log-scale 0.7, as in E43. Latent
coverage given the data is exact; widths are T over the oracle radius Q_q(|W|).

  python experiments/r02_same_information.py [reps] [procs]
Writes results/same_information_v2.csv and results/same_information_v2_summary.csv.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from latentcov.laws import (BLC, DESIGNS, GAUSS, LATENT, LCN, MEDZ, SU, design_classes,
                            draw_noise)
from latentcov.levels import level
from latentcov.rank import rank_for_level

RESULTS = Path(__file__).resolve().parents[1] / 'results'
QS, KS, DELTA = (0.8, 0.9, 0.95), (110, 1000), 0.05


def rules(q):
    """name -> (noisy level, latent class required or None, noise class required or None)."""
    out = {'usual': (q, None, None),
           'Gaussian': (level('gaussian', q).sufficient, BLC, GAUSS),
           'symmetric unimodal': (level('symmetric_unimodal', q).sufficient, BLC, SU)}
    if q == 0.9:
        out['log-concave'] = (level('mean_zero_log_concave', q).sufficient, BLC, LCN)
    out['no shape, median-zero'] = (level('gaussian', q, 'none').sufficient, None, MEDZ)
    out['no shape, log-concave'] = (level('mean_zero_log_concave', q, 'none').sufficient,
                                    None, LCN)
    return out


def covers(rule, latent, design):
    _, lat, noise = rule
    if noise is None:
        return False
    return (lat is None or lat in LATENT[latent].classes) and noise in design_classes(design)


def one(args):
    latent, design, K, reps, seed = args
    rng = np.random.default_rng(seed)
    law = LATENT[latent]
    plan = []
    for q in QS:
        r_or = law.oracle_radius(q)
        for name, rule in rules(q).items():
            plan.append((q, name, rule, rank_for_level(K, rule[0], DELTA), r_or))
    cov = np.full((len(plan), reps), np.nan)
    wid = np.full((len(plan), reps), np.nan)
    for b in range(reps):
        D = 0.577 * rng.lognormal(0, 0.7, K) / math.exp(0.7 ** 2 / 2)
        V = np.sort(np.abs(law.rvs(K, rng) + draw_noise(design, D, rng)))
        for i, (q, _, _, k, r_or) in enumerate(plan):
            if k is not None:
                T = V[k - 1]
                cov[i, b] = law.coverage(T)
                wid[i, b] = T / r_or
    rows = []
    for i, (q, name, rule, k, _) in enumerate(plan):
        ok = cov[i] >= q
        rel = float(np.mean(ok)) if k is not None else np.nan
        rows.append(dict(K=K, q=q, latent=latent, design=design, rule=name, rank=k,
                         level=rule[0], guaranteed=covers(rule, latent, design),
                         reliability=rel,
                         se_reliability=math.sqrt(rel * (1 - rel) / reps) if k else np.nan,
                         mean_coverage=float(np.mean(cov[i])) if k else np.nan,
                         rel_width=float(np.mean(wid[i])) if k else np.nan, reps=reps))
    return rows


def summarize(df):
    """Per (K, q, rule): rank, reliability where covered, width above the usual rank and below
    the no-shape rule of the same noise class, over the settings the rule covers."""
    base = df[df.rule == 'usual'].set_index(['K', 'q', 'latent', 'design']).rel_width
    noshape = {'Gaussian': 'no shape, median-zero',
               'symmetric unimodal': 'no shape, median-zero',
               'log-concave': 'no shape, log-concave'}
    ns = df.set_index(['K', 'q', 'latent', 'design', 'rule']).rel_width
    out = []
    for (K, q, rule), g in df.groupby(['K', 'q', 'rule'], sort=False):
        c = g[g.guaranteed]
        idx = list(zip(c.K, c.q, c.latent, c.design))
        above = 100 * (c.rel_width.values / base.loc[idx].values - 1) if len(c) else np.array([])
        row = dict(K=K, q=q, rule=rule, rank=g['rank'].iloc[0], settings_covered=len(c),
                   min_reliability_covered=c.reliability.min() if len(c) else np.nan,
                   min_reliability_bimodal=c[c.latent == 'bimodal'].reliability.min(),
                   min_reliability_mixed=c[c.design.str.startswith('mixed')].reliability.min(),
                   min_reliability_all=g.reliability.min(),
                   above_usual_min=above.min() if len(above) else np.nan,
                   above_usual_max=above.max() if len(above) else np.nan)
        if rule in noshape:
            row['noshape_rank'] = df[(df.K == K) & (df.q == q) & (df.rule == noshape[rule])][
                'rank'].iloc[0]                              # nan: the no-shape rule has none
        if rule in noshape and len(c) and np.isfinite(row['noshape_rank']):
            other = ns.loc[[i + (noshape[rule],) for i in idx]].values
            below = 100 * (1 - c.rel_width.values / other)
            row.update(below_noshape_min=below.min(), below_noshape_median=np.median(below),
                       below_noshape_max=below.max())
        out.append(row)
    return pd.DataFrame(out)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(lat, des, K, reps, 1000 * K + 37 * i + j)
            for K in KS for i, lat in enumerate(LATENT) for j, des in enumerate(DESIGNS)]
    with Pool(procs) as pool:
        rows = [r for part in pool.imap_unordered(one, jobs) for r in part]
    df = pd.DataFrame(rows).sort_values(['K', 'q', 'latent', 'design', 'rule'])
    df.to_csv(RESULTS / 'same_information_v2.csv', index=False)
    s = summarize(df)
    s.to_csv(RESULTS / 'same_information_v2_summary.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
