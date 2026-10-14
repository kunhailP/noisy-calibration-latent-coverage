"""R05: school-district application (apipop, CRAN `survey`), rules by the information they use.

Each replication splits the districts with at least five schools into 110 training, 110
calibration and the remaining evaluation districts, samples n = 2 or 3 schools per training and
calibration district without replacement, fits a ridge predictor on the training schools, and
centres each district at the population mean of the predictor. The calibration score is the
sample-mean residual V_c; its design variance D_c is estimated from n schools and is therefore
very noisy. The target of an evaluation district is its true mean residual. Coverage is the
share of evaluation districts covered (finite-population coverage), not the probability for one
new district of Theorem 2.

At q = 0.9 and K = 110 every class rank of Theorem 3 equals the usual rank (k = 105). The rules
with the same information (no variance estimates) are also compared at q = 0.95, where the rank
without a shape assumption does not exist at K = 110, and with the contamination levels p_xi(q)
(Proposition 5), xi = 1% and 5%. Fay-Herriot, LatentCP and the Markov rule plug in the D_c as if
known and are run at q = 0.9 only.

  python experiments/fetch_apipop.py           (once)
  python experiments/r05_apipop.py [reps] [procs]
Writes results/apipop.csv and results/apipop_summary.csv.
"""
import math
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from latentcov.contamination import level as contamination_level
from latentcov.kv.procedures import fay_herriot_boot_pac, fay_herriot_reml, shape_free_markov
from latentcov.levels import level
from latentcov.rank import rank_for_level

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
Q, ALPHA, DELTA = 0.9, 0.1, 0.05
QS = (0.9, 0.95)
XIS = (0.01, 0.05)
CONDITIONS = [('api99', 3), ('api99', 2), ('socio', 3), ('socio', 2)]


def load_apipop():
    """Districts with >= 5 schools, covariates cleaned (as in the Biometrika version)."""
    path = ROOT / 'data' / 'apipop.pkl'
    if not path.exists():
        raise SystemExit('data/apipop.pkl missing: run  python experiments/fetch_apipop.py')
    d = pd.read_pickle(path)
    d = d[d.groupby('dnum').dnum.transform('size') >= 5].copy()
    base = ['meals', 'ell', 'mobility', 'not.hsg', 'hsg', 'some.col', 'col.grad', 'grad.sch',
            'avg.ed', 'full', 'emer', 'enroll', 'pct.resp']
    d['yr_rnd'] = (d['yr.rnd'].astype(str) == 'Yes').astype(float)
    d = pd.concat([d, pd.get_dummies(d.stype.astype(str), prefix='st', dtype=float)], axis=1)
    extra = ['yr_rnd'] + [c for c in d.columns if str(c).startswith('st_')]
    for c in base + ['api99', 'api00']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d[base] = d[base].fillna(d[base].median())
    return d, {'socio': base + extra, 'api99': base + extra + ['api99']}


def replication(d, feats, n, rng, n_train=110, n_calib=110):
    from sklearn.linear_model import RidgeCV
    groups = {k: g for k, g in d.groupby('dnum')}
    p = rng.permutation(np.array(list(groups)))
    T, C, E = p[:n_train], p[n_train:n_train + n_calib], p[n_train + n_calib:]
    samp = {c: groups[c].sample(n, random_state=int(rng.integers(1e9))) for c in np.r_[T, C]}
    tr = pd.concat([samp[c] for c in T])
    h = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(tr[feats].values, tr.api00.values)
    V, D = [], []
    for c in C:
        g, s = groups[c], samp[c]
        res = s.api00.values - h.predict(s[feats].values)
        V.append(res.mean())
        D.append((1 - n / len(g)) * res.var(ddof=1) / n)
    We = np.array([(groups[c].api00.values - h.predict(groups[c][feats].values)).mean()
                   for c in E])
    return np.array(V), np.maximum(np.array(D), 1e-6), We


def latentcp_radius(d, q, gamma):
    from scipy.optimize import brentq
    from scipy.special import ndtr
    p = lambda w: ndtr((q - w) / math.sqrt(d)) - ndtr((-q - w) / math.sqrt(d))
    target = 1 - gamma / ALPHA
    if p(0.0) < target:
        return 0.0
    return brentq(lambda w: p(w) - target, 0, q + 10 * math.sqrt(d))


def one(args):
    fset, n, seed = args
    d, feats = load_apipop()
    rng = np.random.default_rng(seed)
    V, D, We = replication(d, feats[fset], n, rng)
    K = len(V)
    S = np.sort(np.abs(V))
    rows = []
    for q in QS:
        levels = [('usual', q), ('ours: Gaussian', level('gaussian', q).sufficient),
                  ('ours: symmetric unimodal', level('symmetric_unimodal', q).sufficient),
                  ('no shape, median-zero', level('gaussian', q, 'none').sufficient)]
        if q == 0.9:
            levels.append(('ours: mean-zero log-concave', level('mean_zero_log_concave', q).sufficient))
        for xi in XIS:
            levels.append((f'contamination {100 * xi:g}%, Gaussian',
                           min(1.0, contamination_level(q, xi, 'gaussian') + 1e-12)))
        for name, lev in levels:
            k = rank_for_level(K, lev, DELTA)
            hw = S[k - 1] if k else np.inf
            rows.append(dict(condition=f'{fset}_n{n}', seed=seed, q=q, method=name, rank=k,
                             width=2 * hw, cov=float(np.mean(np.abs(We) <= hw))))
    ints = {}
    mu, A, vmu = fay_herriot_reml(V, D)
    _, lam = fay_herriot_boot_pac(V, D, mu, A, rng, level=Q, delta=DELTA)
    ints['fay_herriot (plug-in D)'] = (mu, lam * math.sqrt(A + vmu))
    g = ALPHA / 2
    kq = int(math.ceil((K + 1) * (1 - g)))
    qg = S[kq - 1] if kq <= K else np.inf
    # application variant of LatentCP: a new district's variance is unknown, so it is drawn from
    # the calibration estimates D_c; LatentCP itself assumes the forward model (D_new) is known
    dnew = rng.choice(D, 200)
    radii = np.array([latentcp_radius(x, qg, g) for x in dnew])
    for name, (c, hw) in ints.items():
        rows.append(dict(condition=f'{fset}_n{n}', seed=seed, q=Q, method=name, width=2 * hw,
                         cov=float(np.mean(np.abs(We - c) <= hw))))
    rows.append(dict(condition=f'{fset}_n{n}', seed=seed, q=Q, method='latentcp (plug-in D)',
                     width=float(2 * radii.mean()),
                     cov=float(np.mean([np.mean(np.abs(We) <= r) for r in radii]))))
    k_sf = rank_for_level(K, Q + (1 - Q) / 4, DELTA)
    try:
        hw = shape_free_markov(V, D, k_sf, Q, DELTA)[0]
    except ValueError:
        hw = np.inf
    rows.append(dict(condition=f'{fset}_n{n}', seed=seed, q=Q, method='shape_free (plug-in D)',
                     width=2 * hw, cov=float(np.mean(np.abs(We) <= hw))))
    return rows


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    jobs = [(f, n, 70000 + 1000 * i + r) for i, (f, n) in enumerate(CONDITIONS)
            for r in range(reps)]
    with Pool(procs) as pool:
        r = pd.DataFrame([x for xs in pool.imap_unordered(one, jobs, chunksize=2) for x in xs])
    r.to_csv(RESULTS / 'apipop.csv', index=False)
    def wilson(x, n, z=1.959963984540054):
        p = x / n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return c - h, c + h
    out = []
    for (cond, q, m), g in r.groupby(['condition', 'q', 'method'], sort=False):
        ok = int(np.sum(g['cov'] >= q - 1e-12))
        lo, hi = wilson(ok, len(g))
        out.append(dict(condition=cond, q=q, method=m, reps=len(g),
                        rank=g['rank'].iloc[0] if 'rank' in g and g['rank'].notna().all() else np.nan,
                        mean_cov=g['cov'].mean(), mean_cov_se=g['cov'].std(ddof=1) / math.sqrt(len(g)),
                        pr_cov_ge_q=ok / len(g), pr_lo=lo, pr_hi=hi, min_cov=g['cov'].min(),
                        width=g['width'].mean()))
    s = pd.DataFrame(out)
    s.to_csv(RESULTS / 'apipop_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(s.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
