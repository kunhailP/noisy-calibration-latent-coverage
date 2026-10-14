"""R04: comparison with existing methods, organised by the information each one uses.

Setting (Supplementary Table S5 of the Biometrika version, extended): latent residuals with
variance 1 from the six laws of `latentcov.laws`; noise variances D_i = 0.577 L_i / E(L_i), L_i
lognormal with log-scale 0.7; K = 110 and 1000; q = 0.9, delta = 0.05. Three noise scenarios
with the same variances: Gaussian (the competitors' forward model holds), Laplace, and laws
drawn per unit from the symmetric unimodal class (Gaussian, uniform, Laplace, t3). Every rule
outputs a latent interval; its coverage given the data is exact from the latent distribution
function. Reliability is the share of data sets with coverage >= q; LatentCP and the plain
conformal rule target marginal coverage, i.e. mean coverage >= q.

Rules (information used -> guarantee):
  no noise information
    usual          noisy threshold at the usual high-probability rank       none for W
    ours: Gaussian / symmetric unimodal   class rank of Theorem 2            PAC, bi-log-concave W
    no shape, median-zero                  level (1 + q)/2 (Proposition 2)   PAC, any W
  known variances D_i (Gaussian noise assumed)
    latentcp       LatentCP single level gamma = alpha/2 (Zheng, Zhou & Zhu)   marginal
    latentcp_tuned / latentcp_multi   tuned single / two-level, half the data for tuning  marginal
    latentcp_pac   single level at a high-probability rank (our adaptation)  PAC
    shape_free     Markov rule with the known D_i (`kv.procedures`)          PAC, any W
    cohen          deconvolution threshold (Cohen, Goldberger & Tirer, Alg. 1), bin 0.01 / 0.2  none
    fay_herriot    Fay-Herriot with parametric-bootstrap tolerance multiplier  normal model
  known lower bound D_min
    simple_shrink  max{0, T - 0.114 D_min^{1/2}} (Supplementary Corollary S1)  PAC, bi-log-concave W
  known variances, heavy (only with `hetldc`): HetLDC with certified radius, PAC, log-concave W

  python experiments/r04_competitors.py [reps110] [reps1000] [procs] [hetldc]
Writes results/competitors_v2.csv and results/competitors_v2_summary.csv.
"""
import json
import math
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, lsq_linear
from scipy.special import ndtr

from latentcov.kv.procedures import (fay_herriot_boot_pac, fay_herriot_reml, hetldc_certified,
                                     shape_free_markov, simple_shrink_halfwidth)
from latentcov.laws import LATENT, draw_noise
from latentcov.levels import level
from latentcov.rank import rank_for_level

warnings.filterwarnings('ignore')
RESULTS = Path(__file__).resolve().parents[1] / 'results'
Q, ALPHA, DELTA, DBAR = 0.9, 0.1, 0.05, 0.577
SCENARIOS = {'Gaussian': 'Gaussian', 'Laplace': 'Laplace',
             'mixed symmetric unimodal': 'mixed symmetric unimodal'}
GUARANTEE = {
    'usual': ('none', 'none'), 'ours: Gaussian': ('none', 'PAC, bi-log-concave W, Gaussian noise'),
    'ours: symmetric unimodal': ('none', 'PAC, bi-log-concave W, symm. unimodal noise'),
    'no shape, median-zero': ('none', 'PAC, any W, median-zero noise'),
    'latentcp': ('known D_i', 'marginal'), 'latentcp_tuned': ('known D_i', 'marginal'),
    'latentcp_multi': ('known D_i', 'marginal'), 'latentcp_pac': ('known D_i', 'PAC (adaptation)'),
    'shape_free': ('known D_i', 'PAC, any W'), 'cohen': ('known D_i', 'none'),
    'cohen_stable': ('known D_i', 'none'), 'fay_herriot': ('known D_i', 'normal model'),
    'simple_shrink': ('known D_min', 'PAC, bi-log-concave W, Gaussian noise'),
    'hetldc': ('known D_i', 'PAC, log-concave W, Gaussian noise'),
}


def variances(n, rng):
    return DBAR * rng.lognormal(0, .7, n) / math.exp(.7 ** 2 / 2)


def conformal_q(scores, lev):
    n = len(scores)
    k = int(math.ceil((n + 1) * lev))
    return np.inf if k > n else np.sort(scores)[k - 1]


def pac_q(scores, lev):
    k = rank_for_level(len(scores), lev, DELTA)
    return np.inf if k is None else np.sort(scores)[k - 1]


# --- Cohen, Goldberger & Tirer (as in E34) -------------------------------------------------

def cohen(V, D, h=0.05, lam=0.01, step=0.01, masked=True):
    sd = np.sqrt(D.max())
    lo, hi = V.min() - 4 * sd, V.max() + 4 * sd
    edges = np.arange(lo, hi + h, h)
    mids = (edges[:-1] + edges[1:]) / 2
    L = len(mids)
    hist = np.histogram(V, edges)[0] / (len(V) * h)
    m = int(np.ceil(4 * sd / h))
    off = np.arange(-m, m + 1) * h
    kern = np.mean(stats.norm.pdf(off[:, None], scale=np.sqrt(D)[None, :]), axis=1) * h
    # banded Toeplitz matrix A[r, c] = kern[r - c + m] for |r - c| <= m, built in one step
    # (identical to summing kern[i] * np.eye(L, k=-o) over the offsets, which took 95% of a job)
    diff = np.subtract.outer(np.arange(L), np.arange(L))
    A = np.where(np.abs(diff) <= m, kern[np.clip(diff + m, 0, 2 * m)], 0.0)
    mask = hist > 1e-12 if masked else np.ones(L, bool)
    Am = np.vstack([A[mask], np.sqrt(lam) * np.eye(L)])
    bm = np.concatenate([hist[mask], np.zeros(L)])
    pW = lsq_linear(Am, bm, bounds=(0, 1 / h)).x
    cov = lambda q: h * pW[np.abs(mids) <= q].sum()
    q = conformal_q(np.abs(V), 1 - ALPHA)
    while q - step > 0 and cov(q - step) >= 1 - ALPHA:
        q -= step
    return q


# --- LatentCP (as in E34) ------------------------------------------------------------------

def p_gamma(w, d, q):
    return ndtr((q - w) / np.sqrt(d)) - ndtr((-q - w) / np.sqrt(d))


def radius_multi(d, qs, gammas, weights):
    """U(d) = {w : sum_k w_k (1 - p_{gamma_k}(w, d))/gamma_k <= 1/alpha} (their eq. 6)."""
    if not all(np.isfinite(qs)):
        return np.inf
    e = lambda w: sum(wk * (1 - p_gamma(w, d, qk)) / gk for wk, qk, gk in zip(weights, qs, gammas))
    if e(0.0) > 1 / ALPHA:
        return 0.0
    return brentq(lambda w: e(w) - 1 / ALPHA, 0, max(qs) + 10 * np.sqrt(d))


def latentcp_tuned(V, D, rng, multi, Dnew, max_grid=12, max_tune_d=100):
    """Their Algorithms 1-2: tune on a random half (criterion: mean width), calibrate on the
    other. For K = 1000 the gamma grid is thinned to `max_grid` values and the width criterion
    uses `max_tune_d` of the tuning variances (computational limits, documented)."""
    K = len(V)
    idx = rng.permutation(K)
    tun, cal = idx[:K // 2], idx[K // 2:]
    St, Sc = np.abs(V[tun]), np.abs(V[cal])
    m = len(tun)
    grid = [j / (m + 1) for j in range(1, m + 1) if j / (m + 1) <= ALPHA * 0.999]
    if len(grid) > max_grid:
        grid = [grid[i] for i in np.unique(np.linspace(0, len(grid) - 1, max_grid).astype(int))]
    dt = D[tun] if len(tun) <= max_tune_d else rng.choice(D[tun], max_tune_d, replace=False)

    def width(gs, ws):
        qs = [conformal_q(St, 1 - g) for g in gs]
        return np.mean([radius_multi(d, qs, gs, ws) for d in dt])
    cands = [((g,), (1.0,)) for g in grid]
    if multi:
        cands += [((g1, g2), (w, 1 - w)) for i, g1 in enumerate(grid) for g2 in grid[i + 1:]
                  for w in np.arange(0.1, 1.0, 0.1)]
    gs, ws = min(cands, key=lambda c: (width(*c), c[0][0]))
    qs = [conformal_q(Sc, 1 - g) for g in gs]
    return np.array([radius_multi(d, qs, gs, ws) for d in Dnew])


def coverage(law, centre, half):
    half = np.atleast_1d(np.asarray(half, float))
    c = np.where(np.isfinite(half), law.cdf(centre + half) - law.cdf(centre - half), 1.0)
    return float(np.mean(c)), float(np.mean(2 * half))


def one(args):
    latent, scen, K, seed, heavy = args
    rng = np.random.default_rng(seed)
    law = LATENT[latent]
    D = variances(K, rng)
    V = law.rvs(K, rng) + draw_noise(SCENARIOS[scen], D, rng)
    S = np.sort(np.abs(V))
    out = {}
    for name, lev in (('usual', Q), ('ours: Gaussian', level('gaussian', Q).sufficient),
                      ('ours: symmetric unimodal', level('symmetric_unimodal', Q).sufficient),
                      ('no shape, median-zero', level('gaussian', Q, 'none').sufficient)):
        k = rank_for_level(K, lev, DELTA)
        out[name] = coverage(law, 0.0, S[k - 1] if k else np.inf)
    Dnew = variances(400, rng)                       # an unsampled unit has no own variance
    g = ALPHA / 2
    q = conformal_q(S, 1 - g)
    out['latentcp'] = coverage(law, 0.0, [radius_multi(d, [q], [g], [1.0]) for d in Dnew])
    out['latentcp_tuned'] = coverage(law, 0.0, latentcp_tuned(V, D, rng, False, Dnew))
    out['latentcp_multi'] = coverage(law, 0.0, latentcp_tuned(V, D, rng, True, Dnew))
    qp = pac_q(S, 1 - g)
    out['latentcp_pac'] = coverage(law, 0.0, [radius_multi(d, [qp], [g], [1.0]) for d in Dnew])
    k_sf = rank_for_level(K, Q + (1 - Q) / 4, DELTA)  # fixed in advance: level 0.925
    try:
        out['shape_free'] = coverage(law, 0.0, shape_free_markov(V, D, k_sf, Q, DELTA)[0])
    except ValueError:
        out['shape_free'] = coverage(law, 0.0, np.inf)
    out['cohen'] = coverage(law, 0.0, cohen(V, D, h=0.01))
    out['cohen_stable'] = coverage(law, 0.0, cohen(V, D, h=0.2, masked=False))
    mu, A, vmu = fay_herriot_reml(V, D)
    _, lam = fay_herriot_boot_pac(V, D, mu, A, rng, level=Q, delta=DELTA)
    out['fay_herriot'] = coverage(law, mu, lam * math.sqrt(A + vmu))
    out['simple_shrink'] = coverage(law, 0.0, simple_shrink_halfwidth(V, D.min(), q=Q, delta=DELTA))
    if heavy:
        out['hetldc'] = coverage(law, 0.0, hetldc_certified(V, D, q=Q, delta=DELTA)[0])
    r_or = law.oracle_radius(Q)
    return [dict(latent=latent, noise=scen, K=K, seed=seed, method=m, cov=c, width=w,
                 rel_width=w / (2 * r_or)) for m, (c, w) in out.items()]


def summarize(d):
    s = d.groupby(['K', 'noise', 'latent', 'method'], sort=False).agg(
        reps=('cov', 'size'), mean_cov=('cov', 'mean'),
        reliability=('cov', lambda c: float(np.mean(c >= Q - 1e-12))),
        rel_width=('rel_width', 'mean')).reset_index()
    s['information'] = s.method.map(lambda m: GUARANTEE[m][0])
    s['guarantee'] = s.method.map(lambda m: GUARANTEE[m][1])
    return s


def _one_keyed(job):
    return job, one(job)


def _py(x):
    return x.item() if hasattr(x, 'item') else str(x)


def main():
    reps110 = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    reps1000 = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    heavy = len(sys.argv) > 4 and sys.argv[4] == 'hetldc'
    jobs = []
    for K, reps in ((110, reps110), (1000, reps1000)):
        for i, lat in enumerate(LATENT):
            for j, scen in enumerate(SCENARIOS):
                for r in range(reps):
                    jobs.append((lat, scen, K, 10 ** 6 * (K == 1000) + 10 ** 4 * i + 10 ** 3 * j + r,
                                 heavy and K == 110 and scen == 'Gaussian'))
    jobs.sort(key=lambda j: (-j[4], -j[2]))            # heavy and large jobs first
    tag = '_hetldc' if heavy else ''
    # one JSON line per finished job, so an interrupted run resumes where it stopped
    ckpt = RESULTS / f'competitors_v2{tag}.ckpt.jsonl'
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:                # partial last line of a killed run
                continue
            done[tuple(rec['job'])] = rec['rows']
    todo = [j for j in jobs if tuple(j) not in done]
    print(f'{len(done)} jobs restored, {len(todo)} to run', flush=True)
    with Pool(procs) as pool, ckpt.open('a') as f:
        for job, part in pool.imap_unordered(_one_keyed, todo, chunksize=1):
            done[tuple(job)] = part
            f.write(json.dumps({'job': list(job), 'rows': part}, default=_py) + '\n')
            f.flush()
    rows = [r for j in jobs for r in done[tuple(j)]]
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / f'competitors_v2{tag}.csv', index=False)
    s = summarize(d)
    s.to_csv(RESULTS / f'competitors_v2{tag}_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(s.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
