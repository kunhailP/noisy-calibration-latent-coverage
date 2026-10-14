"""R08: the noisy level p_kappa(q) for centred latent residuals (notes/centred.md).

  python experiments/r08_centred.py levels [PROCS] [STARTS]   global search, writes
        results/centred_levels.csv (one row per case, appended as each case finishes)
  python experiments/r08_centred.py validate [PROCS] [LAWS]   adversarial search over laws outside
        the reduced family, writes results/centred_validation.csv
  python experiments/r08_centred.py check3 [PROCS]            three-piece re-optimisation from each
        maximiser, writes results/centred_check3.csv
  python experiments/r08_centred.py closure [PROCS]           exponential limits of the family;
        adds the final level p_kappa (max of the two) to centred_levels.csv
  python experiments/r08_centred.py table                     failure K of the usual rank and the
        ranks, from centred_levels.csv; writes results/centred_ranks.csv and fills the K columns

p_kappa(q) = sup pr(|W + e| < 1) over log-concave W with |E W| <= kappa and pr(|W| < 1) <= q
and noise sigma Z (Gaussian) or uniform on [-r, r] (symmetric unimodal), threshold 1. For
kappa >= 1 it equals Psi_E(q) (proved, notes Step 1); for kappa < 1 it is computed over the
two-piece log-affine family of the localization theorem, by multistart Nelder-Mead, and every
maximiser is re-evaluated in mpmath at 30 digits.
"""
import csv
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from latentcov import centred as C
from latentcov.gaussian import psi_gaussian
from latentcov.noise_classes import psi_su_enclosure

RESULTS = Path(__file__).resolve().parents[1] / 'results'
LEVELS = RESULTS / 'centred_levels.csv'
VALID = RESULTS / 'centred_validation.csv'
RANKS = RESULTS / 'centred_ranks.csv'
QS = (0.8, 0.9, 0.95)
KAPPAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, math.inf)
NOISES = ('gaussian', 'su')
KS = (110, 1000, 10_000, 100_000, 1_000_000)
DELTA = 0.05
FIELDS = ['noise', 'q', 'kappa', 'level', 'excess', 'support', 'scale', 'mean', 'latent',
          'brute_excess', 'brute_latent', 'brute_mean', 'xs', 'betas', 'theta', 'starts',
          'top5_spread', 'seconds']


def psi(noise, q):
    return psi_gaussian(q) if noise == 'gaussian' else sum(psi_su_enclosure(q)) / 2


def _done():
    if not LEVELS.exists():
        return {}
    return {(r['noise'], float(r['q']), float(r['kappa'])): r for r in csv.DictReader(open(LEVELS))}


def _append(path, fields, row):
    new = not path.exists()
    if not new:                                  # keep the header already there (table() adds
        with open(path) as f:                    # the K and rank columns to the levels file)
            fields = next(csv.reader(f))
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', restval='')
        if new:
            w.writeheader()
        w.writerow(row)


def _to_theta(row, kappa):
    """Previous maximiser (mean mu_prev <= kappa) as a start for kappa: same shape and scale."""
    th = [float(v) for v in row['theta'].split(';')]
    mu = float(row['mean'])
    mumax = min(kappa, 1.0)
    if mumax > 0:
        x = min(max(abs(mu) / mumax, 1e-9), 1 - 1e-9)
        th[3] = math.log(x / (1 - x))
    return th


def levels(procs=120, starts=192, noises=NOISES, qs=QS):
    RESULTS.mkdir(exist_ok=True)
    done = _done()
    with Pool(procs) as pool:
        for noise in noises:
            for q in qs:
                prev = []
                for kappa in KAPPAS:
                    key = (noise, q, kappa)
                    if key in done:
                        prev.append(done[key])
                        print('skip', key, flush=True)
                        continue
                    t0 = time.time()
                    if kappa >= 1:
                        # proved: p_kappa = Psi_E (notes Step 1(b)); the search below would only
                        # approach it as the law collapses onto the boundary
                        p = psi(noise, q)
                        row = dict(noise=noise, q=q, kappa=kappa, level=repr(p), excess=repr(p - q),
                                   support='proved (Psi_E, not attained)', starts=0,
                                   seconds=0.0)
                        _append(LEVELS, FIELDS, row)
                        print(row, flush=True)
                        continue
                    extra = [_to_theta(r, kappa) for r in prev if r.get('theta')]
                    rng = np.random.default_rng(int(1000 * q) + int(100 * kappa) +
                                                (7 if noise == 'su' else 0))
                    ths = [C.random_start(rng) for _ in range(starts)] + extra
                    jobs = [(t, noise, q, kappa, 3000) for t in ths]
                    res = pool.map(C._nm, jobs, chunksize=1)
                    res.sort(key=lambda t: -t[0])
                    # polish the eight best in parallel with longer runs
                    pol = pool.map(C._nm, [(th, noise, q, kappa, 10000) for _, th in res[:8]],
                                   chunksize=1)
                    pol.sort(key=lambda t: -t[0])
                    v, th = pol[0]
                    W, sc = C.decode(th, q, kappa)
                    bn = float(C.brute_noisy(W, noise, sc))
                    bl, bm = C.brute_latent(W)
                    row = dict(noise=noise, q=q, kappa=kappa, level=repr(q + v), excess=repr(v),
                               support='numerical (attained; mpmath-checked)', scale=repr(sc),
                               mean=repr(W.mean), latent=repr(W.latent()),
                               brute_excess=repr(bn - q), brute_latent=repr(float(bl)),
                               brute_mean=repr(float(bm)),
                               xs=';'.join(repr(x) for x in W.xs),
                               betas=';'.join(repr(b) for b in W.betas),
                               theta=';'.join(repr(float(x)) for x in th), starts=len(jobs),
                               top5_spread=repr(pol[0][0] - pol[-1][0]),
                               seconds=round(time.time() - t0, 1))
                    _append(LEVELS, FIELDS, row)
                    prev.append(row)
                    print({k: row[k] for k in ('noise', 'q', 'kappa', 'excess', 'brute_excess',
                                               'scale', 'xs', 'betas', 'seconds')}, flush=True)


# ---------------------------------------------------------------- adversarial validation
VFIELDS = ['noise', 'q', 'kappa', 'family', 'law', 'excess', 'mu', 'lam', 'scale', 'p_kappa_excess',
           'margin']


def _validate_job(args):
    noise, q, kappa, kind, seed = args
    rng = np.random.default_rng(seed)
    out = []
    if kind == 'standard':
        laws = [(X.name, X) for X in C.standard_laws(rng)]
    elif kind == 'pla':
        laws = [(f'pla{len(X.betas)}', X) for X in (C.random_pla(rng) for _ in range(4))]
    else:                                   # random three-piece laws (p = 3 count, no reduction)
        laws = []
        for _ in range(4):
            c1, c2 = sorted(rng.uniform(0, 1, 2))
            b = np.sort(rng.normal(0, 1, 3) * 10 ** rng.uniform(0, 2.5))[::-1]
            laws.append(('pla3', C.PLA([0, c1, c2, 1], b)))
    for name, X in laws:
        try:
            v, mu, lam, sc = C.best_over_placement(X, noise, q, kappa, maxfev=300)
        except Exception as e:                       # noqa: BLE001 (report and continue)
            v, mu, lam, sc = float('nan'), None, None, None
            name += f' [error {type(e).__name__}]'
        out.append((name, v, mu, lam, sc))
    return noise, q, kappa, kind, out


def validate(procs=120, laws=24, kinds=('standard', 'pla', 'pla3')):
    lv = _done()
    cases = [(n, q, k) for n in NOISES for q in QS for k in KAPPAS if k < 1]
    have = set()                                 # resume: (case, family) pairs already written
    if VALID.exists():
        have = {(r['noise'], float(r['q']), float(r['kappa']), r['family'])
                for r in csv.DictReader(open(VALID))}
    jobs = []
    for i, (n, q, k) in enumerate(cases):
        for j in range(laws // 4):
            kind = 'standard' if j % 3 == 0 else ('pla' if j % 3 == 1 else 'pla3')
            if kind in kinds and (n, q, k, kind) not in have:
                jobs.append((n, q, k, kind, 10_000 * i + j))
    worst = {}
    with Pool(procs) as pool:
        for n, q, k, kind, out in pool.imap_unordered(_validate_job, jobs):
            pk = float(lv[(n, q, k)]['excess']) if (n, q, k) in lv else float('nan')
            for name, v, mu, lam, sc in out:
                row = dict(noise=n, q=q, kappa=k, family=kind, law=name, excess=repr(v), mu=mu,
                           lam=lam, scale=sc, p_kappa_excess=repr(pk), margin=repr(pk - v))
                _append(VALID, VFIELDS, row)
                if v == v:
                    worst[(n, q, k)] = max(worst.get((n, q, k), -1.0), v)
    for key, v in sorted(worst.items()):
        pk = float(lv[key]['excess']) if key in lv else float('nan')
        print(key, f'best outside the family {v:.4e}  p_kappa - q {pk:.4e}  margin {pk - v:.2e}')


# ---------------------------------------------------------------- three-piece local check
def check3(procs=120):
    """From each two-piece maximiser, insert a third piece and re-optimise (24 starts): the
    localization count with p = 3 constraints allows three pieces; Step 2 of the notes reduces
    to two. Writes results/centred_check3.csv."""
    rows = [r for r in csv.DictReader(open(LEVELS)) if r.get('theta')]
    jobs, keys = [], []
    for r in rows:
        th = [float(v) for v in r['theta'].split(';')]
        rng = np.random.default_rng(len(jobs))
        for t in C.three_piece_starts(th, rng):
            jobs.append((t, r['noise'], float(r['q']), float(r['kappa']), 6000))
            keys.append((r['noise'], float(r['q']), float(r['kappa']), float(r['brute_excess'])))
    with Pool(procs) as pool:
        res = pool.map(C._nm3, jobs, chunksize=1)
    best = {}
    for k, (v, th) in zip(keys, res):
        if k not in best or v > best[k]:
            best[k] = v
    out = [dict(noise=k[0], q=k[1], kappa=k[2], two_piece=k[3], three_piece=v, gain=v - k[3])
           for k, v in sorted(best.items())]
    with open(RESULTS / 'centred_check3.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)


# ---------------------------------------------------------------- failure K and ranks
RFIELDS = ['noise', 'q', 'kappa', 'excess', 'K90_normal', 'K90_exact', 'K50_normal', 'K50_exact'] + \
    [f'{r}_{K}' for K in KS for r in ('usual', 'centred', 'thm3')]


def _closure_job(args):
    """Exponential laws b - E/beta and b + E/beta (the infinite-support limits of the two-piece
    family), optimised over mean and noise scale."""
    noise, q, kappa = args
    best = -1.0
    for X in (C.Standard('exponential', lambda x: -np.expm1(-np.maximum(x, 0)), 1.0, 1.0, (0.0,)),
              C.Standard('reflected', lambda x: np.exp(np.minimum(x, 0)), -1.0, 1.0, (0.0,))):
        best = max(best, C.best_over_placement(X, noise, q, kappa, maxfev=3000)[0])
    return noise, q, kappa, best


def closure(procs=48):
    """Adds exp_limit_excess and the final level p_kappa = max(two-piece, exponential limit) to
    centred_levels.csv (columns p_kappa, p_kappa_minus_q); the two-piece value is checked against
    mpmath (brute_excess), which is the value used."""
    raw = list(csv.DictReader(open(LEVELS)))
    jobs = [(r['noise'], float(r['q']), float(r['kappa'])) for r in raw if float(r['kappa']) < 1]
    with Pool(procs) as pool:
        res = {(n, q, k): v for n, q, k, v in pool.map(_closure_job, jobs, chunksize=1)}
    for r in raw:
        key = (r['noise'], float(r['q']), float(r['kappa']))
        q = key[1]
        if key[2] >= 1:
            r['exp_limit_excess'] = ''
            r['p_kappa'], r['p_kappa_minus_q'] = r['level'], r['excess']
            continue
        two = float(r['brute_excess'])
        e = res[key]
        r['exp_limit_excess'] = repr(e)
        v = max(two, e)
        r['p_kappa'], r['p_kappa_minus_q'] = repr(q + v), repr(v)
        if e > two:
            r['support'] = 'numerical (exponential limit beats two-piece by %.1e)' % (e - two)
        print(key, two, e, e - two, flush=True)
    fields = list(raw[0].keys())
    with open(LEVELS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(raw)


def table():
    from latentcov.rank import rank_for_level
    rows = []
    for r in csv.DictReader(open(LEVELS)):
        q, kappa = float(r['q']), float(r['kappa'])
        p = float(r['p_kappa']) if r.get('p_kappa') else float(r['level'])
        out = dict(noise=r['noise'], q=q, kappa=kappa, excess=p - q)
        for tau, tag in ((0.9, '90'), (0.5, '50')):
            kn, ke = C.failure_K(p, q, tau, DELTA)
            out[f'K{tag}_normal'], out[f'K{tag}_exact'] = kn, ke
        for K in KS:
            out[f'usual_{K}'] = rank_for_level(K, q, DELTA)
            out[f'centred_{K}'] = rank_for_level(K, p, DELTA)
            out[f'thm3_{K}'] = rank_for_level(K, psi(r['noise'], q), DELTA)
        rows.append(out)
        print(out, flush=True)
    with open(RANKS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RFIELDS)
        w.writeheader()
        w.writerows(rows)
    # the same columns merged into centred_levels.csv, one row per case
    raw = list(csv.DictReader(open(LEVELS)))
    extra = [c for c in RFIELDS if c not in ('noise', 'q', 'kappa', 'excess')]
    for r, o in zip(raw, rows):
        r.update({c: o[c] for c in extra})
    fields = list(raw[0].keys()) + [c for c in extra if c not in raw[0]]
    with open(LEVELS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', restval='')
        w.writeheader()
        w.writerows(raw)
    if VALID.exists():
        validation_summary()


def validation_summary():
    """Per case: the largest excess found outside the reduced family against p_kappa(q) - q."""
    lv = _done()
    best = {}
    for r in csv.DictReader(open(VALID)):
        key = (r['noise'], float(r['q']), float(r['kappa']))
        v = float(r['excess'])
        if v == v and (key not in best or v > best[key][0]):
            best[key] = (v, r['law'], r['family'])
    n = {}
    for r in csv.DictReader(open(VALID)):
        key = (r['noise'], float(r['q']), float(r['kappa']))
        n[key] = n.get(key, 0) + 1
    out = []
    for key in sorted(best):
        r0 = lv.get(key, {})
        pk = float(r0.get('p_kappa_minus_q') or r0.get('brute_excess') or 'nan')
        v, law, fam = best[key]
        out.append(dict(noise=key[0], q=key[1], kappa=key[2], laws=n[key], best_outside=v,
                        best_law=law, family=fam, p_kappa_excess=pk, margin=pk - v))
        print(out[-1])
    with open(RESULTS / 'centred_validation_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    if cmd in ('levels', 'all'):
        levels(procs, int(sys.argv[3]) if len(sys.argv) > 3 else 192,
               (sys.argv[4],) if len(sys.argv) > 4 else NOISES,
               (float(sys.argv[5]),) if len(sys.argv) > 5 else QS)
    if cmd in ('validate', 'all'):
        validate(procs, int(sys.argv[3]) if len(sys.argv) > 3 and cmd == 'validate' else 48,
                 tuple(sys.argv[4].split(',')) if len(sys.argv) > 4 and cmd == 'validate'
                 else ('standard', 'pla', 'pla3'))
    if cmd in ('check3', 'all'):
        check3(procs)
    if cmd in ('closure', 'all'):
        closure(procs)
    if cmd in ('table', 'all'):
        table()
