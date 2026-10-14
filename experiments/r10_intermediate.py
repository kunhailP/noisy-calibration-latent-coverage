"""R10: exact latent reliability in intermediate designs (notes/intermediate.md).

Neither extremal (built to break the usual rank) nor benign by construction: miscentred skewed
residuals of a misspecified mean model, noise-to-signal variance ratios from 0.001 to 3, equal or
lognormal heterogeneous noise variances, Gaussian or uniform noise, small K, and estimated
proportions whose noise variance theta(1 - theta)/n depends on the latent value.

Units are i.i.d., so N = #{i : |V_i| < r}, r = Q_q(|W|), is Bin(K, H) with H = pr(|V| < r), and
the latent reliability of rank k is pr{Bin(K, H) <= k - 1} exactly (proof of Theorem 3). H is
computed by quadrature (latentcov.intermediate); no Monte Carlo.

Designs
  independent noise (Theorem 3 applies: log-concave latent law, noise independent of W)
    latent W = W0 + b, W0 standardized (mean 0, variance 1): centred gamma(2), gamma(5),
      exponential, skew-normal(4), half-normal, each also reflected (b >= 0 then covers both
      signs of the bias), normal, Laplace; b solves b / Q_q(|W0 + b|) = zeta,
      zeta in {0, 0.1, 0.2, 0.3, 0.5} (spec) and 0.75 (extension);
    noise sigma eps, eps Gaussian or uniform (variance 1), sigma^2 = rho (homogeneous) or
      rho L / E L with log L ~ N(0, 0.7^2) (heterogeneous, integrated exactly over L);
      rho in {0.1, 0.3, 0.58, 1, 3} (spec) and {0.001, 0.003, 0.01, 0.03} (extension: the
      excess H - q, when positive, peaks at noise sd of a few hundredths of the threshold).
  proportions (Theorem 3's independence assumption fails)
    theta ~ Beta(a, b), W = theta - c, V = W + sqrt(theta (1 - theta)/n) Z; c the population
      mean or the mean -+ half a standard deviation; n in {20, 50, 200} (spec) and 1000
      (extension). Beta(a, b) is log-concave iff a, b >= 1; Beta(0.5, 4.5) is included as a
      latent law outside the class (not even bi-log-concave: its hazard first decreases).
    H_indep_minus_q: the same latent law with independent homogeneous Gaussian noise of the same
      average variance E{theta (1 - theta)}/n, for comparison.

Rules: the usual rank (level q) and the rank of Theorem 3 (level Psi_N(q) for Gaussian noise,
the symmetric unimodal level for uniform noise; the sufficient end of the certified enclosure).

  python experiments/r10_intermediate.py compute [PROCS]   H for every design, appended to
        results/intermediate_H.csv as jobs finish (rerun skips the rows already there)
  python experiments/r10_intermediate.py envelope [PROCS]  sup of H - q over a fine grid of noise
        ratios (rho in [1e-5, 3], n in [10, 1e6]): results/intermediate_envelope.csv
  python experiments/r10_intermediate.py verify [PROCS]    independent checks of H (mpmath at 30
        digits, dense Simpson in log L, Monte Carlo): results/intermediate_check.csv
  python experiments/r10_intermediate.py table [PROCS]     reliabilities: results/intermediate.csv
        (design x q x rule x K, with the minimum over K in [29, 1e6] and the first K below 0.95)
        and results/intermediate_summary.csv
  python experiments/r10_intermediate.py all [PROCS]
"""
import csv
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from latentcov import intermediate as I
from latentcov.levels import level

RESULTS = Path(__file__).resolve().parents[1] / 'results'
HFILE = RESULTS / 'intermediate_H.csv'
ENVELOPE = RESULTS / 'intermediate_envelope.csv'
CHECK = RESULTS / 'intermediate_check.csv'
OUT = RESULTS / 'intermediate.csv'
SUMMARY = RESULTS / 'intermediate_summary.csv'

DELTA = 0.05
QS = (0.8, 0.9, 0.95)
KS = (30, 60, 110, 1000, 10_000, 100_000)
ZETAS_SPEC, ZETAS_EXT = (0.0, 0.1, 0.2, 0.3, 0.5), (0.75,)
RHOS_SPEC, RHOS_EXT = (0.1, 0.3, 0.58, 1.0, 3.0), (0.001, 0.003, 0.01, 0.03)
NS_SPEC, NS_EXT = (20, 50, 200), (1000,)
BETAS = ((2, 18), (1, 9), (3, 12), (1.5, 6), (2, 3), (5, 5), (0.5, 4.5))
CENTRES = (('mean', 0.0), ('mean - sd/2', -0.5), ('mean + sd/2', 0.5))
NOISE_CLASS = {'gaussian': 'gaussian', 'uniform': 'symmetric_unimodal'}

HFIELDS = ['id', 'design', 'group', 'grid', 'law', 'log_concave', 'zeta', 'bias', 'rho', 'scale',
           'noise', 'beta_a', 'beta_b', 'centre_type', 'centre', 'n', 'q', 'r', 'H', 'H_minus_q',
           'psi', 'H_minus_psi', 'quad_err', 'check_diff', 'H_indep_minus_q', 'seconds']


def group_of(name):
    if name in ('normal', 'Laplace'):
        return 'symmetric'
    if 'exponential' in name or 'half-normal' in name:
        return 'J-shaped'           # mode at a support end (a hard edge)
    return 'skewed smooth'


def psi(noise, q):
    return level(NOISE_CLASS[noise], q).sufficient


# ------------------------------------------------------------------ designs
def designs():
    laws = I.latent_laws()
    for q in QS:
        for name in laws:
            for zeta in ZETAS_SPEC + ZETAS_EXT:
                for rho in RHOS_SPEC + RHOS_EXT:
                    for scale in ('homogeneous', 'heterogeneous'):
                        for noise in ('gaussian', 'uniform'):
                            ext = zeta in ZETAS_EXT or rho in RHOS_EXT
                            yield dict(design='independent noise', law=name, group=group_of(name),
                                       grid='extension' if ext else 'spec', zeta=zeta, rho=rho,
                                       scale=scale, noise=noise, q=q,
                                       id=f'{name}|zeta={zeta:g}|rho={rho:g}|{scale}|{noise}')
        for a, b in BETAS:
            for ctype, dc in CENTRES:
                for n in NS_SPEC + NS_EXT:
                    lc = a >= 1 and b >= 1
                    yield dict(design='proportions', law=f'Beta({a:g}, {b:g})',
                               group='proportions' + ('' if lc else ' (not log-concave)'),
                               grid='extension' if n in NS_EXT else 'spec', beta_a=a, beta_b=b,
                               centre_type=ctype, dc=dc, n=n, noise='gaussian',
                               scale='theta(1 - theta)/n', q=q,
                               id=f'Beta({a:g}, {b:g})|c={ctype}|n={n}')


def compute_one(d):
    t0 = time.time()
    q = d['q']
    out = {k: d.get(k, '') for k in HFIELDS}
    if d['design'] == 'independent noise':
        law = I.latent_laws()[d['law']]
        b = I.bias_for_zeta(law, d['zeta'], q)
        r = I.radius(law, b, q)
        if d['scale'] == 'homogeneous':
            D, err = I.excess(law, b, r, d['noise'], math.sqrt(d['rho']))
            check = ''
        else:
            D, err = I.excess_het(law, b, r, d['noise'], d['rho'])
            if d['noise'] == 'gaussian':      # Gauss-Hermite (96 nodes) against adaptive quad
                check = I.excess_het_adaptive(law, b, r, d['noise'], d['rho'])[0] - D
            else:                             # adaptive with breakpoints against dense Simpson
                check = _simpson_het(law, b, r, d['noise'], d['rho']) - D
        out.update(log_concave=law.log_concave, bias=b, r=r)
    else:
        a, bb, n = d['beta_a'], d['beta_b'], d['n']
        law0 = I.beta_law(a, bb)
        c = law0.mean + d['dc'] * law0.sd
        law, shift, sig = I.proportion_design(a, bb, c, n)
        r = I.radius(law, shift, q)
        D, err = I.excess(law, shift, r, 'gaussian', sig)
        # comparison: independent homogeneous noise with the same average variance
        ev = (law.mean * (1 - law.mean) - law.sd ** 2) / n
        Dind = I.excess(law, shift, r, 'gaussian', math.sqrt(ev))[0]
        check = ''
        out.update(log_concave=law.log_concave, bias=law.mean - c, centre=c, r=r,
                   zeta=abs(law.mean - c) / r, rho=ev / law.sd ** 2, H_indep_minus_q=Dind)
    p = psi(d['noise'], q)
    out.update(H=q + D, H_minus_q=D, psi=p, H_minus_psi=q + D - p, quad_err=err,
               check_diff=check, seconds=round(time.time() - t0, 3))
    return out


def _simpson_het(law, b, r, noise, rho, npts=4001):
    """D for heterogeneous scales by composite Simpson in x = log(L)/s on [-9, 9]."""
    from scipy.integrate import simpson
    s = I.LOGNORMAL_S
    xs = np.linspace(-9, 9, npts)
    g = [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) *
         I.excess(law, b, r, noise, math.sqrt(rho * math.exp(s * x - s * s / 2)))[0] for x in xs]
    return float(simpson(g, x=xs))


def _append(path, fields, rows):
    new = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', restval='')
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def compute(procs):
    RESULTS.mkdir(exist_ok=True)
    done = set()
    if HFILE.exists():
        done = {(r['id'], float(r['q'])) for r in csv.DictReader(open(HFILE))}
    jobs = [d for d in designs() if (d['id'], d['q']) not in done]
    # heterogeneous jobs are ~100x slower: start them first
    jobs.sort(key=lambda d: d.get('scale') != 'heterogeneous')
    print(f'{len(done)} done, {len(jobs)} to compute', flush=True)
    t0 = time.time()
    buf = []
    with Pool(procs) as pool:
        for i, row in enumerate(pool.imap_unordered(compute_one, jobs, chunksize=4), 1):
            buf.append(row)
            if len(buf) >= 50 or i == len(jobs):
                _append(HFILE, HFIELDS, buf)
                buf = []
                print(f'{i}/{len(jobs)}  {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------ envelope over the noise ratio
EFIELDS = ['design', 'law', 'group', 'zeta', 'scale', 'noise', 'centre_type', 'q', 'sup_H_minus_q',
           'argmax', 'H_minus_q_at_spec_max', 'psi_minus_q', 'grid_points']
RHO_FINE = np.geomspace(1e-5, 3, 61)
N_FINE = np.unique(np.round(np.geomspace(10, 1e6, 41)).astype(int))


def envelope_one(d):
    q = d['q']
    if d['design'] == 'independent noise':
        law = I.latent_laws()[d['law']]
        b = I.bias_for_zeta(law, d['zeta'], q)
        r = I.radius(law, b, q)
        f = (lambda rho: I.excess(law, b, r, d['noise'], math.sqrt(rho))[0]) \
            if d['scale'] == 'homogeneous' else \
            (lambda rho: I.excess_het(law, b, r, d['noise'], rho)[0])
        grid = RHO_FINE
        vals = np.array([f(x) for x in grid])
        spec = max(vals[grid >= 0.1 - 1e-12])
    else:
        a, bb = d['beta_a'], d['beta_b']
        law0 = I.beta_law(a, bb)
        c = law0.mean + d['dc'] * law0.sd
        grid = N_FINE
        vals = []
        for n in grid:
            law, shift, sig = I.proportion_design(a, bb, c, int(n))
            r = I.radius(law, shift, q)
            vals.append(I.excess(law, shift, r, 'gaussian', sig)[0])
        vals = np.array(vals)
        spec = max(vals[(grid >= 20) & (grid <= 200)])
    i = int(np.argmax(vals))
    return dict(design=d['design'], law=d['law'], group=d['group'], zeta=d.get('zeta', ''),
                scale=d['scale'], noise=d['noise'], centre_type=d.get('centre_type', ''), q=q,
                sup_H_minus_q=vals[i], argmax=float(grid[i]), H_minus_q_at_spec_max=spec,
                psi_minus_q=psi(d['noise'], q) - q,
                grid_points=('rho in [1e-5, 3], 61 log-spaced' if d['design'] ==
                             'independent noise' else 'n in [10, 1e6], 41 log-spaced'))


def envelope(procs):
    seen, jobs = set(), []
    for d in designs():
        key = (d['law'], d.get('zeta'), d['scale'], d['noise'], d.get('centre_type'), d['q'])
        if key not in seen:
            seen.add(key)
            jobs.append(d)
    jobs.sort(key=lambda d: d.get('scale') != 'heterogeneous')
    with Pool(procs) as pool:
        rows = pool.map(envelope_one, jobs, chunksize=1)
    pd.DataFrame(rows, columns=EFIELDS).to_csv(ENVELOPE, index=False)
    print(f'wrote {ENVELOPE} ({len(rows)} rows)')


# ------------------------------------------------------------------ independent checks
def _mp_pdf(name, w):
    import mpmath as mp
    refl = name.startswith('reflected ')
    base = name[len('reflected '):] if refl else name
    w = -w if refl else w
    if base.startswith('centred gamma') or base == 'centred exponential':
        k = mp.mpf(1) if base == 'centred exponential' else mp.mpf(base[len('centred gamma('):-1])
        s = mp.sqrt(k)
        x = k + s * w
        return s * x ** (k - 1) * mp.exp(-x) / mp.gamma(k) if x > 0 else mp.mpf(0)
    if base == 'skew-normal(4)':
        al = mp.mpf(4)
        dd = al / mp.sqrt(1 + al ** 2)
        m = dd * mp.sqrt(2 / mp.pi)
        sd = mp.sqrt(1 - m ** 2)
        x = m + sd * w
        return 2 * sd * mp.npdf(x) * mp.ncdf(al * x)
    if base == 'centred half-normal':
        m = mp.sqrt(2 / mp.pi)
        sd = mp.sqrt(1 - 2 / mp.pi)
        x = m + sd * w
        return 2 * sd * mp.npdf(x) if x >= 0 else mp.mpf(0)
    if base == 'normal':
        return mp.npdf(w)
    if base == 'Laplace':
        return mp.exp(-mp.sqrt(2) * abs(w)) / mp.sqrt(2)
    raise ValueError(name)


def _mp_kernel(noise, w, r, s):
    import mpmath as mp
    if s == 0:
        return mp.mpf(1) if abs(w) < r else mp.mpf(0)
    if noise == 'gaussian':
        return mp.ncdf((r - w) / s) - mp.ncdf((-r - w) / s)
    a = mp.sqrt(3) * s
    return max(mp.mpf(0), min(w + a, r) - max(w - a, -r)) / (2 * a)


def mp_excess(pdf, pts, r, noise, sigma, dps=30):
    """H - q in mpmath: noisy coverage minus latent coverage, each integrated directly (no
    deficit form), pdf and kernel from independent mpmath code; sigma constant or callable."""
    import mpmath as mp
    with mp.workdps(dps):
        r = mp.mpf(r)
        pts = [mp.mpf(p) for p in pts]
        sg = sigma if callable(sigma) else (lambda w: mp.mpf(sigma))
        H = mp.quad(lambda w: pdf(w) * _mp_kernel(noise, w, r, sg(w)), pts)
        inside = [p for p in pts if -r < p < r]
        lat = mp.quad(pdf, [-r] + inside + [r])
        return H - lat, lat


def _pts(law, b, r, s, noise):
    lo, hi = law.lo + b, law.hi + b
    pts = {lo, hi, -r, r} | {k + b for k in law.kinks}
    if noise == 'gaussian':
        pts |= {c + m * s * g for c in (-r, r) for m in (0.5, 1, 2, 4, 8, 16) for g in (-1, 1)}
    else:
        a = math.sqrt(3) * s
        pts |= {c + g * a * m for c in (-r, r) for g in (-1, 1) for m in (0.25, 0.5, 1)}
    return sorted(p for p in pts if lo <= p <= hi)


CHECKS = [  # (law, zeta, rho, noise, q)
    ('reflected centred exponential', 0.0, 0.001, 'gaussian', 0.8),
    ('reflected centred exponential', 0.5, 0.01, 'gaussian', 0.8),
    ('reflected centred exponential', 0.3, 0.003, 'uniform', 0.9),
    ('centred gamma(2)', 0.5, 0.1, 'gaussian', 0.9),
    ('reflected centred gamma(5)', 0.2, 0.01, 'uniform', 0.95),
    ('skew-normal(4)', 0.3, 0.58, 'gaussian', 0.9),
    ('reflected skew-normal(4)', 0.1, 0.001, 'uniform', 0.8),
    ('reflected centred half-normal', 0.5, 0.03, 'gaussian', 0.9),
    ('centred half-normal', 0.75, 0.3, 'uniform', 0.95),
    ('Laplace', 0.2, 1.0, 'gaussian', 0.9),
    ('normal', 0.0, 3.0, 'uniform', 0.8),
]
CHECKS_HET = [('reflected centred exponential', 0.5, 0.01, 'uniform', 0.8),
              ('centred gamma(2)', 0.2, 1.0, 'uniform', 0.9),
              ('reflected centred half-normal', 0.3, 0.003, 'gaussian', 0.95)]
CHECKS_THETA = [((2, 18), 0.0, 20, 0.9), ((1, 9), -0.5, 200, 0.8), ((5, 5), 0.5, 1000, 0.95),
                ((0.5, 4.5), 0.0, 50, 0.9)]


def _check_hom(args):
    import mpmath as mp
    name, zeta, rho, noise, q = args
    law = I.latent_laws()[name]
    b = I.bias_for_zeta(law, zeta, q)
    r = I.radius(law, b, q)
    s = math.sqrt(rho)
    D = I.excess(law, b, r, noise, s)[0]
    Dm, lat = mp_excess(lambda w: _mp_pdf(name, w - b), _pts(law, b, r, s, noise), r, noise, s)
    return dict(check='mpmath 30 digits (homogeneous)', law=name, zeta=zeta, rho=rho,
                noise=noise, q=q, H_minus_q=D, reference=float(Dm), diff=D - float(Dm),
                latent_coverage_error=float(lat - mp.mpf(q)))


def _check_het(args):
    from scipy.integrate import simpson
    name, zeta, rho, noise, q = args
    law = I.latent_laws()[name]
    b = I.bias_for_zeta(law, zeta, q)
    r = I.radius(law, b, q)
    D = I.excess_het(law, b, r, noise, rho)[0]
    s = I.LOGNORMAL_S
    xs = np.linspace(-9, 9, 20001)
    g = [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) *
         I.excess(law, b, r, noise, math.sqrt(rho * math.exp(s * x - s * s / 2)))[0] for x in xs]
    ref = float(simpson(g, x=xs))
    return dict(check='dense Simpson in log L, 20001 points (heterogeneous)', law=name,
                zeta=zeta, rho=rho, noise=noise, q=q, H_minus_q=D, reference=ref, diff=D - ref)


def _check_theta(args):
    import mpmath as mp
    (a, bb), dc, n, q = args
    law0 = I.beta_law(a, bb)
    c = law0.mean + dc * law0.sd
    law, shift, sig = I.proportion_design(a, bb, c, n)
    r = I.radius(law, shift, q)
    D = I.excess(law, shift, r, 'gaussian', sig)[0]
    with mp.workdps(30):
        A, B, C, N = mp.mpf(a), mp.mpf(bb), mp.mpf(c), mp.mpf(n)
        lb = mp.beta(A, B)

        def pdf(w):
            t = w + C
            return t ** (A - 1) * (1 - t) ** (B - 1) / lb if 0 < t < 1 else mp.mpf(0)

        def sg(w):
            t = w + C
            return mp.sqrt(t * (1 - t) / N) if 0 < t < 1 else mp.mpf(0)
        pts = {-c, 1 - c, -r, r}
        for e in (-r, r):
            s0 = sig(e)
            pts |= {e + m * s0 * g for m in (0.5, 1, 2, 4, 8, 16) for g in (-1, 1)}
        pts = sorted(p for p in pts if -c <= p <= 1 - c)
        Dm, lat = mp_excess(pdf, pts, r, 'gaussian', sg)
    # Monte Carlo, 4e7 draws
    rng = np.random.default_rng(20261014 + n)
    hits, tot = 0, 0
    for _ in range(40):
        t = rng.beta(a, bb, 1_000_000)
        v = t - c + np.sqrt(t * (1 - t) / n) * rng.standard_normal(t.size)
        hits += int(np.count_nonzero(np.abs(v) < r))
        tot += t.size
    Hmc = hits / tot
    return dict(check='mpmath 30 digits and Monte Carlo 4e7 (proportions)',
                law=f'Beta({a:g}, {bb:g})', zeta=dc, rho=n, noise='gaussian', q=q, H_minus_q=D,
                reference=float(Dm), diff=D - float(Dm), latent_coverage_error=float(lat - q),
                mc_H_minus_q=Hmc - q, mc_se=math.sqrt(Hmc * (1 - Hmc) / tot))


def verify(procs):
    with Pool(procs) as pool:
        res = [pool.apply_async(_check_hom, (c,)) for c in CHECKS]
        res += [pool.apply_async(_check_het, (c,)) for c in CHECKS_HET]
        res += [pool.apply_async(_check_theta, (c,)) for c in CHECKS_THETA]
        rows = [r.get() for r in res]
    d = pd.DataFrame(rows)
    d.to_csv(CHECK, index=False)
    pd.set_option('display.width', 250)
    print(d.to_string(index=False))


# ------------------------------------------------------------------ reliabilities
GRID = I.k_grid(extra=KS + (1_000_000,))
_RANKS = {}


def _ranks(p):
    key = round(p, 15)
    if key not in _RANKS:
        _RANKS[key] = I.ranks_on_grid(GRID, p, DELTA)
    return _RANKS[key]


def _init_ranks(levels):
    for p in levels:
        _ranks(p)


def table_one(row):
    q, H = float(row['q']), float(row['H'])
    rules = (('usual', q), ('Theorem 3', float(row['psi'])))
    stats_ = {}
    long = []
    for rule, p in rules:
        ks = _ranks(p)
        rel = I.reliability(GRID, ks, H)
        ok = ~np.isnan(rel)
        m6 = ok
        m5 = ok & (GRID <= 100_000)
        m4 = ok & (GRID <= 10_000)
        i4 = np.flatnonzero(m4)[np.nanargmin(rel[m4])] if m4.any() else None
        i6 = np.flatnonzero(m6)[np.nanargmin(rel[m6])] if m6.any() else None
        i5 = np.flatnonzero(m5)[np.nanargmin(rel[m5])] if m5.any() else None
        below95 = np.flatnonzero(ok & (rel < 1 - DELTA - 1e-12))
        below90 = np.flatnonzero(ok & (rel < 0.9))
        st = dict(min_rel_K_le_1e6=rel[i6] if i6 is not None else np.nan,
                  argmin_K_le_1e6=int(GRID[i6]) if i6 is not None else '',
                  min_rel_K_le_1e5=rel[i5] if i5 is not None else np.nan,
                  argmin_K_le_1e5=int(GRID[i5]) if i5 is not None else '',
                  min_rel_K_le_1e4=rel[i4] if i4 is not None else np.nan,
                  first_K_below_095=int(GRID[below95[0]]) if below95.size else 'never',
                  rel_at_first_K_below_095=rel[below95[0]] if below95.size else np.nan,
                  first_K_below_090=int(GRID[below90[0]]) if below90.size else 'never',
                  rel_at_1e6=rel[-1])
        stats_[rule] = st
        for K in KS:
            j = int(np.searchsorted(GRID, K))
            if GRID[j] != K:
                raise AssertionError(K)
            k = int(ks[j])
            long.append(dict(rule=rule, level=p, K=K, k=k if k > 0 else '',
                             reliability=rel[j], **st))
    return long, stats_


def table(procs):
    h = pd.read_csv(HFILE)
    h = h.drop_duplicates(['id', 'q'], keep='last')
    expected = {(d['id'], d['q']) for d in designs()}
    have = set(zip(h['id'], h['q']))
    missing = expected - have
    if missing:
        print(f'warning: {len(missing)} designs missing from {HFILE.name}; run compute first')
    h = h[[k in expected for k in zip(h['id'], h['q'])]].copy()
    ind = h['design'] == 'independent noise'
    h.loc[ind, 'group'] = h.loc[ind, 'law'].map(group_of)
    levels = sorted(set(h['q']) | set(h['psi']))
    t0 = time.time()
    rows = h.to_dict('records')
    with Pool(procs, initializer=_init_ranks, initargs=(levels,)) as pool:
        res = pool.map(table_one, rows, chunksize=16)
    print(f'reliabilities in {time.time() - t0:.0f}s', flush=True)
    # the long file keeps the design keys and the main statistics; everything else per design
    # (r, H, bias, centre, levels, the minimum over K <= 1e4, ...) is in intermediate_designs.csv
    keep = ['design', 'group', 'grid', 'law', 'zeta', 'rho', 'scale', 'noise', 'centre_type', 'n',
            'q', 'H_minus_q', 'H_minus_psi']
    drop = ('level', 'min_rel_K_le_1e4', 'argmin_K_le_1e5', 'rel_at_1e6')
    out = []
    for row, (long, _) in zip(rows, res):
        base = {k: row[k] for k in keep}
        out += [{**base, **{k: v for k, v in x.items() if k not in drop}} for x in long]
    d = pd.DataFrame(out)
    d.to_csv(OUT, index=False, float_format='%.10g')
    print(f'wrote {OUT} ({len(d)} rows)')
    summary(h, res)


def _fk(v):
    return math.inf if v == 'never' else int(v)


def summary(h, res):
    """Per family (latent group x noise x scale, or proportions x centre) x zeta x q."""
    rows = []
    recs = []
    for row, (_, st) in zip(h.to_dict('records'), res):
        recs.append(dict(row, **{f'{r}_{k}': v for r in st for k, v in st[r].items()}))
    # the per-design statistics, one row per design x q (the long file repeats them per K)
    pd.DataFrame(recs).drop(columns=['seconds', 'quad_err', 'check_diff']).to_csv(
        RESULTS / 'intermediate_designs.csv', index=False, float_format='%.12g')
    d = pd.DataFrame(recs)
    d['zeta_key'] = np.where(d['design'] == 'proportions', d['centre_type'],
                             d['zeta'].map(lambda z: f'{z:g}'))
    keys = ['design', 'group', 'noise', 'scale', 'zeta_key', 'q']
    for key, g in d.groupby(keys, sort=False):
        i = g['H_minus_q'].idxmax()
        j = g['usual_min_rel_K_le_1e5'].idxmin()
        fk = g['usual_first_K_below_095'].map(_fk)
        rows.append(dict(zip(keys, key), designs=len(g),
                         designs_spec_grid=int((g['grid'] == 'spec').sum()),
                         designs_H_above_q=int((g['H_minus_q'] > 0).sum()),
                         max_H_minus_q=g['H_minus_q'].max(), argmax_design=g.loc[i, 'id'],
                         max_H_minus_q_spec_grid=g.loc[g['grid'] == 'spec', 'H_minus_q'].max(),
                         max_H_minus_psi=g['H_minus_psi'].max(),
                         min_rel_usual_K_le_1e5=g.loc[j, 'usual_min_rel_K_le_1e5'],
                         at_K=g.loc[j, 'usual_argmin_K_le_1e5'], at_design=g.loc[j, 'id'],
                         min_rel_usual_K_le_1e4=g['usual_min_rel_K_le_1e4'].min(),
                         min_rel_usual_K_le_1e5_spec_grid=g.loc[g['grid'] == 'spec',
                                                                'usual_min_rel_K_le_1e5'].min(),
                         min_rel_thm3_K_le_1e5=g['Theorem 3_min_rel_K_le_1e5'].min(),
                         min_rel_usual_K_le_1e6=g['usual_min_rel_K_le_1e6'].min(),
                         first_K_below_095_usual=('never' if not np.isfinite(fk.min())
                                                  else int(fk.min())),
                         designs_usual_below_095_K_le_1e5=int((fk <= 100_000).sum())))
    s = pd.DataFrame(rows)
    s.to_csv(SUMMARY, index=False, float_format='%.6g')
    print(f'wrote {SUMMARY} ({len(s)} rows)')
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 400)
    print(s.drop(columns=['argmax_design', 'at_design']).to_string(index=False))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    if cmd in ('compute', 'all'):
        compute(procs)
    if cmd in ('envelope', 'all'):
        envelope(procs)
    if cmd in ('verify', 'all'):
        verify(procs)
    if cmd in ('table', 'all'):
        table(procs)
