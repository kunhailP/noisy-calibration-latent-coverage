"""Centred latent residuals: the noisy level p_kappa(q) under a mean constraint (notes/centred.md).

Threshold t = 1 by scaling. For a noise class E (Gaussian 'gaussian' or symmetric unimodal 'su')
and a centring dial kappa >= 0,

    p_kappa(q) = sup{ pr(|W + e| < 1) : W log-concave, |E W| <= kappa, pr(|W| < 1) <= q,
                      e = sigma eps, eps in E, sigma > 0 }.

By the localization theorem of Fradelizi and Guedon (2006), with the coverage constraint and the two
bounds on the mean as three linear constraints, the supremum over W may be taken over point masses and
laws with density exp(-V) on a segment, V convex and piecewise affine with at most THREE pieces.
The search below parametrizes two pieces; the three-piece check (`three_piece_starts`) found gains <= 6e-11. The
two-piece reduction of notes/centred.md (a degrees-of-freedom count) is not proved and not used in
the paper; over symmetric unimodal noise it is over uniform laws on [-r, r] (Khintchine), over
Gaussian noise over sigma. kappa = inf gives Psi_E(q) (Theorem 2 and Supplementary S2).

Laws are represented by `PLA` (piecewise log-affine density, any number of pieces): distribution
function, mean and E(x - W)^+ in closed form, evaluated in log space (expm1-type kernels), so
every quantity is accurate to a few units of 1e-16 in absolute terms; the excesses of interest are
1e-7 to 1e-3. Noisy coverage: uniform noise in closed form through E(x - W)^+, Gaussian noise by
composite Gauss-Legendre quadrature of phi(z) {F(1 + sigma z) - F(-1 + sigma z)}, split at every
kink. `brute_noisy` recomputes either with mpmath (independent, 30 digits) for verification.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

NOISES = ('gaussian', 'su')
_GLX, _GLW = np.polynomial.legendre.leggauss(20)
_ZMAX = 9.0                 # Phi(-9) = 1.1e-19: Gaussian tails beyond are below double resolution
_LN_PHI0 = -0.5 * math.log(2 * math.pi)


# ---------------------------------------------------------------- stable exponential kernels
# Every piece is evaluated from its end with the larger density, so only exponentials of
# nonpositive arguments occur and no log-density of large magnitude is ever subtracted.
def _e1n(z):
    """E1(z) = (e^z - 1)/z = int_0^1 e^{zs} ds for z <= 0 (scalar)."""
    if z > -1e-5:
        return 1 + z / 2 + z * z / 6
    return -math.expm1(z) / -z


def _e2n(z):
    """E2(z) = int_0^1 (1 - s) e^{zs} ds = (e^z - 1 - z)/z^2 for z <= 0 (scalar)."""
    if z > -0.1:
        t, s, n = 1.0, 0.0, 0
        while True:                        # sum z^n / (n + 2)!
            term = t / ((n + 1) * (n + 2))
            s += term
            if abs(term) < 1e-18:
                return s
            n += 1
            t *= z / n
    return (math.expm1(z) - z) / (z * z)


def _e3n(z):
    """E3(z) = int_0^1 s e^{zs} ds = (1 - e^z + z e^z)/z^2 for z <= 0 (scalar)."""
    if z > -0.1:
        t, s, n = 1.0, 0.0, 0
        while True:                        # sum z^n / (n! (n + 2))
            term = t / (n + 2)
            s += term
            if abs(term) < 1e-18:
                return s
            n += 1
            t *= z / n
    return (-math.expm1(z) + z * math.exp(z)) / (z * z)


def _mean01(k):
    """Mean of the law on [0, 1] with density proportional to e^{ks}, k <= 0: E3(k)/E1(k)."""
    return _e3n(k) / _e1n(k)


def _e1n_vec(z):
    small = z > -1e-5
    zz = np.where(small, -1.0, z)
    return np.where(small, 1 + z / 2 + z * z / 6, -np.expm1(zz) / -zz)


# ---------------------------------------------------------------- piecewise log-affine laws
class PLA:
    """Law with density proportional to exp(-V) on [xs[0], xs[-1]], V convex piecewise affine:
    log-density slope betas[j] on [xs[j], xs[j+1]], continuous at the breakpoints. Log-concave iff
    betas is nonincreasing (checked unless check=False). Log-densities at the breakpoints are
    accumulated outwards from the highest one, so they are accurate where the mass is."""

    def __init__(self, xs, betas, check=True):
        xs = [float(x) for x in xs]
        betas = [float(b) for b in betas]
        k = len(betas)
        L = [xs[j + 1] - xs[j] for j in range(len(xs) - 1)]
        if len(L) != k or min(L) <= 0:
            raise ValueError('need increasing breakpoints and one slope per piece')
        if check and any(betas[j + 1] > betas[j] + 1e-12 * (1 + abs(betas[j + 1]))
                         for j in range(k - 1)):
            raise ValueError('slopes must be nonincreasing for a log-concave density')
        raw = [0.0]
        for j in range(k):
            raw.append(raw[-1] + betas[j] * L[j])
        m = max(range(k + 1), key=lambda i: raw[i])
        v = [0.0] * (k + 1)
        for i in range(m + 1, k + 1):
            v[i] = v[i - 1] + betas[i - 1] * L[i - 1]
        for i in range(m - 1, -1, -1):
            v[i] = v[i + 1] - betas[i] * L[i]
        # high end of piece j: log density hi[j], at the right end if beta > 0
        hi = [max(v[j], v[j + 1]) for j in range(k)]
        logm = [hi[j] + math.log(L[j]) + math.log(_e1n(-abs(betas[j]) * L[j])) for j in range(k)]
        M = max(logm)
        logZ = M + math.log(math.fsum(math.exp(x - M) for x in logm))
        self.k = k
        self.xs, self.betas, self.L = xs, betas, L
        self.v = [x - logZ for x in v]              # log density at the breakpoints
        self.mass = [math.exp(x - logZ) for x in logm]
        self.cum = [0.0]
        for x in self.mass:
            self.cum.append(self.cum[-1] + x)
        self.pmean = [xs[j] + L[j] * _mean01(betas[j] * L[j]) if betas[j] <= 0 else
                      xs[j + 1] - L[j] * _mean01(-betas[j] * L[j]) for j in range(k)]
        self.mean = math.fsum(x * y for x, y in zip(self.mass, self.pmean))
        self.first = [0.0]                          # partial sums of mass * piece mean
        for x, y in zip(self.mass, self.pmean):
            self.first.append(self.first[-1] + x * y)
        self._np = None

    def _arrays(self):
        if self._np is None:
            self._np = tuple(np.asarray(a) for a in (self.xs, self.betas, self.L, self.v,
                                                     self.cum))
        return self._np

    def _find(self, x):
        j = 0
        while j < self.k - 1 and x >= self.xs[j + 1]:
            j += 1
        return j

    def _logpdf1(self, j, x):
        b = self.betas[j]
        if b <= 0:
            return self.v[j] + b * (x - self.xs[j])
        return self.v[j + 1] - b * (self.xs[j + 1] - x)

    # -- elementary functionals
    def pdf(self, x):
        xs, betas, L, v, cum = self._arrays()
        x = np.asarray(x, dtype=float)
        j = np.clip(np.searchsorted(xs, x, side='right') - 1, 0, self.k - 1)
        b = betas[j]
        lp = np.where(b <= 0, v[j] + b * (x - xs[j]), v[j + 1] - b * (xs[j + 1] - x))
        inside = (x >= xs[0]) & (x <= xs[-1])
        return np.where(inside, np.exp(lp), 0.0)

    def _partial(self, j, x):
        """mass of [xs[j], x] for x in piece j (scalar)."""
        d = x - self.xs[j]
        if d <= 0:
            return 0.0
        b = self.betas[j]
        if b <= 0:
            return math.exp(self.v[j]) * d * _e1n(b * d)
        return math.exp(self._logpdf1(j, x)) * d * _e1n(-b * d)

    def cdf1(self, x):
        """Distribution function at a float."""
        if x <= self.xs[0]:
            return 0.0
        if x >= self.xs[-1]:
            return 1.0
        j = self._find(x)
        return self.cum[j] + self._partial(j, x)

    def cdf(self, x):
        xs, betas, L, v, cum = self._arrays()
        x = np.asarray(x, dtype=float)
        j = np.clip(np.searchsorted(xs, x, side='right') - 1, 0, self.k - 1)
        d = np.clip(x - xs[j], 0.0, L[j])
        b = betas[j]
        neg = b <= 0
        dr = np.clip(xs[j + 1] - x, 0.0, L[j])          # distance to the right end, unrounded
        lp = np.where(neg, v[j], v[j + 1] - b * dr)       # log density at the larger end of [x_j, x]
        part = np.exp(lp) * d * _e1n_vec(-np.abs(b) * d)
        out = cum[j] + part
        out = np.where(x <= xs[0], 0.0, out)
        return np.where(x >= xs[-1], 1.0, out)

    def stop_loss(self, x):
        """C(x) = E(x - W)^+ = int_{-inf}^x F, at a float."""
        if x <= self.xs[0]:
            return 0.0
        if x >= self.xs[-1]:
            return x - self.mean
        j = self._find(x)
        d = x - self.xs[j]
        b = self.betas[j]
        if d <= 0:
            part = 0.0
        elif b <= 0:       # int_0^d (d - t) f(x_j + t) dt = f(x_j) d^2 E2(b d)
            part = math.exp(self.v[j]) * d * d * _e2n(b * d)
        else:              # = f(x) int_0^d u e^{-b u} du = f(x) d^2 E3(-b d)
            part = math.exp(self._logpdf1(j, x)) * d * d * _e3n(-b * d)
        return x * self.cum[j] - self.first[j] + part

    def latent(self, t=1.0):
        """pr(|W| < t) (= pr(|W| <= t): no atoms)."""
        return self.cdf1(t) - self.cdf1(-t)

    # -- noisy coverage pr(|W + e| < 1)
    def noisy(self, noise, scale):
        if noise == 'gaussian':
            return self._noisy_gauss(scale)
        if noise in ('su', 'uniform'):
            return self._noisy_unif(scale)
        raise ValueError(f'unknown noise {noise!r}')

    def _noisy_unif(self, r):
        """Uniform noise on [-r, r]: (1/2r) int_{-r}^{r} {F(1+u) - F(-1+u)} du via C = int F."""
        C = self.stop_loss
        return ((C(1 + r) - C(1 - r)) - (C(-1 + r) - C(-1 - r))) / (2 * r)

    def _nodes(self, sigma):
        """Gauss-Legendre nodes and weights in z on [-ZMAX, ZMAX], split where 1 + sigma z or
        -1 + sigma z hits a breakpoint. On each subinterval the integrand is smooth with
        exponential rate rho = sigma |beta| of the active pieces: chunks of length <= 1.5, graded
        geometrically (lengths 1/rho, 2/rho, 4/rho, ...) towards both ends when rho is large."""
        xs = self._arrays()[0]
        z = np.concatenate([(xs - 1) / sigma, (xs + 1) / sigma, [-_ZMAX, _ZMAX]])
        bp = np.unique(z[(z >= -_ZMAX) & (z <= _ZMAX)])
        edges = []
        for lo, hi in zip(bp[:-1], bp[1:]):
            zm = (lo + hi) / 2
            rho = 0.0
            for c in (1.0, -1.0):
                w = c + sigma * zm
                if self.xs[0] < w < self.xs[-1]:
                    rho = max(rho, abs(self.betas[self._find(w)]) * sigma)
            wid = hi - lo
            if rho * wid <= 3.0:
                n = max(1, math.ceil(wid / 1.5))
                edges.append(lo + wid * np.arange(n) / n)
                continue
            g = 2.0 ** np.arange(0, 64) / rho
            g = g[g < wid / 2]
            left = lo + np.concatenate([[0.0], g])
            right = hi - g[::-1]
            mid_lo, mid_hi = left[-1], right[0]
            n = max(1, math.ceil((mid_hi - mid_lo) / 1.5))
            middle = mid_lo + (mid_hi - mid_lo) * np.arange(1, n) / n
            edges.append(np.concatenate([left, middle, right]))
        e = np.concatenate(edges + [[_ZMAX]])
        width = np.diff(e)
        start = e[:-1]
        zz = (start[:, None] + width[:, None] * (_GLX[None, :] + 1) / 2).ravel()
        ww = (width[:, None] * _GLW[None, :] / 2).ravel()
        return zz, ww * np.exp(_LN_PHI0 - zz * zz / 2)

    def _noisy_gauss(self, sigma):
        z, w = self._nodes(sigma)
        return float(np.dot(w, self.cdf(1 + sigma * z) - self.cdf(-1 + sigma * z)))


def noisy_unif_quadrature(law, r):
    """Independent check of the uniform closed form: Gauss-Legendre in u, split at kinks."""
    bp = [-r, r]
    for c in (1.0, -1.0):
        u = np.asarray(law.xs) - c
        bp.extend(u[(u > -r) & (u < r)])
    bp = np.unique(bp)
    tot = 0.0
    for lo, hi in zip(bp[:-1], bp[1:]):
        n = max(1, int(np.ceil((hi - lo) * max(1.0, max(abs(b) for b in law.betas)) / 2)))
        for i in range(n):
            a, b = lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n
            u = a + (b - a) * (_GLX + 1) / 2
            tot += np.dot(_GLW * (b - a) / 2, law.cdf(1 + u) - law.cdf(-1 + u))
    return float(tot / (2 * r))


def noisy_gauss_closed(law, sigma):
    """Gaussian noisy coverage from the closed form of Supplementary eq. (S.closed), piece by piece.
    Cancels for small |beta| (terms of size 1/beta); used only as a cross-check."""
    from scipy.special import log_ndtr
    tot = 0.0
    for j in range(len(law.L)):
        a, b, beta, l0 = law.xs[j], law.xs[j + 1], law.betas[j], law.v[j]
        for c, sgn in ((1.0, 1.0), (-1.0, -1.0)):
            # int_a^b f(w) Phi((c - w)/sigma) dw, f = exp(l0 + beta (w - a))
            bnd = (math.exp(l0 + beta * (b - a)) * ndtr((c - b) / sigma)
                   - math.exp(l0) * ndtr((c - a) / sigma)) / beta
            lg = l0 + beta * (c - a) + beta ** 2 * sigma ** 2 / 2
            hi_ = (b - c - beta * sigma ** 2) / sigma
            lo_ = (a - c - beta * sigma ** 2) / sigma
            tail = (math.exp(lg + log_ndtr(hi_)) - math.exp(lg + log_ndtr(lo_))) / beta
            tot += sgn * (bnd + tail)
    return tot


# ---------------------------------------------------------------- brute force (mpmath)
def brute_noisy(law, noise, scale, dps=30):
    """pr(|W + e| < 1) by mpmath quadrature of f(w) g(w), g(w) = pr(|w + e| < 1), split at every
    breakpoint of f and of g; independent of the closed forms above."""
    import mpmath as mp
    with mp.workdps(dps):
        s = mp.mpf(scale)

        def g(w):
            if noise == 'gaussian':
                return mp.ncdf((1 - w) / s) - mp.ncdf((-1 - w) / s)
            lo, hi = max(-1, w - s), min(1, w + s)     # uniform on [w - s, w + s]
            return max(mp.mpf(0), (hi - lo) / (2 * s))

        xs = [mp.mpf(float(x)) for x in law.xs]
        ell = [mp.mpf(0)]
        for j in range(len(law.L)):
            ell.append(ell[-1] + mp.mpf(float(law.betas[j])) * (xs[j + 1] - xs[j]))
        Z = mp.mpf(0)
        for j in range(len(law.L)):
            bj, Lj = mp.mpf(float(law.betas[j])), xs[j + 1] - xs[j]
            Z += mp.exp(ell[j]) * (Lj if bj == 0 else mp.expm1(bj * Lj) / bj)
        cuts = {1, -1}
        if noise != 'gaussian':
            cuts |= {1 + s, 1 - s, -1 + s, -1 - s}
        tot = mp.mpf(0)
        for j in range(len(law.L)):
            a, b, bj = xs[j], xs[j + 1], mp.mpf(float(law.betas[j]))
            pts = sorted({a, b} | {c for c in cuts if a < c < b})
            if noise == 'gaussian':          # refine near +-1 at the noise scale
                extra = [c + k * s for c in (1, -1) for k in range(-12, 13)]
                pts = sorted(set(pts) | {x for x in extra if a < x < b})
            if bj != 0:                      # grade towards the steep end of the piece
                end = b if bj > 0 else a
                sgn = -1 if bj > 0 else 1
                grade = [end + sgn * mp.mpf(2) ** k / abs(bj) for k in range(0, 80)]
                pts = sorted(set(pts) | {x for x in grade if a < x < b})
            f = lambda w, a=a, j=j, bj=bj: mp.exp(ell[j] + bj * (w - a)) / Z
            tot += mp.quad(lambda w: f(w) * g(w), pts)
        return tot


def brute_latent(law, dps=30):
    import mpmath as mp
    with mp.workdps(dps):
        xs = [mp.mpf(float(x)) for x in law.xs]
        ell, Z, inside = [mp.mpf(0)], mp.mpf(0), mp.mpf(0)
        for j in range(len(law.L)):
            ell.append(ell[-1] + mp.mpf(float(law.betas[j])) * (xs[j + 1] - xs[j]))

        def I(j, u, v):
            bj = mp.mpf(float(law.betas[j]))
            if v <= u:
                return mp.mpf(0)
            e0 = mp.exp(ell[j] + bj * (u - xs[j]))
            return e0 * ((v - u) if bj == 0 else mp.expm1(bj * (v - u)) / bj)
        for j in range(len(law.L)):
            Z += I(j, xs[j], xs[j + 1])
            inside += I(j, max(xs[j], mp.mpf(-1)), min(xs[j + 1], mp.mpf(1)))
        mean = mp.mpf(0)
        for j in range(len(law.L)):
            bj, a, b = mp.mpf(float(law.betas[j])), xs[j], xs[j + 1]
            mean += mp.quad(lambda w: w * mp.exp(ell[j] + bj * (w - a)), [a, b])
        return inside / Z, mean / Z


# ---------------------------------------------------------------- the reduced family
def _sig(u):
    return 1 / (1 + math.exp(-u)) if u > -700 else 0.0


def shape(c, beta1, beta2):
    """Two-piece log-affine shape on [0, 1], kink at c in (0, 1), slopes beta1 >= beta2."""
    return PLA([0.0, c, 1.0], [beta1, beta2], check=False)


def place(X, mu, q, t=1.0):
    """W = mu + lam (X - E X) with lam > 0 the unique scale giving pr(|W| < t) = q (|mu| < t:
    the coverage decreases strictly in lam where it is below 1). Returns (W, lam) or None."""
    m0 = X.mean
    if not abs(mu) < t:
        return None

    def cov(ll):
        lam = math.exp(ll)
        return X.cdf1(m0 + (t - mu) / lam) - X.cdf1(m0 + (-t - mu) / lam)
    lo = math.log(0.999 * min((t - mu) / max(1 - m0, 1e-300), (t + mu) / max(m0, 1e-300)))
    hi = lo + 1.0
    while cov(hi) > q:
        hi += 2.0
        if hi > lo + 200:
            return None
    flo = cov(lo) - q
    if flo <= 0:
        return None
    for _ in range(200):                         # bisection: keep cov(hi) <= q
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        if cov(mid) > q:
            lo = mid
        else:
            hi = mid
    for _ in range(60):                          # rounding: make sure the placed law is feasible
        lam = math.exp(hi)
        W = PLA([mu + lam * (x - m0) for x in X.xs], [b / lam for b in X.betas], check=False)
        if W.latent(t) <= q:
            return W, lam
        hi += 4e-16 * max(1.0, abs(hi)) * 2 ** _
    return None


@dataclass
class Maximiser:
    noise: str
    q: float
    kappa: float
    value: float                 # p_kappa(q): best value found (attained, so a lower bound)
    excess: float                # value - q
    scale: float                 # sigma (Gaussian) or r (uniform)
    xs: tuple                    # the maximising law W (two-piece log-affine)
    betas: tuple
    latent: float
    mean: float
    starts: int
    theta: tuple = ()


def decode(theta, q, kappa):
    """theta = (u_c, beta2, log(beta1 - beta2), u_mu, log scale) -> (W, scale) or None.
    The mean is mu = min(kappa, 1) sigmoid(u_mu) >= 0 (W and -W are equivalent: the noise is
    symmetric); for q > 1 - 1/e no law with |E W| >= 1 has noisy coverage above q (notes)."""
    uc, b2, ld, um, ls = theta
    if max(abs(b2), ld) > 1e4:
        return None
    c = min(max(_sig(uc), 1e-12), 1 - 1e-12)
    mumax = min(kappa, 1.0)
    mu = mumax * _sig(um) if mumax > 0 else 0.0
    if mu >= 1.0:
        mu = math.nextafter(1.0, 0.0)
    try:
        X = shape(c, b2 + math.exp(min(ld, 15.0)), b2)
        r = place(X, mu, q)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    if r is None:
        return None
    return r[0], math.exp(ls)


def value(theta, noise, q, kappa):
    """Excess pr(|W + e| < 1) - q at theta (or -1 if theta is not admissible)."""
    d = decode(theta, q, kappa)
    if d is None:
        return -1.0
    W, sc = d
    if not 1e-6 < sc < 50:
        return -1.0
    try:
        with np.errstate(all='ignore'):
            v = W.noisy(noise, sc) - q
    except (ValueError, OverflowError, ZeroDivisionError):
        return -1.0
    return v if math.isfinite(v) else -1.0


def random_start(rng):
    return np.array([rng.normal(0, 2.5), rng.normal(0, 1) * 10 ** rng.uniform(0, 2.3),
                     rng.uniform(-6, 6), rng.normal(0, 3),
                     rng.uniform(math.log(2e-3), math.log(1.0))])


def _nm(args):
    """Nelder-Mead from one start (restarted twice from its own optimum)."""
    from scipy.optimize import minimize
    th, noise, q, kappa, maxfev = args
    f = lambda t: -1e5 * value(t, noise, q, kappa)
    best = (value(th, noise, q, kappa), np.asarray(th, float))
    for _ in range(3):
        r = minimize(f, best[1], method='Nelder-Mead',
                     options={'maxfev': maxfev, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
        v = value(r.x, noise, q, kappa)
        if v > best[0] + 1e-15:
            best = (v, r.x)
        else:
            break
    return best


def p_kappa(q, kappa, noise='gaussian', starts=256, seed=0, procs=1, maxfev=3000, extra=()):
    """Best value of the reduced problem over `starts` random Nelder-Mead starts (plus `extra`
    starts), polished. The value is attained by an explicit law and noise, so p_kappa(q) >= value
    up to rounding; that it is the supremum is numerical."""
    if not q > 1 - math.exp(-1):
        raise ValueError('the reduction to |E W| < 1 needs q > 1 - 1/e')
    rng = np.random.default_rng(seed)
    ths = [random_start(rng) for _ in range(starts)] + [np.asarray(e, float) for e in extra]
    jobs = [(t, noise, q, kappa, maxfev) for t in ths]
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            res = pool.map(_nm, jobs, chunksize=1)
    else:
        res = [_nm(j) for j in jobs]
    res.sort(key=lambda t: -t[0])
    v, th = res[0]
    v, th = _nm((th, noise, q, kappa, 20 * maxfev))
    W, sc = decode(th, q, kappa)
    return Maximiser(noise, q, kappa, q + v, v, sc, tuple(W.xs), tuple(W.betas), W.latent(),
                     W.mean, len(jobs), tuple(th)), res


# ---------------------------------------------------------------- check: three pieces
def decode3(theta, q, kappa):
    """Three-piece version of decode: theta = (u_c1, u_c2, beta3, log(b2 - b3), log(b1 - b2),
    u_mu, log scale), kinks c1 < c2 in (0, 1)."""
    u1, u2, b3, l2, l1, um, ls = theta
    c1 = min(max(_sig(u1), 1e-12), 1 - 2e-12)
    c2 = c1 + (1 - c1) * min(max(_sig(u2), 1e-12), 1 - 1e-12)
    if not (0 < c1 < c2 < 1) or abs(b3) > 1e4:
        return None
    b2 = b3 + math.exp(min(l2, 15.0))
    b1 = b2 + math.exp(min(l1, 15.0))
    mumax = min(kappa, 1.0)
    mu = mumax * _sig(um) if mumax > 0 else 0.0
    try:
        r = place(PLA([0.0, c1, c2, 1.0], [b1, b2, b3], check=False), min(mu, 1 - 1e-15), q)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return None if r is None else (r[0], math.exp(ls))


def value3(theta, noise, q, kappa):
    d = decode3(theta, q, kappa)
    if d is None or not 1e-6 < d[1] < 50:
        return -1.0
    with np.errstate(all='ignore'):
        v = d[0].noisy(noise, d[1]) - q
    return v if math.isfinite(v) else -1.0


def _nm3(args):
    from scipy.optimize import minimize
    th, noise, q, kappa, maxfev = args
    f = lambda t: -1e5 * value3(t, noise, q, kappa)
    best = (value3(th, noise, q, kappa), np.asarray(th, float))
    for _ in range(3):
        r = minimize(f, best[1], method='Nelder-Mead',
                     options={'maxfev': maxfev, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
        v = value3(r.x, noise, q, kappa)
        if v > best[0] + 1e-15:
            best = (v, r.x)
        else:
            break
    return best


def three_piece_starts(theta2, rng, n=24):
    """Three-piece starts around a two-piece optimum theta2: an extra kink inserted at random
    inside one of the two pieces, with a random small change of slope."""
    uc, b2, ld, um, ls = theta2
    c = min(max(_sig(uc), 1e-9), 1 - 1e-9)
    b1 = b2 + math.exp(min(ld, 15.0))
    out = []
    for _ in range(n):
        if rng.random() < 0.5:                       # split the left piece at c1 < c
            c1, c2 = c * rng.uniform(0.05, 0.95), c
            s1, s2, s3 = b1 + abs(rng.normal(0, 1)) * (1 + abs(b1)) * 0.1, b1, b2
        else:                                        # split the right piece at c2 > c
            c1, c2 = c, c + (1 - c) * rng.uniform(0.05, 0.95)
            s1, s2, s3 = b1, b2, b2 - abs(rng.normal(0, 1)) * (1 + abs(b2)) * 0.1
        s2 = min(s2, s1 - 1e-9)
        s3 = min(s3, s2 - 1e-9)
        lg = lambda x: math.log(min(max(x, 1e-12), 1 - 1e-12) / (1 - min(max(x, 1e-12), 1 - 1e-12)))
        out.append(np.array([lg(c1), lg(min(max((c2 - c1) / (1 - c1), 1e-9), 1 - 1e-9)), s3,
                             math.log(s2 - s3), math.log(s1 - s2), um, ls]))
    return out


# ---------------------------------------------------------------- validation: laws outside the family
class Standard:
    """A log-concave law X given by a vectorised distribution function, its mean, a scale (sd)
    and its kinks (points where the density is not smooth, e.g. support ends)."""

    def __init__(self, name, cdf, mean, sd, kinks=()):
        self.name, self._cdf, self.mean, self.sd, self.kinks = name, cdf, mean, sd, tuple(kinks)

    def cdf(self, x):
        return self._cdf(np.asarray(x, dtype=float))


class Placed:
    """W = mu + lam (X - E X) for a `Standard` X; noisy coverage by composite Gauss-Legendre."""

    def __init__(self, X, mu, lam):
        self.X, self.mu, self.lam = X, mu, lam
        self.mean = mu
        self.kinks = [mu + lam * (k - X.mean) for k in X.kinks]

    def cdf(self, w):
        return self.X.cdf(self.X.mean + (np.asarray(w, dtype=float) - self.mu) / self.lam)

    def latent(self):
        return float(self.cdf(1.0) - self.cdf(-1.0))

    def _gl(self, lo, hi, bps, h, fun):
        bp = np.unique(np.concatenate([[lo, hi], [b for b in bps if lo < b < hi]]))
        wid = np.diff(bp)
        n = np.maximum(1, np.ceil(wid / h).astype(int))
        idx = np.repeat(np.arange(len(wid)), n)
        pos = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        width = wid[idx] / n[idx]
        start = bp[:-1][idx] + pos * width
        x = (start[:, None] + width[:, None] * (_GLX[None, :] + 1) / 2).ravel()
        w = (width[:, None] * _GLW[None, :] / 2).ravel()
        return float(np.dot(w, fun(x)))

    def noisy(self, noise, scale):
        h_law = 0.5 * self.lam * self.X.sd
        k = np.asarray(self.kinks, dtype=float)
        if noise == 'gaussian':
            s = scale
            bps = np.concatenate([(k - 1) / s, (k + 1) / s])
            fun = lambda z: np.exp(_LN_PHI0 - z * z / 2) * (self.cdf(1 + s * z) - self.cdf(-1 + s * z))
            return self._gl(-_ZMAX, _ZMAX, bps, min(1.0, h_law / s), fun)
        r = scale
        bps = np.concatenate([k - 1, k + 1])
        fun = lambda u: self.cdf(1 + u) - self.cdf(-1 + u)
        return self._gl(-r, r, bps, min(r / 2, h_law), fun) / (2 * r)


def standard_laws(rng):
    """Random members of standard log-concave families (shape parameters drawn at random).
    Distribution functions through scipy.special (scipy.stats objects are too slow here)."""
    from scipy import special as sp
    E = np.euler_gamma
    clip = lambda x, lo, hi: np.minimum(np.maximum(x, lo), hi)
    out = [Standard('normal', sp.ndtr, 0.0, 1.0),
           Standard('logistic', sp.expit, 0.0, math.pi / math.sqrt(3)),
           Standard('laplace', lambda x: np.where(x < 0, 0.5 * np.exp(np.minimum(x, 0)),
                                                  1 - 0.5 * np.exp(-np.maximum(x, 0))),
                    0.0, math.sqrt(2), (0.0,)),
           Standard('gumbel', lambda x: np.exp(-np.exp(-clip(x, -6, 700))), E,
                    math.pi / math.sqrt(6)),
           Standard('gumbel_left', lambda x: -np.expm1(-np.exp(clip(x, -700, 6))), -E,
                    math.pi / math.sqrt(6)),
           Standard('exponential', lambda x: -np.expm1(-np.maximum(x, 0)), 1.0, 1.0, (0.0,)),
           Standard('reflected_exponential', lambda x: np.exp(np.minimum(x, 0)), -1.0, 1.0,
                    (0.0,))]
    a = 1 + rng.exponential(2.0)
    out.append(Standard(f'gamma({a:.3g})', lambda x, a=a: sp.gammainc(a, np.maximum(x, 0)),
                        a, math.sqrt(a), (0.0,)))
    out.append(Standard(f'reflected_gamma({a:.3g})',
                        lambda x, a=a: sp.gammaincc(a, np.maximum(-x, 0)), -a, math.sqrt(a),
                        (0.0,)))
    kw = 1 + rng.exponential(1.5)
    m1, m2 = math.gamma(1 + 1 / kw), math.gamma(1 + 2 / kw)
    out.append(Standard(f'weibull({kw:.3g})',
                        lambda x, k=kw: -np.expm1(-np.maximum(x, 0) ** k), m1,
                        math.sqrt(m2 - m1 * m1), (0.0,)))
    al, be = 1 + rng.exponential(3), 1 + rng.exponential(3)
    mb = al / (al + be)
    sb = math.sqrt(al * be / ((al + be) ** 2 * (al + be + 1)))
    out.append(Standard(f'beta({al:.3g},{be:.3g})',
                        lambda x, al=al, be=be: sp.betainc(al, be, clip(x, 0, 1)), mb, sb,
                        (0.0, 1.0)))
    L = math.exp(rng.uniform(-1, 3))
    te_m = 1 - L / math.expm1(L)                       # mean of Exp(1) truncated to [0, L]
    te_v = 1 - L * L * math.exp(L) / math.expm1(L) ** 2
    out.append(Standard(f'truncexp({L:.3g})',
                        lambda x, L=L: -np.expm1(-clip(x, 0, L)) / -math.expm1(-L), te_m,
                        math.sqrt(te_v), (0.0, L)))
    out.append(Standard(f'reflected_truncexp({L:.3g})',
                        lambda x, L=L: 1 + np.expm1(-clip(-x, 0, L)) / -math.expm1(-L), -te_m,
                        math.sqrt(te_v), (-L, 0.0)))
    ta, tb = sorted(rng.normal(0, 2, 2))
    tb += 0.1
    Za = sp.ndtr(tb) - sp.ndtr(ta)
    pa, pb = math.exp(-ta * ta / 2) / math.sqrt(2 * math.pi), math.exp(-tb * tb / 2) / math.sqrt(2 * math.pi)
    tm = (pa - pb) / Za
    tv = 1 + (ta * pa - tb * pb) / Za - tm * tm
    out.append(Standard(f'truncnorm({ta:.3g},{tb:.3g})',
                        lambda x, ta=ta, tb=tb, Za=Za: (sp.ndtr(clip(x, ta, tb)) - sp.ndtr(ta)) / Za,
                        tm, math.sqrt(tv), (ta, tb)))
    return out


def random_pla(rng, pieces=None):
    """Random log-concave law with density exp(-V), V convex piecewise affine with many pieces."""
    k = pieces or int(rng.integers(3, 40))
    xs = np.cumsum(np.r_[0.0, rng.exponential(1.0, k)])
    xs = xs / xs[-1]
    slopes = np.sort(rng.normal(0, 1, k) * 10 ** rng.uniform(0, 2.5))[::-1]
    return PLA(xs, slopes)


def _placed_pla(X, mu, lam):
    return PLA([mu + lam * (x - X.mean) for x in X.xs], [b / lam for b in X.betas], check=False)


def _lam_q(X, mu, q):
    """Scale with pr(|mu + lam (X - EX)| < 1) = q (bisection in log lam), or None."""
    cdf = X.cdf1 if isinstance(X, PLA) else (lambda x: float(X.cdf(x)))
    m0, sd = X.mean, (X.sd if isinstance(X, Standard) else 1.0)

    def cov(ll):
        lam = math.exp(ll)
        return cdf(m0 + (1 - mu) / lam) - cdf(m0 + (-1 - mu) / lam)
    lo, hi = math.log(1e-3 / sd), math.log(1e-3 / sd) + 1
    while cov(lo) <= q:
        lo -= 3
        if lo < -60:
            return None
    while cov(hi) > q:
        hi += 3
        if hi > 60:
            return None
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid in (lo, hi) or hi - lo < 1e-15:
            break
        if cov(mid) > q:
            lo = mid
        else:
            hi = mid
    return math.exp(hi)


def best_over_placement(X, noise, q, kappa, rng=None, maxfev=400):
    """sup over mean mu in [-min(kappa,1), min(kappa,1)], extra dilation (latent coverage <= q)
    and noise scale of the excess of mu + lam (X - EX). Returns (excess, mu, lam, scale)."""
    from scipy.optimize import minimize
    mumax = min(kappa, 1 - 1e-9)
    is_pla = isinstance(X, PLA)

    def build(t):
        mu = mumax * math.tanh(t[0]) if mumax > 0 else 0.0
        lq = _lam_q(X, mu, q)
        if lq is None:
            return None
        lam = lq * math.exp(abs(t[2]))
        return (_placed_pla(X, mu, lam) if is_pla else Placed(X, mu, lam)), mu, lam

    def val(t):
        sc = math.exp(t[1])
        if not 1e-5 < sc < 20:
            return -1.0, None
        try:
            b = build(t)
            if b is None:
                return -1.0, None
            W = b[0]
            if W.latent() > q + 1e-15:
                return -1.0, None
            with np.errstate(all='ignore'):
                v = W.noisy(noise, sc) - q
            return (v if math.isfinite(v) else -1.0), (b[1], b[2], sc)
        except (ValueError, OverflowError, ZeroDivisionError):
            return -1.0, None
    starts = [np.array([m, ls, 0.0]) for m in (0.0, 0.7, -0.7, 2.5, -2.5)
              for ls in np.log([0.003, 0.01, 0.03, 0.1, 0.3])]
    vals = [(val(t)[0], t) for t in starts]
    vals.sort(key=lambda z: -z[0])
    best = (-1.0, None)
    for v0, t0 in vals[:3]:
        r = minimize(lambda t: -1e5 * val(t)[0], t0, method='Nelder-Mead',
                     options={'maxfev': maxfev, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
        v, info = val(r.x)
        if v > best[0]:
            best = (v, info)
    if best[1] is None:
        return -1.0, None, None, None
    return (best[0],) + best[1]


# ---------------------------------------------------------------- reliability of the usual rank
def failure_K(p, q, tau, delta=0.05):
    """Normal approximation and exact binomial scan of the K at which the usual rank's latent
    reliability pr{Bin(K, p) <= k_K - 1}, k_K the usual rank, first falls below tau.

    Returns (K_normal, K_exact). Normal approximation: the reliability is about
    Phi(z_{1-delta} - sqrt(K) (p - q)/sqrt(q(1-q))), so K_normal = q(1-q) (z_{1-delta} - z_tau)^2
    /(p - q)^2 (for tau = 1/2 this is z^2 q(1-q)/(p - q)^2 of Section 4). Exact: scan of a log grid
    from K_normal/30 to 30 K_normal; the reliability is a sawtooth in K (integer rank), so the
    first grid K below tau is refined by bisection against the previous grid point (to 0.05%).
    K_exact is nan beyond 1e13."""
    from scipy import stats
    from latentcov.rank import rank_for_level
    if p <= q:
        return math.inf, math.inf
    z = stats.norm.ppf(1 - delta) - stats.norm.ppf(tau)
    Kn = q * (1 - q) * z * z / (p - q) ** 2

    def rel(K):
        K = int(K)
        k = rank_for_level(K, q, delta)
        if k is None:                          # no valid usual rank at this K: nothing to fail
            return 1.0
        return float(stats.binom.cdf(k - 1, K, p))
    if Kn > 1e13:
        return Kn, math.nan
    kmin = math.ceil(math.log(delta) / math.log(q))     # smallest K with a usual rank
    grid = np.unique(np.round(np.geomspace(max(kmin, Kn / 30), Kn * 30, 241)).astype(np.int64))
    prev = None
    for K in grid:
        if rel(K) < tau:
            lo = int(prev) if prev is not None else kmin
            hi = int(K)
            while hi - lo > max(1, lo // 2000):
                mid = (lo + hi) // 2
                if rel(mid) < tau:
                    hi = mid
                else:
                    lo = mid
            return Kn, hi
        prev = K
    return Kn, math.nan
