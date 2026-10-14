"""Exact noisy coverage H = pr(|V| < r), r = Q_q(|W|), for intermediate designs (R10).

With i.i.d. units, N = #{i : |V_i| < r} is Bin(K, H), so the latent reliability of rank k is
pr{Bin(K, H) <= k - 1} exactly (proof of Theorem 3). This module computes H accurately for

    W = W0 + b              W0 a standardized latent law (mean 0, variance 1), bias b >= 0;
    V = W + sigma eps       eps Gaussian or uniform (variance 1), sigma fixed, random (lognormal
                            variance, independent of W) or a function of W (proportions).

H is computed as q + D with D = int f(w) {k(w) - 1(|w| < r)} dw, where k(w) = pr(|w + e| < r):
the integrand vanishes away from +-r when the noise is small, so the excess H - q is obtained
without cancellation. scipy quad on pieces split at +-r, +-r +- multiples of the noise scale and
the kinks of f; the error is checked against mpmath (experiments/r10_intermediate.py verify).
"""
import math
import warnings

import numpy as np
from scipy import integrate, special, stats
from scipy.optimize import brentq

TAIL = 1e-18          # latent mass dropped beyond the integration range (each side)
LOGNORMAL_S = 0.7     # log-scale of the variance multiplier L in the heterogeneous designs


# ------------------------------------------------------------------ latent laws
class Law:
    """Latent law with scalar pdf, cdf and survival function (fast scalar code for quad).

    lo, hi: integration range (support ends, or quantiles at TAIL); kinks: points where the
    density is not smooth (support ends of a positive density, the Laplace mode)."""

    def __init__(self, name, pdf, cdf, sf, lo, hi, kinks=(), log_concave=True, mean=0.0,
                 sd=1.0):
        self.name, self.pdf, self.cdf, self.sf = name, pdf, cdf, sf
        self.lo, self.hi, self.kinks = lo, hi, tuple(kinks)
        self.log_concave, self.mean, self.sd = log_concave, mean, sd

    def reflected(self):
        return Law('reflected ' + self.name, lambda w: self.pdf(-w), lambda w: self.sf(-w),
                   lambda w: self.cdf(-w), -self.hi, -self.lo, tuple(-k for k in self.kinks),
                   self.log_concave, -self.mean, self.sd)

    def coverage(self, r, b=0.0):
        """pr(|W0 + b| < r) = pr(-r - b < W0 < r - b), from both tails for accuracy."""
        if r <= 0:
            return 0.0
        return 1.0 - self.cdf(-r - b) - self.sf(r - b)


def _gamma(k):
    s = math.sqrt(k)
    lg = math.lgamma(k)

    def pdf(w):
        x = k + s * w
        if x <= 0:
            return 0.0
        return s * math.exp((k - 1) * math.log(x) - x - lg)

    def cdf(w):
        x = k + s * w
        return 0.0 if x <= 0 else float(special.gammainc(k, x))

    def sf(w):
        x = k + s * w
        return 1.0 if x <= 0 else float(special.gammaincc(k, x))

    hi = (stats.gamma.isf(TAIL, k) - k) / s
    name = 'centred exponential' if k == 1 else f'centred gamma({k:g})'
    return Law(name, pdf, cdf, sf, -s, hi, kinks=(-s,))


def _skew_normal(alpha):
    d = alpha / math.sqrt(1 + alpha * alpha)
    m = d * math.sqrt(2 / math.pi)
    sd = math.sqrt(1 - m * m)

    def pdf(w):
        x = m + sd * w
        return sd * 2 * math.exp(-x * x / 2) / math.sqrt(2 * math.pi) * float(special.ndtr(alpha * x))

    def cdf(w):
        x = m + sd * w
        return float(special.ndtr(x) - 2 * special.owens_t(x, alpha))

    def sf(w):
        x = m + sd * w
        return float(special.ndtr(-x) + 2 * special.owens_t(x, alpha))

    sn = stats.skewnorm(alpha)
    lo, hi = (sn.ppf(TAIL) - m) / sd, (sn.isf(TAIL) - m) / sd
    return Law(f'skew-normal({alpha:g})', pdf, cdf, sf, lo, hi)


def _half_normal():
    """|Z| standardized: a hard edge where the density is largest (flat there)."""
    m = math.sqrt(2 / math.pi)
    sd = math.sqrt(1 - 2 / math.pi)
    c = 2 / math.sqrt(2 * math.pi)

    def pdf(w):
        x = m + sd * w
        return 0.0 if x < 0 else sd * c * math.exp(-x * x / 2)

    def cdf(w):
        x = m + sd * w
        return 0.0 if x <= 0 else float(special.erf(x / math.sqrt(2)))

    def sf(w):
        x = m + sd * w
        return 1.0 if x <= 0 else float(2 * special.ndtr(-x))

    return Law('centred half-normal', pdf, cdf, sf, -m / sd, (9.5 - m) / sd, kinks=(-m / sd,))


def _normal():
    c = 1 / math.sqrt(2 * math.pi)
    return Law('normal', lambda w: c * math.exp(-w * w / 2), lambda w: float(special.ndtr(w)),
               lambda w: float(special.ndtr(-w)), -9.5, 9.5)


def _laplace():
    s2 = math.sqrt(2)
    lo = math.log(2 * TAIL) / s2           # e^{sqrt2 w}/2 = TAIL
    return Law('Laplace', lambda w: math.exp(-s2 * abs(w)) / s2,
               lambda w: 0.5 * math.exp(s2 * w) if w < 0 else 1 - 0.5 * math.exp(-s2 * w),
               lambda w: 0.5 * math.exp(-s2 * w) if w > 0 else 1 - 0.5 * math.exp(s2 * w),
               lo, -lo, kinks=(0.0,))


def latent_laws():
    """The latent laws of R10, standardized to mean 0 and variance 1, all log-concave. Skewed
    laws come with their reflections, so that a bias b >= 0 covers both signs of the bias. The
    exponential and the half-normal have a hard edge (a support end with positive density)."""
    skewed = [_gamma(2), _gamma(5), _gamma(1), _skew_normal(4.0), _half_normal()]
    out = {L.name: L for L in skewed}
    out.update({L.name: L for L in (L.reflected() for L in skewed)})
    out['normal'], out['Laplace'] = _normal(), _laplace()
    return out


def beta_law(a, b):
    """Beta(a, b) on [0, 1] (not standardized), for the proportion designs."""
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def pdf(t):
        if t <= 0 or t >= 1:
            return 0.0
        return math.exp((a - 1) * math.log(t) + (b - 1) * math.log1p(-t) - lb)

    m = a / (a + b)
    sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return Law(f'Beta({a:g}, {b:g})', pdf,
               lambda t: float(special.betainc(a, b, min(max(t, 0.0), 1.0))),
               lambda t: float(special.betainc(b, a, min(max(1 - t, 0.0), 1.0))),
               0.0, 1.0, kinks=(0.0, 1.0), log_concave=(a >= 1 and b >= 1), mean=m, sd=sd)


# ------------------------------------------------------------------ radius and bias
def radius(law, b, q):
    """r = Q_q(|W0 + b|), the latent q-quantile of |W| (law continuous: coverage(r) = q)."""
    hi = 1.0
    while law.coverage(hi, b) < q:
        hi *= 2
    return brentq(lambda r: law.coverage(r, b) - q, 0.0, hi, xtol=1e-15, rtol=8.9e-16,
                  maxiter=500)


def bias_for_zeta(law, zeta, q):
    """b >= 0 with b / Q_q(|W0 + b|) = zeta (zeta in [0, 1))."""
    if zeta == 0:
        return 0.0
    f = lambda b: b / radius(law, b, q) - zeta     # noqa: E731
    grid = np.linspace(0, 20, 401)
    prev = grid[0]
    for x in grid[1:]:
        if f(x) > 0:
            return brentq(f, prev, x, xtol=1e-14, rtol=8.9e-16)
        prev = x
    raise ValueError(f'no bias gives zeta = {zeta} for {law.name}')


# ------------------------------------------------------------------ kernel deficits
def deficit_gauss(w, r, s):
    """pr(|w + sZ| < r) - 1(|w| < r)."""
    if s <= 0:
        return 0.0
    if w >= r:
        return float(special.ndtr((r - w) / s) - special.ndtr((-r - w) / s))
    if w <= -r:
        return float(special.ndtr((w + r) / s) - special.ndtr((w - r) / s))
    return -float(special.ndtr((w - r) / s) + special.ndtr((-r - w) / s))


def deficit_unif(w, r, a):
    """pr(|w + aU| < r) - 1(|w| < r), U uniform on [-1, 1] (half-width a)."""
    if a <= 0:
        return 0.0
    cut = max(w + a - r, 0.0) + max(a - r - w, 0.0)
    k = max(2 * a - cut, 0.0) / (2 * a)
    return -cut / (2 * a) if abs(w) < r else k


def _quad_pieces(fun, pts):
    tot, err = 0.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', integrate.IntegrationWarning)
        for x0, x1 in zip(pts[:-1], pts[1:]):
            if x1 > x0:
                v, e = integrate.quad(fun, x0, x1, epsabs=1e-16, epsrel=1e-13, limit=400)
                tot += v
                err += e
    return tot, err


_MULT = (0.5, 1, 2, 4, 8, 16)


def excess(law, b, r, noise, scale):
    """D = H - q for W = W0 + b with r = Q_q(|W|) and fixed noise scale.

    noise 'gaussian': e = scale * Z; 'uniform': e = scale * sqrt(3) U (variance scale^2);
    scale may be a function of w (noise depending on the latent residual), Gaussian only.
    Returns (D, quadrature error estimate)."""
    lo, hi = law.lo + b, law.hi + b
    pts = {lo, hi, -r, r} | {k + b for k in law.kinks}
    if callable(scale):
        assert noise == 'gaussian'
        sfun = scale
        for c in (-r, r):
            s0 = sfun(c) if lo < c < hi else 0.0
            pts |= {c + m * s0 * sg for m in _MULT for sg in (-1, 1)}

        def fun(w):
            return law.pdf(w - b) * deficit_gauss(w, r, sfun(w))
    elif noise == 'gaussian':
        s = float(scale)
        pts |= {c + m * s * sg for c in (-r, r) for m in _MULT for sg in (-1, 1)}

        def fun(w):
            return law.pdf(w - b) * deficit_gauss(w, r, s)
    elif noise == 'uniform':
        a = math.sqrt(3) * float(scale)
        pts |= {c + sg * a for c in (-r, r) for sg in (-1, 1)}
        pts |= {c + sg * a * m for c in (-r, r) for sg in (-1, 1) for m in (0.25, 0.5)}

        def fun(w):
            return law.pdf(w - b) * deficit_unif(w, r, a)
    else:
        raise ValueError(noise)
    pts = sorted(p for p in pts if lo <= p <= hi)
    return _quad_pieces(fun, pts)


def lognormal_nodes(n=96, s=LOGNORMAL_S, wmin=1e-22):
    """Gauss-Hermite nodes and weights for L / E L, log L ~ N(0, s^2) (weights sum to 1)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    w = w / math.sqrt(math.pi)
    m = w > wmin
    return np.exp(math.sqrt(2) * s * x[m] - s * s / 2), w[m] / w[m].sum()


def excess_het(law, b, r, noise, rho, n=96):
    """D for heterogeneous scales sigma^2 = rho L / E L, log L ~ N(0, s^2), s = 0.7.

    Gaussian noise: D(sigma) is analytic in sigma > 0, so Gauss-Hermite in log L with n nodes
    (agrees with adaptive quad in log L to 3e-16 at n = 96). Uniform noise: D(a) has kinks where
    a breakpoint +-r +- a of the kernel meets a kink of the latent density or the other end
    (a = r), which spoils Gauss-Hermite (errors ~1e-6); adaptive quad in log L split there."""
    if noise == 'gaussian':
        mult, wts = lognormal_nodes(n)
        tot, err = 0.0, 0.0
        for m, w in zip(mult, wts):
            v, e = excess(law, b, r, noise, math.sqrt(rho * m))
            tot += w * v
            err += w * e
        return tot, err
    return excess_het_adaptive(law, b, r, noise, rho)


def excess_het_adaptive(law, b, r, noise, rho, s=LOGNORMAL_S, xmax=9.0):
    """D for heterogeneous scales by adaptive quad over x = log(L)/s ~ N(0, 1), split where the
    noise scale hits a kink of D (uniform noise: half-width a = |k +- r| for kinks k of the
    latent density, and a = r). Returns (D, error estimate)."""
    def a_of(x):
        return math.sqrt(3 * rho * math.exp(s * x - s * s / 2))

    def x_of(a):                                  # inverse of a_of
        return (math.log(a * a / (3 * rho)) + s * s / 2) / s

    pts = {-xmax, xmax}
    if noise == 'uniform':
        crit = {r} | {abs(k + b + sg * r) for k in law.kinks for sg in (-1, 1)}
        pts |= {x_of(a) for a in crit if a > 0}
    pts = sorted(p for p in pts if -xmax <= p <= xmax)

    def g(x):
        sc = math.sqrt(rho * math.exp(s * x - s * s / 2))
        return math.exp(-x * x / 2) / math.sqrt(2 * math.pi) * excess(law, b, r, noise, sc)[0]
    return _quad_pieces(g, pts)


# ------------------------------------------------------------------ proportions
def proportion_design(a, bb, c, n):
    """theta ~ Beta(a, bb), W = theta - c, V = W + sqrt(theta (1 - theta)/n) Z.

    Returns (law, shift, sigma(w)) for `excess` (W = theta + shift, shift = -c)."""
    law = beta_law(a, bb)

    def sigma(w):
        t = w + c
        return math.sqrt(t * (1 - t) / n) if 0 < t < 1 else 0.0
    return law, -c, sigma


# ------------------------------------------------------------------ ranks and reliability
def ranks_on_grid(Ks, p, delta=0.05):
    """rank_for_level(K, p, delta) for an array of K (0 where no rank is valid): vectorized
    binomial quantile, with rank_for_level itself wherever the binomial cdf is within 1e-9 of
    1 - delta (or the quantile fails to verify)."""
    from latentcov.rank import rank_for_level
    Ks = np.asarray(Ks, dtype=np.int64)
    k = stats.binom.ppf(1 - delta, Ks, p).astype(np.int64) + 1
    c1 = stats.binom.cdf(k - 1, Ks, p)
    c0 = stats.binom.cdf(k - 2, Ks, p)
    bad = (np.abs(c1 - (1 - delta)) < 1e-9) | (np.abs(c0 - (1 - delta)) < 1e-9) | \
        (c1 < 1 - delta) | (c0 >= 1 - delta) | (k > Ks)
    for i in np.flatnonzero(bad):
        kk = rank_for_level(int(Ks[i]), p, delta)
        k[i] = 0 if kk is None else kk
    return k


def reliability(Ks, ks, H):
    """pr{Bin(K, H) <= k - 1}; nan where k = 0 (no valid rank)."""
    Ks, ks = np.asarray(Ks), np.asarray(ks)
    out = stats.binom.cdf(ks - 1, Ks, H)
    return np.where(ks > 0, out, np.nan)


def k_grid(kmin=29, dense_to=3000, kmax=1_000_000, n=12000, extra=()):
    """All K in [kmin, dense_to], a geometric grid of about n points up to kmax, and extra."""
    g = np.unique(np.round(np.geomspace(dense_to, kmax, n)).astype(np.int64))
    return np.unique(np.concatenate([np.arange(kmin, dense_to + 1), g,
                                     np.asarray(extra, dtype=np.int64)]))
