"""Rigorous enclosures of the one-sided constants c_{p,q} and of the split constant C_{p,q}.

All inequalities that decide a bound are checked in ball arithmetic (Arb via python-flint);
floating point is used only to propose candidate points. A ball is turned into a double only
with outward rounding (`fup`, `fdown`), sums of bounds are formed as balls, and the u-grid ends
are exactly u_min and u_max. The results therefore do not rest on
the double-precision closed forms used elsewhere.

One-sided problem (Lemma A). For the exponential tail Y = b - E, E ~ Exp(u), Z ~ N(0, 1),

    kappa_u(b) = P(Y + Z <= 0) = Phi(-b) + exp(-u b + u^2/2) Phi(b - u),

and c_{p,q} = sup_{u > 0} v_u, v_u = b_u(p) - log(1/q)/u with kappa_u(b_u(p)) = p.

Monotonicity. kappa_u(b) decreases in b, and in u because E = E_1/u decreases in u. Hence
b_u(p) is nonincreasing in u, and on [u0, u1]

    v_u <= b_u0(p) - log(1/q)/u1.                                        (box bound)

A ball evaluation kappa_u(b) < p proves b_u(p) < b (upper candidate); kappa_u(b) > p proves
b_u(p) > b (lower candidate).

Ends. For u >= u_max, v_u <= b_{u_max}(p). For small u take eps = u^3 and
b(u) = (L_p + u^2/2 - log(1 - eps))/u, L_p = log(1/p). Then exp(-u b + u^2/2) = p (1 - eps) and
b >= L_p/u, so Phi(-b) <= phi(L_p/u) u / L_p by the Mills ratio. If phi(L_p/u)/u^2 <= p L_p the
two terms of kappa_u(b) sum to at most p, so b_u(p) <= b(u). The function phi(L/u)/u^2
increases on (0, L/sqrt(2)), so checking the inequality at u_min <= L_p/sqrt(2) covers
(0, u_min]. For p >= q the resulting bound on v_u is increasing in u, so its value at u_min
bounds the whole end.

Split constant (Theorem 5+). c_{p',q'} <= g iff q' <= exp(-u (b_u(p') - g)) for every u > 0,
iff 1 - q' >= phi_g(1 - p') where

    phi_g(a) = max(0, sup_u 1 - exp(-G_u)),   G_u = u (b_u(1 - a) - g).

Hence C_{p,q} <= g iff  Psi_g = sup_{a_L + a_R = 1 - p} phi_g(a_L) + phi_g(a_R) <= 1 - q
(choose beta_L = phi_g(a_L), beta_R = 1 - q - beta_L; a strict inequality leaves room for a
margin). phi_g(a) is nondecreasing in a because b_u(1 - a) is. G_u is bounded on [u0, u1] by
u1 (b_u0 - g) when b_u0 - g >= 0 and by 0 otherwise, and the small-u end by
L + u^2/2 - log(1 - u^3) + u max(-g, 0) at u_min, as above.
"""
import heapq
import math

import numpy as np
from flint import arb, ctx
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

ctx.prec = 128
SQRT2 = arb(2).sqrt()


def A(v):
    """Exact ball for a float or a decimal string."""
    return arb(v) if isinstance(v, str) else arb(float(v))


def fup(x):
    """Smallest double found with double >= every point of the ball x (outward rounding)."""
    f = float(x.upper())
    while not arb(f) >= x:
        f = math.nextafter(f, math.inf)
    return f


def fdown(x):
    """Double <= every point of the ball x."""
    f = float(x.lower())
    while not arb(f) <= x:
        f = math.nextafter(f, -math.inf)
    return f


def log_grid(u_min, u_max, n):
    """Geometric grid of doubles whose ends are exactly u_min and u_max, so the boxes
    [grid[i], grid[i+1]] cover [u_min, u_max] without gaps."""
    g = np.exp(np.linspace(math.log(u_min), math.log(u_max), n + 1))
    g[0], g[-1] = u_min, u_max
    return np.maximum.accumulate(g)


def Phi(x):
    return (-x / SQRT2).erfc() / 2


def phi(x):
    return (-x * x / 2).exp() / (2 * arb.pi()).sqrt()


def kappa(u, b):
    """Ball for kappa_u(b); u, b are floats (exact binary numbers)."""
    u, b = A(u), A(b)
    return Phi(-b) + (-u * b + u * u / 2).exp() * Phi(b - u)


def _kappa_float(b, u):
    return ndtr(-b) + np.exp(-u * b + u * u / 2 + log_ndtr(b - u))


def _root(u, p):
    return brentq(lambda b: _kappa_float(b, u) - p, -40.0, 40.0 / u + 40.0, xtol=1e-15, rtol=1e-15,
                  maxiter=400)


def b_bounds(u, p):
    """Floats (lo, hi) with lo < b_u(p) < hi proved in ball arithmetic. p is a ball."""
    b = _root(u, float(p.mid()))
    d = 1e-13 * max(1.0, abs(b))
    hi = lo = None
    for _ in range(60):
        if hi is None and kappa(u, b + d) < p:
            hi = b + d
        if lo is None and kappa(u, b - d) > p:
            lo = b - d
        if hi is not None and lo is not None:
            return lo, hi
        d *= 4
    raise RuntimeError(f'no enclosure of b_u(p) at u = {u}')


def small_u_end(p, candidates=None):
    """Largest u_min from the candidates for which the small-u argument applies at level p."""
    L = -p.log()
    Lf = float(L.mid())
    if candidates is None:
        candidates = Lf * np.array([0.70, 0.5, 0.35, 0.25, 0.18, 0.14, 0.11, 0.09, 0.075, 0.06, 0.05])
    for u in candidates:
        u = float(u)
        ua = A(u)
        if ua < L / SQRT2 and phi(L / ua) / (ua * ua) < p * L:
            return u
    raise RuntimeError('small-u end not established')


def _eps_term(u):
    ua = A(u)
    return ua * ua / 2 - (1 - ua ** 3).log()


def c_enclosure(p, q, u_max=64.0, n0=400, tol=1e-9, max_iter=200000):
    """Rigorous (lo, hi) with lo <= c_{p,q} <= hi, for p >= q (decimal strings or floats)."""
    p, q = A(p), A(q)
    assert not (p < q), 'needs p >= q'
    Lq = -q.log()
    # lower bound from the best of the proposal points
    u_min = small_u_end(p)
    grid = log_grid(u_min, u_max, n0)
    bs = [b_bounds(float(u), p) for u in grid]
    lo_f = max(fdown(A(bl) - Lq / A(u)) for u, (bl, _) in zip(grid, bs))
    # ends
    Lp = -p.log()
    ua = A(u_min)
    end_small = fup((Lp - Lq + _eps_term(u_min)) / ua)
    end_large = bs[-1][1]
    # box bounds on [u_i, u_{i+1}]
    def box(u0, bh0, u1):
        return fup(A(bh0) - Lq / A(u1))
    heap = [(-box(grid[i], bs[i][1], grid[i + 1]), float(grid[i]), float(grid[i + 1]), bs[i][1])
            for i in range(n0)]
    heapq.heapify(heap)
    it = 0
    while heap and -heap[0][0] > lo_f + tol and it < max_iter:
        _, u0, u1, bh0 = heapq.heappop(heap)
        um = math.sqrt(u0 * u1)
        bl, bh = b_bounds(um, p)
        vl = fdown(A(bl) - Lq / A(um))
        if vl > lo_f:
            lo_f = vl
        heapq.heappush(heap, (-box(u0, bh0, um), u0, um, bh0))
        heapq.heappush(heap, (-box(um, bh, u1), um, u1, bh))
        it += 1
    hi = max(-heap[0][0], end_small, end_large)
    return lo_f, hi, dict(u_min=u_min, boxes=len(heap), iters=it, end_small=end_small,
                          end_large=end_large)


def phi_upper(a, g, u_max=64.0, n0=200, tol=1e-7, max_iter=100000):
    """Rigorous upper bound on phi_g(a) (a, g floats or decimal strings)."""
    a, g = A(a), A(g)
    p1 = 1 - a
    L = -p1.log()
    u_min = small_u_end(p1)
    gneg = -g if g < 0 else arb(0)          # g is an exact double, so one branch is certain
    ua = A(u_min)
    G_end = fup(L + _eps_term(u_min) + ua * gneg)
    bh_max = b_bounds(u_max, p1)[1]
    if not (A(bh_max) - g < 0):
        raise RuntimeError('increase u_max')
    grid = log_grid(u_min, u_max, n0)
    bs = [b_bounds(float(u), p1) for u in grid]

    def box(u0, bh0, u1):
        d = A(bh0) - g
        if d < 0:
            return -math.inf
        return fup(A(u1) * d)

    def point(u, bl):
        return fdown(A(u) * (A(bl) - g))

    lo = max(point(u, bl) for u, (bl, _) in zip(grid, bs))
    heap = [(-box(grid[i], bs[i][1], grid[i + 1]), float(grid[i]), float(grid[i + 1]), bs[i][1])
            for i in range(n0)]
    heapq.heapify(heap)
    it = 0
    while heap and -heap[0][0] > lo + tol and it < max_iter:
        _, u0, u1, bh0 = heapq.heappop(heap)
        um = math.sqrt(u0 * u1)
        bl, bh = b_bounds(um, p1)
        lo = max(lo, point(um, bl))
        heapq.heappush(heap, (-box(u0, bh0, um), u0, um, bh0))
        heapq.heappush(heap, (-box(um, bh, u1), um, u1, bh))
        it += 1
    G = max(-heap[0][0], G_end)
    if G <= 0:
        return 0.0
    return fup(1 - (-A(G)).exp())


def split_certificate(p, q, g, n0=16, max_rounds=40, pool=None, tol=1e-7):
    """Checks C_{p,q} <= g. On an interval a_L in [a0, a1] of [0, T/2], T = 1 - p, the bound is
    phi(a1) + phi(T - a0) (phi nondecreasing; a_R = T - a_L). Intervals whose bound is not below
    1 - q are halved, up to max_rounds times. Returns (worst interval bound, budget, ok, info)."""
    pa, qa = A(p), A(q)
    T = 1 - pa
    budget = 1 - qa                       # compared as a ball
    cache = {0.0: 0.0}                    # phi at a = 0 is 0 (no feasible law)

    def need(points):
        new = sorted({x for x in points if x not in cache})
        args = [(x, g, 64.0, 200, tol) for x in new]
        vals = pool.starmap(phi_upper, args) if pool is not None else [phi_upper(*a) for a in args]
        cache.update(zip(new, vals))

    # nodes as exact dyadic fractions of T; phi (nondecreasing) is evaluated at ends rounded up
    def up(fr):
        return fup(T * fr) if fr > 0 else 0.0

    ivs = [(i / (2 * n0), (i + 1) / (2 * n0)) for i in range(n0)]     # fractions of T, a_L <= T/2
    worst = math.inf
    for rnd in range(max_rounds):
        need([up(b) for _, b in ivs] + [up(1 - a) for a, _ in ivs])
        bad, worst = [], -math.inf
        for a, b in ivs:
            v = A(cache[up(b)]) + A(cache[up(1 - a)])    # exact at 128 bits
            worst = max(worst, fup(v))
            if not v < budget:
                bad.append((a, b))
        if not bad:
            return worst, fdown(budget), True, dict(rounds=rnd, evals=len(cache))
        ivs = [iv for iv in ivs if iv not in bad] + \
              [h for a, b in bad for h in ((a, (a + b) / 2), ((a + b) / 2, b))]
    return worst, fdown(budget), False, dict(rounds=max_rounds, evals=len(cache), bad=bad[:5])
