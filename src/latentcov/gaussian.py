"""Exact coverage transfer for Gaussian noise (notes/gaussian_exact.md).

For Z ~ N(0, 1) and a = log(1/q), Psi_N(q) = sup_{s > 0} G(q, s) with

    G(q, s) = E min(1, q e^{-s Z}) = Phi(-a/s) + q e^{s^2/2} Phi(a/s - s).

With m(b) = phi(b)/Phi(b) and A(b) = m(b){b + m(b)} = 1 - var(Z | Z <= b), which decreases
from 1 to 0, the maximiser for q > 1/e is s* = m(b*) with A(b*) = a, and

    Psi_N(q) = Phi(-c) + phi(c)/m(b*),   c = b* + m(b*);   Psi_N'(q) = e^{m(b*)^2/2} Phi(b*).

Psi_N' increases in q, so Psi_N is strictly convex on (1/e, 1) and p_N(q) = Psi_N(q). For
q <= 1/e, Psi_N(q) = 1/2. Enclosures below are computed in Arb ball arithmetic.
"""
import math

from flint import arb

from latentcov.interval import A as ball, Phi, fdown, fup, phi


def m(b):
    """Inverse Mills ratio phi(b)/Phi(b) of a ball b."""
    return phi(b) / Phi(b)


def A_of(b):
    """A(b) = m(b)(b + m(b)) = 1 - var(Z | Z <= b) for a ball b."""
    mb = m(b)
    return mb * (b + mb)


def G(q, s):
    """Ball for E min(1, q e^{-sZ}); q, s balls with s > 0."""
    a = -q.log()
    return Phi(-a / s) + q * (s * s / 2).exp() * Phi(a / s - s)


def psi_tilde(b):
    """Psi_N along the curve of maximisers: Phi(-c) + phi(c)/m(b), c = b + m(b)."""
    mb = m(b)
    c = b + mb
    return Phi(-c) + phi(c) / mb


def b_star_bounds(q, tol=1e-15):
    """Doubles lo < b* < hi with A(lo) > a > A(hi) proved in ball arithmetic (A decreases)."""
    a = -ball(q).log()
    # b* -> -inf as q -> 1/e and -> +inf as q -> 1: widen until the root is bracketed
    lo, hi = -40.0, 40.0
    while not A_of(ball(lo)) > a:
        lo *= 2
        if lo < -1e12:
            raise ValueError(f'cannot bracket b* at q = {q}: too close to 1/e')
    while not A_of(ball(hi)) < a:
        hi *= 2
        if hi > 1e12:
            raise ValueError(f'cannot bracket b* at q = {q}: too close to 1')
    while hi - lo > tol * max(1.0, abs(hi)):
        mid = (lo + hi) / 2
        v = A_of(ball(mid))
        if v > a:
            lo = mid
        elif v < a:
            hi = mid
        else:
            break
    return lo, hi


def psi_gaussian_enclosure(q):
    """Rigorous [lo, hi] for Psi_N(q), q a float or decimal string in (1/e, 1).

    Lower end: G(q, s) at s = m(b_lo), a value of the supremand. Upper end: psi_tilde(b_hi),
    which is Psi_N at q(b_hi) = e^{-A(b_hi)} > q, and Psi_N increases."""
    qa = ball(q)
    if not qa > arb(-1).exp():
        raise ValueError('the closed form needs q > 1/e (Psi_N = 1/2 below)')
    lo, hi = b_star_bounds(q)
    lower = G(qa, m(ball(lo)))
    upper = psi_tilde(ball(hi))
    return fdown(lower), fup(upper)


def psi_gaussian(q):
    """Psi_N(q) in double precision from the closed form (q in (0, 1))."""
    if q <= math.exp(-1):
        return 0.5
    lo, hi = psi_gaussian_enclosure(q)
    return (lo + hi) / 2
