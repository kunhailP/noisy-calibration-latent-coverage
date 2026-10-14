"""Noisy levels for contaminated bi-log-concave latent laws (notes/contamination.md).

For latent laws (1 - c) G0 + c H with G0 bi-log-concave and H arbitrary, and Gaussian or
symmetric unimodal noise (beta = 1/2), the noisy level that guarantees latent coverage q is

    p_c(q) = (1 - c) Psi(q / (1 - c)) + c / 2,          1/e < q/(1 - c) < 1,

and the shape-free level (1 + q)/2 for c >= 1 - q. It is exact under the condition
Psi'(lambda((p_c - c)/(1 - c))) >= 1/2, which `condition_holds` checks; the condition holds for every
c whenever q >= Psi(q_beta), Psi'(q_beta) = 1/2 (0.52634 Gaussian, 0.52267 symmetric unimodal), so the
functions here require q >= Q_MIN. Below that the closed form can be too low (an invalid level).
"""
import math

from scipy.optimize import brentq
from scipy.stats import beta as beta_dist

from latentcov.gaussian import psi_gaussian
from latentcov.noise_classes import psi_su_enclosure

BETA = 0.5
Q_MIN = 0.53     # above Psi(q_beta) for both classes


def psi(q, noise):
    if q <= math.exp(-1):
        return 0.5
    if q >= 1:
        return 1.0
    if noise == 'gaussian':
        return psi_gaussian(q)
    if noise == 'symmetric_unimodal':
        lo, hi = psi_su_enclosure(q)
        return (lo + hi) / 2
    raise ValueError(f'contamination levels need gaussian or symmetric_unimodal noise, got {noise!r}')


def level(q, c, noise='gaussian'):
    """p_c(q); double precision."""
    if not 0 <= c < 1:
        raise ValueError(f'c must lie in [0, 1), got {c}')
    if not Q_MIN <= q < 1:
        raise ValueError(f'the closed form is proved for q >= {Q_MIN}, got {q}')
    if c >= 1 - q:
        return 1 - BETA * (1 - q)
    q0 = q / (1 - c)
    if q0 <= math.exp(-1):
        raise ValueError('the formula needs q/(1 - c) > 1/e')
    return (1 - c) * psi(q0, noise) + c * (1 - BETA)


def condition_holds(q, c, noise='gaussian', h=1e-6):
    """Condition (C): Psi'(lambda((p_c - c)/(1 - c))) >= 1/2, by a central difference."""
    if c >= 1 - q or c == 0:
        return True
    p = level(q, c, noise)
    P = (p - c) / (1 - c)
    if P <= 0.5:
        return False
    lam = brentq(lambda x: psi(x, noise) - P, math.exp(-1) + 1e-9, 1 - 1e-12, xtol=1e-14)
    return (psi(lam + h, noise) - psi(lam - h, noise)) / (2 * h) >= BETA


def tolerated(K, k, q, delta=0.05, noise='gaussian'):
    """Largest contamination c* for which rank k is valid: root of p_c(q) = p_k, with p_k the
    delta-quantile of Beta(k, K + 1 - k). Returns None if k is not valid even for c = 0, and 1.0
    if k is valid for every c in [0, 1) (p_k at or above the shape-free level)."""
    pk = float(beta_dist.ppf(delta, k, K + 1 - k))
    if pk < level(q, 0.0, noise):
        return None
    if pk >= 1 - BETA * (1 - q):
        return 1.0
    return brentq(lambda c: level(q, c, noise) - pk, 0.0, 1 - q - 1e-12, xtol=1e-12)
