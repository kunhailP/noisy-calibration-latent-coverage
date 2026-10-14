"""Fast checks of latentcov.intermediate (R10, notes/intermediate.md)."""
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.special import ndtr

from latentcov import intermediate as I
from latentcov.gaussian import psi_gaussian
from latentcov.rank import rank_for_level

LAWS = I.latent_laws()


@pytest.mark.parametrize('name', sorted(LAWS))
def test_laws_are_standardized(name):
    from scipy.integrate import quad
    law = LAWS[name]
    pts = [k for k in law.kinks if law.lo < k < law.hi]
    m = [quad(lambda w: w ** j * law.pdf(w), law.lo, law.hi, points=pts or None, limit=400)[0]
         for j in range(3)]
    assert m[0] == pytest.approx(1, abs=1e-12)
    assert m[1] == pytest.approx(0, abs=1e-12)
    assert m[2] == pytest.approx(1, abs=1e-11)
    for w in (-0.7, 0.2, 1.3):
        hi = min(w, law.hi)
        kinks = [k for k in law.kinks if law.lo < k < hi] + [min(0.0, hi - 1)]
        assert law.cdf(w) == pytest.approx(quad(law.pdf, law.lo, hi, points=kinks,
                                                limit=400)[0], abs=1e-12)
        assert law.cdf(w) + law.sf(w) == pytest.approx(1, abs=1e-15)


@pytest.mark.parametrize('q', [0.8, 0.9, 0.95])
@pytest.mark.parametrize('rho', [0.001, 0.1, 1.0, 3.0])
def test_normal_latent_gaussian_noise_closed_form_and_anderson(q, rho):
    """W ~ N(b, 1), V ~ N(b, 1 + rho): H in closed form; for b = 0, H < q (Anderson)."""
    law = LAWS['normal']
    for b in (0.0, 0.4):
        r = I.radius(law, b, q)
        D = I.excess(law, b, r, 'gaussian', math.sqrt(rho))[0]
        s = math.sqrt(1 + rho)
        exact = ndtr((r - b) / s) - ndtr((-r - b) / s) - q
        assert D == pytest.approx(exact, abs=1e-14)
        if b == 0:
            assert D < 0
    # heterogeneous scales: Gauss-Hermite against an adaptive integral of the closed form
    r = I.radius(law, 0.0, q)
    D = I.excess_het(law, 0.0, r, 'gaussian', rho)[0]
    ref = I.excess_het_adaptive(law, 0.0, r, 'gaussian', rho)[0]
    assert D == pytest.approx(ref, abs=1e-14) and D < 0


def test_anderson_uniform_noise_symmetric_laws():
    for name in ('normal', 'Laplace'):
        law = LAWS[name]
        r = I.radius(law, 0.0, 0.9)
        for rho in (1e-4, 0.01, 1.0):
            assert I.excess(law, 0.0, r, 'uniform', math.sqrt(rho))[0] < 0


def test_bias_solves_zeta():
    for name in ('centred gamma(2)', 'reflected centred exponential', 'skew-normal(4)'):
        law = LAWS[name]
        for zeta in (0.1, 0.5):
            b = I.bias_for_zeta(law, zeta, 0.9)
            r = I.radius(law, b, 0.9)
            assert b / r == pytest.approx(zeta, abs=1e-13)
            assert law.coverage(r, b) == pytest.approx(0.9, abs=1e-15)


def test_reflected_exponential_against_mpmath():
    """W = b + 1 - E (reflected centred exponential with bias), Gaussian noise: H - q in
    mpmath (direct coverage integrals, 25 digits) against the deficit-form quadrature."""
    law = LAWS['reflected centred exponential']
    q, rho = 0.8, 0.01
    b = I.bias_for_zeta(law, 0.5, q)
    r = I.radius(law, b, q)
    s = math.sqrt(rho)
    D = I.excess(law, b, r, 'gaussian', s)[0]
    with mp.workdps(25):
        B, R, S = mp.mpf(b), mp.mpf(r), mp.mpf(s)
        edge = B + 1
        pdf = lambda w: mp.exp(w - edge)                                    # noqa: E731
        ker = lambda w: mp.ncdf((R - w) / S) - mp.ncdf((-R - w) / S)        # noqa: E731
        pts = [-mp.inf, -R - 10 * S, -R, -R + 10 * S, R - 10 * S, R, edge]
        H = mp.quad(lambda w: pdf(w) * ker(w), pts)
        lat = mp.exp(min(R, edge) - edge) - mp.exp(-R - edge)
    assert float(lat) == pytest.approx(q, abs=1e-15)
    assert D == pytest.approx(float(H - lat), abs=1e-14)
    assert D > 0                          # a hard edge just outside the threshold gains coverage
    assert q + D < psi_gaussian(q)        # and stays below the class level (Theorem 3)


def test_heterogeneous_uniform_against_monte_carlo():
    law = LAWS['centred gamma(2)']
    q, rho = 0.9, 0.3
    b = I.bias_for_zeta(law, 0.3, q)
    r = I.radius(law, b, q)
    D = I.excess_het(law, b, r, 'uniform', rho)[0]
    rng = np.random.default_rng(1)
    n = 4_000_000
    w = (rng.gamma(2.0, size=n) - 2) / math.sqrt(2) + b
    L = rng.lognormal(0, I.LOGNORMAL_S, n) / math.exp(I.LOGNORMAL_S ** 2 / 2)
    v = w + np.sqrt(3 * rho * L) * rng.uniform(-1, 1, n)
    H = np.mean(np.abs(v) < r)
    assert abs(H - (q + D)) < 5 * math.sqrt(H * (1 - H) / n)


def test_proportions_against_monte_carlo():
    a, bb, n = 1.0, 9.0, 50
    law0 = I.beta_law(a, bb)
    assert law0.log_concave and not I.beta_law(0.5, 4.5).log_concave
    c = law0.mean - 0.5 * law0.sd
    law, shift, sig = I.proportion_design(a, bb, c, n)
    r = I.radius(law, shift, 0.9)
    D = I.excess(law, shift, r, 'gaussian', sig)[0]
    rng = np.random.default_rng(2)
    m = 4_000_000
    t = rng.beta(a, bb, m)
    v = t - c + np.sqrt(t * (1 - t) / n) * rng.standard_normal(m)
    H = np.mean(np.abs(v) < r)
    assert abs(H - (0.9 + D)) < 5 * math.sqrt(H * (1 - H) / m)


def test_ranks_on_grid_match_rank_for_level():
    Ks = np.array([29, 30, 58, 59, 60, 110, 966, 1000, 2999, 10_000, 54_321, 100_000])
    for p in (0.8, 0.9, 0.95, psi_gaussian(0.9), psi_gaussian(0.95)):
        ks = I.ranks_on_grid(Ks, p)
        for K, k in zip(Ks, ks):
            want = rank_for_level(int(K), p)
            assert k == (0 if want is None else want), (K, p)
    rel = I.reliability(Ks, I.ranks_on_grid(Ks, 0.9), 0.9)
    assert np.all(rel >= 0.95)
