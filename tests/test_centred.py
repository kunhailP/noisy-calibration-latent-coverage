"""Fast checks of latentcov.centred (notes/centred.md)."""
import math

import numpy as np
import pytest

from latentcov.centred import (PLA, Placed, Standard, brute_latent, brute_noisy, decode,
                               noisy_gauss_closed, noisy_unif_quadrature, place, shape, value)
from latentcov.gaussian import psi_gaussian
from latentcov.noise_classes import psi_su_enclosure


def test_closed_forms_against_mpmath():
    law = PLA([-3.0, -0.4, 1.07], [2.5, -0.7])
    lat, mean = brute_latent(law)
    assert law.latent() == pytest.approx(float(lat), abs=1e-15)
    assert law.mean == pytest.approx(float(mean), abs=1e-15)
    for noise, s in (('gaussian', 0.05), ('su', 0.1), ('gaussian', 0.7), ('su', 1.3)):
        assert law.noisy(noise, s) == pytest.approx(float(brute_noisy(law, noise, s)), abs=2e-15)
    assert noisy_unif_quadrature(law, 0.1) == pytest.approx(law.noisy('su', 0.1), abs=1e-15)
    assert noisy_gauss_closed(law, 0.05) == pytest.approx(law.noisy('gaussian', 0.05), abs=1e-14)


def test_steep_and_flat_pieces():
    for xs, bs in (([-1.5, 0.9, 1.02], [40.0, -300.0]), ([-2.0, 2.0], [0.0]),
                   ([-1.2, 0.0, 1.3], [1e-9, -1e-9])):
        law = PLA(xs, bs)
        for noise, s in (('gaussian', 0.02), ('su', 0.05)):
            assert law.noisy(noise, s) == pytest.approx(float(brute_noisy(law, noise, s)),
                                                        abs=5e-14)


def test_placement_is_feasible_and_centred():
    X = shape(0.3, 5.0, -2.0)
    for mu in (0.0, 0.4, 0.95):
        W, lam = place(X, mu, 0.9)
        assert W.latent() <= 0.9 and W.latent() == pytest.approx(0.9, abs=1e-13)
        assert W.mean == pytest.approx(mu, abs=1e-12)


def _limit_law(noise, q, s):
    """Exponential tail of Supplementary S2.2, W = 1 + s (a - E1), with the extremal noise."""
    if noise == 'gaussian':
        from latentcov.levels import psi_gaussian_float
        _, u = psi_gaussian_float(q)
        scale = s * u
    else:
        lo, hi = psi_su_enclosure(q)
        # maximising uniform half-width r solves (1 + r) e^{-r} = (1 + log q)/q
        from scipy.optimize import brentq
        r = brentq(lambda r: (1 + r) * math.exp(-r) - (1 + math.log(q)) / q, 1e-9, 50)
        scale = s * r
    a = math.log(1 / q)
    L = 60.0
    return PLA([1 + s * (a - L), 1 + s * a], [1 / s]), scale


@pytest.mark.parametrize('noise,psi', [('gaussian', psi_gaussian(0.9)),
                                        ('su', sum(psi_su_enclosure(0.9)) / 2)])
def test_kappa_infinity_limit(noise, psi):
    """kappa >= 1: the exponential tail with mean in (-1, 1) approaches Psi_E(q)."""
    W, sc = _limit_law(noise, 0.9, 1e-4)
    assert abs(W.mean) < 1
    assert W.latent() == pytest.approx(0.9, abs=1e-12)
    assert W.noisy(noise, sc) == pytest.approx(psi, abs=2e-7)


def test_centred_exponential_q09_gaussian():
    """The centred exponential at q = 0.9 has noisy coverage <= q at every scale."""
    from scipy import stats
    X = Standard('exponential', stats.expon.cdf, 1.0, 1.0, (0.0,))
    for sign in (1, -1):
        Xs = X if sign == 1 else Standard('refl', lambda x: stats.expon.sf(-x), -1.0, 1.0, (0.0,))
        from latentcov.centred import _lam_q
        lam = _lam_q(Xs, 0.0, 0.9)
        W = Placed(Xs, 0.0, lam)
        assert W.latent() == pytest.approx(0.9, abs=1e-12)
        for sig in np.geomspace(1e-3, 1.0, 13):
            assert W.noisy('gaussian', sig) - 0.9 <= 1e-14


def test_small_noise_expansion_positive_at_kappa0():
    """Proposition of Step 5: a slightly tilted centred uniform law has excess ~ (s^2/2)
    (f'(1) - f'(-1)) > 0, so p_0(q) > q."""
    X = PLA([0.0, 1.0], [0.4])
    W, _ = place(X, 0.0, 0.9)
    f1, fm1 = float(W.pdf(1.0)), float(W.pdf(-1.0))
    beta = W.betas[0]
    slope_gap = beta * (f1 - fm1)
    assert slope_gap > 0
    for s in (0.01, 0.005):
        ex = W.noisy('gaussian', s) - 0.9
        assert ex > 0
        assert ex == pytest.approx(s * s / 2 * slope_gap, rel=0.05)
        exu = W.noisy('su', s) - 0.9
        assert exu == pytest.approx(s * s / 6 * slope_gap, rel=0.05)


def test_value_monotone_in_kappa_at_fixed_shape():
    """Raising kappa only enlarges the feasible set: the same theta (same shape, same relative
    position mu/min(kappa, 1)) is compared through the supremum over mu on a grid."""
    th = np.array([1.0, 3.0, 0.5, 0.0, math.log(0.05)])
    best = []
    for kappa in (0.0, 0.2, 0.5):
        vals = [value(np.r_[th[:3], um, th[4]], 'su', 0.9, kappa)
                for um in np.linspace(-8, 8, 33)]
        best.append(max(vals))
    assert best[0] <= best[1] + 1e-15 <= best[2] + 2e-15


def test_reported_levels_spot_values():
    """Spot values of results/centred_levels.csv, re-evaluated from the stored maximisers."""
    import csv
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / 'results' / 'centred_levels.csv'
    if not f.exists():
        pytest.skip('run make centred first')
    rows = list(csv.DictReader(open(f)))
    by = {(r['noise'], float(r['q']), float(r['kappa'])): r for r in rows}
    col = 'p_kappa_minus_q' if 'p_kappa_minus_q' in rows[0] else 'brute_excess'
    for noise in ('gaussian', 'su'):
        for q in (0.8, 0.9, 0.95):
            ex = [float(by[(noise, q, k)][col]) for k in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75)]
            ex.append(float(by[(noise, q, 1.0)]['excess']))          # Psi_E(q) - q, proved
            assert all(b >= a - 1e-12 for a, b in zip(ex, ex[1:]))
            assert ex[0] > 0
        r = by[(noise, 0.9, 0.0)]
        xs = [float(v) for v in r['xs'].split(';')]
        bs = [float(v) for v in r['betas'].split(';')]
        W = PLA(xs, bs, check=False)
        assert W.latent() <= 0.9 + 1e-15 and abs(W.mean) <= 1e-12
        assert W.noisy(noise, float(r['scale'])) - 0.9 == pytest.approx(float(r['brute_excess']),
                                                                       abs=1e-13)
