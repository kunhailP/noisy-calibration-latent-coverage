import math

import numpy as np
import pytest
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm, truncnorm

from latentcov.gaussian import psi_gaussian, psi_gaussian_enclosure
from latentcov.levels import LEGACY_GAUSSIAN, psi_gaussian_float


def m(b):
    return math.exp(norm.logpdf(b) - log_ndtr(b))


def test_A_is_one_minus_truncated_variance_and_decreasing():
    for b in (-4, -1, 0, 1, 4):
        assert m(b) * (b + m(b)) == pytest.approx(1 - truncnorm(-np.inf, b).var(), abs=1e-10)
    bs = np.linspace(-12, 12, 5001)
    A = np.array([m(b) * (b + m(b)) for b in bs])
    assert np.all(np.diff(A) < 0) and A.max() < 1 and A.min() > 0


def test_derivative_increases_along_maximisers():
    bs = np.linspace(-12, 12, 5001)
    logD = np.array([m(b) ** 2 / 2 + log_ndtr(b) for b in bs])
    assert np.all(np.diff(logD) > 0)


@pytest.mark.parametrize('q', [0.3679, 0.368, 0.4, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.999999])
def test_closed_form_matches_direct_maximisation(q):
    lo, hi = psi_gaussian_enclosure(q)
    assert 0 < hi - lo < 1e-15
    # near 1/e the supremand is flat in s, so the double-precision maximum is good to ~1e-14
    assert psi_gaussian_float(q)[0] == pytest.approx((lo + hi) / 2, abs=1e-14)


def test_below_one_over_e_the_supremum_is_one_half():
    q = 0.3
    a = -math.log(q)
    G = lambda s: ndtr(-a / s) + math.exp(math.log(q) + s * s / 2 + log_ndtr(a / s - s))
    vals = [G(s) for s in np.geomspace(0.01, 500, 400)]
    assert np.all(np.diff(vals) > -1e-15) and vals[-1] < 0.5 and vals[-1] > 0.499
    assert psi_gaussian(q) == 0.5


def test_convex_on_fine_grid_and_consistent_with_legacy_brackets():
    qs = np.linspace(0.38, 0.999, 400)
    P = np.array([psi_gaussian(q) for q in qs])
    assert np.all(np.diff(P, 2) > 0)
    for q, (lo, hi) in LEGACY_GAUSSIAN.items():
        assert lo < psi_gaussian_enclosure(q)[0] and psi_gaussian_enclosure(q)[1] <= hi
