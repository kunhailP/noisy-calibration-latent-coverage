import math

import numpy as np
import pytest
from scipy.optimize import brentq

from latentcov.contamination import condition_holds, level, psi, tolerated
from latentcov.rank import rank_for_level


@pytest.mark.parametrize('noise', ['gaussian', 'symmetric_unimodal'])
def test_ends_and_monotone(noise):
    for q in (0.8, 0.9):
        assert level(q, 0.0, noise) == pytest.approx(psi(q, noise), abs=1e-15)
        assert level(q, 1 - q - 1e-12, noise) == pytest.approx((1 + q) / 2, abs=1e-9)
        cs = np.linspace(0, 1 - q - 1e-9, 40)
        vals = [level(q, c, noise) for c in cs]
        assert all(b >= a - 1e-14 for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize('noise', ['gaussian', 'symmetric_unimodal'])
@pytest.mark.parametrize('q,c', [(0.9, 0.01), (0.8, 0.05)])
def test_closed_form_is_the_infimum(noise, q, c):
    """At p = p_c(q) the infimum over the split of noisy coverage equals q, at the layer x = 1/2."""
    assert condition_holds(q, c, noise)
    p = level(q, c, noise)
    lam = lambda P: 0.0 if P <= 0.5 else brentq(lambda x: psi(x, noise) - P,
                                                math.exp(-1) + 1e-9, 1 - 1e-13, xtol=1e-13)
    xs = np.unique(np.concatenate([np.linspace(0, 1, 301), [0.5]]))
    J = [(1 - c) * lam((p - c * x) / (1 - c)) + c * max(0.0, 1 - 2 * (1 - x)) for x in xs]
    i = int(np.argmin(J))
    assert J[i] == pytest.approx(q, abs=1e-9)
    assert xs[i] == pytest.approx(0.5)


def test_tolerated_contamination():
    assert tolerated(110, 105, 0.9) == pytest.approx(0.0124, abs=2e-4)
    k = rank_for_level(1000, level(0.9, 0.0))
    assert 0 < tolerated(1000, k, 0.9) < 0.002
    assert tolerated(1000, rank_for_level(1000, 0.9), 0.9) is None      # usual rank, K = 1000


def test_scope_and_edges():
    with pytest.raises(ValueError):
        level(0.4, 0.3)                       # closed form too low there (brute force 0.65)
    assert tolerated(1000, 962, 0.9) == 1.0   # shape-free rank: valid for every contamination
