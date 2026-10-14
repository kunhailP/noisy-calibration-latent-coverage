import pytest

from latentcov.contamination import psi
from latentcov.misspec import standard_noises, two_end_lower_bound

NOISES = {n.name: n for n in standard_noises()}


@pytest.mark.parametrize('name,cls', [('Gaussian', 'gaussian'), ('uniform', 'symmetric_unimodal')])
def test_single_law_reproduces_class_level(name, cls):
    """The extremal law of each class attains the class level."""
    level, _, _ = two_end_lower_bound(0.9, NOISES[name])
    assert level == pytest.approx(psi(0.9, cls), abs=2e-7)


def test_centred_exponential_and_two_point():
    assert two_end_lower_bound(0.9, NOISES['centred exponential'])[0] == pytest.approx(0.9051755, abs=2e-7)
    assert two_end_lower_bound(0.9, NOISES['two-point symmetric'])[0] == pytest.approx((1 + 0.81) / 2, abs=1e-9)
