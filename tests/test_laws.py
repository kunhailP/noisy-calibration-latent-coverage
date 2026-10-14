import numpy as np
import pytest
from scipy import stats

from latentcov.interval import bimodal_blc_certificate
from latentcov.laws import (BLC, DESIGNS, GAUSS, LATENT, LCN, MEDZ, SU, bimodal_params,
                            design_classes, draw_noise)


@pytest.mark.parametrize('name', list(LATENT))
def test_latent_laws_standardized_and_cdf_matches_sampler(name):
    law = LATENT[name]
    x = law.rvs(200_000, np.random.default_rng(1))
    assert abs(x.mean()) < 0.02 and abs(x.var() - 1) < 0.05
    assert stats.kstest(x[:20_000], law.cdf).pvalue > 1e-3


def test_bimodal_law_is_certified_bilogconcave_and_bimodal():
    ok, worst, _ = bimodal_blc_certificate(*bimodal_params())
    assert ok and worst < 0
    p, m1, m2, s = bimodal_params()
    w = np.linspace(m1 - 1, m2 + 1, 4001)
    f = p * stats.norm.pdf(w, m1, s) + (1 - p) * stats.norm.pdf(w, m2, s)
    assert ((np.diff(np.sign(np.diff(f))) != 0).sum()) == 3      # two modes and an antimode
    assert LATENT['bimodal'].classes == {BLC}


@pytest.mark.parametrize('design', list(DESIGNS))
def test_noise_designs_have_unit_variance(design):
    e = draw_noise(design, np.full(400_000, 1.0), np.random.default_rng(2))
    assert abs(e.mean()) < 0.01 and abs(e.var() - 1) < 0.04


def test_design_memberships():
    assert design_classes('Gaussian') == {GAUSS, SU, LCN, MEDZ}
    assert design_classes('mixed symmetric unimodal') == {SU, MEDZ}
    assert design_classes('mixed log-concave') == {LCN}
    assert design_classes('mixed across classes') == set()
