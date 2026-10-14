import math

import pytest
from scipy import stats

from latentcov import level, rank, rank_for_level
from latentcov.rank import min_calibration_size


def brute(K, p, delta=0.05):
    for k in range(1, K + 1):
        if stats.binom.cdf(k - 1, K, p) >= 1 - delta:
            return k
    return None


@pytest.mark.parametrize('K', [1, 5, 28, 29, 30, 46, 110, 287, 1000, 3001])
@pytest.mark.parametrize('p', [0.8, 0.9, 0.901, 0.9017873, 0.95])
def test_rank_for_level_matches_brute_force(K, p):
    assert rank_for_level(K, p) == brute(K, p)


def test_table1_of_the_biometrika_version():
    """Ranks of Table 1 at q = 0.9, delta = 0.05."""
    cases = [  # (noise, latent, K = 1000, K = 10000) as (necessary, sufficient)
        ('symmetric_unimodal', 'symmetric_unimodal', (916, 916), (9050, 9050)),
        ('gaussian', 'bi_log_concave', (917, 917), (9058, 9058)),       # exact now; was 9058-9060
        ('symmetric_unimodal', 'bi_log_concave', (918, 918), (9068, 9068)),
        ('mean_zero_log_concave', 'bi_log_concave', (921, 922), (9101, 9109)),
        ('gaussian', 'none', (962, 962), (9537, 9537)),
        ('mean_zero_log_concave', 'none', (974, 974), (9664, 9664)),
    ]
    for noise, latent, r3, r4 in cases:
        for K, want in ((1000, r3), (10000, r4)):
            r = rank(K, 0.9, 0.05, noise, latent)
            assert (r.k_necessary, r.k) == want, (noise, latent, K)


def test_smallest_calibration_sizes():
    assert min_calibration_size(level('gaussian', 0.9).sufficient) == 29
    assert min_calibration_size(level('gaussian', 0.95).sufficient) == 59
    assert min_calibration_size(level('symmetric_unimodal', 0.9).sufficient) == 29
    assert min_calibration_size(level('mean_zero_log_concave', 0.9).sufficient) == 31
    assert min_calibration_size(level('gaussian', 0.9, 'none').sufficient) == 59
    assert min_calibration_size(level('mean_zero_log_concave', 0.9, 'none').sufficient) == 80
    assert rank(28, 0.9).k is None and rank(29, 0.9).k == 29


def test_levels_are_ordered_and_shape_free_values():
    for q in (0.8, 0.9, 0.95):
        g, su = level('gaussian', q), level('symmetric_unimodal', q)
        assert q < g.necessary < g.sufficient < g.necessary + 1e-15
        assert g.sufficient <= su.necessary          # Gaussian noise is symmetric unimodal
        assert level('gaussian', q, 'none').sufficient == pytest.approx((1 + q) / 2)
    assert level('mean_zero_log_concave', 0.9, 'none').sufficient == pytest.approx(1 - 0.1 / math.e)
    assert level('symmetric_unimodal', 0.9).necessary == pytest.approx(0.9017873124, abs=1e-9)


def test_unknown_inputs_raise():
    with pytest.raises(ValueError):
        level('mean_zero_log_concave', 0.85)         # no certified level stored
    with pytest.raises(ValueError):
        level('laplace', 0.9)
    with pytest.raises(ValueError):
        level('mean_zero_log_concave', 0.9, 'symmetric_unimodal')
    with pytest.raises(ValueError):
        rank_for_level(0, 0.9)


def test_one_sided_levels():
    for noise in ('gaussian', 'symmetric_unimodal'):
        one, two = level(noise, 0.9, sided='one'), level(noise, 0.9)
        assert (one.necessary, one.sufficient) == (two.necessary, two.sufficient)
    lc = level('mean_zero_log_concave', 0.9, sided='one')
    assert lc.sufficient == 0.9052 < level('mean_zero_log_concave', 0.9).sufficient
    r = rank(10_000, 0.9, noise='mean_zero_log_concave', sided='one')
    assert (r.k_necessary, r.k) == (9101, 9101)          # bracket narrow enough: exact here
    with pytest.raises(ValueError):
        level('gaussian', 0.9, sided='left')


def test_one_sided_anderson_needs_q_at_least_one_half():
    # for t < 0, pr(W + e <= t) >= pr(W <= t): the noisy threshold is anti-conservative
    with pytest.raises(ValueError):
        level('gaussian', 0.2, 'symmetric_unimodal', sided='one')
    assert level('gaussian', 0.5, 'symmetric_unimodal', sided='one').sufficient == 0.5
    assert level('gaussian', 0.2, 'symmetric_unimodal').sufficient == 0.2
