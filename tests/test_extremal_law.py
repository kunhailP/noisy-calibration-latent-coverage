import sys
from pathlib import Path

import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from r01_reliability_curves import ExtremalLaw  # noqa: E402

from latentcov.levels import level  # noqa: E402
from latentcov.rank import rank_for_level  # noqa: E402


@pytest.mark.parametrize('q', [0.8, 0.9, 0.95])
def test_noisy_coverage_at_latent_quantile_is_psi(q):
    law = ExtremalLaw(q)
    H = law.H(law.r_q())
    assert H == pytest.approx(law.psi, abs=1e-10)
    g = level('gaussian', q)
    assert H == pytest.approx(g.sufficient, abs=1e-12)      # the extremal law attains Psi_N(q)


def test_usual_rank_fails_and_certified_rank_holds_at_q08():
    law = ExtremalLaw(0.8)
    H = law.H(law.r_q())
    K = 100_000
    usual = stats.binom.cdf(rank_for_level(K, 0.8) - 1, K, H)
    cert = stats.binom.cdf(rank_for_level(K, level('gaussian', 0.8).sufficient) - 1, K, H)
    assert usual < 0.05 and cert > 0.95
