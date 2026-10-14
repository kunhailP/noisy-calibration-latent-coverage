"""Conformal ranks for latent coverage from noisy scores (Theorem 2 of the Biometrika version).

With T = |V|_(k) the k-th smallest noisy score among K calibration units,

    pr_D{ pr(|W_new| <= T | D) >= q } >= pr{ Bin(K, level) <= k - 1 },

for every latent law in the class and every noise law in the class, with laws and scales that may
differ between units and are unknown. The rank of a level is the smallest k with
pr{Bin(K, level) <= k - 1} >= 1 - delta, or None when even k = K fails.

Binomial tails are evaluated in double precision; within 1e-10 of 1 - delta they are recomputed
with mpmath at 50 digits, so a near-tie cannot flip the rank through rounding.
"""
import math
from dataclasses import dataclass

from scipy import stats

from latentcov.levels import Level, level as _level

_TIE = 1e-10


def _binom_cdf(k, K, p):
    """pr{Bin(K, p) <= k}, with a high-precision recheck near ties handled by the caller."""
    return float(stats.binom.cdf(k, K, p))


def _binom_cdf_mp(k, K, p):
    import mpmath
    with mpmath.workdps(50):
        # pr{Bin(K, p) <= k} = I_{1-p}(K - k, k + 1)
        return mpmath.betainc(K - k, k + 1, 0, 1 - mpmath.mpf(p), regularized=True)


def _ok(k, K, p, delta):
    v = _binom_cdf(k - 1, K, p)
    if abs(v - (1 - delta)) < _TIE:
        import mpmath
        with mpmath.workdps(50):
            return _binom_cdf_mp(k - 1, K, p) >= 1 - mpmath.mpf(delta)
    return v >= 1 - delta


def rank_for_level(K, p, delta=0.05):
    """Smallest k in 1..K with pr{Bin(K, p) <= k - 1} >= 1 - delta, or None."""
    if K < 1:
        raise ValueError(f'K must be positive, got {K}')
    if not 0 < delta < 1:
        raise ValueError(f'delta must lie in (0, 1), got {delta}')
    if not 0 <= p <= 1:
        raise ValueError(f'level must lie in [0, 1], got {p}')
    if p >= 1 or not _ok(K, K, p, delta):
        return None
    lo, hi = 0, K                     # _ok(lo) false (k = 0 never covers), _ok(hi) true
    guess = int(stats.binom.ppf(1 - delta, K, p)) + 1
    for k in (guess - 2, guess + 2):
        if 1 <= k <= K:
            if _ok(k, K, p, delta):
                hi = min(hi, k)
            else:
                lo = max(lo, k)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _ok(mid, K, p, delta):
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class Rank:
    K: int
    delta: float
    level: Level
    k_necessary: int | None     # no smaller rank is valid for every law in the class
    k: int | None               # valid rank (Theorem 2); None if no rank is valid

    @property
    def exact(self):
        return self.k == self.k_necessary


def rank(K, q=0.9, delta=0.05, noise='gaussian', latent='bi_log_concave', sided='two'):
    """Valid rank for latent coverage q with reliability 1 - delta.

    For sided = 'two' the scores are |V_i| and the interval is [-T, T]; for sided = 'one' the
    scores are the signed V_i and the interval is (-inf, T].

    `k` uses the certified sufficient level and is the rank to report; `k_necessary` uses the
    certified lower bound on Psi_E(q), so the smallest valid rank lies in [k_necessary, k]."""
    lv = _level(noise, q, latent, sided)
    return Rank(K, delta, lv, rank_for_level(K, lv.necessary, delta),
                rank_for_level(K, lv.sufficient, delta))


def min_calibration_size(p, delta=0.05):
    """Smallest K for which a valid rank exists at level p: pr{Bin(K, p) <= K - 1} = 1 - p^K."""
    if not 0 < p < 1:
        return None
    return math.ceil(math.log(delta) / math.log(p))


def marginal_rank(K, q):
    """Split-conformal rank ceil(q (K + 1)) (marginal coverage of the noisy score)."""
    return math.ceil(q * (K + 1))
