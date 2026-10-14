"""Noisy levels that transfer latent coverage q, by noise class and latent shape assumption.

For a two-sided interval [-t, t] and a class E of noise laws, the smallest valid conformal rank
(Theorem 2 of the Biometrika version) lies between the ranks given by two levels:

    necessary   a certified lower bound on Psi_E(q): no smaller level works for every latent law
                in the class (converse of Theorem 2);
    sufficient  a certified p >= p_E(q): noisy coverage p at t gives latent coverage q at t.

When the two coincide the smallest rank is exact. Each entry records how it is supported.
"""
import math
from dataclasses import dataclass

from latentcov.gaussian import psi_gaussian_enclosure
from latentcov.noise_classes import psi_su_enclosure

PROVED = 'proved'
CERTIFIED = 'certified in ball arithmetic'
CONJECTURED = 'conjectured'


@dataclass(frozen=True)
class Level:
    noise: str
    latent: str
    q: float
    necessary: float
    sufficient: float
    support: str
    source: str

    @property
    def exact(self):
        return self.necessary == self.sufficient


# Gaussian noise, bi-log-concave latent law: p_N(q) = Psi_N(q) exactly (notes/gaussian_exact.md),
# enclosed in ball arithmetic by latentcov.gaussian. The Biometrika version only had the certified
# brackets below (legacy E27, legacy/interval_constants.json, key "slack"); kept for comparison.
LEGACY_GAUSSIAN = {0.8: (0.8043, 0.8050), 0.9: (0.90079, 0.9010), 0.95: (0.95015, 0.9502)}

# Mean-zero log-concave noise, bi-log-concave latent law (legacy E41, legacy/noise_classes.json):
# lower end the centred exponential law, upper end the certified split at p = 0.906.
_LOG_CONCAVE = {0.9: (0.9051755, 0.906)}
# One-sided: Psi_LC(0.9) itself, certified below 0.9052 (legacy E41, key "one_sided").
_LOG_CONCAVE_ONE_SIDED = {0.9: (0.9051755, 0.9052)}

NOISE_CLASSES = ('gaussian', 'symmetric_unimodal', 'mean_zero_log_concave')
LATENT_SHAPES = ('bi_log_concave', 'symmetric_unimodal', 'none')


def _check_q(q):
    if not 0 < q < 1:
        raise ValueError(f'q must lie in (0, 1), got {q}')


def level(noise, q, latent='bi_log_concave', sided='two'):
    """Level for latent coverage q under the given noise class and latent shape assumption.

    latent = 'bi_log_concave'      the paper's assumption (Theorem 1);
    latent = 'symmetric_unimodal'  with symmetric unimodal noise only: Anderson's theorem, level q;
    latent = 'none'                no shape assumption, the bound 1 - (1 - p)/beta_E (Proposition 2),
                                   attained by unimodal mean-zero latent laws.

    sided = 'two' for [-t, t]; sided = 'one' for (-inf, t], where the level is Psi_E(q) exactly
    (Proposition 1) and only log F_W concave is needed in place of bi-log-concavity. For Gaussian
    and symmetric unimodal noise the two coincide, as p_E = Psi_E. For latent = 'none' the
    one-sided level is the same 1 - beta_E (1 - q): sufficient because W > t and e >= 0 give
    W + e > t, attained by a layer of mass (1 - q)/beta_E just above t.
    """
    _check_q(q)
    q = float(q)
    if sided not in ('one', 'two'):
        raise ValueError(f"sided must be 'one' or 'two', got {sided!r}")
    if latent == 'symmetric_unimodal':
        if noise not in ('gaussian', 'symmetric_unimodal'):
            raise ValueError("Anderson's theorem needs symmetric unimodal noise")
        # one-sided: pr(W + e <= t) = 1/2 + sign(t) pr(|W + e| <= |t|)/2, so Anderson makes the
        # noisy threshold conservative only for t >= 0, i.e. q >= 1/2; below, it is anti-conservative
        if sided == 'one' and q < 0.5:
            raise ValueError('one-sided Anderson level needs q >= 1/2')
        return Level(noise, latent, q, q, q, PROVED, 'Anderson (1955)')
    if latent == 'none':
        beta = {'gaussian': 0.5, 'symmetric_unimodal': 0.5,
                'mean_zero_log_concave': math.exp(-1)}.get(noise)
        if beta is None:
            raise ValueError(f'unknown noise class {noise!r}')
        p = 1 - beta * (1 - q)
        return Level(noise, latent, q, p, p, PROVED,
                     'Proposition 2' + (' (one-sided analogue)' if sided == 'one' else ''))
    if latent != 'bi_log_concave':
        raise ValueError(f'unknown latent shape {latent!r}; choose from {LATENT_SHAPES}')
    if noise == 'symmetric_unimodal':
        if not q > math.exp(-1):
            raise ValueError('the closed form needs q > 1/e')
        lo, hi = psi_su_enclosure(q)
        return Level(noise, latent, q, lo, hi, PROVED + '; ' + CERTIFIED,
                     'Theorem 1(i), Supplementary Proposition S1')
    if noise == 'gaussian':
        if not q > math.exp(-1):
            raise ValueError('the closed form needs q > 1/e')
        lo, hi = psi_gaussian_enclosure(q)
        return Level(noise, latent, q, lo, hi, PROVED + '; ' + CERTIFIED,
                     'Gaussian theorem, notes/gaussian_exact.md')
    if noise != 'mean_zero_log_concave':
        raise ValueError(f'unknown noise class {noise!r}; choose from {NOISE_CLASSES}')
    values, source = ((_LOG_CONCAVE, 'Theorem 1(ii), legacy E41') if sided == 'two' else
                      (_LOG_CONCAVE_ONE_SIDED, 'Theorem 1(ii) one-sided, legacy E41'))
    key = next((k for k in values if math.isclose(k, q, abs_tol=1e-12)), None)
    if key is None:
        raise ValueError(f'no certified level for {noise} noise at q = {q}; '
                         f'available: {sorted(values)}')
    lo, hi = values[key]
    return Level(noise, latent, q, lo, hi, CERTIFIED, source)


def psi_gaussian_float(q):
    """(Psi_N(q), maximiser s*) by direct numerical maximisation in double precision; an
    independent check of the closed form in latentcov.gaussian, which is the certified value."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import log_ndtr, ndtr
    _check_q(q)
    a = -math.log(q)

    def f(s):
        return ndtr(-a / s) + math.exp(math.log(q) + s * s / 2 + log_ndtr(a / s - s))
    ss = np.geomspace(1e-7, 1e4, 6000)         # s* -> inf as q -> 1/e and -> 0 as q -> 1
    i = int(np.argmax([f(s) for s in ss]))
    lo, hi = math.log(ss[max(i - 1, 0)]), math.log(ss[min(i + 1, len(ss) - 1)])
    r = minimize_scalar(lambda ls: -f(math.exp(ls)), bounds=(lo, hi), method='bounded',
                        options={'xatol': 1e-14})
    return -r.fun, math.exp(r.x)
