"""Ball-arithmetic enclosures and branch and bound for the centred level p^0_N(q)
(notes/centred_certificate.md; Gaussian noise, kappa = 0).

Domain (proved reductions in the note): W = Y - E Y with Y one of
  form 'M'  Y on [0, L1 + L2], log-density -r1 y on [0, L1], -r1 L1 - r2 (y - L1) on
            [L1, L1 + L2], 0 <= r1 <= r2 (mode at the left end; increasing laws by reflection);
  form 'T'  Y on [-Ll, Lr], log-density a y (y < 0), -g y (y > 0), a, g >= 0 (mode at the kink,
            which is at W = -E Y; reflection lets us take E Y <= 0);
  form 'P'  Y on [0, L], log-density -r y (one piece);
two-piece laws (M, T) with pr(|W| < 1) = q exactly, one-piece laws (P) with pr(|W| < 1) <= q,
and Gaussian noise sigma Z, sigma in [sigma_0, sigma_1]; the ends of sigma are analytic.

Objective. With D = H - pr(|W| < 1) the excess of the noisy over the latent coverage and
cov = pr(|W| < 1), every feasible law has H - q <= D + lam (cov - q) for every lam in [0, 1]
(cov <= q), with equality on cov = q for every lam. In the unnormalized density g = e^{ell}
(ell = 0 at the mode, Z = int g), D = (T(p_-) - T(p_+))/Z with p_+- = E Y +- 1 and

    T(p) = int g(y) K_p(y) dy,  K_p(y) = Phibar((p - y)/sigma) (y < p), -Phibar((y - p)/sigma) (y > p),

and cov - q = ((1 - q) Z - M_out)/Z with M_out the mass outside (p_-, p_+). On a piece decaying
at rate rho >= 0 from its end h, with n = clamp(p, h, h + len) and ell(n) its log-density,

    int_piece g K_p = sigma e^{ell(n)} {J((p - b)^+/s, -rho s, (n - h)/s) - J((h - p)^+/s, rho s, (b - n)/s)}

(b = h + len, s = sigma), J(d, rho, M) = int_0^M e^{-rho t} Phibar(d + t) dt, which decreases in d
and rho and increases in M. E Y and Z are monotone in every coordinate (association
inequality), so on a box they are enclosed by two (four) point evaluations at corners; J and the
masses A(rho, l) = int_0^l e^{-rho t} dt are enclosed at corners of their argument boxes. All
point evaluations are Arb balls; floating point (the `_f` functions) only chooses split
directions. A box is closed when the upper end of the enclosure of the objective is below
eps = target - q, or when the coverage enclosure excludes q (two-piece) or lies above q (one-piece).
"""
import heapq
import math
import time

from flint import arb, ctx
from scipy.special import log_ndtr

ctx.prec = 128

INF = math.inf
ZERO = arb(0)
ONE = arb(1)
SQRT2 = arb(2).sqrt()
INV_SQRT2PI = 1 / (2 * arb.pi()).sqrt()
TINY = arb(2) ** -40             # |rho| below this is replaced in J (monotone, so still valid)


def _isinf(x):
    return isinstance(x, float) and math.isinf(x)


def A_(x):
    """Exact ball of a double, or INF passed through."""
    return x if _isinf(x) else arb(x)


# ------------------------------------------------------------------ point evaluations (balls)
def _pbar(x):
    return (x / SQRT2).erfc() / 2


def _phi(x):
    return (-(x * x) / 2).exp() * INV_SQRT2PI


def _psi(x):
    """int_x^inf Phibar = phi(x) - x Phibar(x)."""
    return _phi(x) - x * _pbar(x)


def _pbar_diff(a, b):
    """Phibar(a) - Phibar(b) for a <= b, without cancellation in the far tails."""
    if b <= 0:
        return _pbar(-b) - _pbar(-a)
    if a >= 0:
        return _pbar(a) - _pbar(b)
    return 1 - _pbar(-a) - _pbar(b)


def _J(d, rho, M):
    """J(d, rho, M) = int_0^M e^{-rho t} Phibar(d + t) dt at a point (balls; M may be INF, rho
    may be +-INF). Closed form by parts:
      J = {Phibar(d) - e^{-rho M} Phibar(d + M) - e^{rho d + rho^2/2} [Phibar(d + rho) - Phibar(d + rho + M)]}/rho."""
    if not _isinf(M) and M.is_zero():
        return ZERO
    if _isinf(rho):
        if rho > 0:
            return ZERO
        return INF
    if _isinf(d):
        return ZERO
    if rho.is_zero():
        return _psi(d) if _isinf(M) else _psi(d) - _psi(d + M)
    g = (rho * d + rho * rho / 2).exp()
    if _isinf(M):
        return (_pbar(d) - g * _pbar(d + rho)) / rho
    return (_pbar(d) - (-rho * M).exp() * _pbar(d + M) - g * _pbar_diff(d + rho, d + rho + M)) / rho


def J_up(d, rho, M):
    """Upper bound of J at a point; rho may be moved down, d down, M up (J is monotone)."""
    if not _isinf(M) and M <= 0:
        return ZERO
    if not _isinf(rho) and abs(rho) < TINY:
        rho = ZERO if rho >= 0 else -TINY
    v = _J(d, rho, M)
    return v if _isinf(v) else v.upper()


def J_dn(d, rho, M):
    if not _isinf(M) and M <= 0:
        return ZERO
    if not _isinf(rho) and abs(rho) < TINY:
        rho = ZERO if rho <= 0 else TINY
    v = _J(d, rho, M)
    if _isinf(v):
        raise ValueError('J_dn infinite')
    return v.lower()


def _A(rho, L):
    """int_0^L e^{-rho t} dt at a point, rho >= 0 (INF allowed), L >= 0 (INF allowed)."""
    if _isinf(rho):
        return ZERO
    if _isinf(L):
        return INF if rho.is_zero() else 1 / rho
    if L.is_zero():
        return ZERO
    if rho.is_zero():
        return L
    return -(-rho * L).expm1() / rho


def _E2(x):
    """int_0^1 s e^{-x s} ds = (1 - e^{-x}(1 + x))/x^2, x >= 0 a ball (series for small x)."""
    if x < arb(2) ** -12:
        s, t = ZERO, ONE                      # sum (-x)^n / (n! (n + 2))
        for n in range(14):
            s += t / (n + 2)
            t = -t * x / (n + 1)
        return s + arb(0, (abs(x) ** 14).upper() * 2)
    return (-(-x).expm1() - x * (-x).exp()) / (x * x)


def _B(rho, L):
    """int_0^L t e^{-rho t} dt at a point."""
    if _isinf(rho):
        return ZERO
    if _isinf(L):
        return INF if rho.is_zero() else 1 / (rho * rho)
    if L.is_zero():
        return ZERO
    return L * L * _E2(rho * L)


def A_up(rho, L):
    v = _A(rho, L)
    return v if _isinf(v) else v.upper()


def A_dn(rho, L):
    v = _A(rho, L)
    if _isinf(v):
        raise ValueError('A_dn infinite')
    return v.lower()


# ------------------------------------------------------------------ forms: E Y and Z at points
def ey_z_point(form, c):
    """(E Y, Z) at a point; c is a tuple of exact balls or INF."""
    if form == 'M':
        r1, r2, L1, L2 = c
        m1, mu1 = _A(r1, L1), _B(r1, L1)
        if _isinf(L1):
            m2 = mu2 = ZERO
        else:
            e1 = (-r1 * L1).exp()
            a2 = _A(r2, L2)
            m2 = e1 * a2
            mu2 = e1 * (L1 * a2 + _B(r2, L2))
        Z = m1 + m2
        return (mu1 + mu2) / Z, Z
    if form == 'T':
        a, g, Ll, Lr = c
        ml, mr = _A(a, Ll), _A(g, Lr)
        Z = ml + mr
        return (_B(g, Lr) - _B(a, Ll)) / Z, Z
    if form == 'P':
        r, L = c
        Z = _A(r, L)
        return _B(r, L) / Z, Z
    raise ValueError(form)


# Corner rules (lower-corner index pattern): E Y and Z are monotone in every coordinate.
#   M: E Y and Z decrease in r1, r2 and increase in L1, L2.
#   T: E Y increases in a, Lr and decreases in g, Ll; Z decreases in a, g, increases in Ll, Lr.
#   P: E Y and Z decrease in r and increase in L.
_EY_DIR = {'M': (-1, -1, 1, 1), 'T': (1, -1, -1, 1), 'P': (-1, 1)}
_Z_DIR = {'M': (-1, -1, 1, 1), 'T': (-1, -1, 1, 1), 'P': (-1, 1)}


def _corner(box, dirs, low):
    """Corner minimizing (low=True) or maximizing a function with the given monotone directions."""
    out = []
    for (lo, hi), s in zip(box, dirs):
        take_lo = (s > 0) == low
        out.append(A_(lo) if take_lo else A_(hi))
    return tuple(out)


def ey_z_enclosure(form, box):
    """((EY_lo, EY_hi), (Z_lo, Z_hi)) as exact balls (upper ends may be INF)."""
    e_lo, z_lo = ey_z_point(form, _corner(box, _EY_DIR[form], True))
    e_hi, z_hi = ey_z_point(form, _corner(box, _EY_DIR[form], False))
    if _EY_DIR[form] != _Z_DIR[form]:
        _, z_lo = ey_z_point(form, _corner(box, _Z_DIR[form], True))
        _, z_hi = ey_z_point(form, _corner(box, _Z_DIR[form], False))
    lo = lambda v: v if _isinf(v) else v.lower()
    up = lambda v: v if _isinf(v) else v.upper()
    return (lo(e_lo), up(e_hi)), (lo(z_lo), up(z_hi))


# ------------------------------------------------------------------ interval helpers
def _dn(x):
    return x.lower()


def _up(x):
    return x.upper()


def _clamp_cap(v, cap):
    """max(0, min(v, cap)) for an exact ball v and cap (ball or INF)."""
    if not _isinf(cap) and v > cap:
        v = cap
    return v if v > 0 else ZERO


def _pos(v):
    return v if v > 0 else ZERO


def _times_pos(x_lo, x_hi, s_lo, s_hi):
    """Interval x * s for s in [s_lo, s_hi] >= 0 (x signed); exact ends, INF-free x."""
    lo = x_lo * (s_lo if x_lo >= 0 else s_hi)
    hi = x_hi * (s_hi if x_hi >= 0 else s_lo)
    return _dn(lo), _up(hi)


class Piece:
    """A piece [h, h + len] on which ell decreases at rate rho >= 0 from ell(h) = lh. Intervals
    are pairs of exact balls (len and rho upper ends may be INF). mirror=True means the actual
    piece is the reflection y -> -y of this one (it decays towards -y from -h)."""
    __slots__ = ('h', 'lh', 'rho', 'ln', 'mirror', 'crude_mass')

    def __init__(self, h, lh, rho, ln, mirror=False, crude_mass=None):
        self.h, self.lh, self.rho, self.ln, self.mirror = h, lh, rho, ln, mirror
        self.crude_mass = crude_mass          # set -> only the crude bounds are used


def _exp_lo(x):
    return ZERO if _isinf(x) else _dn(x.exp())


def _exp_up(x):
    return INF if _isinf(x) else _up(x.exp())


def _piece_eval(pc, p_lo, p_hi, s_lo, s_hi):
    """Enclosures of int g K_p, of the mass below p and of the mass above p for one piece
    (non-mirrored orientation). Returns six exact balls (C_lo, C_hi, bl_lo, bl_hi, ab_lo, ab_hi)."""
    h_lo, h_hi = pc.h
    lh_lo, lh_hi = pc.lh
    r_lo, r_hi = pc.rho
    n_lo, n_hi = pc.ln
    elh_lo, elh_hi = _exp_lo(lh_lo), _exp_up(lh_hi)
    m_hi = A_up(r_lo, n_hi)
    m_hi = INF if _isinf(m_hi) else _up(elh_hi * m_hi)
    if pc.crude_mass is not None:
        mh = pc.crude_mass
        return _dn(-mh / 2), _up(mh / 2), ZERO, mh, ZERO, mh
    # x = clamp(p - h, 0, len), y = clamp(h + len - p, 0, len)
    x_lo = _clamp_cap(_dn(p_lo - h_hi), n_lo)
    x_hi = _clamp_cap(_up(p_hi - h_lo), n_hi)
    y_lo = _clamp_cap(_dn(h_lo + n_lo - p_hi), n_lo)
    y_hi = INF if _isinf(n_hi) else _clamp_cap(_up(h_hi + n_hi - p_lo), n_hi)
    pb_lo = ZERO if _isinf(n_hi) else _pos(_dn(p_lo - h_hi - n_hi))
    pb_hi = _pos(_up(p_hi - h_lo - n_lo))
    hp_lo = _pos(_dn(h_lo - p_hi))
    hp_hi = _pos(_up(h_hi - p_lo))
    dL_lo, dL_hi = _dn(pb_lo / s_hi), _up(pb_hi / s_lo)
    ML_lo, ML_hi = _dn(x_lo / s_hi), _up(x_hi / s_lo)
    dR_lo, dR_hi = _dn(hp_lo / s_hi), _up(hp_hi / s_lo)
    MR_lo = _dn(y_lo / s_hi)
    MR_hi = INF if _isinf(y_hi) else _up(y_hi / s_lo)
    rs_lo = _dn(r_lo * s_lo)
    rs_hi = INF if _isinf(r_hi) else _up(r_hi * s_hi)
    # ell(n) = lh - rho x
    if _isinf(r_hi):
        ln_lo = -INF if x_hi > 0 else lh_lo
    else:
        ln_lo = -INF if _isinf(lh_lo) else _dn(lh_lo - r_hi * x_hi)
    ln_hi = _up(lh_hi - r_lo * x_lo)
    e_lo, e_hi = _exp_lo(ln_lo), _exp_up(ln_hi)
    # J terms
    JL_hi = J_up(dL_lo, -rs_hi if not _isinf(rs_hi) else -INF, ML_hi)
    JL_lo = J_dn(dL_hi, -rs_lo, ML_lo)
    JR_hi = J_up(dR_lo, rs_lo, MR_hi)
    JR_lo = J_dn(dR_hi, rs_hi, MR_lo)
    if _isinf(JL_hi) or _isinf(JR_hi):
        C_lo, C_hi = (-INF, INF)
    else:
        df_lo, df_hi = _dn(JL_lo - JR_hi), _up(JL_hi - JR_lo)
        se_lo, se_hi = _dn(s_lo * e_lo), _up(s_hi * e_hi)
        C_lo, C_hi = _times_pos(df_lo, df_hi, se_lo, se_hi)
    if not _isinf(m_hi):
        half = _up(m_hi / 2)
        C_lo = _dn(-half) if _isinf(C_lo) else max(C_lo, _dn(-half))
        C_hi = half if _isinf(C_hi) else min(C_hi, half)
    # masses below / above p
    bl_lo = _dn(elh_lo * A_dn(r_hi, x_lo))
    a = A_up(r_lo, x_hi)
    bl_hi = INF if _isinf(a) else _up(elh_hi * a)
    ab_lo = _dn(e_lo * A_dn(r_hi, y_lo))
    a = A_up(r_lo, y_hi)
    ab_hi = INF if (_isinf(a) or _isinf(e_hi)) else _up(e_hi * a)
    if not _isinf(m_hi):
        bl_hi = m_hi if _isinf(bl_hi) else min(bl_hi, m_hi)
        ab_hi = m_hi if _isinf(ab_hi) else min(ab_hi, m_hi)
    return C_lo, C_hi, bl_lo, bl_hi, ab_lo, ab_hi


def _pieces(form, box):
    if form == 'M':
        (r1l, r1h), (r2l, r2h), (L1l, L1h), (L2l, L2h) = [(A_(a), A_(b)) for a, b in box]
        p1 = Piece((ZERO, ZERO), (ZERO, ZERO), (r1l, r1h), (L1l, L1h))
        if _isinf(L1h):
            # piece 2 starts beyond L1_lo: crude bounds by its mass e^{-r1 L1} A(r2, L2)
            a = A_up(r2l, L2h)
            mh = INF if _isinf(a) else _up((-r1l * L1l).exp() * a)
            p2 = Piece(None, None, None, None, crude_mass=mh)
        else:
            lh = (_dn(-r1h * L1h), _up(-r1l * L1l))
            p2 = Piece((L1l, L1h), lh, (r2l, r2h), (L2l, L2h))
        return [p1, p2]
    if form == 'T':
        (al, ah), (gl, gh), (Lll, Llh), (Lrl, Lrh) = [(A_(a), A_(b)) for a, b in box]
        return [Piece((ZERO, ZERO), (ZERO, ZERO), (al, ah), (Lll, Llh), mirror=True),
                Piece((ZERO, ZERO), (ZERO, ZERO), (gl, gh), (Lrl, Lrh))]
    if form == 'P':
        (rl, rh), (Ll, Lh) = [(A_(a), A_(b)) for a, b in box]
        return [Piece((ZERO, ZERO), (ZERO, ZERO), (rl, rh), (Ll, Lh))]
    raise ValueError(form)


def _T_and_masses(pieces, p_lo, p_hi, s_lo, s_hi):
    """Enclosure of T(p) and of the masses below and above p, summed over pieces."""
    T_lo = T_hi = ZERO
    bl_lo = bl_hi = ab_lo = ab_hi = ZERO
    for pc in pieces:
        if pc.mirror:
            C_lo, C_hi, b1, b2, a1, a2 = _piece_eval(pc, _dn(-p_hi), _up(-p_lo), s_lo, s_hi)
            C_lo, C_hi = _dn(-C_hi), _up(-C_lo)
            b1, b2, a1, a2 = a1, a2, b1, b2
        else:
            C_lo, C_hi, b1, b2, a1, a2 = _piece_eval(pc, p_lo, p_hi, s_lo, s_hi)
        T_lo = -INF if (_isinf(T_lo) or _isinf(C_lo)) else _dn(T_lo + C_lo)
        T_hi = INF if (_isinf(T_hi) or _isinf(C_hi)) else _up(T_hi + C_hi)
        bl_lo, ab_lo = _dn(bl_lo + b1), _dn(ab_lo + a1)
        bl_hi = INF if (_isinf(bl_hi) or _isinf(b2)) else _up(bl_hi + b2)
        ab_hi = INF if (_isinf(ab_hi) or _isinf(a2)) else _up(ab_hi + a2)
    return (T_lo, T_hi), (bl_lo, bl_hi), (ab_lo, ab_hi)


def _sub(a, b, up):
    """a - b with INF ends (a lower/upper end, b the opposite end)."""
    if _isinf(a) or _isinf(b):
        v = (a if _isinf(a) else 0.0) - (b if _isinf(b) else 0.0)
        return v
    return _up(a - b) if up else _dn(a - b)


LAMS = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0)


def box_eval(form, box, sig, q, eps, lams=LAMS):
    """Enclosure on a box. box: list of (lo, hi) doubles (hi may be INF) in the form's
    coordinates; sig = (s_lo, s_hi) doubles; q, eps exact balls (eps = target - q).
    Returns dict(status, ub, cov) where status is 'pruned' (no feasible law in the box),
    'ok' (objective < eps proved) or 'open'; ub is an upper bound (float) of the objective
    min over lam (INF if unbounded)."""
    s_lo, s_hi = arb(sig[0]), arb(sig[1])
    (ey_lo, ey_hi), (z_lo, z_hi) = ey_z_enclosure(form, box)
    if _isinf(ey_hi) or _isinf(z_hi) or _isinf(ey_lo):
        return dict(status='open', ub=INF, why='unbounded')
    # a priori conditions (notes): M: 1/9 <= E Y <= 1.72; P: E Y >= 1/9; T: -1.222 <= E Y <= 0
    if form == 'M':
        if ey_hi < ONE / 9 or ey_lo > arb('1.72'):
            return dict(status='pruned', ub=-INF, why='EY')
    elif form == 'P':
        if ey_hi < ONE / 9:
            return dict(status='pruned', ub=-INF, why='EY')
    elif form == 'T':
        if ey_lo > 0 or ey_hi < arb('-1.222'):
            return dict(status='pruned', ub=-INF, why='EY')
    pieces = _pieces(form, box)
    pm_lo, pm_hi = _dn(ey_lo - 1), _up(ey_hi - 1)          # p_- = E Y - 1 (W = -1)
    pp_lo, pp_hi = _dn(ey_lo + 1), _up(ey_hi + 1)          # p_+ = E Y + 1 (W = +1)
    (Tm_lo, Tm_hi), (blm_lo, blm_hi), _ = _T_and_masses(pieces, pm_lo, pm_hi, s_lo, s_hi)
    (Tp_lo, Tp_hi), _, (abp_lo, abp_hi) = _T_and_masses(pieces, pp_lo, pp_hi, s_lo, s_hi)
    # coverage deficit dcov = (1 - q) Z - M_out,  M_out = mass(Y <= p_-) + mass(Y >= p_+)
    mo_lo = _dn(blm_lo + abp_lo)
    mo_hi = INF if (_isinf(blm_hi) or _isinf(abp_hi)) else _up(blm_hi + abp_hi)
    dc_lo = -INF if _isinf(mo_hi) else _dn((1 - q) * z_lo - mo_hi)
    dc_hi = _up((1 - q) * z_hi - mo_lo)
    if form in ('M', 'T'):
        if dc_lo > 0 or dc_hi < 0:
            return dict(status='pruned', ub=-INF, why='cov')
    else:
        if dc_lo > 0:
            return dict(status='pruned', ub=-INF, why='cov')
    # D numerator T(p_-) - T(p_+)
    Dn_hi = _sub(Tm_hi, Tp_lo, True)
    best = INF
    for lam in lams:
        la = arb(lam)
        if lam == 0:
            N_hi = Dn_hi
        else:
            N_hi = INF if (_isinf(Dn_hi) or _isinf(dc_hi)) else _up(Dn_hi + la * dc_hi)
        if _isinf(N_hi):
            continue
        ub = _up(N_hi / z_lo) if N_hi > 0 else _up(N_hi / z_hi)
        if ub < best if not _isinf(best) else True:
            best = ub
    if _isinf(best):
        return dict(status='open', ub=INF, why='inf')
    ok = best < eps
    return dict(status='ok' if ok else 'open', ub=float(best.upper()) if not _isinf(best) else INF,
                cov=((dc_lo, dc_hi)))


# ------------------------------------------------------------------ floating point mirror
def _pbar_f(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def _logpbar_f(x):
    return float(log_ndtr(-x))


def _psi_f(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi) - x * _pbar_f(x)


def _J_f(d, rho, M):
    if M <= 0 or rho == INF or d == INF:
        return 0.0
    if rho == -INF:
        return INF
    if abs(rho) < 1e-7:
        return _psi_f(d) - (0.0 if M == INF else _psi_f(d + M))
    lg = rho * d + rho * rho / 2
    t1 = _pbar_f(d)
    if M == INF:
        t3 = math.exp(min(lg + _logpbar_f(d + rho), 700))
        return (t1 - t3) / rho
    t2 = math.exp(min(-rho * M + _logpbar_f(d + M), 700))
    a, b = d + rho, d + rho + M
    if b <= 0:
        la, lb = float(log_ndtr(a)), float(log_ndtr(b))
        ldiff = lb + math.log1p(-math.exp(la - lb)) if la < lb else -INF
    elif a >= 0:
        la, lb = _logpbar_f(a), _logpbar_f(b)
        ldiff = la + math.log1p(-math.exp(lb - la)) if lb < la else -INF
    else:
        ldiff = math.log(max(1 - _pbar_f(-a) - _pbar_f(b), 1e-300))
    t3 = math.exp(min(lg + ldiff, 700)) if ldiff > -INF else 0.0
    return (t1 - t2 - t3) / rho


def _A_f(rho, L):
    if rho == INF or L <= 0:
        return 0.0
    if L == INF:
        return 1 / rho if rho > 0 else INF
    if rho == 0:
        return L
    return -math.expm1(-rho * L) / rho


def _B_f(rho, L):
    if rho == INF or L <= 0:
        return 0.0
    if L == INF:
        return 1 / rho ** 2 if rho > 0 else INF
    x = rho * L
    if x < 1e-4:
        return L * L * (0.5 - x / 3 + x * x / 8)
    return (-math.expm1(-x) - x * math.exp(-x)) / rho ** 2


def ey_z_f(form, c):
    if form == 'M':
        r1, r2, L1, L2 = c
        m1, mu1 = _A_f(r1, L1), _B_f(r1, L1)
        if L1 == INF:
            m2 = mu2 = 0.0
        else:
            e1 = math.exp(-r1 * L1)
            a2 = _A_f(r2, L2)
            m2, mu2 = e1 * a2, e1 * (L1 * a2 + _B_f(r2, L2))
        Z = m1 + m2
        return (mu1 + mu2) / Z, Z
    if form == 'T':
        a, g, Ll, Lr = c
        Z = _A_f(a, Ll) + _A_f(g, Lr)
        return (_B_f(g, Lr) - _B_f(a, Ll)) / Z, Z
    r, L = c
    Z = _A_f(r, L)
    return _B_f(r, L) / Z, Z


def _piece_f(h, lh, rho, ln, p, s):
    """(int g K_p, mass below p, mass above p) for a non-mirrored piece at a point."""
    b = h + ln
    x = min(max(p - h, 0.0), ln)
    y = min(max(b - p, 0.0), ln)
    e = math.exp(lh - rho * x) if rho < INF else (math.exp(lh) if x == 0 else 0.0)
    if rho == INF:
        return 0.0, 0.0, 0.0
    JL = _J_f(max(p - b, 0.0) / s, -rho * s, x / s)
    JR = _J_f(max(h - p, 0.0) / s, rho * s, y / s)
    return s * e * (JL - JR), math.exp(lh) * _A_f(rho, x), e * _A_f(rho, y)


def _pieces_f(form, c):
    if form == 'M':
        r1, r2, L1, L2 = c
        out = [(0.0, 0.0, r1, L1, False)]
        if L1 < INF:
            out.append((L1, -r1 * L1, r2, L2, False))
        return out
    if form == 'T':
        a, g, Ll, Lr = c
        return [(0.0, 0.0, a, Ll, True), (0.0, 0.0, g, Lr, False)]
    r, L = c
    return [(0.0, 0.0, r, L, False)]


def point_f(form, c, s, q=0.9, lam=0.0):
    """(objective D + lam (cov - q), cov, E Y) at a point, double precision."""
    ey, Z = ey_z_f(form, c)
    pm, pp = ey - 1, ey + 1
    Tm = Tp = blm = abp = 0.0
    for h, lh, rho, ln, mir in _pieces_f(form, c):
        if mir:
            C, b, a = _piece_f(h, lh, rho, ln, -pm, s)
            Tm += -C
            blm += a
            C, b, a = _piece_f(h, lh, rho, ln, -pp, s)
            Tp += -C
            abp += b
        else:
            C, b, a = _piece_f(h, lh, rho, ln, pm, s)
            Tm += C
            blm += b
            C, b, a = _piece_f(h, lh, rho, ln, pp, s)
            Tp += C
            abp += a
    D = (Tm - Tp) / Z
    cov = 1 - (blm + abp) / Z
    return D + lam * (cov - q), cov, ey


def to_pla(form, c):
    """The law W = Y - E Y as a latentcov.centred.PLA (for checks)."""
    from latentcov.centred import PLA
    ey, _ = ey_z_f(form, c)
    if form == 'M':
        r1, r2, L1, L2 = c
        return PLA([-ey, L1 - ey, L1 + L2 - ey], [-r1, -r2], check=False)
    if form == 'T':
        a, g, Ll, Lr = c
        return PLA([-Ll - ey, -ey, Lr - ey], [a, -g], check=False)
    r, L = c
    return PLA([-ey, L - ey], [-r])


# ------------------------------------------------------------------ branch and bound
def _split_point(lo, hi, log=False):
    if hi == INF:
        return max(2 * lo, lo + 2.0)
    if log and lo > 0:
        return math.sqrt(lo * hi)
    return (lo + hi) / 2


def _rep(lo, hi):
    """A representative finite point of [lo, hi] for the float heuristics."""
    if hi == INF:
        return max(1.5 * lo, lo + 1.0)
    return (lo + hi) / 2


def choose_split(form, box, sig, q, lam):
    """Coordinate with the largest smear |f(x + h e_i) - f(x - h e_i)| at the box centre
    (double precision; only a heuristic). Index len(box) means sigma."""
    c = [_rep(lo, hi) for lo, hi in box]
    sc = math.sqrt(sig[0] * sig[1])
    best, arg = -1.0, 0
    for i, (lo, hi) in enumerate(box + [tuple(sig)]):
        if hi - lo <= 0:
            continue
        if i == len(box):
            pts = [(c, lo), (c, hi)]
        else:
            a, b = (lo, hi) if hi < INF else (lo, 1e6)
            ca, cb = list(c), list(c)
            ca[i], cb[i] = a, b
            pts = [(ca, sc), (cb, sc)]
        try:
            v = [point_f(form, tuple(x), s, q, lam)[0] for x, s in pts]
            sm = abs(v[1] - v[0])
            if not math.isfinite(sm):
                sm = 1e300
        except (ValueError, OverflowError, ZeroDivisionError):
            sm = 1e300
        if hi == INF:
            sm *= 1.0 + 1e-9                  # prefer finite splits on ties
        if sm > best:
            best, arg = sm, i
    return arg


def bnb(form, box, sig, q='0.9', target='0.9001', max_boxes=10 ** 6, log_every=0,
        record_depth=True, lam_split=1e-4, time_limit=None):
    """Branch and bound on one initial box (depth-first with an explicit stack). Returns a dict
    with ok (every leaf closed), counts, the largest leaf upper bound among closed leaves, the
    open boxes if the budget ran out, and per-depth statistics."""
    qa, ta = arb(q), arb(target)
    eps = ta - qa
    stack = [(list(box), tuple(sig), 0)]
    n = n_ok = n_pr = 0
    worst = -INF
    t0 = time.time()
    depth_ub = {}
    while stack:
        b, s, dep = stack.pop()
        n += 1
        r = box_eval(form, b, s, qa, eps)
        if record_depth:
            dmax = depth_ub.get(dep, (-INF, 0))
            ub = r['ub']
            depth_ub[dep] = (max(dmax[0], ub if r['status'] != 'pruned' else -INF), dmax[1] + 1)
        if r['status'] == 'pruned':
            n_pr += 1
            continue
        if r['status'] == 'ok':
            n_ok += 1
            worst = max(worst, r['ub'])
            continue
        if n >= max_boxes or (time_limit and time.time() - t0 > time_limit):
            stack.append((b, s, dep))
            return dict(ok=False, boxes=n, closed=n_ok, pruned=n_pr, worst=worst,
                        open=len(stack), open_boxes=stack[-20:], seconds=time.time() - t0,
                        depth=depth_ub, last_ub=r['ub'])
        i = choose_split(form, b, s, float(q), lam_split)
        if i == len(b):
            m = _split_point(s[0], s[1], log=True)
            stack.append((b, (m, s[1]), dep + 1))
            stack.append((b, (s[0], m), dep + 1))
        else:
            lo, hi = b[i]
            m = _split_point(lo, hi)
            b1, b2 = list(b), list(b)
            b1[i], b2[i] = (lo, m), (m, hi)
            stack.append((b2, s, dep + 1))
            stack.append((b1, s, dep + 1))
        if log_every and n % log_every == 0:
            print(f'{n} boxes, stack {len(stack)}, worst {worst:.3e}, {time.time() - t0:.0f}s',
                  flush=True)
    return dict(ok=True, boxes=n, closed=n_ok, pruned=n_pr, worst=worst, open=0,
                seconds=time.time() - t0, depth=depth_ub)


# ------------------------------------------------------------------ analytic ends in sigma
def sigma_small_end(eps):
    """Largest double sigma_0 (from a grid) with sqrt(10) s^2 + 2 sqrt(10) s phi(1/(2 s)) <= eps
    proved in ball arithmetic: then D <= eps for every feasible law and sigma <= sigma_0
    (notes, Lemma S). eps is a ball."""
    S = arb(10).sqrt()
    s = float((eps / S).sqrt().upper())
    for _ in range(200):
        sa = arb(s)
        if S * sa * sa + 2 * S * sa * _phi(1 / (2 * sa)) <= eps:
            return s
        s *= 0.999
    raise RuntimeError('no sigma_0')


def sigma_large_end(target):
    """sigma_1 with 2 phi(0)/sigma_1 < target: H <= 2 phi(0)/sigma for every law."""
    s = 0.9
    assert 2 * INV_SQRT2PI / arb(s) < arb(target)
    return s
