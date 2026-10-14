# Centred latent residuals: the level p_kappa(q)

Status (2026-10-14): reduction (Steps 1-4) written out here, not yet independently checked; the analytic
facts of Step 5 and the rank of Step 6 are proved; the values of p_kappa(q) for kappa < 1 are
numerical (global multistart over the reduced family, each maximiser rechecked in mpmath at 30
digits); the adversarial search over 3700 laws outside the family found nothing above them. Code: `src/latentcov/centred.py`,
`experiments/r08_centred.py`, `tests/test_centred.py`; numbers in `results/centred_levels.csv`.
In the manuscript (2026-10-14) as Proposition 5 / Table 5 (main, Sec. 4.2, symbol zeta for kappa) and
Supplement S3.4 (proof) / S7 (Table S-centred); the proposition states at most THREE affine pieces
(the localization statement itself); two pieces is the computational reduction (Step 2).
Before submission: check the theorem statement in the primary text of Fradelizi & Guedon (2006).

Purpose: the usual rank fails only at grossly miscentred latent laws, as the extremal laws show. This note gives the exact level for
latent laws whose mean is within kappa threshold-units of the centre, so that the paper can say
how large the needed correction is for centred residuals, and from which K the usual rank fails.

## Setting

Threshold t = 1 by scaling. Noise class E: Gaussian N = {sigma Z} or symmetric unimodal SU (scale
mixtures of uniform laws on [-r, r], Khintchine). Latent laws: LC, the log-concave laws on the
line including point masses. (The reduction below uses convexity of LC in the sense of the
localization theorem; it is not stated for B.) Centring dial kappa >= 0:

    p_kappa(q) = sup { pr(|W + e| < 1) : W in LC, |E W| <= kappa, pr(|W| < 1) <= q,
                       e = sigma eps independent of W, eps in E, sigma > 0 }.

kappa = 0: mean-zero residuals; kappa = inf: no centring. In calibration units, kappa bounds
|E W| / Q_q(|W|): the bias of the point prediction relative to the half-width of the oracle
latent interval.

*Conventions.* Both coverages are taken on the open interval. This is what the rank needs: with
r = Q_q(|W|) = inf{x : pr(|W| <= x) >= q}, pr(|W| < r) = lim_{x up r} pr(|W| <= x) <= q always,
and the count in the proof of Theorem 3 is N = #{i : |V_i| < r}. So the transfer is applied at r
itself and no limit s -> r is needed (see Step 6). For continuous noise (Gaussian, uniform with
r > 0) the noisy coverage is the same on the closed interval; a nondegenerate log-concave W has no
atoms, so only point masses distinguish the two latent conventions, and a point mass with
pr(|W| < 1) <= q < 1 sits at |w0| >= 1 and has noisy coverage at most 1/2.

## Step 1: what the paper already gives (kappa = inf, and kappa >= 1)

(a) p_inf(q) = Psi_E(q) for E = N, SU and q > 1/e. Upper bound: Supplementary Lemma lem:twoend with the convexity of
Psi_E (main Theorem 2(iii) for N, Supplementary Proposition prop:su-exact for SU, via Lemma lem:convex): pr(|W + e| <= 1) > Psi_E(q) gives pr(|W| <= 1) > q for W
in B, and for nondegenerate LC W this is pr(|W| < 1) > q; point masses are covered above. Lower
bound: the one-end construction of Supplementary sec:s-oneend, W = 1 + s(a + eta - E_1), a = log(1/q),
has pr(|W| < 1) = q e^{-eta} < q and noisy coverage -> E min(1, q e^{-eta - eps}) as s -> 0.

(b) p_kappa(q) = Psi_E(q) for every kappa >= 1 (proved). The law in (a) has mean
1 - s(1 - a - eta) in (-1, 1) for small s and eta because a < 1 (q > 1/e). So it is feasible for
every kappa >= 1.

(c) For q > 1 - 1/e, laws with |E W| >= 1 never beat q. Reduce to Gaussian or uniform noise first
(Step 3). Then W + e is log-concave (Prekopa), and for a log-concave X, pr(X >= E X) >= 1/e
(Grunbaum's inequality in one dimension, fact 4 of Supplementary sec:s-prelim). If E W = mu >= 1, then
pr(|W + e| < 1) <= pr(W + e < mu) <= 1 - 1/e < q, and symmetrically for mu <= -1. Hence for
q > 1 - 1/e only means in (-1, 1) matter, p_kappa = p_{min(kappa, 1-)}, and the dial runs over
[0, 1). All q in this note (0.8, 0.9, 0.95) exceed 1 - 1/e = 0.632.

(d) Elementary: p_kappa(q) is nondecreasing in kappa and in q, and p_kappa(q) >= q. For the
last, take any W in LC with mean 0 and pr(|W| < 1) = q (the uniform law on [-1/q, 1/q]); as
sigma -> 0 its noisy coverage tends to pr(|W| <= 1) = q.

## Step 2: localization (the reduction over W)

**Theorem (Fradelizi and Guedon 2006, Theorem 1; dimension one).** Let K be a compact interval,
f_1, ..., f_p upper semicontinuous on K, and P_f the set of log-concave probability measures mu
supported in K with int f_i dmu >= 0, i = 1, ..., p. Every extreme point nu of the convex hull of
P_f is a point mass, or has a density e^{-V} on its support (a segment) with V the maximum of at
most p affine functions, i.e. V convex piecewise affine with at most p pieces.

Citation: M. Fradelizi and O. Guedon, A generalized localization theorem and geometric
inequalities for convex bodies, Adv. Math. 204 (2006) 509-529, doi 10.1016/j.aim.2005.05.020
(bib key to add: `fradelizi2006`). The one-constraint case is Fradelizi and Guedon (2004), Discrete
Comput. Geom. 31, 327-335 (`fradelizi2004`, already used in Supplementary prop:reduction): extreme
points are point masses and log-affine laws on segments. **The primary text of the 2006 paper could
not be opened (paywall, bot protection), so its verbatim statement and theorem number are
unverified.** The dimension-one statement above is taken from two secondary sources that agree:
Melbourne, Nayar and Roberto, "Minimum entropy of a log-concave variable with fixed variance",
arXiv:2309.01840, Theorem 3.11 (stated for p = 2 and cited as "[FG06] Theorem 1"), and Aravinda,
arXiv:2212.09115, Definition 1.1 and the restatement of [FG06, Proposition 2]: e^{-U} has k + 1
degrees of freedom iff U is the maximum of k (and not fewer) affine functions. Murawski,
arXiv:2211.05210, Theorem 4.1, restates the same result as "Theorem 2 from [FG06]" (with the
number of pieces equal to the number of active constraints); the theorem number should be checked
against the journal text before it goes into the paper.

*Degrees of freedom* (FG 2006): the degree of freedom of a log-concave f = e^{-U} is the largest
d for which there are eta > 0 and linearly independent h_1, ..., h_d on the support of f with
f + sum delta_i h_i log-concave for all |delta_i| <= eta. The proof of the theorem is: (i)
Krein-Milman / Milman: a convex upper semicontinuous functional attains its maximum over the
compact set P_f at an extreme point of its closed convex hull, and such points lie in P_f;
(ii) an extreme point with d degrees of freedom and m linear equations to preserve (normalization
plus the active constraints) must have d <= m, otherwise a nonzero delta in the kernel of the
m x d system gives f = (f_+ + f_-)/2 with f_+- = f +- delta.h in P_f; (iii) Proposition 2:
d = k + 1 iff U is the maximum of k affine functions.

**Application.** Fix a continuous noise law (Gaussian sigma > 0, or uniform r > 0), so that
g(w) = pr(|w + e| < 1) is continuous and bounded; Phi(mu) = int g dmu is linear and continuous for
weak convergence. Constraints

    f_1 = q - 1_{(-1,1)},   f_2 = kappa - w,   f_3 = kappa + w,

so int f_1 dmu >= 0 is pr(|W| < 1) <= q and f_2, f_3 give |E W| <= kappa. Upper semicontinuity:
1_{(-1,1)} is lower semicontinuous because (-1, 1) is open, so f_1 is upper semicontinuous (this
is where the open convention is needed: mu -> mu(-1, 1) is lower semicontinuous, so
{mu(-1,1) <= q} is weakly closed); f_2, f_3 are continuous on K. The set of log-concave
probabilities on K, point masses included, is weakly closed, so P_f is compact.

The theorem with p = 3 gives at most three pieces (this is what the paper uses). **Two pieces
suffice** by the following degrees-of-freedom count (ii) -- an argument through the PROOF of the
theorem, not its statement, and not proved here in full; the paper does not rely on it: the constraints f_2, f_3 can both be active only if E W = kappa =
-kappa, i.e. kappa = 0, and then they impose the single linear equation int w h = 0. So at an
extreme point at most three linear equations act on the perturbation (normalization, coverage if
active, the mean if active), d <= 3, and by (iii) V is the maximum of at most two affine
functions. Inactive constraints stay strict under small perturbations, since |int f_i delta.h|
<= eta sup_K |f_i| ||h||_1. (This uses the proof of the theorem, not only its statement; the
numerical search of Step 4 also covers three-piece laws, as a check.)

**Truncation to compact support** (as in Supplementary prop:reduction, but the mean constraint needs
care). Let q > 1 - 1/e and W in LC feasible, nondegenerate, mean mu with |mu| < 1 (Step 1(c)).
(i) If pr(|W| < 1) = q, dilate about the mean: W_a = mu + a(W - mu), a > 1. The event
|W_a| < 1 is W in (mu - (1 + mu)/a, mu + (1 - mu)/a), an interval strictly inside (-1, 1) when
|mu| < 1; the support of W crosses -1 or 1 in its interior (0 < q < 1), where the density is
positive, so pr(|W_a| < 1) < q, with the same mean. (ii) Now with pr(|W| < 1) < q: condition on
[-M, M] and translate back to mean mu. The conditioned law is log-concave, and as M -> inf its
latent coverage and its mean converge, so for M large the translated law is feasible. (iii)
Noisy coverage is continuous under these operations (g bounded and continuous, weak convergence),
so sup over LC = sup over compactly supported LC, and on each K the theorem applies. Point masses
are feasible as they are (|w0| >= 1) and have noisy coverage <= 1/2.

Hence, with F2 = {point masses} U {densities prop. to exp(-V) on a segment, V convex with at
most two affine pieces},

    p_kappa(q) = sup over the noise scale of sup { pr(|W + e| < 1) : W in F2, feasible }.

The supremum over W need not be attained (the support may run off to infinity, slopes may
diverge), but it is a supremum over a five-parameter family.

## Step 3: the reduction over the noise

For a fixed W the noisy coverage is linear in the law of eps. SU: eps = R U, U uniform on
[-1, 1], R >= 0 (Khintchine), so pr(|W + sigma R U| < 1) = E{pr(|W + sigma R U| < 1 | R)} <=
sup_{r >= 0} pr(|W + rU| < 1), and r = 0 gives the latent coverage <= q. So the SU supremum is
over uniform laws on [-r, r], r > 0. Gaussian: over sigma > 0. (Both before Step 2, which needs a
continuous kernel.)

Hence p_kappa(q) is the supremum of an explicit function of six real parameters (two-piece law,
noise scale), subject to the coverage and mean constraints. The family contains the extremal laws
of Step 1 in the limit.

## Step 4: computation

Parametrization with exact constraints. A two-piece shape X on [0, 1] (kink c in (0, 1), slopes
beta1 >= beta2), placed as W = mu + lam (X - E X): the mean is mu in [0, min(kappa, 1)) (W and -W
are equivalent: symmetric noise, symmetric interval), and lam is the unique scale with
pr(|W| < 1) = q (for |mu| < 1 the event is an interval around mu shrinking in lam, so the
coverage decreases strictly where it is below 1; bisection, then a check that the placed law has
coverage <= q in floating point). Five free parameters (four at kappa = 0). Latent coverage
strictly below q is allowed in the adversarial check (extra dilation).

Evaluation. Distribution function, mean and E(x - W)^+ of piecewise log-affine laws in closed
form, in log space with expm1 kernels (no cancellation for any slope). Uniform noise: noisy
coverage = {C(1+r) - C(1-r) - C(-1+r) + C(-1-r)}/(2r) with C(x) = E(x - W)^+. Gaussian noise:
int phi(z) {F(1 + sigma z) - F(-1 + sigma z)} dz by 20-point Gauss-Legendre on chunks split at
every kink (z in [-9, 9]). Absolute accuracy of the noisy coverage: 1e-15 (checked against mpmath
at 30 digits, against the closed form of Supplementary eq:closed and against Gauss-Legendre for the uniform
form; tests/test_centred.py). The excesses of interest are 1e-7 to 1e-3, so double precision has
six or more spare digits; the concern about 1e-6 differences is about cancellation in
closed forms with 1/beta terms, which this route avoids.

Optimizer. Random multistart Nelder-Mead (adaptive, restarted from its own optimum) over the five
parameters, parallel over starts, best start polished with a long run. Every reported maximiser is
re-evaluated by `brute_noisy` / `brute_latent` (mpmath quadrature of f(w) g(w) at 30 digits,
independent code): the reported value is attained by an explicit law and noise, so it is a valid
*lower* bound on p_kappa(q) (a law that the usual rank must face). That it is the supremum is
numerical (multistart; no certificate).

## Step 5: what is proved about p_kappa(q)

(a) p_kappa(q) = Psi_E(q) for kappa >= 1 (Step 1(b)); for q > 1 - 1/e, p_kappa(q) depends on
kappa only through min(kappa, 1) (Step 1(c)).

(b) **p_kappa(q) > q for every q in (0, 1), every kappa >= 0 and both classes (proved).** So
no threshold q* with p_0(q) = q exists: the usual rank is not valid even for exactly centred
log-concave residuals with Gaussian noise, and by the argument of Proposition 3 its latent
reliability tends to 0 as K -> inf at a fixed such law. Proof: small-noise expansion. Let W have
density f prop. to e^{beta w} on [a, b], beta > 0 small, with mean 0, pr(|W| < 1) = q and
a < -1 < 1 < b (for beta = 0 the uniform law on [-1/q, 1/q] has these properties with room to
spare, and the placement W = lam (X - E X) of Step 4 depends continuously on beta). f is smooth
near +-1, so with F the distribution function and x = +-1,
E F(x + sigma Z) - F(x) = sigma^2 f'(x)/2 + O(sigma^4) (the part of the Gaussian law that leaves
the smooth range costs exp(-c/sigma^2)), and E F(x + rU) - F(x) = r^2 f'(x)/6 + O(r^4) exactly
by Taylor for r below the distance to a, b. Hence

    pr(|W + e| < 1) - q = (v/2) {f'(1) - f'(-1)} + O(v^2),   v = var(e),

and f'(1) - f'(-1) = beta {f(1) - f(-1)} > 0. (tests/test_centred.py checks the expansion
numerically.) In words: centred laws whose density *rises* through the right end of the interval
(and so rises through the left end less steeply) gain coverage from small noise, at order v.

(c) Elementary: nondecreasing in kappa and q; p_kappa(q) >= q.

(d) Not proved: a closed form, or an analytic bound below Psi_E(q), for 0 <= kappa < 1. The order
of the excess in kappa near 1 and the shape of the maximisers are numerical (Numbers below).

## Step 6: the rank (proved, given the level)

**Proposition (centred rank).** In the setting of Section 2, let W_1, ..., W_K, W_new be i.i.d.
with a law G in LC, r = Q_q(|W|) and |E W| <= kappa r; the noises e_i = sigma_i eps_i are
independent of the W's, with laws of eps_i in E (Gaussian or symmetric unimodal) and scales that
may differ between units. Then

    pr_D{ pr(|W_new| <= T | D) >= q } >= pr{ Bin(K, p_kappa(q)) <= k - 1 }.

The same holds for independent W_i with different laws G_i in LC if Q_q(|W_i|) >= r_0 and
|E W_i| <= kappa r_0 for every i, r_0 = Q_q(|W_new|) (the dial is relative to r_0, not to
Q_q(|W_i|)). If p_kappa(q) <= q the usual rank would be valid outright; by Step 5(b) this never
happens.

Proof. As for Theorem 3: latent coverage >= q iff T >= r iff N = #{i : |V_i| < r} <= k - 1. If
r = 0 there is nothing to prove. Otherwise W_i/r is log-concave, |E(W_i/r)| <= kappa and
pr(|W_i/r| < 1) = pr(|W_i| < r) <= q (r is the lower quantile: pr(|W| <= x) < q for x < r), and
e_i/r has a law in the class (closed under scaling), so pr(|V_i| < r) = pr(|W_i/r + e_i/r| < 1)
<= p_kappa(q) unit by unit, by the definition of p_kappa as a supremum. N is a sum of independent
indicators with these probabilities. In the heterogeneous case the same at r_0, using
pr(|W_i| < r_0) <= q.

*On the limit s -> r.* Theorem 3 transfers at thresholds s < r with closed intervals and lets
s -> r. Here that route would be wrong as stated: the dial at threshold s is |E W|/s <= kappa r/s,
slightly *larger* than kappa, so one would need right-continuity of kappa -> p_kappa(q), which
is not proved. Taking open intervals in the definition of p_kappa and transferring at r itself
avoids the limit. The price is only the open-interval convention, which is also what makes the
coverage constraint upper semicontinuous in Step 2.

*Converse (sharpness).* If W*, sigma* is a feasible law and noise with pr(|W*| < 1) = q, density
positive on both sides of the boundary, and noisy coverage H*, then Q_q(|W*|) = 1 and, with the
same law and noise in every unit, the latent reliability of rank k is exactly
pr{Bin(K, H*) <= k - 1}. The maximisers in the table are such laws, so every reported level is a
rigorous *necessary* level (up to the mpmath recheck): no rank with a lower level is valid for
all centred log-concave laws. It is *sufficient* only to the extent that the multistart found the
supremum.

## Numbers (results/centred_levels.csv; delta = 0.05)

p_kappa(q) - q. Rows kappa >= 1 are Psi_E(q) - q (proved, Step 1(b)); rows kappa < 1 are the
best value over the two-piece family and its exponential limits (attained, mpmath-checked; the
supremum is numerical).

| kappa | Gaussian q = 0.8 | 0.9 | 0.95 | SU q = 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|
| 0 | 1.70e-4 | 2.82e-6 | 1.09e-7 | 4.67e-4 | 1.33e-5 | 6.88e-7 |
| 0.05 | 2.67e-4 | 1.12e-5 | 1.60e-6 | 6.94e-4 | 4.64e-5 | 8.12e-6 |
| 0.1 | 3.92e-4 | 3.10e-5 | 6.96e-6 | 9.73e-4 | 1.15e-4 | 3.11e-5 |
| 0.2 | 7.40e-4 | 1.27e-4 | 6.92e-5 | 1.69e-3 | 4.00e-4 | 2.31e-4 |
| 0.3 | 1.24e-3 | 4.13e-4 | 1.15e-4 | 2.69e-3 | 1.12e-3 | 3.36e-4 |
| 0.5 | 3.28e-3 | 7.41e-4 | 1.51e-4 | 6.40e-3 | 1.69e-3 | 4.15e-4 |
| 0.75 | 4.39e-3 | 8.03e-4 | 1.59e-4 | 7.70e-3 | 1.79e-3 | 4.31e-4 |
| >= 1 | 4.41e-3 | 8.03e-4 | 1.59e-4 | 7.72e-3 | 1.79e-3 | 4.31e-4 |

The kappa = 0 values agree with a search over log-affine laws only (Gaussian q = 0.8: 1.7e-4; 2.4e-6 at q = 0.9
and 1.2e-5 for uniform noise there; two pieces give 2.82e-6 and 1.33e-5).
The dial matters more than q: already kappa = 0.05 multiplies the q = 0.9 excess by 4 and the
q = 0.95 excess by 15, and kappa = 0.3 is within a factor 2-4 of the uncentred level.

K at which the usual rank's latent reliability at the worst law falls below 0.9 / below 0.5
(normal approximation; the exact binomial scan agrees within 4% for K > 1e5 and within 26% below,
where the integer rank makes the reliability a sawtooth in K; both in the csv):

| kappa | Gauss 0.8 | Gauss 0.9 | Gauss 0.95 | SU 0.8 | SU 0.9 | SU 0.95 |
|---|---|---|---|---|---|---|
| 0 | 7.3e5 / 1.5e7 | 1.5e9 / 3.1e10 | 5.3e11 / 1.1e13 | 1.0e5 / 2.0e6 | 6.7e7 / 1.4e9 | 1.3e10 / 2.7e11 |
| 0.05 | 3.0e5 / 6.1e6 | 9.4e7 / 1.9e9 | 2.5e9 / 5.0e10 | 4.6e4 / 9.0e5 | 5.5e6 / 1.1e8 | 9.5e7 / 1.9e9 |
| 0.1 | 1.4e5 / 2.8e6 | 1.2e7 / 2.5e8 | 1.3e8 / 2.7e9 | 2.3e4 / 4.6e5 | 9.0e5 / 1.8e7 | 6.5e6 / 1.3e8 |
| 0.2 | 4.0e4 / 7.9e5 | 7.4e5 / 1.5e7 | 1.3e6 / 2.7e7 | 7.8e3 / 1.5e5 | 7.9e4 / 1.5e6 | 1.2e5 / 2.4e6 |
| 0.3 | 1.5e4 / 2.8e5 | 7.2e4 / 1.4e6 | 4.9e5 / 9.8e6 | 3.2e3 / 6.0e4 | 1.0e4 / 1.9e5 | 6.0e4 / 1.1e6 |
| 0.5 | 2.1e3 / 4.1e4 | 2.3e4 / 4.4e5 | 2.8e5 / 5.6e6 | 6.4e2 / 1.1e4 | 4.4e3 / 8.6e4 | 3.9e4 / 7.5e5 |
| >= 1 | 1.4e3 / 2.2e4 | 1.9e4 / 3.8e5 | 2.6e5 / 5.1e6 | 4.2e2 / 7.3e3 | 4.0e3 / 7.6e4 | 3.6e4 / 6.9e5 |

Ranks: usual / level p_kappa(q) / Theorem 3 (level Psi_E(q)), Gaussian noise, q = 0.9:

| kappa | K = 110 | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|---|
| 0 | 105/105/105 | 916/916/917 | 9050/9050/9058 | 90157/90157/90237 | 900494/900497/901296 |
| 0.1 | 105/105/105 | 916/916/917 | 9050/9050/9058 | 90157/90160/90237 | 900494/900525/901296 |
| 0.3 | 105/105/105 | 916/917/917 | 9050/9054/9058 | 90157/90198/90237 | 900494/900906/901296 |
| 0.5 | 105/105/105 | 916/917/917 | 9050/9057/9058 | 90157/90230/90237 | 900494/901233/901296 |

Symmetric unimodal noise, q = 0.9: kappa = 0 gives 105/916/9050/90158/900508 against the usual
105/916/9050/90157/900494 and Theorem 3's 105/918/9068/90334/902278; kappa = 0.3 gives
105/917/9061/90268/901616. q = 0.8 and 0.95 and all kappa are in results/centred_ranks.csv.

Reading: for exactly centred residuals the centred rank equals the usual rank up to K = 1e4
(Gaussian, q >= 0.9) and differs by a few ranks at 1e6; the usual rank's failure needs K of
order 1e9 (q = 0.9) or 1e6-1e7 (q = 0.8). With a bias of 10-30% of the half-width the failure
K drops to 1e4-1e7, and the centred rank moves most of the way to Theorem 3's.

## Maximising laws (numerical)

All maximisers have pr(|W| < 1) = q exactly, the mean at the dial (|E W| = kappa), and a hard
edge just outside one end of the interval (right end at 1.01-1.20) with the density rising or flat
towards it; the noise scale is a few hundredths of the threshold (Gaussian sd 0.006-0.11,
uniform half-width 0.014-0.22). The noise carries part of the thin layer beyond the edge inside;
because the density does not fall towards the edge, it carries less mass out. Three types:

* q = 0.8, kappa <= 0.2: the (reflected) exponential law W = b - E/beta, b = 1.06-1.10,
  beta = 0.94-1.11, i.e. the centred (kappa-shifted) exponential with its edge just outside the
  threshold; the second piece degenerates and the support runs off to -inf (the supremum is
  approached, not attained; the exponential limit and the two-piece fit agree to 6e-11, and the
  table takes the larger). For SU noise at kappa = 0 the mirror image.
* q = 0.9, 0.95, kappa <= 0.1: "flat top, hard edge, steep tail": a nearly flat density (log-slope
  0.08-0.33) on (-1.02...-1.09, 1.02...1.09), edge just outside +1, and beyond the kink just
  outside -1 a steep exponential left tail (log-slope 6-84, i.e. tail scale 0.01-0.16) that holds
  the mean at kappa. Not a member of any standard family: every standard law tried (normal,
  logistic, Laplace, Gumbel, gamma, Weibull, beta, truncated normal and exponential, both
  orientations) has excess <= 1e-11, i.e. <= 0 up to rounding, at q = 0.9 and 0.95 when
  kappa = 0 (both classes), and also at kappa = 0.05 except SU noise at q = 0.95 (4.7e-6, a
  reflected truncated exponential). Common centred laws thus gain nothing; the class contains laws
  that do better.
* kappa >= 0.2 (all q), and q = 0.95 at kappa = 0.1: one log-affine piece (truncated
  exponential, increasing) on [a, b], b just outside 1, a just outside or inside -1; as
  kappa -> 1 the edge moves to 1 and the slope grows: the law collapses onto the extremal
  exponential tail of Section 3, whose mean tends to 1.

## Validation

* kappa = inf / kappa >= 1: the exponential tail of Supplementary sec:s-oneend with mean in (-1, 1)
  reproduces Psi_N(0.9) = 0.9008035 and Psi_SU(0.9) = 0.9017873 within 2e-7 at s = 1e-4
  (tests/test_centred.py), and the kappa = 0.75 search values are within 1e-6 to 2e-5 of Psi.
* Accuracy: every maximiser recomputed in mpmath (30 digits, independent quadrature code):
  |double - mpmath| <= 1e-14 (Gaussian) and <= 1e-13 (uniform). Latent coverage <= q + 5e-16 at
  every maximiser; |mean| <= kappa + 1e-12, except the exponential-limit cases (supports out to
  -3e6, where placement rounding gives |mean| - kappa up to 3e-9; the excess changes by less than
  1e-11, and the exponential limit evaluated with exact mean agrees to 4e-12).
* Robustness of the search: 192 random starts plus warm starts from the smaller kappa per case;
  the 8 best polished runs agree to 8e-9 (to 1e-13 in most cases); the values are monotone in
  kappa (tested).
* Three pieces (results/centred_check3.csv): re-optimising with an extra kink from each maximiser
  (24 starts) gains at most 6e-11, only in the q = 0.8 Gaussian exponential-limit cases, i.e.
  the same convergence slack as above; elsewhere |gain| < 1e-13.
* Adversarial search outside the family (results/centred_validation*.csv): for each of the 42
  cases (kappa < 1), 88 laws - 56 random members of normal, logistic, Laplace, Gumbel (both
  orientations), exponential (both), gamma(a >= 1) (both), Weibull(k >= 1), beta(a, b >= 1),
  truncated exponential (both), truncated normal; 16 random log-concave laws exp(-V) with V convex
  piecewise affine with 3-40 pieces; 16 random three-piece laws - each optimised over its mean
  (|mu| <= kappa), extra dilation (latent coverage <= q) and noise scale (3700 laws in all). None
  exceeds the reported p_kappa(q) - q (smallest margin 5e-15, after the exponential limits were
  folded into the level; before, the plain exponential beat the two-piece fit by 6e-11 at
  Gaussian q = 0.8, kappa = 0.05 and 0.2). Median margin 2.3e-4. The closest competitors are
  exponential, truncated exponential and three-piece laws, i.e. members or limits of the family,
  and at kappa >= 0.3 truncated normals.

## Open points

* The primary statement of Fradelizi and Guedon (2006), Theorem 1 (number and exact hypotheses),
  must be checked against the journal text; the two-piece count uses the degrees-of-freedom
  argument of its proof, written out in Step 2, and should be stated as our lemma with that
  credit. Bib entry `fradelizi2006` to add.
* The values for kappa < 1 are numerical lower bounds that a global multistart did not improve;
  no certificate. A ball-arithmetic branch and bound over the five parameters (as for
  mean-zero log-concave noise) would make them proved; the parameter space is non-compact
  (exponential limits), so the ends need analytic bounds as in Supplementary sec:s-lc.
* The reduction is for LC latent laws. For B (bi-log-concave) the localization theorem does not
  apply; whether the centred level is the same over B is open.
* Mean-zero log-concave noise is not treated (its kappa = inf level is itself only bracketed).
* Small-noise expansion (Step 5(b)) suggests the kappa = 0 excess for smooth laws is of order
  var(e) times f'(1) - f'(-1); the maximisers instead use a hard edge at distance O(sigma) from
  the threshold. An analytic description of the kappa -> 0 behaviour (e.g. p_0(q) - q as q -> 1)
  is not attempted.

