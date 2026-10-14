# Contaminated bi-log-concave latent laws

Status (2026-10-14): V1 closed form = brute force (tests); V3 exact reliability (R07); V4 independent
check: Steps 1, 3, 4 verified, Step 2 correct under (C), which holds for every c when
q >= Psi(q_beta) (0.526 Gaussian, 0.523 SU; proof: P0(1) >= q for q > 1/2, so lambda(P0(1)) >= q_beta).
The code now refuses q < 0.53. V2 adversarial search over actual laws (2026-10-14): differential evolution over 10 latent families
(9 log-concave + the bimodal law in B, reflected or not) x 2 atoms for H x Gaussian or two-uniform
mixture noise, q in {0.6, 0.8, 0.9, 0.95}, c in {0.01, 0.05, 0.1}: 240 cases, max of
min(P - p_c, q - L) = -1.2e-14 (double-precision noise, at q = 0.95, c = 0.1 >= 1 - q); no law with
noisy coverage >= p_c and latent < q. In the manuscript as Proposition 4 (main) and S3.3 / S7.

Purpose: answer "what if the latent law is not bi-log-concave?" with an exact sensitivity result
instead of a single counterexample, and give the user a dial (the contamination a rank tolerates).

## Setting

Noise class E, closed under scaling and negation, beta = beta_E = inf_{eps in E} pr(eps >= 0) > 0.
Contamination level e0 in [0, 1) (written eps in code; here `c` to avoid the noise symbol):

    B_c = { (1 - c) G0 + c H : G0 in B, H any law on the real line }.

W ~ (1 - c) G0 + c H, noise e = sigma*eps with law of eps in E, independent of W, threshold t > 0.
Lambda_E(P): the latent coverage at t guaranteed for every W in B by noisy coverage P at t
(Lemma S2 bound, 1 - sup{phi(aL) + phi(aR) : aL + aR <= 1 - P}); for Gaussian and symmetric
unimodal noise Lambda_E = lambda_E = Psi_E^{-1} on (1/2, 1) (Lemma S3), and 0 for P <= 1/2.
f(x) = max{0, 1 - (1 - x)/beta}: the shape-free guarantee (Proposition 2 lower bound).

## Step 1: decomposition bound (any E with beta > 0)

Write W = W0 with prob. 1 - c and W = W_H with prob. c (W0 ~ G0, W_H ~ H, both independent of e).
Then with P0 = pr(|W0 + e| <= t), x = pr(|W_H + e| <= t):

    noisy  = (1 - c) P0 + c x,
    latent = (1 - c) pr(|W0| <= t) + c pr(|W_H| <= t) >= (1 - c) Lambda_E(P0) + c f(x).

Hence noisy coverage >= p implies latent coverage >= lambda_c(p), where

    lambda_c(p) = inf { J(x) : x in [0,1], P0(x) in [0,1] },  J(x) = (1-c) Lambda_E(P0(x)) + c f(x),
    P0(x) = (p - c x)/(1 - c)

(Lambda_E and f are nondecreasing, so "noisy >= p" reduces to equality). lambda_c is nondecreasing
in p. Define the level p_c(q) = inf{p : lambda_c(p) >= q}; noisy coverage above p_c(q) gives latent
coverage q for every W in B_c and every noise law in E.

## Step 2: evaluation for Gaussian and symmetric unimodal noise (beta = 1/2)

lambda = Psi^{-1} is increasing and concave on (1/2, 1), with lambda'(P) = 1/Psi'(lambda(P)).
Assume p < 1 - c*beta and the condition

    (C)   P0(1) = (p - c)/(1 - c) > 1/2   and   Psi'(lambda(P0(1))) >= beta.

* On x in [0, 1 - beta], f = 0 and J is nonincreasing in x, so the infimum over this piece is at
  x = 1 - beta (feasible because p <= 1 - c beta).
* On x in [1 - beta, 1], P0(x) runs over [P0(1), P0(1 - beta)] inside (1/2, 1), where lambda is
  concave; J is concave (concave of affine plus linear), so its minimum is at an endpoint. By
  concavity, lambda(P0(1-beta)) - lambda(P0(1)) <= lambda'_+(P0(1)) * c beta/(1 - c), so
  J(1) - J(1 - beta) >= c {1 - beta lambda'_+(P0(1))} >= 0 under (C).

Therefore, under (C),

    lambda_c(p) = J(1 - beta) = (1 - c) lambda( (p - c(1 - beta)) / (1 - c) ),
    p_c(q)      = (1 - c) Psi( q/(1 - c) ) + c (1 - beta),         for 1/e < q/(1 - c) < 1.

The worst H is a layer just outside the threshold with noisy coverage exactly 1 - beta and latent
coverage 0. If (C) fails the closed form can be too LOW (V4: q = 0.4, c = 0.3 gives 0.5747, brute force 0.65):
Lambda jumps from 0 to 1/e at P = 1/2, so c f(x*) with P0(x*) = 1/2 is a further candidate. We state
the result only for q >= Psi(q_beta).

Endpoints: c = 0 gives Psi(q). c = 1 - q gives q + (1 - q)(1 - beta) = 1 - beta(1 - q), the
shape-free level; for c >= 1 - q the level is the shape-free one (H alone can carry the shape-free
extremal layer of mass 1 - q, and the shape-free bound holds for every law).

Monotone in c (beta = 1/2): d/dc p_c(q) = -Psi(q0) + q0 Psi'(q0) + 1 - beta, q0 = q/(1 - c).
Psi is C^1 and convex on (0, 1) (constant 1/2 below 1/e, derivative 0 there for both classes), so
its tangent at q0 has intercept Psi(q0) - q0 Psi'(q0) <= Psi(0+) = 1/2; hence the derivative is
>= 0, strictly when q0 > 1/e by strict convexity.

## Step 3: sharpness (Gaussian and symmetric unimodal noise)

Let p < p_c(q), q0 = q/(1 - c) in (1/e, 1), a0 = log(1/q0), and eps* the extremal noise law of the
class at q0 (Gaussian: s* Z; SU: uniform on [-r, r]); in both, pr(eps* < 0) = 1/2 = 1 - beta and
eps* has no atom at 0. Take eta, eta' > 0 small and, for s > 0,
    W0 = t + s (a0 + eta - E_1)      (log-concave),     H = point mass at t + s eta',
    noise s eps*.
As s -> 0: pr(|W0 + s eps*| <= t) -> E min(1, q0 e^{-eta - eps*}), pr(|t + s eta' + s eps*| <= t)
-> pr(eps* < -eta'), and as eta, eta' -> 0 the noisy coverage tends to
(1 - c) Psi(q0) + c/2 = p_c(q) > p, while the latent coverage is (1 - c) q0 e^{-eta} < q.
So no level below p_c(q) suffices, and p_c(q) is exact.

## Step 4: the rank

Theorem 3 holds with B_c in place of B and p_c(q) in place of p_E(q): for s < r = Q_q(|W|),
pr(|W| <= s) < q, so pr(|V_i| <= s) <= p for every p > p_c(q); hence pr(|V_i| < r) <= p_c(q), unit
by unit. The converse of Step 3, with the same law in every unit, makes the rank sharp for
Gaussian and SU noise (infimum along these laws). Corollary 1 holds with G_i in B_c.

Tolerated contamination of a rank: pr{Bin(K, l) <= k - 1} = pr{Beta(k, K + 1 - k) > l} is
decreasing in l, so rank k is valid for every law in B_c iff p_c(q) <= p_k, the delta-quantile
of Beta(k, K + 1 - k) (the order-statistic level of Proposition S5). As p_c(q) is continuous and
nondecreasing in c, the largest tolerated contamination is the root c*(k) of p_c(q) = p_k.

Population mixtures: if calibration and new units are i.i.d. draws from a population whose latent
law is in B_c (at most a fraction c departs from bi-log-concavity) and the noise is independent of W
given the component label, with law in E, Theorem 3 applies to the population law with level p_c(q). This
turns the negative example of §S3.2 into a quantitative statement.

## Numbers (Gaussian noise, delta = 0.05; closed form, checked against brute force)

q = 0.9: usual ranks 105 / 916 / 9050 at K = 110 / 1e3 / 1e4; shape-free 109 / 962 / 9537.

| c | p_c(0.9) | ranks K = 110 / 1e3 / 1e4 |
|---|---|---|
| 0 | 0.90080 | 105 / 917 / 9058 |
| 0.005 | 0.90322 | 105 / 919 / 9082 |
| 0.01 | 0.90563 | 105 / 922 / 9105 |
| 0.02 | 0.91049 | 106 / 926 / 9153 |
| 0.05 | 0.92517 | 107 / 940 / 9296 |
| 0.10 | 0.95000 | 109 / 962 / 9537 |

q = 0.8: c = 0, 0.01, 0.05, 0.1, 0.2 give 0.80441, 0.80893, 0.82731, 0.85093, 0.90000.
Rule of thumb: p_c(q) ~ Psi(q) + 0.48 c at q = 0.9.

## Open points to check

- Value of q_beta (Psi'(q_beta) = 1/2) for both classes, and that (C) holds for the tabled (q, c).
- Mean-zero log-concave noise (beta = 1/e): Step 1 holds with certified Lambda; Step 2 needs
  convexity of the two-sided transfer, not proved. Proven lower bound on any valid level (V4): with
  the centred exponential, (1 - c) max_sigma E min(1, q0 e^{-sigma(E1 - 1)}) + c (1 - 1/e); NOT
  (1 - c) Psi_LC(q0) + c (1 - 1/e), since extremality of the centred exponential is open.
- One-sided intervals: same proof with the one-sided lambda and the same f gives the same p_c (V4).
