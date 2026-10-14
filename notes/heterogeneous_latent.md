# Heterogeneous latent laws

Status: proof complete (a corollary of the proof of Theorem 2, no new tools). Numerical check in
`experiments/r06_heterogeneous_latent.py`, outputs `results/heterogeneous_latent*.csv`.

Purpose: answer the question "must all units share one latent law?". It clarifies how
far Theorem 2 applies; it is not a separate headline result. One paragraph in the main text plus
the remarks below in the supplement.

## Statement

Setting of §3, except that $W_1, \ldots, W_K, W_{\rm new}$ are independent with laws
$G_1, \ldots, G_K, G_0$ that may differ. Noise as before: $e_i = \sigma_i\epsilon_i$, laws in
$\mathcal E$, scales unknown, independent of the $W$'s. Let $r_0 = Q_q(|W_{\rm new}|)$.

**Corollary.** If $G_i \in \mathcal B$ and $Q_q(|W_i|) \ge r_0$ for every $i = 1, \ldots, K$, then
$$\pr_{\mathcal D}\{\pr(|W_{\rm new}| \le T \mid \mathcal D) \ge q\} \ge \pr\{{\rm Bin}(K, p_{\mathcal E}(q)) \le k - 1\}.$$
So the rank of Theorem 2 is unchanged. $G_0$ need not be bi-log-concave.

The condition holds when $|W_{\rm new}|$ is stochastically smaller than every $|W_i|$, i.e. the new
unit is no harder to cover than any calibration unit. It compares only the $q$-quantiles of $|W|$.

**Proof.** As in Theorem 2, $\pr(|W_{\rm new}| \le T \mid \mathcal D) \ge q \iff T \ge r_0 \iff N \le k - 1$
with $N = \#\{i : |V_i| < r_0\}$; this step uses only the law of $W_{\rm new}$. For $s < r_0$,
$s \le Q_q(|W_i|)$ gives $\pr(|W_i| \le s) < q$, and the transfer of §2 at threshold $s$, applied
to $G_i \in \mathcal B$ and the noise of unit $i$, gives $\pr(|V_i| \le s) \le p$ for every
$p > p_{\mathcal E}(q)$. Letting $s \uparrow r_0$, $\pr(|V_i| < r_0) \le p_{\mathcal E}(q)$, and $N$ is
stochastically smaller than ${\rm Bin}\{K, p_{\mathcal E}(q)\}$. $\square$

The converse half of Theorem 2 is the special case $G_i = G_0$, so the rank stays sharp.

## Remarks

1. **General form.** Without the quantile condition, set $c_i = \pr(|W_i| < r_0)$. The same
   argument gives $\pr(|V_i| < r_0) \le p_{\mathcal E}(c_i+)$, so $N$ is stochastically smaller
   than a sum of independent Bernoulli$\{p_{\mathcal E}(c_i+)\}$. By the Lemma below, the
   reliability is at least $\pr\{{\rm Bin}(K, \bar\pi) \le k - 1\}$ with
   $\bar\pi = K^{-1}\sum_i p_{\mathcal E}(c_i+)$, whenever $k \ge K\bar\pi + 2$.
   So a few easier calibration units are absorbed by slack elsewhere. The $c_i$ are not
   identifiable, so this matters for interpretation, not for computing a rank. Hoeffding
   (1956, Theorem 4) has the same comparison under the weaker $k - 1 \ge K\bar\pi$. The proof
   below is self-contained, and the extra unit of slack is harmless: the ranks of Theorem 2
   exceed $K\bar\pi$ by about $1.645\{K\bar\pi(1 - \bar\pi)\}^{1/2}$.

2. **Normalized scores.** If known scales $s_i > 0$, fixed before calibration (for instance from
   the training fit or area sizes), use the scores $|V_i|/s_i$ and the interval
   $[-s_{\rm new}T, s_{\rm new}T]$. Here $W_i/s_i \in \mathcal B$ and $e_i/s_i$ has a law in
   $\mathcal E$, because both classes are closed under scaling. The corollary then needs
   $Q_q(|W_i|/s_i) \ge Q_q(|W_{\rm new}|/s_{\rm new})$. In particular, latent laws that differ only
   in a known scale give Theorem 2 exactly. This is the usual normalized conformal score, and it
   covers heteroscedastic latent residuals.

3. **A link between units is necessary.** Suppose instead the new unit is a random draw from a
   heterogeneous population, so $W_{\rm new}$ follows the mixture of the $G_i$. Shape assumptions
   on the components then give nothing. Point masses lie in $\mathcal B$, their mixtures are
   arbitrary laws, and Proposition 2 applies: the guarantee is the shape-free one. Concretely,
   take atoms at $0$ and at $1 + \eta$, the latter with weight just above $1 - q$, and small
   Gaussian noise. Then the noisy level is about $(1 + q)/2$, while latent coverage at $r_0$
   is just below $q$. If instead the population (mixture) law is itself bi-log-concave and the
   noise is independent of the unit's type, Theorem 2 applies directly to the mixture.

## Sums of independent Bernoulli variables (for Remark 1)

Let $S = \sum_{i=1}^n X_i$ with independent $X_i \sim$ Bernoulli$(p_i)$, mean $\mu = \sum_i p_i$,
$\bar p = \mu/n$, and $a_k = \pr(S = k)$.

**Lemma A (mode).** If $j \ge \mu$ is an integer, then $a_{j+1} \le a_j$.

*Proof.* Fix $n$, $j$ and $\mu \le j$, and let $\Delta(p) = a_{j+1} - a_j$ on the compact set
$P_\mu = \{p \in [0,1]^n : \sum_i p_i = \mu\}$. Fix a pair $(i, l)$, let $U$ be the sum of the
other $n - 2$ variables with probabilities $u_k = \pr(U = k)$, and write $s = p_i + p_l$ and
$\pi = p_i p_l$. Since $q_iq_l = 1 - s + \pi$ and $p_iq_l + q_ip_l = s - 2\pi$,
$$a_k = (1 - s)u_k + s\,u_{k-1} + \pi(u_k - 2u_{k-1} + u_{k-2}),$$
so with $s$ fixed, $\Delta = \alpha(s) + \gamma\pi$, where $\gamma$ depends only on $U$. On the
segment $\{p_i + p_l = s\}$ the product $\pi$ is largest at $p_i = p_l$ and smallest at the
endpoints, where one of the two is $0$ or $1$.

Among the maximisers of $\Delta$ on $P_\mu$, a closed set, take one with the smallest
$\sum_i p_i^2$. Suppose two of its coordinates are in $(0,1)$ and unequal. Then $p_i$ lies
strictly inside its segment. If $\gamma < 0$, moving to an endpoint strictly increases $\Delta$,
which contradicts maximality. If $\gamma \ge 0$, setting both to $s/2$ does not decrease $\Delta$
and strictly decreases $\sum p^2$, which contradicts the choice. So all fractional coordinates
of this maximiser are equal, to $a$ say. The maximiser is then $S = m + B$, with $m$ coordinates
equal to $1$, $B \sim {\rm Bin}(r, a)$, zeros dropped, and $m + ra = \mu \le j$.

For this law, $i = j - m \ge ra \ge 0$. If $r = 0$ or $i \ge r$, then $a_{j+1} = 0$. Otherwise
$a_{j+1}/a_j = \pr(B = i + 1)/\pr(B = i) = (r - i)a/\{(i + 1)(1 - a)\}$. This is at most $1$
because $i \ge ra > ra - (1 - a)$. Hence $\max_{P_\mu}\Delta \le 0$. $\square$

**Lemma B (comparison).** If $c$ is an integer with $c \ge n\bar p + 1$, then
$\pr(S \le c) \ge \pr\{{\rm Bin}(n, \bar p) \le c\}$.

*Proof.* Replace $(p_i, p_l)$ by their mean $m$ and let $R$ be the sum of the other variables.
The law of $X_i + X_l$ gains $d = m^2 - p_ip_l = (p_i - p_l)^2/4 \ge 0$ at $0$ and at $2$ and
loses $2d$ at $1$, so
$$\Delta\pr(S \le c) = d\{F_R(c) - 2F_R(c-1) + F_R(c-2)\} = d\{\pr(R = c) - \pr(R = c-1)\} \le 0.$$
The inequality is Lemma A for $R$ with $j = c - 1$: the mean of $R$ is at most $n\bar p \le c - 1$.
Each such replacement keeps $\sum_i p_i$, so the condition persists. Averaging the largest and
smallest coordinates reduces $V = \sum_i (p_i - \bar p)^2$ by $(p_{\max} - p_{\min})^2/2 \ge V/(2n)$,
so iterating drives $p$ to $(\bar p, \ldots, \bar p)$. Since $\pr(S \le c)$ is continuous in $p$
and never increases along the way, the limit gives the claim. $\square$

Numerical check: on 20,000 random probability vectors ($n \le 40$; uniform, U-shaped, and near
$\{0, a, 1\}$), Lemma A had maximal violation $0$ and Lemma B $9 \times 10^{-16}$ (rounding).
This also held under Hoeffding's weaker condition $c \ge n\bar p$. Test:
`tests/test_bernoulli_sums.py`.

## Numerical check (R06)

Gaussian noise. Calibration latent laws are exponential tails $b - E/u$, the extremal laws, with
$Q_q(|W_i|) = r_0 + \tau_i/u_i$ ($\tau$ in units of the tail scale $1/u$) and noise scale
adversarial per unit; each $\pi_i = \pr(|V_i| < r_0)$ is computed from a closed form.

*Part 1: unit parameters i.i.d.* The indicators are i.i.d. with $H = E\pi_i$, and the
reliability is $\pr\{{\rm Bin}(K, H) \le k - 1\}$. $H$ is exact in the homogeneous and point-mass
rows. In the heterogeneous rows it is a Monte Carlo mean over 4000 draws, with standard error
about $2 \times 10^{-4}$, so those reliabilities carry Monte Carlo error. This does not matter
here, because $H - \Psi_N$ is about $-0.01$, fifty standard errors.

| design, q = 0.9 | $H - \Psi_N(q)$ | Theorem 2 rank, rel. at $K = 10^4$ / $10^6$ | usual rank, $10^4$ / $10^6$ |
|---|---|---|---|
| homogeneous extremal ($\tau = 0$, $u = 200$) | $0$ (to $10^{-14}$) | 0.952 / 0.950 | 0.918 / 0.150 |
| homogeneous, $r_i = 0.999995$, $u = 200$ ($\tau = -0.001$) | $+8.8 \times 10^{-4}$ | 0.915 / 0.094 | 0.864 / 0.000 |
| homogeneous, $r_i = 0.999995$, $u = 2000$ ($\tau = -0.01$) | $+8.9 \times 10^{-3}$ | 0.086 / 0.000 | 0.050 / 0.000 |
| heterogeneous, $\tau_i = \lvert N(0, 0.02^2)\rvert$ (MC) | $-0.014$ | 1.000 / 1.000 | 1.000 / 1.000 |
| 10% of units easier, $\tau = -0.02$ (MC) | $-0.010$ | 1.000 / 1.000 | 1.000 / 1.000 |
| random unit (mixture of atoms) | $+0.049$ | 0.000 / 0.000 | 0.000 / 0.000 |

At $q = 0.8$ the pattern is the same. Under the mixture design, the shape-free rank (level
$(1 + q)/2$) keeps reliability $\ge 0.95$ at every $K$, as Remark 3 predicts.

*Part 2: fixed units, independent but not identically distributed.* One draw of the parameters
of $K$ units is held fixed, and the reliability is the Poisson-binomial probability, computed
exactly by recursion (`results/heterogeneous_latent_fixed.csv`). When the condition holds,
$\max_i \pi_i = \Psi_N(q)$, attained by units with $\tau_i \approx 0$, as the corollary allows. The
Theorem 2 rank then has reliability $0.9989$ ($K = 10^3$) and $1.0000$ ($10^4$) at $q = 0.9$, and
$0.9955$ and $1.0000$ at $q = 0.8$. With 10% easier units, $\max_i \pi_i$ exceeds $\Psi_N$
($0.9187$ at $q = 0.9$), but the reliability stays $\ge 0.997$.

Reading:
- The bound is attained by homogeneous extremal laws. Heterogeneity across units that satisfy
  the condition only adds slack, the same message as R03 for heterogeneous noise scales.
- The quantile condition is sufficient, not necessary. Some designs that violate it keep the
  guarantee (row 5, Part 2, Remark 1). But a violation can break the guarantee however small it
  is. At the homogeneous extremal law, lowering every $r_i$ from $1$ to $0.999995$ already
  drops the reliability of the Theorem 2 rank to $0.094$ at $K = 10^6$. Whether a violation is
  small is relative to the latent tail scale: with a tail ten times steeper, the same violation
  gives $0.086$ at $K = 10^4$. This is the sharpness of Theorem 2 seen from the other side: at
  the extremal law the condition has no slack.
- Correct message: *the quantile ordering is sufficient; some designs that violate it keep the
  guarantee, but even a small violation can break it, depending on the latent law and on $K$.*
  No latent shape assumption on the calibration laws can replace the link between units
  (Remark 3).

## For the paper

- Discussion §6 currently says the method "requires a bi-log-concave latent law shared by the
  calibration units and the new one". Replace this with the corollary: the laws may differ,
  provided each calibration unit's law is bi-log-concave and no easier than the new unit's at
  the $q$-quantile of $|W|$. State Remark 3 as the reason some link is needed.
- Remark 2 (normalized scores) is the practical form for small areas of different sizes.
