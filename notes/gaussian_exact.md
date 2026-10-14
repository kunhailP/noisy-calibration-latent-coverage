# Exact two-sided level for Gaussian noise

Status: proof complete. Every step is checked numerically (`tests/test_gaussian.py`). The two
cited facts (F2), (F3) are inequalities (3) and (4) of Sampford (1953), checked against the
paper's text (transcription supplied by the author, 2026-10-07; the PDF itself has not been read
here). The algebra of Sampford's proof of (4) was re-derived line by line.

## Setting

Gaussian noise class $\mathcal N = \{\sigma Z : \sigma > 0\}$, closed under scaling and negation,
$\beta_{\mathcal N} = 1/2$. With $a = \log(1/q)$,

$$\Psi_{\mathcal N}(q) = \sup_{s>0} G(q,s),\qquad
G(q,s) = E\min(1, q e^{-sZ}) = \Phi(-a/s) + q e^{s^2/2}\,\Phi(a/s - s).$$

Write $m(b) = \phi(b)/\Phi(b)$ (inverse Mills ratio of the left tail) and
$A(b) = m(b)\{b + m(b)\} = 1 - \operatorname{var}(Z \mid Z \le b)$.

## Facts used

- (F1) $m'(b) = -m(b)\{b + m(b)\} = -A(b)$. Direct differentiation.
- Notation of Sampford (1953): $R_x$ is Mills' ratio, $\nu(x) = 1/R_x = \phi(x)/\{1 - \Phi(x)\}$
  and $\lambda_S(x) = \nu'(x) = \nu(\nu - x)$. By symmetry $m(b) = \nu(-b)$, so
  $A(b) = -m'(b) = \lambda_S(-b)$ and $A'(b) = -\lambda_S'(-b)$.
- (F2) $0 < A(b) < 1$ for every finite $b$: Sampford's inequality (3), $0 < \lambda_S < 1$ for all
  finite $x$, proved there from $\operatorname{var}(Z \mid Z \ge x) = 1 - \nu(\nu - x) > 0$. The
  limits $A(b) \to 1$ as $b \to -\infty$ and $A(b) \to 0$ as $b \to \infty$ are elementary: as
  $x \to \infty$, $\nu = x + x^{-1} + O(x^{-3})$, so $\lambda_S \to 1$; as $x \to -\infty$, $\nu \to 0$
  faster than $|x|$ grows, so $\lambda_S \to 0$.
- (F3) $A$ is strictly decreasing: Sampford's inequality (4), $\lambda_S'(x) = \nu\{(\nu - x)(2\nu - x) - 1\} > 0$
  for all finite $x$. Conjectured for $x > 0$ by Birnbaum (1950), proved for large $x$ by Murty
  (1952), proved for all finite $x$ by Sampford. His argument: $\varphi = (\nu - x)(2\nu - x)$ tends
  to $1$ as $x \to \infty$ and to $\infty$ as $x \to -\infty$, and $\varphi' = \nu(\varphi - 1) + 2(\nu - x)(\lambda_S - 1) < \nu(\varphi - 1)$.
  So $\varphi$ cannot reach $1$: at the first point where it did, $\varphi' < 0$ would take it
  below $1$, and the later return towards $1$ would force an interior minimum with
  $\varphi \le 1$ and $\varphi' = 0$, which is impossible. The identity for $\varphi'$ was re-derived
  by expanding both sides to $4\nu\lambda_S - 3x\lambda_S - 3\nu + 2x$.
- Citation: M. R. Sampford (1953), Some inequalities on Mill's ratio and related functions,
  *Ann. Math. Statist.* 24(1), 130–132, doi:10.1214/aoms/1177729093 (bibliographic data checked
  against Crossref, 2026-10-07), inequalities (3) and (4).
- (I) Identity: $q e^{s^2/2}\phi(a/s - s) = \phi(a/s)$, because
  $(a/s)^2 - (a/s - s)^2 = 2a - s^2$.

## Theorem

(i) If $q \le e^{-1}$, then $\Psi_{\mathcal N}(q) = 1/2$, approached as $s \to \infty$ and not attained.

(ii) If $q > e^{-1}$, then $G(q, \cdot)$ has a unique maximiser $s^* = m(b^*)$, where $b^*$ is the
unique solution of $A(b^*) = \log(1/q)$, and

$$\Psi_{\mathcal N}(q) = \Phi(-c) + \phi(c)/m(b^*),\qquad c = b^* + m(b^*).$$

(iii) $\Psi_{\mathcal N}$ is continuously differentiable and convex on $(0,1)$, strictly convex on
$(e^{-1}, 1)$, with $\Psi_{\mathcal N}'(q) = e^{m(b^*)^2/2}\,\Phi(b^*)$ there and $0$ below.

(iv) Hence $\phi_{\mathcal N}$ is convex and $p_{\mathcal N}(q) = \Psi_{\mathcal N}(q)$ for every
$q \in (e^{-1}, 1)$: the two-sided level equals the one-sided one, and the smallest valid rank of
Theorem 2 is exact for Gaussian noise, as for symmetric unimodal noise.

At $q = 0.9$: $b^* = 2.045457$, $s^* = 0.0502739$, $\Psi_{\mathcal N}(0.9) = 0.90080346944649\ldots$,
certified to $2.3 \times 10^{-16}$ (`latentcov.gaussian.psi_gaussian_enclosure`).

## Proof

**Derivative in $s$.** Let $b = a/s - s$. By (I),
$$\partial_s G = \phi(a/s)\,\frac{a}{s^2} + q e^{s^2/2}\{s\Phi(b) - \phi(b)(a/s^2 + 1)\}
= q s e^{s^2/2}\Phi(b) - \phi(a/s) = \phi(a/s)\Big\{\frac{s}{m(b)} - 1\Big\},$$
using $q e^{s^2/2}\phi(b) = \phi(a/s)$ twice. So $\operatorname{sign}\partial_s G = \operatorname{sign}\{s - m(b)\}$.

**Sign change.** For fixed $a > 0$, $s \mapsto b = a/s - s$ is a strictly decreasing bijection of
$(0,\infty)$ onto $\mathbb R$, and $s^2 + bs = a$. The map $t \mapsto t^2 + bt$ is strictly
increasing on $t > -b/2$. Both $s$ and $m(b)$ lie there: $s = \{-b + (b^2 + 4a)^{1/2}\}/2 > -b/2$,
and $m(b) > -b$ because $A(b) = m(b)\{b + m(b)\} > 0$ with $m(b) > 0$ (F2). Hence

$$s > m(b) \iff a = s^2 + bs > m(b)^2 + b\,m(b) = A(b).$$

As $s$ increases, $b$ decreases and $A(b)$ increases strictly (F3), through all of $(0,1)$ (F2).

- If $a \ge 1$, then $a > A(b)$ for every $s$, so $G(q,\cdot)$ is strictly increasing. As
  $s \to \infty$, $\Phi(-a/s) \to 1/2$ and, by (I) and Mills' inequality,
  $q e^{s^2/2}\Phi(b) \le q e^{s^2/2}\phi(b)/|b| = \phi(a/s)/|b| \to 0$. This proves (i).
- If $a < 1$, $\partial_s G$ is positive before and negative after the unique $s^*$ with
  $A(b^*) = a$, so $s^*$ is the unique maximiser, and $s^* = m(b^*)$. At $s^*$, by (I),
  $q e^{s^2/2}\Phi(b) = q e^{s^2/2}\phi(b)/s = \phi(a/s)/s$ and $a/s = b^* + s^* = c$. This proves (ii).

**Derivative in $q$.** With $s$ fixed, by (I),
$$\partial_a G = -\frac{\phi(a/s)}{s} - q e^{s^2/2}\Phi(b) + q e^{s^2/2}\frac{\phi(b)}{s} = -q e^{s^2/2}\Phi(b),$$
so $\partial_q G = e^{s^2/2}\Phi(b)$. The maximiser is unique, and $G$ is continuously
differentiable. Moreover $s^*(q) = m(b^*(q))$ is continuous, because $b^*(q) = A^{-1}(\log 1/q)$
and $A$ is a continuous strictly monotone bijection. By Danskin's theorem,
$\Psi_{\mathcal N}'(q) = D(b^*)$ with $D(b) = e^{m(b)^2/2}\Phi(b)$.

**Monotonicity.** $q(b) = e^{-A(b)}$ increases strictly from $e^{-1}$ to $1$ (F2, F3). By (F1),
$$\frac{d}{db}\log D(b) = m m' + \frac{\phi(b)}{\Phi(b)} = m\{1 - A(b)\} = m(b)\operatorname{var}(Z\mid Z\le b) > 0.$$
So $\Psi'_{\mathcal N}$ is strictly increasing on $(e^{-1}, 1)$.

As $q \downarrow e^{-1}$, $b^* \to -\infty$, and $m(b) = |b| + |b|^{-1} + O(|b|^{-3})$, so
$m^2 = b^2 + 2 + o(1)$ and $D(b) \sim e\,\phi(b)e^{b^2/2}/|b| = e/\{(2\pi)^{1/2}|b|\} \to 0$. This
matches the derivative $0$ of the constant part, which proves (iii).

**Two-sided level.** $\Psi_{\mathcal N}$ maps $(e^{-1},1)$ onto $(1/2, 1)$, strictly increasing and
convex, so its inverse $\lambda_{\mathcal N}$ is concave there. Then
$\phi_{\mathcal N}(a) = 1 - \lambda_{\mathcal N}(1 - a)$ is convex on $[0, 1/2)$ with $\phi(0) = 0$,
hence superadditive there. In the definition of $p_{\mathcal N}(q)$ the failure probabilities
satisfy $a_L + a_R \le 1 - p < 1/2$, since $p \ge \Psi_{\mathcal N}(q) > 1/2$. The argument of
§2.2 of the earlier version then gives $p_{\mathcal N}(q) = \Psi_{\mathcal N}(q)$. This
proves (iv). $\square$

## Consequences for the paper

- The open question "convexity of $\phi_{\mathcal E}$ for Gaussian noise" is closed. Table 1's
  Gaussian row becomes exact: level $0.9008035$ and ranks 917 / 9058 (was 9058–9060).
- Corollary 1 becomes exact: $R^{\mathcal B}_{p,0.9} \le 1$ for all $x$ iff $p \ge \Psi_{\mathcal N}(0.9)$.
  The slack $10^{-3}$ is replaced by $8.035 \times 10^{-4}$.
- $q > e^{-1}$ is the same threshold as for symmetric unimodal noise (Theorem 1(i)): in both
  classes the transfer is nontrivial exactly above $e^{-1}$. Below it, Gaussian noise gives
  $\Psi = 1/2$, the shape-free value. Worth stating.
- The proof uses only (F1)–(F3), that is, Sampford's two inequalities. They are existing
  results and are cited, not claimed. What is new is the reduction: the first-order condition
  is $s = m(b)$, the envelope derivative is $e^{m^2/2}\Phi(b)$, and its monotonicity is exactly
  Sampford's (3). The analysis also explains why $e^{-1}$ is the threshold: it is $e^{-\sup A}$. The same scheme (first-order condition = Mills ratio of the
  noise, envelope derivative = tilted cdf) may extend to other single noise laws $\epsilon$,
  through $G_\epsilon(q,s) = E\min(1, q e^{-s\epsilon})$; for log-concave $\epsilon$ the analogue
  of (F3) is the monotonicity of $\operatorname{var}(\epsilon_{\text{tilted}} \mid \cdot)$. Open.
