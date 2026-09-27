# Noisy calibration, latent coverage and the location of the mean

Code, results and manuscript for the paper *Noisy calibration, latent coverage and the location of the mean* by Kun Woo Park (in preparation for *Biometrika*). Every number in the paper and its Supplementary Material can be regenerated from this repository.

## The question

A prediction interval for a latent quantity, such as the true mean of an area that a survey did not sample, is often calibrated on noisy proxies. The latent residual `W` is observed only through `V = W + e`, with `e ~ N(0, D)` and `D` known. The threshold `t` at which `|V|` has coverage `p` is observable. The question is which radius gives `W` coverage `q`, uniformly over all log-concave laws of `W`. Writing `x = D/t²`, the answer is `t R_{p,q}(x)`, the sharp radius studied in the paper. It is sharp for this class and this single constraint; it is not claimed to be optimal among all procedures that use the whole calibration sample.

## Main results

1. **Reduction.** `R_{p,q}` is a supremum over point masses and log-affine laws on a segment, with closed-form Gaussian convolutions (Proposition 1).
2. **No restriction on the mean.** `R_{q,q}(x) <= 1 + c_q x^{1/2}` for every `x`, sharp as `x -> 0`. Ball arithmetic gives `c_0.9` in `[0.0190618, 0.0190619]` (Theorem 1, Lemma 1).
3. **Mean away from the boundary.** If `|E(W)| <= B < t`, then `Q_q(|W|) <= B + {(t - B)² + D}^{1/2} <= t + D/{2(t - B)}` for every `D` (Theorem 2). Asymmetric laws with mean zero attain the order `D` (Proposition 2).
4. **Transition.** A shortfall of order `D^{1/2}` requires the mean within `O(D^{1/2})` of the boundary. At distance `κ D^{1/2}` the limiting coefficient is the one-dimensional extremum `L_q(κ)`, which equals `c_q` up to `κ*`; `κ*` is about 20 at `q = 0.9` (Lemma 2, Theorem 3, Figure 1).
5. **Slack.** A calibration slack of `10^{-3}` at `q = 0.9` removes the shortfall at every noise level (Corollary 1).
6. **Finite-sample procedure.** With heterogeneous known noise variances, an order-statistic level and a per-data-set certified radius give conditional reliability for every log-concave latent law (Proposition 3, Table 1).

Two further results are computed here but are not yet in the manuscript. `e33_edge.py` studies the noise-dominated regime near the feasibility edge `x_p`: there `R_{p,q}(x) ~ M_q (x_p - x)^{1/2}`, with `M_q = sup Q_q(|W|)/E(W²)^{1/2}` over log-concave laws (`M_0.9 = 1.8532`). `e34_competitors.py` compares the procedure with LatentCP and with deconvolution conformal prediction on the data sets of Table 1.

## Layout

```
paper/          main.tex and supplement.tex (Biometrika 2025 class), figures, references, PDFs
src/uai/        library: closed forms and constants (extremal), certificates (certify, interval),
                procedures, estimated-variance rules, closed-form latent laws
experiments/    one script per experiment, numbered as in the Supplementary Material
results/        outputs of the experiments (CSV/JSON) used in the paper
tests/          fast checks of constants, counterexamples and bounds
```

## Installation

Python 3.10 or later.

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,figures]"
make test
```

`python-flint` provides the Arb ball arithmetic. Scripts take the number of worker processes as an argument (`PROCS`, default 8). Set `OMP_NUM_THREADS=1` when running many workers.

## Reproducing the paper

| Item | Command | Output |
|---|---|---|
| Constants `c_q`, `C_{p,q}`, slack intervals (Lemma 1, Corollary 1) | `make constants` | `results/interval_constants.json` |
| Centring and transition values, Figure 1 | `make quick figures` | `results/centering.json`, `paper/fig_transition.pdf` |
| Certified `R_{p,q}`, boundary map (Supplement S2–S3) | `make certified` | `results/certified_R.csv`, `results/boundary_map.csv` |
| Table 1 and the shape-free rule | `make simulations` | `results/hetldc_synth_summary.csv`, `results/shape_free_summary.csv` |
| Comparison with LatentCP and deconvolution | `make comparison` | `results/competitors_summary.csv` |
| Estimated variances (Supplement S4) | `make estimated` | `results/estimated_scale_summary.csv`, `results/areawise_variance_summary.csv` |
| School-district application | `make apipop` | `results/hetldc_apipop_summary.csv` |
| Plug-in counterexamples, oracle-matched widths | `make supplement-extras` | `results/hetero_kernel.csv`, `results/conditional_synth_exact_*.csv` |
| Manuscript and supplement | `make paper` | `paper/main.pdf`, `paper/supplement.pdf` |

`make certified` and `make simulations` take hours on a few cores; the other targets take minutes.

## Precision

The constants `c_q` and `C_{p,q}` and the slack intervals are certified in Arb ball arithmetic with outward rounding. The upper bounds of `R_{p,q}` come from a branch and bound. Its closed forms are evaluated in double precision, and boxes are cleared with a margin of `10^{-9}`; this is not interval arithmetic. The transition coefficients, the centred-law coefficients and the edge constants `M_q` are computed in double precision and are not certified.

## Citation

```
@unpublished{park2026noisy,
  author = {Park, Kun Woo},
  title  = {Noisy calibration, latent coverage and the location of the mean},
  year   = {2026},
  note   = {Manuscript}
}
```
