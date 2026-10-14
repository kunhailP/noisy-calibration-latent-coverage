# Latent coverage from noisy calibration

Code, proofs and manuscript for **"Latent coverage from noisy calibration"** (Kun Woo Park).

Prediction intervals for a latent quantity, such as the true mean of an unsampled small area or a
response measured with error, are often calibrated on noisy proxies with the usual conformal
threshold. The paper asks when this threshold covers the latent target with high probability if the
noise law and variance are unknown and differ between units, and what the smallest correction is.

- **Manuscript:** [`paper/main.pdf`](paper/main.pdf) and the Supplementary Material
  [`paper/supplement.pdf`](paper/supplement.pdf) (Statistica Sinica template).
- **Main results.** Unimodality of the latent law gains nothing over no shape assumption. Under
  bi-log-concavity the noisy level that guarantees latent coverage `q` is `Psi(q)`, exact for
  Gaussian noise (closed form) and symmetric unimodal noise, and certified for mean-zero
  log-concave noise at `q = 0.9`. Calibrating at that level gives a conformal rank that keeps the
  latent guarantee with probability `1 - delta` for every noise law and scale in the class, without
  noise variances; the usual rank is nearly valid, and fails only when the residual density
  increases at an end of the interval. Extensions cover centred residuals, contamination of the
  latent law, latent laws that differ between units, and misspecified noise classes.
- **Status of each claim** (proved, certified in ball arithmetic, numerical) is listed in Table S1
  of the Supplementary Material; derivations are in [`notes/`](notes/).

The certified modules `interval.py` and `noise_classes.py` come unchanged from an earlier version of
this work (package `uai` renamed `latentcov`); `legacy/` keeps the certificate outputs they produced.

## Layout

```
src/latentcov/  interval, noise_classes (earlier version); levels, rank, gaussian, contamination, centred,
                intermediate;
                kv/ procedures that use known noise variances (competitors in R04, R05)
experiments/    r01-r10, fetch_apipop.py and supp_tables.py (see below)
results/        outputs of the experiments
tests/          fast checks
notes/          proofs and derivations (gaussian_exact, heterogeneous_latent, contamination, centred, intermediate)
paper/          main.tex and supplement.tex (Statistica Sinica templates), refs.bib, supp_tables/
legacy/         certificate files of the earlier version
data/           downloaded data, not redistributed (created by `make data`)
```

## Usage

```python
from latentcov import rank
r = rank(K=10_000, q=0.9, delta=0.05, noise='gaussian', latent='bi_log_concave')
r.k, r.k_necessary, r.level.support      # 9058, 9058, 'proved; certified in ball arithmetic'
```

`noise` is `gaussian`, `symmetric_unimodal` or `mean_zero_log_concave`; `latent` is
`bi_log_concave`, `symmetric_unimodal` (Anderson) or `none` (no shape assumption); `sided` is `two`
(interval `[-T, T]`) or `one` (`(-inf, T]`). `k` is the valid rank; the smallest valid rank lies
in `[k_necessary, k]`, and the two coincide for Gaussian and symmetric unimodal noise. `k` is
`None` when no rank is valid.

## Reproduce

Python 3.12; exact versions in `requirements-lock.txt`.

```
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt && .venv/bin/pip install -e .
make test
make all            # or one target at a time, as below
```

| target | script | what | time (32 cores) | output in `results/` |
|---|---|---|---|---|
| `reliability` | r01 | exact latent reliability at the Gaussian-extremal law, q = 0.8, 0.9, 0.95 | seconds | `reliability_*`, `fig_reliability_q.*` |
| `same-information` | r02 | rules with the same information; bimodal latent law, mixed noise | minutes | `same_information_v2*` |
| `mixed-noise` | r03 | exact stress test under mixed noise laws | minutes | `mixed_noise_stress*`, `fig_mixed_noise.*` |
| `competitors` | r04 | comparison with existing methods (LatentCP, Fay-Herriot, deconvolution, HetLDC) | hours | `competitors_v2_hetldc*` |
| `data` | fetch_apipop | pinned CRAN `survey` 4.5, checked by SHA-256 | seconds | `data/apipop.pkl` |
| `apipop` | r05 | school-district application | minutes | `apipop*` |
| `heterogeneous` | r06 | exact check of the heterogeneous-latent corollary | seconds | `heterogeneous_latent.csv` |
| `contamination` | r07 | exact reliability under contaminated bi-log-concave latent laws | seconds | `contamination*.csv` |
| `centred` | r08 | level p_kappa(q) for centred latent residuals, adversarial check, failure K and ranks (notes/centred.md) | about an hour | `centred_levels.csv`, `centred_validation*.csv`, `centred_check3.csv`, `centred_ranks.csv` |
| `misspec` | r09 | misspecified noise class: levels of single noise laws and the reliability of the class ranks | minutes | `noise_misspec.csv` |
| `intermediate` | r10 | exact reliability in intermediate designs: miscentred skewed residuals, noise-to-signal ratios, heterogeneous scales, uniform noise, small K, proportions (notes/intermediate.md) | minutes | `intermediate*.csv` |
| `tables` | supp_tables | tables of the Supplementary Material, from `results/` | seconds | `paper/supp_tables/*.tex` |
| `paper` | (pdflatex, bibtex) | `paper/main.pdf` and `paper/supplement.pdf` with cross references | a minute | `paper/` |

The Makefile sets one BLAS thread per process (`OMP_NUM_THREADS=1` etc.); set the same when
running a script directly, or the parallel experiments oversubscribe the cores. `PROCS` sets the
number of worker processes (default 32). R04 writes one line per finished job to
`results/competitors_v2_hetldc.ckpt.jsonl` and, rerun with the same arguments, skips the jobs
already there; delete that file to start afresh.

## Citation

Park, K. W. Latent coverage from noisy calibration. Manuscript, 2026.
