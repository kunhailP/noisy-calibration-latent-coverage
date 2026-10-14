PY ?= .venv/bin/python
PROCS ?= 32

# one BLAS thread per worker: the experiments parallelise over processes, and multithreaded BLAS
# inside each worker oversubscribes the cores (R04 made no progress until this was set)
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

.PHONY: intermediate misspec test reliability same-information mixed-noise competitors data apipop heterogeneous contamination centred all tables paper

test:
	$(PY) -m pytest -q

# seconds: exact latent reliability at the Gaussian-extremal law, q = 0.8, 0.9, 0.95
reliability:
	$(PY) experiments/r01_reliability_curves.py

# minutes: rules with the same information, bimodal latent law, mixed noise
same-information:
	$(PY) experiments/r02_same_information.py 2000 $(PROCS)

# exact reliability under mixed noise laws at the extremal latent law (minutes, one core per case)
mixed-noise:
	$(PY) experiments/r03_mixed_noise_stress.py

# hours on 32 cores: comparison with existing methods; resumes from its checkpoint if interrupted
competitors:
	$(PY) experiments/r04_competitors.py 300 200 $(PROCS) hetldc

# downloads the pinned CRAN `survey` 4.5 tarball and checks api.rda against its SHA-256
data/apipop.pkl:
	$(PY) experiments/fetch_apipop.py

data: data/apipop.pkl

# school-district application
apipop: data/apipop.pkl
	$(PY) experiments/r05_apipop.py 120 $(PROCS)

# seconds: exact check of the heterogeneous-latent corollary
heterogeneous:
	$(PY) experiments/r06_heterogeneous_latent.py

# seconds: exact reliability under contaminated bi-log-concave latent laws
contamination:
	$(PY) experiments/r07_contamination.py

# about an hour on 160 cores: level p_kappa(q) for centred latent residuals (global search over the
# two-piece family, mpmath recheck), adversarial search outside the family, failure K and ranks
centred:
	$(PY) experiments/r08_centred.py levels $(PROCS)
	$(PY) experiments/r08_centred.py validate $(PROCS)
	$(PY) experiments/r08_centred.py check3 $(PROCS)
	$(PY) experiments/r08_centred.py closure $(PROCS)
	$(PY) experiments/r08_centred.py table

all: test reliability same-information mixed-noise competitors apipop heterogeneous contamination centred misspec intermediate

# minutes: misspecified noise class, exact reliability of the class ranks at the extremal law
misspec:
	$(PY) experiments/r09_noise_misspec.py $(PROCS)

# minutes on 48 cores: exact reliability in intermediate designs (miscentred skewed residuals,
# noise-to-signal ratios, heterogeneous scales, proportions); resumes from results/intermediate_H.csv
intermediate:
	$(PY) experiments/r10_intermediate.py all $(PROCS)

# tables of the Supplementary Material, written from results/ (no recomputation)
tables:
	$(PY) experiments/supp_tables.py

# main.pdf and supplement.pdf with cross references (xr) and bibliographies
paper: tables
	cd paper && for i in 1 2; do pdflatex -interaction=nonstopmode main >/dev/null; \
	  pdflatex -interaction=nonstopmode supplement >/dev/null; done; \
	  bibtex main >/dev/null; bibtex supplement >/dev/null; \
	  for i in 1 2 3; do pdflatex -interaction=nonstopmode main >/dev/null; \
	  pdflatex -interaction=nonstopmode supplement >/dev/null; done
