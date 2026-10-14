"""R07: exact latent reliability under contaminated bi-log-concave latent laws (notes/contamination.md).

Latent law (threshold 1, Gaussian noise N(0, s^2), s = 1e-3):
    W = (1 - c) * [exponential tail W0 = 1 + (s/u)(a0 - E), a0 = log(1/q0), q0 = q/(1 - c)]
        + c * [point mass at 1 + s*eta'],
with u the maximiser of Gamma(q0, .) (Theorem 2). Then pr(|W| <= 1) = (1 - c) q0 = q, so the latent
q-quantile of |W| is r = 1, and every calibration unit has
    H = pr(|V| < 1) = (1 - c) pr(|W0 + e| < 1) + c pr(|1 + s eta' + e| < 1),
computed in closed form; as s, eta' -> 0, H -> p_c(q) = (1 - c) Psi(q/(1 - c)) + c/2. The latent
reliability of rank k is pr{Bin(K, H) <= k - 1} exactly (proof of Theorem 3).

Rules: the usual rank (level q), the rank of Theorem 3 for Gaussian noise (level Psi(q), c = 0),
and the contamination rank (level p_c(q)).

  python experiments/r07_contamination.py
Writes results/contamination.csv and results/contamination_levels.csv.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import log_ndtr, ndtr

from latentcov.contamination import condition_holds, level
from latentcov.levels import psi_gaussian_float
from latentcov.rank import rank_for_level

RESULTS = Path(__file__).resolve().parents[1] / 'results'
S, ETA, DELTA = 1e-3, 1e-6, 0.05
QS = (0.8, 0.9)
CS = (0.0, 0.005, 0.01, 0.02, 0.05)
KS = [110, 1000, 10_000, 100_000, 1_000_000]


def unit_H(q, c):
    q0 = q / (1 - c)
    _, u = psi_gaussian_float(q0)
    lam, A = u / S, 1 + S * math.log(1 / q0) / u    # rate and right end of W0

    def noisy_cdf(x):                               # pr(W0 + e <= x)
        z = x - A
        return ndtr(z / S) + math.exp(lam * z + (lam * S) ** 2 / 2 + log_ndtr(-z / S - lam * S))
    p0 = noisy_cdf(1) - noisy_cdf(-1)
    h = 1 + S * ETA
    ph = ndtr((1 - h) / S) - ndtr((-1 - h) / S)
    return (1 - c) * p0 + c * ph


def main():
    rows, lv = [], []
    for q in QS:
        for c in CS:
            H = unit_H(q, c)
            pc = level(q, c)
            assert condition_holds(q, c), (q, c)
            lv.append(dict(q=q, c=c, level=pc, H=H, H_minus_level=H - pc))
            for K in KS:
                for rule, p in (('usual', q), ('Theorem 3 (c = 0)', level(q, 0.0)),
                                ('contamination rank', pc)):
                    k = rank_for_level(K, p, DELTA)
                    rel = float(stats.binom.cdf(k - 1, K, H)) if k is not None else np.nan
                    rows.append(dict(q=q, c=c, K=K, rule=rule, level=p, rank=k, reliability=rel))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / 'contamination.csv', index=False)
    pd.DataFrame(lv).to_csv(RESULTS / 'contamination_levels.csv', index=False)
    pd.set_option('display.width', 200)
    print(pd.DataFrame(lv).to_string(index=False))
    print(d.pivot_table(index=['q', 'c', 'K'], columns='rule', values='reliability').round(4)
          .to_string())


if __name__ == '__main__':
    main()
