"""Tables of the Supplementary Material, written from the outputs in results/ (no recomputation).

  python experiments/supp_tables.py
Writes paper/supp_tables/*.tex, each a tabular body that supplement.tex inputs.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / 'results', ROOT / 'paper' / 'supp_tables'
OUT.mkdir(exist_ok=True)
BLC = ['normal', 'Laplace', 'centred gamma', 'truncated exponential', 'bimodal']


def f3(x):
    return '--' if pd.isna(x) else f'{x:.3f}'


def pct(x, nd=1):
    return '--' if pd.isna(x) else f'{x:.{nd}f}'


def rng_pct(a, b, nd=1):
    if pd.isna(a):
        return '--'
    a, b = round(a, nd), round(b, nd)
    if a == b:
        return f'{a:.{nd}f}'
    return f'{a:.{nd}f} to {b:.{nd}f}' if a < 0 else f'{a:.{nd}f}--{b:.{nd}f}'


def write(name, lines):
    (OUT / name).write_text('\n'.join(lines) + '\n')
    print('wrote', OUT / name)


def r01():
    """Exact reliability at the Gaussian-extremal law, all rules, quoted K."""
    d = pd.read_csv(RES / 'reliability_curves.csv')
    Ks = [110, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
    d = d[d.K.isin(Ks)]
    rules = ['marginal', 'usual', 'gaussian_certified', 'symmetric_unimodal']
    lines = []
    for q in (0.8, 0.9, 0.95):
        for i, K in enumerate(Ks):
            row = [f'${q}$' if i == 0 else '', f'$10^{{{int(round(math.log10(K)))}}}$'
                   if K != 110 else '$110$']
            for r in rules:
                x = d[(d.q == q) & (d.K == K) & (d.rule == r)].iloc[0]
                row += [f'{int(x["rank"])}', f3(x.reliability)]
            lines.append(' & '.join(row) + r' \\')
        if q != 0.95:
            lines.append(r'\addlinespace')
    write('r01_reliability.tex', lines)


def r03():
    """Mixed noise laws at the extremal latent law: H, excess, reliability of both ranks."""
    s = pd.read_csv(RES / 'mixed_noise_stress_summary.csv')
    lab = {'uniform 1.00': 'uniform', 'uniform 0.50': 'uniform $1/2$, rest',
           'uniform 0.25': 'uniform $1/4$, rest', 'centred exp 1.00': 'centred exp.',
           'centred exp 0.50': 'centred exp.\\ $1/2$, rest',
           'centred exp 0.25': 'centred exp.\\ $1/4$, rest',
           'centred exp both signs': 'both signs, $1/2$ each'}
    lines = []
    for _, x in s.iterrows():
        m = next(v for k, v in lab.items() if x.mixture.startswith(k))
        cls = 'SU' if x.noise_class == 'symmetric_unimodal' else 'LC'
        row = [cls, f'${x.q}$', m, 'yes' if x.hetero_scales else 'no',
               f'{x.H:.6f}', f'{x.excess * 1e3:.3f}']
        for K in (1000, 10_000, 100_000):
            row += [f3(x[f'usual K={K}']), f3(x[f'class rank K={K}'])]
        lines.append(' & '.join(row) + r' \\')
    write('r03_mixed.tex', lines)


def r02():
    """Rules with the same information: rank, minimum reliability, width over the usual rank and
    saving over the no-shape rule, both K and all q."""
    s = pd.read_csv(RES / 'same_information_v2_summary.csv')
    order = ['usual', 'Gaussian', 'symmetric unimodal', 'log-concave', 'no shape, median-zero',
             'no shape, log-concave']
    name = {'usual': 'usual', 'Gaussian': 'Gaussian', 'symmetric unimodal': 'symm.\\ unimodal',
            'log-concave': 'mean-zero log-concave', 'no shape, median-zero':
            'no shape, median zero', 'no shape, log-concave': 'no shape, log-concave'}
    lines = []
    for q in (0.8, 0.9, 0.95):
        for K in (110, 1000):
            d = s[(s.q == q) & (s.K == K)].set_index('rule')
            first = True
            for r in order:
                if r not in d.index:
                    continue
                x = d.loc[r]
                k = '--' if pd.isna(x['rank']) else f'{int(x["rank"])}'
                rel = x.min_reliability_all if r == 'usual' else x.min_reliability_covered
                row = [f'${q}$' if first and K == 110 else '', f'${K}$' if first else '',
                       name[r], k, f3(rel), rng_pct(x.above_usual_min, x.above_usual_max),
                       rng_pct(x.below_noshape_min, x.below_noshape_max, 0)
                       + ('' if pd.isna(x.below_noshape_median)
                          else f' ({x.below_noshape_median:.0f})')]
                lines.append(' & '.join(row) + r' \\')
                first = False
            if not (q == 0.95 and K == 1000):
                lines.append(r'\addlinespace')
    write('r02_sameinfo.tex', lines)


def r04():
    """Comparison with existing methods: per K and noise scenario, over the bi-log-concave laws,
    smallest mean coverage, minimum reliability and width relative to the usual rank."""
    s = pd.read_csv(RES / 'competitors_v2_hetldc_summary.csv')
    u = s[s.method == 'usual'][['K', 'noise', 'latent', 'rel_width']].rename(
        columns={'rel_width': 'w0'})
    s = s.merge(u, on=['K', 'noise', 'latent'])
    s['dw'] = 100 * (s.rel_width / s.w0 - 1)
    order = ['ours: Gaussian', 'ours: symmetric unimodal', 'no shape, median-zero', 'latentcp',
             'latentcp_tuned', 'latentcp_multi', 'latentcp_pac', 'shape_free', 'fay_herriot',
             'hetldc', 'simple_shrink', 'cohen', 'cohen_stable']
    name = {'ours: Gaussian': 'Theorem~\\ref{M-thm:rank}, Gaussian', 'ours: symmetric unimodal':
            'Theorem~\\ref{M-thm:rank}, symm.\\ unimodal', 'no shape, median-zero': 'no shape, median zero',
            'latentcp': 'LatentCP', 'latentcp_tuned': 'LatentCP, tuned',
            'latentcp_multi': 'LatentCP, two levels', 'latentcp_pac': 'LatentCP, PAC rank',
            'shape_free': 'Markov rule', 'fay_herriot': 'Fay--Herriot', 'hetldc': 'HetLDC',
            'simple_shrink': 'known lower bound', 'cohen': 'deconvolution, bin $0.01$',
            'cohen_stable': 'deconvolution, bin $0.2$'}
    scen = ['Gaussian', 'Laplace', 'mixed symmetric unimodal']
    for K in (110, 1000):
        lines = []
        d = s[(s.K == K) & s.latent.isin(BLC)]
        for m in order:
            row = [name[m]]
            have = False
            for sc in scen:
                x = d[(d.method == m) & (d.noise == sc)]
                if x.empty:
                    row += ['--', '--', '--']
                    continue
                have = True
                row += [f3(x.mean_cov.min()), f3(x.reliability.min()),
                        rng_pct(x.dw.min(), x.dw.max(), 1)]
            if have:
                lines.append(' & '.join(row) + r' \\')
        write(f'r04_competitors_K{K}.tex', lines)
    # t3 control: reliability and mean coverage of every rule
    lines = []
    d = s[s.latent == 't3']
    for m in order:
        row = [name[m]]
        have = False
        for K in (110, 1000):
            x = d[(d.method == m) & (d.K == K)]
            if x.empty:
                row += ['--', '--']
                continue
            have = True
            row += [f'{x.mean_cov.min():.3f}', f3(x.reliability.min())]
        if have:
            lines.append(' & '.join(row) + r' \\')
    write('r04_t3.tex', lines)


def r05():
    """Application: per condition, mean and minimum coverage, share of replications with coverage
    >= 0.9 and mean width, for every rule."""
    s = pd.read_csv(RES / 'apipop_summary.csv')
    cond = {'api99_n3': 'with 1999 index, $n = 3$', 'api99_n2': 'with 1999 index, $n = 2$',
            'socio_n3': 'socio-demographic only, $n = 3$',
            'socio_n2': 'socio-demographic only, $n = 2$'}
    name = {'usual': 'usual rank', 'ours: Gaussian': 'Theorem~\\ref{M-thm:rank} (all three classes)',
            'no shape, median-zero': 'no shape, median zero',
            'fay_herriot (plug-in D)': 'Fay--Herriot, plug-in $\\hat D_c$',
            'latentcp (plug-in D)': 'LatentCP, plug-in $\\hat D_c$',
            'shape_free (plug-in D)': 'Markov rule, plug-in $\\hat D_c$'}
    lines = []
    for c, lab in cond.items():
        d = s[s.condition == c].set_index('method')
        w0 = d.loc['usual', 'width']
        lines.append(r'\multicolumn{6}{l}{\emph{' + lab + r'}} \\')
        for m, nm in name.items():
            x = d.loc[m]
            lines.append(' & '.join([nm, f'{x.mean_cov:.3f}', f'{x.min_cov:.3f}',
                                     f'{x.pr_cov_ge_90:.3f}', f'{x.width:.1f}',
                                     pct(100 * (x.width / w0 - 1), 1)]) + r' \\')
        if c != 'socio_n2':
            lines.append(r'\addlinespace')
    write('r05_apipop.tex', lines)


R06_LABEL = {
    'homogeneous extremal': 'homogeneous extremal ($\\tau = 0$, $u = 200$)',
    'homogeneous, slightly easier (r_i = 0.999995, u = 200)':
        'all units easier, $\\tau = -10^{-3}$, $u = 200$',
    'homogeneous, slightly easier (r_i = 0.999995, u = 2000)':
        'all units easier, $\\tau = -10^{-2}$, $u = 2000$',
    'heterogeneous, condition holds': 'heterogeneous, condition holds',
    '10% easier units (tau = -0.02)': '$10\\%$ of units easier, $\\tau = -0.02$',
    'random unit (mixture of point masses, sigma = 1e-4)': 'new unit a random unit (mixture)',
}


def r06():
    """Heterogeneous latent laws, Part 1 (i.i.d. unit parameters)."""
    d = pd.read_csv(RES / 'heterogeneous_latent.csv')
    lines = []
    for q in (0.8, 0.9):
        x = d[d.q == q]
        for design in x.design.unique():
            y = x[x.design == design]
            row = [f'${q}$' if design == x.design.unique()[0] else '', R06_LABEL[design],
                   f'{1e3 * y.H_minus_psi.iloc[0] + 0.0:+.2f}'.replace('-0.00', '0.00').replace('+0.00', '0.00')]
            for K in (10_000, 1_000_000):
                for rule in ('usual', 'Gaussian class (Theorem 2)', 'no latent shape'):
                    z = y[(y.K == K) & (y.rule == rule)]
                    row.append(f3(z.reliability.iloc[0]) if len(z) else '--')
            lines.append(' & '.join(row) + r' \\')
        if q == 0.8:
            lines.append(r'\addlinespace')
    write('r06_hetero.tex', lines)


def r07():
    """Contaminated latent laws at the extremal law: level and exact reliability of three ranks."""
    d = pd.read_csv(RES / 'contamination.csv')
    rules = ['usual', 'Theorem 3 (c = 0)', 'contamination rank']
    lines = []
    for q in (0.8, 0.9):
        x = d[d.q == q]
        for i, c in enumerate(sorted(x.c.unique())):
            y = x[x.c == c]
            lev = y[y.rule == 'contamination rank'].level.iloc[0]
            row = [f'${q}$' if i == 0 else '', f'${100 * c:g}\\%$', f'{lev:.5f}']
            for K in (1000, 10_000, 100_000):
                for r in rules:
                    row.append(f3(y[(y.K == K) & (y.rule == r)].reliability.iloc[0]))
            lines.append(' & '.join(row) + r' \\')
        if q == 0.8:
            lines.append(r'\addlinespace')
    write('r07_contam.tex', lines)


def sci(x):
    if pd.isna(x):
        return '--'
    m, e = f'{x:.1e}'.split('e')
    return f'${m} \\times 10^{{{int(e)}}}$'


def r08():
    """Centred log-concave residuals: excess of the level, failure sizes and ranks (R08)."""
    d = pd.read_csv(RES / 'centred_levels.csv')
    lines = []
    cases = [('gaussian', 0.8), ('gaussian', 0.9), ('gaussian', 0.95), ('su', 0.8), ('su', 0.9),
             ('su', 0.95)]
    for j, (noise, q) in enumerate(cases):
        x = d[(d.noise == noise) & (d.q == q)].sort_values('kappa')
        x = x[x.kappa.isin([0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])]
        lab = ('Gaussian' if noise == 'gaussian' else 'symm.\\ unimodal') + f', ${q}$'
        for i, (_, y) in enumerate(x.iterrows()):
            z = '$\\ge 1$' if y.kappa == 1 else f'${y.kappa:g}$'
            row = [lab if i == 0 else '', z, sci(y.excess), sci(y.K90_exact), sci(y.K50_exact)]
            row += [str(int(y[c])) for c in ('usual_10000', 'centred_10000', 'thm3_10000',
                                             'usual_1000000', 'centred_1000000', 'thm3_1000000')]
            lines.append(' & '.join(row) + r' \\')
        if j < len(cases) - 1:
            lines.append(r'\addlinespace')
    write('r08_centred.tex', lines)


def r09():
    """Misspecified noise class at q = 0.9: level bound and reliability of the class ranks (R09)."""
    d = pd.read_csv(RES / 'noise_misspec.csv')
    d = d[d.q == 0.9]
    lines = []
    for name in d.noise.unique():
        x = d[d.noise == name]
        g = x[x.rule == 'Gaussian'].iloc[0]
        rel = {r: x[x.rule == r].rel_10000.iloc[0] for r in x.rule}
        k50 = g['first_K_below_0.5']
        row = [name, g.noise_class, f'{g.level_lower_bound:.5f}',
               f3(rel['Gaussian']), f3(rel['symmetric unimodal']), f3(rel['mean-zero log-concave']),
               '--' if pd.isna(k50) else f'{int(k50):,}'.replace(',', '{,}')]
        lines.append(' & '.join(row) + r' \\')
    write('r09_misspec.tex', lines)


def r10():
    """Intermediate designs (R10): per group of latent laws and q, exact summary."""
    d = pd.read_csv(RES / 'intermediate_summary.csv')
    g = d.groupby(['group', 'q']).agg(designs=('designs', 'sum'), above=('designs_H_above_q', 'sum'),
                                     maxHq=('max_H_minus_q', 'max'),
                                     usual=('min_rel_usual_K_le_1e5', 'min'),
                                     thm3=('min_rel_thm3_K_le_1e5', 'min')).reset_index()
    order = ['symmetric', 'skewed smooth', 'J-shaped', 'proportions', 'proportions (not log-concave)']
    lines = []
    for j, grp in enumerate(order):
        x = g[g.group == grp].sort_values('q')
        for i, (_, y) in enumerate(x.iterrows()):
            ex = y.maxHq
            exs = f'${ex * 1e4:.1f}$' if ex > 0 else f'${ex * 1e4:.1f}$'
            row = [grp if i == 0 else '', f'${y.q}$', str(int(y.designs)), str(int(y.above)), exs,
                   f3(y.usual), f3(y.thm3)]
            lines.append(' & '.join(row) + r' \\')
        if j < len(order) - 1:
            lines.append(r'\addlinespace')
    write('r10_intermediate.tex', lines)


if __name__ == '__main__':
    r01(), r02(), r03(), r04(), r05(), r06(), r07(), r08(), r09(), r10()
