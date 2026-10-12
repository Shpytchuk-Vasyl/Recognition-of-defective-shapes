"""Таблиці й рисунки 3–6 з результатів серії навчань.

Запуск: python experiments/analysis.py article/results article/figures
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

DECISION = 0.5
SCALABLE = ('line', 'bump', 'slit', 'dent')
TYPES = ('line', 'bump', 'blob', 'slit', 'dent')
MATRIX_H = (0.10, 0.14, 0.20)
TYPE_UA = {'line': 'відросток', 'bump': 'наріст-півколо', 'blob': 'наріст на ніжці',
           'slit': 'проріз', 'dent': 'вмʼятина'}
COMP_UA = {'L': 'L: відросток', 'B': 'B: наріст-півколо', 'D': 'D: вмʼятина',
           'M': 'М: без наросту-півкола', 'S': 'С: усі три нарости', 'A': 'A: усі пʼять',
           'T05': 'T = 0.05', 'T10': 'T = 0.10', 'T15': 'T = 0.15'}
STYLE = {'L': ('#1f77b4', 'o', '-'), 'B': ('#ff7f0e', 's', '-'), 'D': ('#2ca02c', '^', '-'),
         'M': ('#9467bd', 'v', '--'), 'S': ('#d62728', 'D', '--'), 'A': ('#111111', 'P', '-.'),
         'T05': ('#8c564b', 'o', ':'), 'T10': ('#e377c2', 's', ':'), 'T15': ('#7f7f7f', '^', ':')}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 110,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def load(results):
    meta = pd.read_csv(results / 'test_meta.csv', index_col='id', keep_default_na=False)
    log = pd.read_csv(results / 'train_log.csv')
    scores = {}
    for path in sorted((results / 'scores').glob('*.npz')):
        comp, seed = path.stem.rsplit('_s', 1)
        scores[(comp, int(seed))] = dict(np.load(path))
    return meta, log, scores


def comps_of(scores):
    order = list(COMP_UA)
    present = {comp for comp, _ in scores}
    return [c for c in order if c in present] + sorted(present - set(order))


def detection(scores, meta, comp, mask):
    seeds = sorted(seed for c, seed in scores if c == comp)
    return np.array([(scores[(comp, s)]['test'][mask] > DECISION).mean() for s in seeds])


def pod_curve(scores, meta, comp, defect):
    grid = meta[(meta['set'] == 'grid') & (meta['defect'] == defect)]
    levels = np.array(sorted(grid['h'].unique()))
    seeds = sorted(seed for c, seed in scores if c == comp)
    rates = np.array([[(scores[(comp, s)]['test'][grid.index[grid['h'] == h]] > DECISION).mean()
                       for h in levels] for s in seeds])
    return levels, rates


def crossing(levels, rate, q):
    """Розмір, за якого частка виявлених уперше досягає q; ±inf - поза сіткою."""
    if rate[0] >= q:
        return -np.inf
    for i in range(1, len(levels)):
        if rate[i] >= q:
            x0, x1 = np.log(levels[i - 1]), np.log(levels[i])
            t = (q - rate[i - 1]) / (rate[i] - rate[i - 1])
            return float(np.exp(x0 + t * (x1 - x0)))
    return np.inf


def fmt_level(values, levels):
    values = np.asarray(values)
    if np.all(np.isposinf(values)):
        return f'> {levels[-1]:.2f}'
    if np.all(np.isneginf(values)):
        return f'≤ {levels[0]:.2f}'
    finite = values[np.isfinite(values)]
    text = f'{finite.mean():.3f}' + (f' ± {finite.std():.3f}' if len(finite) > 1 else '')
    if len(finite) < len(values):
        text += f' ({len(values) - len(finite)} з {len(values)} поза сіткою)'
    return text


def table_in_distribution(log):
    grouped = log.groupby('comp')
    table = pd.DataFrame({
        'сідів': grouped.size(),
        'accuracy': grouped['own_acc'].agg(lambda v: f'{v.mean():.4f} ± {v.std(ddof=0):.4f}'),
        'AUC': grouped['own_auc'].agg(lambda v: f'{v.mean():.4f} ± {v.std(ddof=0):.4f}'),
        'епох (голова + донавчання)': grouped.apply(
            lambda g: f"{g['epochs_head'].mean():.1f} + {g['epochs_tune'].mean():.1f}",
            include_groups=False),
    })
    return table.reindex([c for c in COMP_UA if c in table.index])


def table_matrix(scores, meta):
    rows = {}
    clean = meta['set'] == 'clean'
    for comp in comps_of(scores):
        row = {'FPR ідеальні': detection(scores, meta, comp, clean.values).mean()}
        for defect in TYPES:
            band = meta['h'].isin((0.20, 0.28, 0.40) if defect == 'blob' else MATRIX_H)
            mask = ((meta['set'] == 'grid') & (meta['defect'] == defect) & band).values
            row[defect] = detection(scores, meta, comp, mask).mean()
        for eps in sorted(meta.loc[meta['set'] == 'irregular', 'eps'].unique()):
            mask = ((meta['set'] == 'irregular') & (meta['eps'] == eps)).values
            row[f'FPR eps={eps:g}'] = detection(scores, meta, comp, mask).mean()
        rows[comp] = row
    return pd.DataFrame(rows).T


def table_levels(scores, meta):
    rows = []
    for comp in comps_of(scores):
        for defect in SCALABLE:
            levels, rates = pod_curve(scores, meta, comp, defect)
            rows.append({'склад': comp, 'дефект': defect,
                         'a50': fmt_level([crossing(levels, r, 0.5) for r in rates], levels),
                         'a90': fmt_level([crossing(levels, r, 0.9) for r in rates], levels),
                         '_a50': [crossing(levels, r, 0.5) for r in rates]})
    return pd.DataFrame(rows)


def figure_matrix(matrix, path):
    columns = ['FPR ідеальні', *TYPES, *[c for c in matrix.columns if c.startswith('FPR eps')]]
    data = matrix[columns].astype(float).values * 100
    fig, ax = plt.subplots(figsize=(1.0 + 0.78 * len(columns), 0.5 + 0.42 * len(matrix)))
    ax.imshow(data, cmap='Greys', vmin=0, vmax=100, aspect='auto')
    for (i, j), value in np.ndenumerate(data):
        ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                color='white' if value > 55 else 'black', fontsize=8)
    names = ['хибні\nідеальні', *[TYPE_UA[t].replace(' ', '\n', 1).replace('-', '-\n', 1) for t in TYPES],
             *[f'хибні\nε = {c.split("=")[1]}' for c in columns if c.startswith('FPR eps')]]
    ax.set_xticks(range(len(columns)), names, fontsize=8)
    ax.set_yticks(range(len(matrix)), [COMP_UA.get(c, c) for c in matrix.index])
    ax.axvline(len(TYPES) + 0.5, color='white', lw=3)
    ax.axvline(0.5, color='white', lw=3)
    fig.savefig(path)
    plt.close(fig)


def figure_pod(scores, meta, comps, path):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for ax, defect in zip(axes.ravel(), ('line', 'slit', 'bump', 'dent')):
        for comp in comps:
            levels, rates = pod_curve(scores, meta, comp, defect)
            color, marker, style = STYLE.get(comp, ('k', 'o', '-'))
            ax.plot(levels, rates.mean(0) * 100, color=color, marker=marker, ls=style,
                    ms=3.5, lw=1.2, label=COMP_UA.get(comp, comp))
            if len(rates) > 1:
                ax.fill_between(levels, rates.min(0) * 100, rates.max(0) * 100, color=color, alpha=0.12, lw=0)
        ax.axhline(50, color='#999', lw=0.6, ls=':')
        ax.axhline(90, color='#999', lw=0.6, ls=':')
        ax.set_xscale('log')
        ax.set_title(f'{TYPE_UA[defect]} ({"надлишок" if defect in ("line", "bump") else "нестача"})')
        ax.grid(alpha=0.25, which='both')
    levels = sorted(meta.loc[meta['set'] == 'grid', 'h'].unique())
    for ax in axes[1]:
        ax.set_xlabel('розмір дефекту h, частки масштабу фігури')
        ax.set_xticks(levels, [f'{h:g}' for h in levels], fontsize=7)
        ax.minorticks_off()
    for ax in axes[:, 0]:
        ax.set_ylabel('виявлено, %')
    axes[0, 1].legend(fontsize=7, loc='upper left')
    fig.savefig(path)
    plt.close(fig)


def figure_tolerance(levels_table, config, path):
    tolerances = {c: spec['tolerance'] for c, spec in config['compositions'].items() if 'tolerance' in spec}
    rows = levels_table[levels_table['склад'].isin(tolerances)]
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    lo, hi = 0.03, 0.20
    ax.plot([lo, hi], [lo, hi], color='#888', lw=1.2, ls='--', label='a50 = T', zorder=1)
    markers = {'line': 'o', 'bump': 's', 'slit': '^', 'dent': 'v'}
    colors = {'line': '#1f77b4', 'bump': '#ff7f0e', 'slit': '#2ca02c', 'dent': '#d62728'}
    for k, defect in enumerate(SCALABLE):
        first = True
        for comp, T in tolerances.items():
            values = np.array(rows[(rows['склад'] == comp) & (rows['дефект'] == defect)]['_a50'].iloc[0])
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            x = T + (k - 1.5) * 0.004
            ax.errorbar(x, finite.mean(), yerr=finite.std() if len(finite) > 1 else None,
                        marker=markers[defect], ms=9, color=colors[defect], capsize=4, lw=1.5,
                        elinewidth=1.5, ls='none', zorder=3, label=TYPE_UA[defect] if first else None)
            first = False
    ticks = sorted(tolerances.values())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xticks(ticks, [f'{t:.2f}' for t in ticks])
    ax.set_yticks([0.05, 0.10, 0.15, 0.20], ['0.05', '0.10', '0.15', '0.20'])
    ax.tick_params(labelsize=11)
    ax.set_xlabel('заданий допуск T', fontsize=12)
    ax.set_ylabel('виміряний поріг виявлення a50', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10.5, loc='upper left')
    fig.savefig(path)
    plt.close(fig)


def figure_irregular(scores, meta, comps, path):
    eps_levels = sorted(meta.loc[meta['set'] == 'irregular', 'eps'].unique())
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for comp in comps:
        rates = np.array([detection(scores, meta, comp,
                                    ((meta['set'] == 'irregular') & (meta['eps'] == e)).values)
                          for e in eps_levels]).T
        color, marker, style = STYLE.get(comp, ('k', 'o', '-'))
        ax.plot(eps_levels, rates.mean(0) * 100, color=color, marker=marker, ls=style, ms=3.5, lw=1.2,
                label=COMP_UA.get(comp, comp))
    ax.set_xlabel('амплітуда нерівності ε, частки масштабу фігури')
    ax.set_ylabel('хибні тривоги, %')
    ax.set_xticks(eps_levels, [f'{e:g}' for e in eps_levels])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def main(results, figures):
    results, figures = Path(results), Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    config = json.loads((results / 'config.json').read_text(encoding='utf-8'))
    meta, log, scores = load(results)
    comps = comps_of(scores)
    base = [c for c in comps if not c.startswith('T')]
    tolerance = [c for c in comps if c.startswith('T')]

    indist = table_in_distribution(log)
    matrix = table_matrix(scores, meta)
    levels_table = table_levels(scores, meta)

    indist.to_csv(results / 'table_in_distribution.csv')
    matrix.to_csv(results / 'table_matrix.csv', float_format='%.4f')
    levels_table.drop(columns='_a50').to_csv(results / 'table_a50_a90.csv', index=False)

    figure_matrix(matrix, figures / 'fig_matrix.png')
    figure_pod(scores, meta, base, figures / 'fig_pod.png')
    if tolerance:
        figure_pod(scores, meta, tolerance, figures / 'fig_pod_tolerance.png')
        figure_tolerance(levels_table, config, figures / 'fig_tolerance.png')
    figure_irregular(scores, meta, comps, figures / 'fig_irregular.png')

    pd.set_option('display.width', 200)
    print('== на власному тесті ==\n', indist, '\n')
    print('== матриця перенесення (частка вердиктів «дефект») ==\n', (matrix * 100).round(1), '\n')
    print('== a50 / a90 ==\n', levels_table.drop(columns='_a50').to_string(index=False))


if __name__ == '__main__':
    main(*sys.argv[1:3])
