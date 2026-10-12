"""Фото фігур, намальованих маркером, через увесь конвеєр: таблиця 5 і рисунок 7.

Запуск: python experiments/real_eval.py article/real article/results article/figures
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from model import load_trained
from segmenter import CumulativeHistogramSegmenter, load_rgb, mask_to_input

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

DECISION = 0.5
PHOTO_TYPES = ('.jpg', '.jpeg', '.png')
ORDER = ('L', 'B', 'D', 'M', 'S', 'A', 'T05', 'T10', 'T15')
LABEL = {'M': 'М', 'S': 'С', 'T05': 'T0,05', 'T10': 'T0,10', 'T15': 'T0,15'}
TYPE_UA = {'clean': 'без дефекту', 'line': 'відросток', 'bump': 'наріст-півколо',
           'blob': 'наріст на ніжці', 'slit': 'проріз', 'dent': 'вмʼятина'}
# для рис. 7 - дефекти без коректора, бо їхні маски найчистіші
EXAMPLES = ('IMG_20261012_160354-01', 'IMG_20261012_160354-02', 'IMG_20261012_161316-03',
            'IMG_20261012_161800-02', 'IMG_20261012_163438-01', 'IMG_20261012_163438-03')


def parse(path):
    parts = path.stem.split('_')
    if len(parts) < 3 or parts[-1] not in TYPE_UA:
        raise ValueError(f'{path.name}: очікується <номер>_<фігура>_<дефект>, дефект з {list(TYPE_UA)}')
    return parts[-2], parts[-1]


def segment_all(real_dir, masks_dir):
    cleaned, raw = CumulativeHistogramSegmenter(cleanup=True), CumulativeHistogramSegmenter()
    rows, inputs, raw_inputs = [], [], []
    for path in sorted(p for p in Path(real_dir).iterdir() if p.suffix.lower() in PHOTO_TYPES):
        shape, defect = parse(path)
        rgb = load_rgb(path)
        result, plain = cleaned.segment(rgb), raw.segment(rgb)
        for folder, mask in ((masks_dir, result['mask']), (masks_dir / 'raw', plain['mask'])):
            Image.fromarray(np.where(mask > 0, 0, 255).astype(np.uint8)).save(folder / f'{path.stem}.png')
        rows.append({'photo': path.name, 'shape': shape, 'defect': defect, 'label': int(defect != 'clean'),
                     's_opt': result['s_opt'], 'target': result['target'],
                     'fill': round(float(result['mask'].mean()), 4),
                     'fill_raw': round(float(plain['mask'].mean()), 4)})
        inputs.append(mask_to_input(result['mask']))
        raw_inputs.append(mask_to_input(plain['mask']))
    return pd.DataFrame(rows), np.stack(inputs), np.stack(raw_inputs)


def summary(table, comps):
    rows = {}
    for comp in comps:
        verdict = table[comp] > DECISION
        row = {'хибні тривоги (без дефекту)': verdict[table['label'] == 0].mean(),
               'точність': (verdict == (table['label'] == 1)).mean()}
        for defect in ('line', 'bump', 'blob', 'slit', 'dent'):
            subset = table['defect'] == defect
            if subset.any():
                row[TYPE_UA[defect]] = verdict[subset].mean()
        rows[comp] = row
    return pd.DataFrame(rows).T


def figure_examples(table, real_dir, masks_dir, comps, path, per_type=1):
    chosen = [i for stem in EXAMPLES for i in table.index[table['photo'].str.startswith(stem)]]
    if len(chosen) == len(EXAMPLES):
        return draw_examples(table, real_dir, masks_dir, comps, path, chosen)
    picks, used = [], set()
    for defect in TYPE_UA:
        for _ in range(per_type * (2 if defect == 'clean' else 1)):
            pool = [i for i in table.index[table['defect'] == defect] if i not in picks]
            fresh = [i for i in pool if table.loc[i, 'shape'] not in used]
            if pool:
                picks.append((fresh or pool)[0])
                used.add(table.loc[picks[-1], 'shape'])
    draw_examples(table, real_dir, masks_dir, comps, path, picks)


def draw_examples(table, real_dir, masks_dir, comps, path, picks):
    shown = [c for c in ('S', 'M', 'T10') if c in comps] or comps[:2]
    fig, axes = plt.subplots(2, len(picks), figsize=(1.45 * len(picks), 3.4))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for k, i in enumerate(picks):
        row = table.loc[i]
        photo = Image.open(Path(real_dir) / row['photo'])
        photo.thumbnail((400, 400))
        axes[0, k].imshow(photo)
        axes[0, k].set_title(TYPE_UA[row['defect']], fontsize=8)
        axes[1, k].imshow(Image.open(masks_dir / f'{Path(row["photo"]).stem}.png'), cmap='gray', vmin=0, vmax=255)
        verdicts = '\n'.join(f'{LABEL.get(c, c)}: {row[c]:.2f}' for c in shown)
        axes[1, k].set_xlabel(verdicts, fontsize=7)
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel('фото', fontsize=8)
    axes[1, 0].set_ylabel('маска 128×128', fontsize=8)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(real_dir, results, figures):
    results, figures = Path(results), Path(figures)
    masks_dir = results / 'real_masks'
    (masks_dir / 'raw').mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    table, inputs, raw_inputs = segment_all(real_dir, masks_dir)
    raw_table = table.copy()
    models = {path.name.split('_s0')[0]: path for path in (results / 'models').glob('*_s0.weights.h5')}
    comps = [c for c in ORDER if c in models] + sorted(set(models) - set(ORDER))
    for comp in comps:
        model = load_trained(models[comp])
        table[comp] = model.predict(inputs, verbose=0).ravel().round(4)
        raw_table[comp] = model.predict(raw_inputs, verbose=0).ravel().round(4)

    table.to_csv(results / 'real_scores.csv', index=False)
    raw_table.to_csv(results / 'real_scores_raw.csv', index=False)
    result = summary(table, comps)
    result.to_csv(results / 'table_real.csv', float_format='%.4f')
    raw_result = summary(raw_table, comps)
    raw_result.to_csv(results / 'table_real_raw.csv', float_format='%.4f')
    figure_examples(table, real_dir, masks_dir, comps, figures / 'fig_real_examples.png')

    print(f'фото: {len(table)} (без дефекту {int((table["label"] == 0).sum())}, '
          f'з дефектом {int(table["label"].sum())})')
    print('== з постобробкою маски ==')
    print((result * 100).round(1).to_string())
    print('== без постобробки (як у застосунку) ==')
    print((raw_result * 100).round(1).to_string())
    small = table[table['fill'] < 0.05]
    if len(small):
        print('\nпідозріло мала фігура в масці (перевірте сегментацію):', ', '.join(small['photo']))


if __name__ == '__main__':
    main(*sys.argv[1:4])
