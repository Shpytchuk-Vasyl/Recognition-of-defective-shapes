"""Стійкість вердикту до повороту, дзеркала, масштабу, яскравості, розмиття і JPEG: таблиця 6.

Запуск: python experiments/augment_real.py article/real/sheets article/results
"""
import csv
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from cut_sheets import find_figures, sheets
from model import load_trained
from real_eval import ORDER
from segmenter import CumulativeHistogramSegmenter, load_rgb, mask_to_input

DECISION = 0.5
ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
SCALES = (1.0, 0.5, 0.25)
INNER = 1.1      # половина сторони кадру в розмірах фігури
PHOTOMETRIC = {
    'темніше ×0,7': lambda im: ImageEnhance.Brightness(im).enhance(0.7),
    'світліше ×1,3': lambda im: ImageEnhance.Brightness(im).enhance(1.3),
    'розмиття r = 2': lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    'JPEG, якість 20': lambda im: jpeg(im, 20),
}


def jpeg(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')


def window(rgb, cy, cx, half):
    h, w = rgb.shape[:2]
    y, x = int(round(cy)) - half, int(round(cx)) - half
    y0, x0, y1, x1 = max(0, y), max(0, x), min(h, y + 2 * half), min(w, x + 2 * half)
    part = rgb[y0:y1, x0:x1]
    return np.pad(part, ((y0 - y, y + 2 * half - y1), (x0 - x, x + 2 * half - x1), (0, 0)), mode='reflect')


def geometric(rgb, box, angle, mirror, scale):
    y0, y1, x0, x1 = box
    inner = int(round(INNER * max(y1 - y0, x1 - x0)))
    # повертаємо більший фрагмент аркуша (> √2·inner), щоб кути кадру заповнив справжній папір:
    # однорідна заливка дає в гістограмі пік, на який стає поріг
    outer = int(np.ceil(1.45 * inner))
    image = Image.fromarray(window(rgb, (y0 + y1) / 2, (x0 + x1) / 2, outer))
    if mirror:
        image = ImageOps.mirror(image)
    if angle:
        image = image.rotate(angle, resample=Image.BICUBIC)
    image = image.crop((outer - inner, outer - inner, outer + inner, outer + inner))
    if scale != 1.0:
        image = image.resize((max(1, round(image.width * scale)),) * 2, Image.LANCZOS)
    return image


def expected_mask(mask, angle, mirror, segmenter):
    big = Image.fromarray((mask * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    if mirror:
        big = ImageOps.mirror(big)
    if angle:
        big = big.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)
    return segmenter.fit_to_frame((np.asarray(big) > 127).astype(np.uint8))


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def labelled_figures(sheet_dir):
    with open(Path(sheet_dir).parent / 'sheets.csv', encoding='utf-8') as fh:
        labels = {}
        for row in csv.DictReader(fh):
            labels.setdefault(row['sheet'], {})[int(row['figure'])] = row
    for path in sheets(sheet_dir):
        if path.name not in labels:
            continue
        rgb = load_rgb(path)
        for k, box in enumerate(find_figures(rgb), start=1):
            row = labels[path.name][k]
            if row['defect'] != 'skip':
                yield f'{path.stem}-{k:02d}', row['shape'], row['defect'], rgb, box


def run(sheet_dir, results):
    results = Path(results)
    segmenter = CumulativeHistogramSegmenter(cleanup=True)
    rows, inputs = [], []
    for name, shape, defect, rgb, box in labelled_figures(sheet_dir):
        original = geometric(rgb, box, 0, False, 1.0)
        base = segmenter.segment(np.asarray(original))['mask']
        variants = [('геометрія', a, m, s, None) for s in SCALES for m in (False, True) for a in ANGLES]
        variants += [('фото', 0, False, 1.0, photo) for photo in PHOTOMETRIC]
        for group, angle, mirror, scale, photo in variants:
            image = PHOTOMETRIC[photo](original) if photo else geometric(rgb, box, angle, mirror, scale)
            mask = segmenter.segment(np.asarray(image))['mask']
            rows.append({'photo': name, 'shape': shape, 'defect': defect, 'label': int(defect != 'clean'),
                         'group': group, 'angle': angle, 'mirror': mirror, 'scale': scale,
                         'photometric': photo or '', 'original': group == 'геометрія' and not angle
                         and not mirror and scale == 1.0,
                         'iou': round(iou(mask, expected_mask(base, angle, mirror, segmenter)), 4)})
            inputs.append(mask_to_input(mask))
    table = pd.DataFrame(rows)
    X = np.stack(inputs)

    models = {p.name.split('_s0')[0]: p for p in (results / 'models').glob('*_s0.weights.h5')}
    comps = [c for c in ORDER if c in models]
    for comp in comps:
        table[comp] = load_trained(models[comp]).predict(X, batch_size=128, verbose=0).ravel().round(4)
    table.to_csv(results / 'real_augmented.csv', index=False)
    return table, comps


def summarize(table, comps, results):
    results = Path(results)
    geo = table[table['group'] == 'геометрія']
    original = table[table['original']].set_index('photo')

    rows = {}
    for comp in comps:
        verdict = geo[comp] > DECISION
        same = verdict.values == (original.loc[geo['photo'], comp].values > DECISION)
        row = {'хибні тривоги': verdict[geo['label'] == 0].mean(),
               'виявлено дефектів': verdict[geo['label'] == 1].mean(),
               'точність': (verdict == (geo['label'] == 1)).mean(),
               'вердикт як в оригіналу': same.mean()}
        for scale in SCALES:
            part = geo['scale'] == scale
            row[f'хибні, масштаб {scale:g}'] = verdict[part & (geo['label'] == 0)].mean()
            row[f'виявлено, масштаб {scale:g}'] = verdict[part & (geo['label'] == 1)].mean()
        rows[comp] = row
    by_model = pd.DataFrame(rows).T
    by_model.to_csv(results / 'table_real_augmented.csv', float_format='%.4f')

    photo = table[table['group'] == 'фото']
    photometric = {}
    for comp in comps:
        verdict = photo[comp] > DECISION
        same = verdict.values == (original.loc[photo['photo'], comp].values > DECISION)
        photometric[comp] = {name: same[(photo['photometric'] == name).values].mean() for name in PHOTOMETRIC}
    photometric = pd.DataFrame(photometric).T
    photometric.to_csv(results / 'table_real_photometric.csv', float_format='%.4f')

    quality = table.groupby(['group', 'scale'])['iou'].agg(['mean', 'min', lambda v: (v < 0.8).mean()])
    quality.columns = ['IoU середній', 'IoU мінімальний', 'частка IoU < 0,8']

    pd.set_option('display.width', 250)
    print(f'варіантів: {len(table)} ({len(geo)} геометричних, {len(photo)} фотометричних)')
    print('\n== якість сегментації варіантів (IoU з перетвореною маскою оригіналу) ==')
    print(quality.round(3).to_string())
    print('\n== геометричні варіанти, % ==')
    print((by_model * 100).round(1).to_string())
    print('\n== частка фотометричних варіантів з тим самим вердиктом, що в оригіналу, % ==')
    print((photometric * 100).round(1).to_string())


if __name__ == '__main__':
    table, comps = run(*sys.argv[1:3])
    summarize(table, comps, sys.argv[2])
