"""Вирізає фігури з фото аркушів за розміткою article/real/sheets.csv.

Запуск: python experiments/cut_sheets.py preview article/real/sheets <тека>
        python experiments/cut_sheets.py cut article/real/sheets article/real
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from segmenter import CumulativeHistogramSegmenter, load_rgb

PHOTO_TYPES = ('.jpg', '.jpeg', '.png')
MARGIN = 0.6


def find_figures(rgb):
    """Габарити фігур у порядку читання: рядками згори вниз, у рядку зліва направо."""
    gray = CumulativeHistogramSegmenter.to_gray(rgb)
    h, w = gray.shape
    labels, _ = ndimage.label(gray < 130)
    edge = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    boxes = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if index in edge or box is None:
            continue
        area = int((labels[box] == index).sum())
        if area < 2000 or area > 0.1 * h * w:
            continue
        boxes.append((box[0].start, box[0].stop, box[1].start, box[1].stop))

    boxes.sort(key=lambda b: (b[0] + b[1]) / 2)
    rows, current = [], []
    for box in boxes:
        if current and box[0] > max(b[1] for b in current):
            rows.append(current)
            current = []
        current.append(box)
    if current:
        rows.append(current)
    return [box for row in rows for box in sorted(row, key=lambda b: b[2])]


def crop(rgb, box):
    y0, y1, x0, x1 = box
    margin = int(MARGIN * max(y1 - y0, x1 - x0))
    return rgb[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin]


def sheets(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in PHOTO_TYPES)


def preview(sheet_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sheets(sheet_dir):
        rgb = load_rgb(path)
        tiles = []
        for k, box in enumerate(find_figures(rgb), start=1):
            tile = Image.fromarray(crop(rgb, box))
            tile.thumbnail((260, 260))
            canvas = Image.new('RGB', (270, 290), 'white')
            canvas.paste(tile, ((270 - tile.width) // 2, 25))
            ImageDraw.Draw(canvas).text((5, 5), f'{k}', fill='red')
            tiles.append(canvas)
        if not tiles:
            print(f'{path.name}: фігур не знайдено')
            continue
        cols = min(6, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * 270, rows * 290), 'white')
        for k, tile in enumerate(tiles):
            sheet.paste(tile, ((k % cols) * 270, (k // cols) * 290))
        sheet.save(out_dir / f'{path.stem}.jpg', quality=90)
        print(f'{path.name}: {len(tiles)} фігур')


def cut(sheet_dir, out_dir, labels_path=None):
    out_dir = Path(out_dir)
    labels_path = Path(labels_path or Path(sheet_dir).parent / 'sheets.csv')
    with open(labels_path, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    by_sheet = {}
    for row in rows:
        by_sheet.setdefault(row['sheet'], {})[int(row['figure'])] = row
    written = 0
    for path in sheets(sheet_dir):
        labels = by_sheet.get(path.name)
        if labels is None:
            print(f'{path.name}: немає розмітки, пропущено')
            continue
        rgb = load_rgb(path)
        boxes = find_figures(rgb)
        if len(boxes) != len(labels):
            raise ValueError(f'{path.name}: знайдено {len(boxes)} фігур, а в розмітці {len(labels)}')
        for k, box in enumerate(boxes, start=1):
            row = labels[k]
            if row['defect'] == 'skip':
                continue
            name = f'{path.stem}-{k:02d}_{row["shape"]}_{row["defect"]}.png'
            Image.fromarray(crop(rgb, box)).save(out_dir / name)
            written += 1
    print(f'вирізано фрагментів: {written}')


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    {'preview': preview, 'cut': cut}[mode](*args)
