"""Порт js/segmentation/cumulativeHistogramSegmenter.js; збігається з ним піксель у піксель (cleanup теж).

cleanup=True - постобробка маски для фото: найбільша зв'язна область, що не
торкається краю кадру, із заповненими порожнинами.
"""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


def js_round(x):
    """Math.round з JS: половина округлюється вгору, а не до парного."""
    return int(np.floor(x + 0.5))


class CumulativeHistogramSegmenter:
    def __init__(self, levels=256, min_prominence=0.002, significance=0.3,
                 target='auto', manual_level=None, size=128, cleanup=False):
        self.levels = levels
        self.min_prominence = min_prominence
        self.significance = significance
        self.target = target
        self.manual_level = manual_level
        self.size = size
        self.cleanup = cleanup

    def segment(self, rgb):
        gray = self.to_gray(rgb)
        target = self.resolve_target(gray)

        vf = np.cumsum(np.bincount(gray.ravel(), minlength=self.levels).astype(np.float64)) / gray.size
        vfg = (np.arange(self.levels) + 1) / self.levels
        difference = vf - vfg

        extrema = self.mark_edge_extrema(self.find_extrema(difference))
        manual = self.picked_level()
        auto = self.select_extremum(extrema, target)
        s_opt = manual if manual is not None else (auto['s'] if auto else self.median_level(vf))

        binary = (gray <= s_opt) if target == 'dark' else (gray > s_opt)
        if self.cleanup:
            binary = self.main_object(binary)
        mask = self.fit_to_frame(binary.astype(np.uint8))
        if mask is None:
            raise ValueError('Після сегментації не лишилося жодного пікселя фігури.')
        return {'mask': mask, 's_opt': int(s_opt), 'target': target, 'extrema': extrema,
                'vf': vf, 'vfg': vfg, 'difference': difference, 'binary': binary}

    @staticmethod
    def main_object(binary):
        labels, count = ndimage.label(binary, structure=np.ones((3, 3), bool))
        if count == 0:
            return binary
        edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        inner = sizes.copy()
        inner[edge] = 0
        # фігура на весь кадр торкається краю - тоді беремо просто найбільшу область
        keep = int(np.argmax(inner)) if inner.max() > 0 else int(np.argmax(sizes))
        return ndimage.binary_fill_holes(labels == keep)

    @staticmethod
    def to_gray(rgb):
        r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
        return (r * 0.299 + g * 0.587 + b * 0.114).astype(np.uint8)   # | 0 у JS

    def resolve_target(self, gray):
        if self.target != 'auto':
            return self.target
        h, w = gray.shape
        # кути враховуються двічі - як у JS
        total = int(gray[0].sum()) + int(gray[h - 1].sum()) + int(gray[:, 0].sum()) + int(gray[:, w - 1].sum())
        return 'dark' if total / (2 * w + 2 * h) > self.levels / 2 else 'light'

    def find_extrema(self, d):
        last = len(d) - 1
        found = []
        for s in range(1, last):
            prev, cur, nxt = d[s - 1], d[s], d[s + 1]
            if cur > prev and cur >= nxt:
                found.append({'s': s, 'value': d[s], 'type': 'max'})
            elif cur < prev and cur <= nxt:
                found.append({'s': s, 'value': d[s], 'type': 'min'})
        if d[last] != d[last - 1]:
            found.append({'s': last, 'value': d[last], 'type': 'max' if d[last] > d[last - 1] else 'min'})
        return self.filter_prominence(found)

    def filter_prominence(self, extrema):
        kept = []
        for point in extrema:
            if not kept:
                kept.append(point)
                continue
            previous = kept[-1]
            if previous['type'] == point['type']:
                if abs(point['value']) > abs(previous['value']):
                    kept[-1] = point
                continue
            if abs(point['value'] - previous['value']) >= self.min_prominence:
                kept.append(point)
        return kept

    @staticmethod
    def mark_edge_extrema(extrema):
        marked = [{**point, 'ignored': False} for point in extrema]
        if marked and marked[0]['type'] == 'min':
            marked[0]['ignored'] = True
        if len(marked) > 1 and marked[-1]['type'] == 'max':
            marked[-1]['ignored'] = True
        return marked

    def picked_level(self):
        if self.manual_level is None or not np.isfinite(self.manual_level):
            return None
        return min(self.levels - 1, max(0, js_round(self.manual_level)))

    def select_extremum(self, extrema, target):
        candidates = [point for point in extrema if not point['ignored']]
        pool = candidates or extrema
        if not pool:
            return None
        strongest = max(abs(point['value']) for point in pool)
        significant = [point for point in pool if abs(point['value']) >= strongest * self.significance]
        return significant[-1] if target == 'light' else significant[0]

    def median_level(self, vf):
        above = np.nonzero(vf >= 0.5)[0]
        return int(above[0]) if len(above) else self.levels >> 1

    def fit_to_frame(self, mask):
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        bx0, by0 = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max()) - bx0 + 1, int(ys.max()) - by0 + 1

        size = self.size
        scale = size / max(bw, bh)
        dw, dh = max(1, js_round(bw * scale)), max(1, js_round(bh * scale))
        ox, oy = (size - dw) // 2, (size - dh) // 2

        def spans(origin, length, count, limit):
            lo = np.array([int(np.floor(origin + i * length / count)) for i in range(count)])
            hi = np.array([max(lo[i] + 1, int(np.ceil(origin + (i + 1) * length / count)))
                           for i in range(count)])
            return lo, np.minimum(hi, limit)

        y0, y1 = spans(by0, bh, dh, h)
        x0, x1 = spans(bx0, bw, dw, w)

        integral = np.zeros((h + 1, w + 1), np.int64)
        integral[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
        total = (integral[y1[:, None], x1[None, :]] - integral[y0[:, None], x1[None, :]]
                 - integral[y1[:, None], x0[None, :]] + integral[y0[:, None], x0[None, :]])
        count = (y1 - y0)[:, None] * (x1 - x0)[None, :]
        filled = (count > 0) & (total >= 0.5 * count)

        out = np.zeros((size, size), np.uint8)
        out[oy:oy + dh, ox:ox + dw] = filled
        return out


def load_rgb(path):
    """Як у браузері: орієнтація з EXIF, прозорі пікселі - білі."""
    image = ImageOps.exif_transpose(Image.open(path))
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGBA')
        white = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    return np.array(image.convert('RGB'))


def mask_to_input(mask):
    """Як toTensor у застосунку: фігура 0, тло 255, три канали."""
    plane = np.where(mask > 0, 0, 255).astype(np.float32)
    return np.repeat(plane[:, :, None], 3, axis=2)
