"""Параметричний генератор силуетів: фігури з дефектами розміру h і нерівністю контуру eps."""
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 128
CANVAS = 512
CENTER = CANVAS / 2

SHAPES = ('square', 'rectangle', 'circle', 'ellipse', 'pentagon', 'hexagon')
EXCESS = ('line', 'bump', 'blob')
MISSING = ('slit', 'dent')
DEFECTS = EXCESS + MISSING

TRAIN_RANGES = {
    'line': (0.35, 0.70),
    'bump': (0.12, 0.22),
    'blob': (0.46, 0.77),
    'slit': (0.20, 0.45),
    'dent': (0.12, 0.22),
}


def ellipse_outline(rx, ry, steps=96):
    t = np.linspace(0, 2 * np.pi, steps, endpoint=False)
    return list(zip(rx * np.cos(t), ry * np.sin(t)))


def unit_shape(kind, rng):
    if kind == 'square':
        return [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    if kind == 'rectangle':
        b = rng.uniform(0.45, 0.72)
        return [(-1, -b), (1, -b), (1, b), (-1, b)]
    if kind == 'circle':
        return ellipse_outline(1.0, 1.0)
    if kind == 'ellipse':
        return ellipse_outline(1.0, rng.uniform(0.5, 0.78))
    sides = 5 if kind == 'pentagon' else 6
    angles = -np.pi / 2 + np.arange(sides) * 2 * np.pi / sides
    return list(zip(np.cos(angles), np.sin(angles)))


def place(points, scale, rotation, center=CENTER):
    c, s = np.cos(rotation), np.sin(rotation)
    return np.array([(center + scale * (x * c - y * s), center + scale * (x * s + y * c))
                     for x, y in points])


def densify(points, step):
    out = []
    for p, q in zip(points, np.roll(points, -1, axis=0)):
        k = max(1, int(np.ceil(np.hypot(*(q - p)) / step)))
        out.extend(p + (q - p) * j / k for j in range(k))
    return np.array(out)


def wobble(points, amp, rng, k_min=2, k_max=8):
    d = points - CENTER
    r = np.hypot(d[:, 0], d[:, 1])
    theta = np.arctan2(d[:, 1], d[:, 0])
    ks = np.arange(k_min, k_max + 1)
    a = rng.normal(size=len(ks)) / ks
    phi = rng.uniform(0, 2 * np.pi, len(ks))
    f = (a * np.cos(np.outer(theta, ks) + phi)).sum(axis=1)
    f /= np.abs(f).max()
    return CENTER + d * ((r + amp * f) / r)[:, None]


def anchor(points, smooth, rng, where=None):
    """Точка контуру і зовнішня нормаль у ній; where = (номер ребра, частка вздовж нього)."""
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if where is None:
        i = rng.choice(len(points), p=lengths / lengths.sum())
        u = rng.uniform(0, 1) if smooth else rng.uniform(0.2, 0.8)
    else:
        i, u = where
    a = points[i] + u * edges[i]
    n = np.array([edges[i][1], -edges[i][0]]) / lengths[i]
    if np.dot(n, a - points.mean(axis=0)) < 0:
        n = -n
    return a, n


def segment(p0, p1, width):
    d = (p1 - p0) / np.hypot(*(p1 - p0))
    o = np.array([-d[1], d[0]]) * width / 2
    return [tuple(p0 + o), tuple(p1 + o), tuple(p1 - o), tuple(p0 - o)]


def disc(c, r):
    return [c[0] - r, c[1] - r, c[0] + r, c[1] + r]


def render(kind, rng, defect=None, h=None, eps=0.0, info=None, scale=None, rotation=None, where=None):
    image = Image.new('L', (CANVAS, CANVAS), 255)
    pen = ImageDraw.Draw(image)

    s = CANVAS * 0.28 * (rng.uniform(0.85, 1.15) if scale is None else scale)
    nominal = place(unit_shape(kind, rng), s, rng.uniform(0, 2 * np.pi) if rotation is None else rotation)
    outline = wobble(densify(nominal, 4.0), eps * s, rng) if eps > 0 else nominal
    pen.polygon([tuple(p) for p in outline], fill=0)
    if info is not None:
        info.update(s=s, nominal=nominal)

    if defect is None:
        return image

    a, n = anchor(nominal, smooth=kind in ('circle', 'ellipse'), rng=rng, where=where)
    reach = h * s
    if defect == 'line':
        pen.polygon(segment(a - n * s * 0.06, a + n * reach, s * rng.uniform(0.04, 0.09)), fill=0)
    elif defect == 'bump':
        pen.ellipse(disc(a, reach), fill=0)
    elif defect == 'blob':
        r = 0.3 * reach
        c = a + n * (reach - r)
        stalk = min(s * rng.uniform(0.03, 0.07), 1.2 * r)
        pen.polygon(segment(a - n * s * 0.06, c, stalk), fill=0)
        pen.ellipse(disc(c, r), fill=0)
    elif defect == 'slit':
        pen.polygon(segment(a + n * s * 0.06, a - n * reach, s * rng.uniform(0.04, 0.09)), fill=255)
    elif defect == 'dent':
        pen.ellipse(disc(a, reach), fill=255)
    else:
        raise ValueError(f'невідомий дефект: {defect}')
    return image


def fit_to_frame(image):
    pixels = np.array(image)
    ys, xs = np.where(pixels < 128)
    if len(xs) == 0:
        return None
    crop = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = crop.size
    scale = IMG_SIZE / max(w, h)
    crop = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    frame = Image.new('L', (IMG_SIZE, IMG_SIZE), 255)
    frame.paste(crop, ((IMG_SIZE - crop.size[0]) // 2, (IMG_SIZE - crop.size[1]) // 2))
    return np.where(np.array(frame) < 128, 0, 255).astype(np.uint8)


def sample(kind, rng, defect=None, h=None, eps=0.0):
    return fit_to_frame(render(kind, rng, defect, h, eps))


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def composition_set(types, n, rng, ranges=TRAIN_RANGES):
    images, labels = np.empty((n, IMG_SIZE, IMG_SIZE), np.uint8), np.empty(n, np.float32)
    for i in range(n):
        kind = SHAPES[rng.integers(len(SHAPES))]
        if rng.random() < 0.5:
            images[i], labels[i] = sample(kind, rng), 0.0
        else:
            defect = types[rng.integers(len(types))]
            images[i] = sample(kind, rng, defect, rng.uniform(*ranges[defect]))
            labels[i] = 1.0
    return images, labels


def tolerance_set(T, n, rng):
    """Явний допуск T: відхилення h < T і eps < T - норма, h >= T - брак."""
    small = ('line', 'bump', 'slit', 'dent')
    images, labels = np.empty((n, IMG_SIZE, IMG_SIZE), np.uint8), np.empty(n, np.float32)
    for i in range(n):
        kind = SHAPES[rng.integers(len(SHAPES))]
        part = i % 3
        if part == 0:
            eps = rng.uniform(0, T) if rng.random() < 0.5 else 0.0
            images[i], labels[i] = sample(kind, rng, eps=eps), 0.0
        elif part == 1:
            defect = small[rng.integers(len(small))]
            images[i], labels[i] = sample(kind, rng, defect, log_uniform(rng, 0.01, T)), 0.0
        else:
            defect = DEFECTS[rng.integers(len(DEFECTS))]
            lo = max(T, 0.25) if defect == 'blob' else T
            hi = 0.77 if defect == 'blob' else 0.5
            images[i], labels[i] = sample(kind, rng, defect, log_uniform(rng, lo, hi)), 1.0
    order = rng.permutation(n)
    return images[order], labels[order]


GRID_H = (0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40)
BLOB_H = (0.20, 0.28, 0.40)
GRID_EPS = (0.01, 0.02, 0.03, 0.05, 0.08)


def test_suite(rng, per_shape_clean=200, per_cell=120, per_eps=600):
    images, meta = [], []

    def add(kind, set_name, defect=None, h=None, eps=0.0):
        images.append(sample(kind, rng, defect, h, eps))
        meta.append({'set': set_name, 'shape': kind, 'defect': defect or '',
                     'h': h if h is not None else 0.0, 'eps': eps,
                     'label': 1 if defect else 0})

    for kind in SHAPES:
        for _ in range(per_shape_clean):
            add(kind, 'clean')
    for defect in DEFECTS:
        for h in (BLOB_H if defect == 'blob' else GRID_H):
            for j in range(per_cell):
                add(SHAPES[j % len(SHAPES)], 'grid', defect, h)
    for eps in GRID_EPS:
        for j in range(per_eps):
            add(SHAPES[j % len(SHAPES)], 'irregular', eps=eps)
    return np.stack(images), meta


def clean_set(n, rng):
    return np.stack([sample(SHAPES[i % len(SHAPES)], rng) for i in range(n)])
