"""Рисунки 1–2: схема конвеєра і приклади дефектів.

Запуск: python experiments/figures.py article/figures
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

import shapes as S

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({'font.size': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

TYPE_UA = {'line': 'відросток', 'bump': 'наріст-півколо', 'blob': 'наріст на ніжці',
           'slit': 'проріз', 'dent': 'вмʼятина'}


def figure_pipeline(path):
    fig, ax = plt.subplots(figsize=(7.4, 2.1))
    ax.set_xlim(-0.1, 10.55)
    ax.set_ylim(-0.8, 1.8)
    ax.axis('off')
    w = 1.3

    def box(x, y, text, width=w, h=0.66, fill='#f2f2f2'):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), width, h, boxstyle='round,pad=0.03,rounding_size=0.06',
                                    fc=fill, ec='#333', lw=0.8))
        ax.text(x + width / 2, y, text, ha='center', va='center', fontsize=6.4, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#333', shrinkA=0, shrinkB=0))

    top, bottom, middle = 1.25, -0.25, 0.5
    steps = ['Фото фігури\n(маркер\nна папері)', 'Градації\nсірого', 'Кумулятивні\nгістограми,\nD(s) = V − V_g',
             'Поріг s_opt\n(екстремум D)', 'Бінаризація,\nпостобробка']
    gap = 0.22
    xs = [i * (w + gap) for i in range(len(steps))]
    for i, (x, text) in enumerate(zip(xs, steps)):
        box(x, top, text)
        if i:
            arrow(xs[i - 1] + w, top, x, top)
    end = xs[-1] + w
    box(0.0, bottom, 'Генератор силуетів: фігура, тип дефекту, розмір h, нерівність ε',
        width=end, h=0.5, fill='#e8eef7')

    frame_x, right_x = end + 0.55, end + 1.85
    box(frame_x, middle, 'Кадр\n128×128', width=1.0, fill='#ffffff')
    arrow(end, top, frame_x, middle + 0.22)
    arrow(end, bottom, frame_x, middle - 0.22)
    box(right_x, middle + 0.62, 'MobileNetV2\n+ Dense', width=1.2, h=0.56)
    box(right_x, middle - 0.62, 'P(дефект)\n> 0.5 → брак', width=1.2, h=0.56, fill='#ffffff')
    arrow(frame_x + 1.0, middle + 0.12, right_x, middle + 0.5)
    arrow(right_x + 0.6, middle + 0.34, right_x + 0.6, middle - 0.34)
    ax.text(0, top + 0.5, 'етап 1: сегментація за кумулятивною гістограмою (лише для фото)',
            fontsize=6.6, color='#555')
    ax.text(0, bottom - 0.45, 'синтетичні силуети вже бінарні й подаються в кадр без сегментації',
            fontsize=6.6, color='#555')
    fig.savefig(path)
    plt.close(fig)


# дефект, фігура, поворот (рад), точка дефекту (ребро, частка вздовж нього)
EXAMPLE_ROWS = [
    ('line', 'square', 0.25, (0, 0.5)),
    ('bump', 'circle', 0.0, (84, 0.5)),
    ('blob', 'pentagon', 0.0, (0, 0.5)),
    ('slit', 'rectangle', 0.25, (0, 0.5)),
    ('dent', 'hexagon', 0.0, (0, 0.5)),
    ('irregular', 'ellipse', 0.4, None),
]
SIZES = (0.05, 0.10, 0.20, 0.40)
EPSILONS = (0.02, 0.03, 0.05, 0.08)


def figure_examples(path):
    fig, axes = plt.subplots(len(EXAMPLE_ROWS), 1 + len(SIZES), figsize=(5.6, 7.6))
    for r, (kind, shape, rotation, where) in enumerate(EXAMPLE_ROWS):
        values = (None, *(EPSILONS if kind == 'irregular' else SIZES))
        for c, value in enumerate(values):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            rng = np.random.default_rng(10 + r)
            fixed = dict(scale=1.0, rotation=rotation)
            if value is None:
                image = S.fit_to_frame(S.render(shape, rng, **fixed))
                ax.set_xlabel('без дефекту', fontsize=7)
            elif kind == 'irregular':
                image = S.fit_to_frame(S.render(shape, rng, eps=value, **fixed))
                ax.set_xlabel(f'ε = {value:g}', fontsize=7)
            elif kind == 'blob' and value < 0.2:
                ax.text(0.5, 0.5, 'не задається\n(h < 0.2)', ha='center', va='center',
                        fontsize=6.5, color='#888', transform=ax.transAxes)
                ax.set_xlabel(f'h = {value:g}', fontsize=7)
                for spine in ax.spines.values():
                    spine.set_color('#ddd')
                continue
            else:
                image = S.fit_to_frame(S.render(shape, rng, kind, value, where=where, **fixed))
                ax.set_xlabel(f'h = {value:g}', fontsize=7)
            # поле лише для показу: кадр вписаний по габаритах, і фігура торкається краю
            ax.imshow(np.pad(image, 8, constant_values=255), cmap='gray', vmin=0, vmax=255)
            for spine in ax.spines.values():
                spine.set_color('#999')
        label = 'нерівність\n(норма)' if kind == 'irregular' else TYPE_UA[kind].replace(' ', '\n', 1)
        axes[r, 0].set_ylabel(label, fontsize=7.5)
    fig.subplots_adjust(wspace=0.08, hspace=0.32)
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    figure_pipeline(out / 'fig_pipeline.png')
    figure_examples(out / 'fig_examples.png')
