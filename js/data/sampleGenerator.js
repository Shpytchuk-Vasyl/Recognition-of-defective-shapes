import {
  StaticCanvas, Rect, Circle, Ellipse, Polygon, Line,
} from '../../vendor/fabric.min.mjs';

const INK = '#000';
const PAPER = '#fff';

/**
 * Генератор тестових зображень: базові фігури та їх копії із зайвими
 * поверхнями — тонкою лінією, невеликим виступом і наростом на ніжці.
 *
 * Імена файлів детерміновані (`<фігура>_<варіант>.png`), тож повторна
 * генерація перезаписує ті самі файли, а не плодить нові.
 */
export class SampleGenerator {
  constructor({ size = 256 } = {}) {
    this.size = size;
    this.center = size / 2;
    this.radius = size * 0.3;
  }

  get shapes() {
    const c = this.center;
    const R = this.radius;
    return [
      { key: 'square', angle: 0, outline: this.rectOutline(R * 0.85, R * 0.85),
        object: () => new Rect(this.centered({ width: R * 1.7, height: R * 1.7 })) },

      { key: 'rectangle', angle: -Math.PI / 2, outline: this.rectOutline(R * 1.05, R * 0.6),
        object: () => new Rect(this.centered({ width: R * 2.1, height: R * 1.2 })) },

      { key: 'circle', angle: -Math.PI / 5, outline: this.ellipseOutline(R, R),
        object: () => new Circle(this.centered({ radius: R })) },

      { key: 'ellipse', angle: Math.PI * 0.75, outline: this.ellipseOutline(R * 1.15, R * 0.68),
        object: () => new Ellipse(this.centered({ rx: R * 1.15, ry: R * 0.68 })) },

      { key: 'pentagon', angle: Math.PI / 2, outline: this.regularOutline(5, R),
        object: () => new Polygon(this.regularOutline(5, R), { fill: INK }) },

      { key: 'hexagon', angle: 0, outline: this.regularOutline(6, R),
        object: () => new Polygon(this.regularOutline(6, R), { fill: INK }) },
    ].map(shape => ({ ...shape, center: { x: c, y: c } }));
  }

  get variants() {
    return ['clean', 'line', 'bump', 'blob'];
  }

  /** Усі зразки як [{ name, blob }]. */
  async build() {
    const samples = [];
    for (const shape of this.shapes) {
      for (const variant of this.variants) {
        samples.push({
          name: `${shape.key}_${variant}.png`,
          blob: await this.toBlob(this.render(shape, variant)),
        });
      }
    }
    return samples;
  }

  /** Один зразок як HTMLCanvasElement. */
  render(shape, variant) {
    const element = document.createElement('canvas');
    element.width = this.size;
    element.height = this.size;

    const canvas = new StaticCanvas(element, {
      width: this.size,
      height: this.size,
      backgroundColor: PAPER,
      renderOnAddRemove: false,
    });

    canvas.add(shape.object(), ...this.defect(shape, variant));
    canvas.renderAll();

    const rendered = canvas.toCanvasElement();
    canvas.dispose();
    return rendered;
  }

  /** Обʼєкти зайвої поверхні для заданого варіанта. */
  defect(shape, variant) {
    if (variant === 'clean') return [];

    const R = this.radius;
    const anchor = this.rayHit(shape.outline, shape.center, shape.angle);
    if (!anchor) return [];

    const dx = Math.cos(shape.angle);
    const dy = Math.sin(shape.angle);
    const inside = { x: anchor.x - dx * R * 0.06, y: anchor.y - dy * R * 0.06 };

    if (variant === 'line') {
      const tip = { x: anchor.x + dx * R * 0.55, y: anchor.y + dy * R * 0.55 };
      return [this.stroke(inside, tip, R * 0.07)];
    }

    if (variant === 'bump') {
      return [new Circle(this.at(anchor, { radius: R * 0.17 }))];
    }

    const blob = { x: anchor.x + dx * R * 0.42, y: anchor.y + dy * R * 0.42 };
    return [
      this.stroke(inside, blob, R * 0.055),
      new Circle(this.at(blob, { radius: R * 0.19 })),
    ];
  }

  /* ─────────── контури для прив'язки дефекту ─────────── */

  rectOutline(halfW, halfH) {
    const c = this.center;
    return [
      { x: c - halfW, y: c - halfH }, { x: c + halfW, y: c - halfH },
      { x: c + halfW, y: c + halfH }, { x: c - halfW, y: c + halfH },
    ];
  }

  ellipseOutline(rx, ry, steps = 96) {
    const c = this.center;
    return Array.from({ length: steps }, (_, i) => {
      const t = i / steps * Math.PI * 2;
      return { x: c + rx * Math.cos(t), y: c + ry * Math.sin(t) };
    });
  }

  regularOutline(sides, r) {
    const c = this.center;
    return Array.from({ length: sides }, (_, i) => {
      const t = -Math.PI / 2 + i * Math.PI * 2 / sides;
      return { x: c + r * Math.cos(t), y: c + r * Math.sin(t) };
    });
  }

  /** Точка перетину променя з центру та контуру фігури. */
  rayHit(points, center, angle) {
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    let nearest = null;

    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      const ex = b.x - a.x;
      const ey = b.y - a.y;
      const den = ex * dy - dx * ey;
      if (Math.abs(den) < 1e-9) continue;

      const t = (ex * (a.y - center.y) - ey * (a.x - center.x)) / den;
      const u = (dx * (a.y - center.y) - dy * (a.x - center.x)) / den;
      if (t > 0 && u >= 0 && u <= 1 && (!nearest || t < nearest.t)) {
        nearest = { t, x: center.x + dx * t, y: center.y + dy * t };
      }
    }
    return nearest;
  }

  /* ─────────── дрібні помічники ─────────── */

  centered(options) {
    return { originX: 'center', originY: 'center', left: this.center, top: this.center, fill: INK, ...options };
  }

  at(point, options) {
    return { originX: 'center', originY: 'center', left: point.x, top: point.y, fill: INK, ...options };
  }

  stroke(from, to, width) {
    return new Line([from.x, from.y, to.x, to.y], {
      stroke: INK, strokeWidth: width, strokeLineCap: 'round',
    });
  }

  toBlob(canvas) {
    return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
  }
}
