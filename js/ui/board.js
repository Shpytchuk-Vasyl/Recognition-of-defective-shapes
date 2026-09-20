import {
  Canvas, Circle, Ellipse, Rect, Triangle, Polygon, Line, PencilBrush,
} from '../../vendor/fabric.min.mjs';

const INK = '#000';
const PAPER = '#fff';
const SNAP = 15;          // крок прилипання кута, градуси
const MIN_RADIUS = 3;     // коротший штрих вважаємо промахом

/**
 * Кожна фігура будується з одиничним розміром, а протяжка задає їй
 * радіус і поворот, тож resize() однаково працює для всіх.
 */
const SHAPES = {
  circle: p => new Circle({ ...base(p), radius: 1 }),
  ellipse: p => new Ellipse({ ...base(p), rx: 1, ry: 0.62 }),
  square: p => new Rect({ ...base(p), width: Math.SQRT2, height: Math.SQRT2 }),
  rectangle: p => new Rect({ ...base(p), width: Math.SQRT2, height: Math.SQRT2 * 0.6 }),
  triangle: p => new Triangle({ ...base(p), width: 1.8, height: 1.8 }),
  pentagon: p => new Polygon(regularPoints(5), base(p)),
  hexagon: p => new Polygon(regularPoints(6), base(p)),
};

export class Board {
  constructor(el, { tool = () => 'brush', brushSize = () => 18 } = {}) {
    this.getTool = tool;
    this.getBrushSize = brushSize;

    this.canvas = new Canvas(el, { backgroundColor: PAPER, selection: false });
    this.canvas.freeDrawingBrush = new PencilBrush(this.canvas);

    this.draft = null;
    this.origin = null;

    this.canvas.on('mouse:down', e => this.onDown(e));
    this.canvas.on('mouse:move', e => this.onMove(e));
    this.canvas.on('mouse:up', () => this.onUp());

    this.syncTool();
  }

  syncTool() {
    const tool = this.getTool();
    const painting = tool === 'brush' || tool === 'eraser';
    const selecting = tool === 'select';

    this.canvas.isDrawingMode = painting;
    this.canvas.selection = selecting;
    this.canvas.defaultCursor = selecting ? 'default' : 'crosshair';
    this.canvas.forEachObject(object => {
      object.selectable = selecting;
      object.evented = selecting;
    });
    if (painting) {
      this.canvas.freeDrawingBrush.color = tool === 'eraser' ? PAPER : INK;
      this.canvas.freeDrawingBrush.width = this.getBrushSize();
    }
    if (!selecting) this.canvas.discardActiveObject();
    this.canvas.requestRenderAll();
  }

  onDown(e) {
    const create = SHAPES[this.getTool()];
    if (!create) return;
    this.origin = this.canvas.getScenePoint(e.e);
    this.draft = create(this.origin);
    this.canvas.add(this.draft);
  }

  /** Довжина протяжки задає розмір, напрямок — поворот. */
  onMove(e) {
    if (!this.draft) return;
    const point = this.canvas.getScenePoint(e.e);
    const dx = point.x - this.origin.x;
    const dy = point.y - this.origin.y;
    const radius = Math.hypot(dx, dy);

    resize(this.draft, this.origin, radius);
    if (radius > 8 && !(this.draft instanceof Circle)) {
      const angle = Math.atan2(dy, dx) * 180 / Math.PI;
      this.draft.set({ angle: e.e.shiftKey ? angle : Math.round(angle / SNAP) * SNAP });
    }
    this.draft.setCoords();
    this.canvas.requestRenderAll();
  }

  onUp() {
    if (!this.draft) return;
    if (radiusOf(this.draft) < MIN_RADIUS) this.canvas.remove(this.draft);
    this.draft = null;
    this.origin = null;
    this.syncTool();
    this.canvas.requestRenderAll();
  }

  /** Видаляє виділене; повертає true, якщо було що видаляти. */
  deleteSelection() {
    const active = this.canvas.getActiveObjects();
    if (!active.length) return false;
    this.canvas.remove(...active);
    this.canvas.discardActiveObject();
    this.canvas.requestRenderAll();
    return true;
  }

  clear() {
    this.canvas.remove(...this.canvas.getObjects());
    this.canvas.backgroundColor = PAPER;
    this.canvas.requestRenderAll();
  }

  snapshot() {
    return this.canvas.toCanvasElement();
  }

  get isEmpty() { return this.canvas.getObjects().length === 0; }

  demo(withDefect) {
    this.clear();
    const c = this.canvas.width / 2;
    const R = this.canvas.width * 0.28;
    this.canvas.add(new Circle({ ...base({ x: c, y: c }), radius: R }));

    if (withDefect) {
      const angle = -Math.PI / 5;
      const blobR = R * 0.22;
      const bx = c + Math.cos(angle) * (R + blobR * 1.8);
      const by = c + Math.sin(angle) * (R + blobR * 1.8);
      this.canvas.add(new Line(
        [c + Math.cos(angle) * R * 0.98, c + Math.sin(angle) * R * 0.98, bx, by],
        { stroke: INK, strokeWidth: blobR * 0.32, strokeLineCap: 'round' },
      ));
      this.canvas.add(new Circle({ ...base({ x: bx, y: by }), radius: blobR }));
    }
    this.canvas.requestRenderAll();
  }
}

function base(p) {
  return {
    originX: 'center', originY: 'center', left: p.x, top: p.y,
    fill: INK, selectable: false, evented: false,
  };
}

/** Правильний багатокутник з описаним радіусом 1, зцентрований за габаритами. */
function regularPoints(sides) {
  const points = Array.from({ length: sides }, (_, i) => {
    const angle = -Math.PI / 2 + i * 2 * Math.PI / sides;
    return { x: Math.cos(angle), y: Math.sin(angle) };
  });
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  return points.map(p => ({ x: p.x - cx, y: p.y - cy }));
}

function resize(shape, origin, radius) {
  const r = Math.max(1, radius);
  if (shape instanceof Circle) shape.set({ radius: r });
  else if (shape instanceof Ellipse) shape.set({ rx: r, ry: r * 0.62 });
  else if (shape instanceof Polygon) shape.set({ scaleX: r, scaleY: r });
  else shape.set({ scaleX: r, scaleY: r });
  shape.set({ left: origin.x, top: origin.y });
}

function radiusOf(shape) {
  if (shape instanceof Circle) return shape.radius;
  if (shape instanceof Ellipse) return shape.rx;
  return Math.max(shape.width * shape.scaleX, shape.height * shape.scaleY) / 2;
}
