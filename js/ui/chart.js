import uPlot from '../../vendor/uPlot.esm.js';

/**
 * Обгортка uPlot під графіки методу.
 *
 * spec = {
 *   x, xLabel, yLabel, yRange, height,
 *   series: [{ label, color, dash?, points?, size? }] - points - лише маркери без лінії,
 *   zeroLine, verticals: [числа по осі X]
 * }
 */
export class Chart {
  constructor(container) {
    this.container = container;
    this.spec = null;
    this.plot = null;

    new ResizeObserver(() => this.plot?.setSize(this.size())).observe(container);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.render());
  }

  show(spec) {
    this.spec = spec;
    this.render();
  }

  clear() {
    this.spec = null;
    this.destroy();
    this.container.replaceChildren();
  }

  destroy() {
    this.plot?.destroy();
    this.plot = null;
  }

  token(name, fallback) {
    return getComputedStyle(this.container).getPropertyValue(name).trim() || fallback;
  }

  size() {
    return {
      width: Math.max(280, this.container.clientWidth),
      height: this.spec?.height ?? 240,
    };
  }

  render() {
    if (!this.spec) return;
    this.destroy();
    this.container.replaceChildren();

    const { x, series, xLabel, yLabel, yRange, zeroLine, verticals } = this.spec;
    const ink = this.token('--text-muted', '#85847d');
    const grid = { stroke: this.token('--grid', '#e6e5e0'), width: 1 };
    const axis = { stroke: ink, grid, ticks: grid };

    this.plot = new uPlot({
      ...this.size(),
      padding: [10, 12, 0, 0],
      cursor: { y: false, points: { size: 8 } },
      legend: { live: true },
      scales: { x: { time: false }, y: yRange ? { range: yRange } : {} },
      axes: [{ ...axis, label: xLabel }, { ...axis, label: yLabel, labelSize: 34 }],
      series: [{ label: xLabel }, ...series.map(one => this.seriesOptions(one))],
      plugins: [this.guides(zeroLine, verticals ?? [], ink)],
    }, [x, ...series.map(one => one.values)], this.container);
  }

  seriesOptions({ label, color, dash, points, size = 9 }) {
    if (!points) return { label, stroke: color, width: 2, dash };
    return {
      label,
      stroke: color,
      width: 0,
      paths: () => null,
      points: { show: true, size, stroke: color, fill: color },
    };
  }

  /** Пунктирна нульова лінія та вертикалі (наприклад, обраний поріг). */
  guides(zeroLine, verticals, color) {
    return {
      hooks: {
        draw: u => {
          const { ctx, bbox } = u;
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;

          if (zeroLine) {
            const y = Math.round(u.valToPos(0, 'y', true)) + 0.5;
            stroke(ctx, bbox.left, y, bbox.left + bbox.width, y);
          }
          for (const value of verticals) {
            const x = Math.round(u.valToPos(value, 'x', true)) + 0.5;
            stroke(ctx, x, bbox.top, x, bbox.top + bbox.height);
          }
          ctx.restore();
        },
      },
    };
  }
}

function stroke(ctx, x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}
