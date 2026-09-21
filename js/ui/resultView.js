import { maskToCanvas } from '../core/imageUtils.js';

export class ResultView {
  constructor({ verdict, preview, metrics, charts, chartNote, maskCaption, onPickLevel }) {
    this.verdict = verdict;
    this.preview = preview;
    this.metrics = metrics;
    this.charts = charts;          // { cumulative, difference }
    this.chartNote = chartNote;
    this.maskCaption = maskCaption;
    this.onPickLevel = onPickLevel;
    this.rows = {};
  }

  setVerdict(kind, text) {
    this.verdict.className = 'verdict verdict-' + kind;
    this.verdict.textContent = text;
  }

  showSegmentation(segmentation) {
    const { size } = segmentation;
    maskToCanvas(segmentation.mask, size, size, this.preview);
    if (this.maskCaption) this.maskCaption.textContent = `Сегментована маска, ${size} × ${size}`;
    this.showCurves(segmentation.curves);
    this.rows = { ...segmentation.metrics };
    this.showMetrics(this.rows);
  }

  showClassification(classification) {

    this.setVerdict(classification.defect ? 'bad' : 'ok', (classification.defect
      ? 'Зображення З ДЕФЕКТОМ: нейромережа виявила зайву поверхню'
      : 'Зображення БЕЗ ДЕФЕКТУ: суцільна фігура')
      + ` (ймовірність дефекту ${(classification.probability * 100).toFixed(1)} %)`);
    this.rows = { ...this.rows, ...classification.metrics };
    this.showMetrics(this.rows);
  }

  showCurves(curves) {
    const { levels, vf, vfg, difference, extrema, sOpt } = curves;
    const color = name => getComputedStyle(this.verdict).getPropertyValue(name).trim();

    this.charts.cumulative.show({
      x: levels,
      xLabel: 'яскравість s',
      yLabel: 'накопичена частка',
      yRange: [0, 1.02],
      verticals: [sOpt],
      series: [
        { label: 'V_F(s) - зображення', values: Array.from(vf), color: color('--series-1') },
        { label: 'V_FG(s) - ідеальна', values: Array.from(vfg), color: color('--series-2'), dash: [6, 4] },
      ],
    });

    const kept = extrema.filter(point => !point.ignored && point.s !== sOpt);
    const dropped = extrema.filter(point => point.ignored && point.s !== sOpt);
    const selected = [{ s: sOpt, value: difference[sOpt] }];

    this.charts.difference.show({
      x: levels,
      xLabel: 'яскравість s',
      yLabel: 'D(s) = V_F − V_FG',
      zeroLine: true,
      verticals: [sOpt],
      onPick: value => this.onPickLevel?.(value),
      series: [
        { label: 'D(s)', values: Array.from(difference), color: color('--series-1') },
        ...markerSeries('Екстремуми', kept, levels.length, color('--series-2')),
        ...markerSeries('Крайові, відкинуті', dropped, levels.length, color('--text-muted')),
        ...markerSeries('Обраний поріг', selected, levels.length, color('--series-3'), 12),
      ],
    });

    this.chartNote.textContent =
      `Обраний поріг яскравості ${sOpt}; знайдено ${extrema.length} екстремумів, `
      + `${extrema.filter(point => point.ignored).length} відкинуто як крайові. `
      + 'Клацніть по графіку, щоб узяти за поріг будь-яку точку кривої.';
  }

  showMetrics(rows) {
    this.metrics.innerHTML = Object.entries(rows)
      .map(([key, value]) => `<tr><td>${key}</td><td>${value}</td></tr>`).join('');
  }
}

function markerSeries(label, points, length, color, size) {
  if (!points.length) return [];
  const values = new Array(length).fill(null);
  for (const point of points) values[point.s] = point.value;
  return [{ label, values, color, points: true, size }];
}
