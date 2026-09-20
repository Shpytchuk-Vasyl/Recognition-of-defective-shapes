export class CumulativeHistogramSegmenter {
  constructor({
    levels = 256,          // кількість градацій яскравості
    minProminence = 0.002, // мінімальна виразність екстремуму
    significance = 0.3,    // частка від найбільшого |D|, щоб екстремум був вагомим
    target = 'auto',       // 'dark' | 'light' | 'auto' - де шукати обʼєкт
    size = 128,            // сторона кадру на виході
  } = {}) {
    this.levels = levels;
    this.minProminence = minProminence;
    this.significance = significance;
    this.target = target;
    this.size = size;
  }

  get name() { return 'Кумулятивна гістограма'; }


  segment(imageData) {
    const { width, height } = imageData;
    const gray = this.toGray(imageData);
    const target = this.resolveTarget(gray, width, height);

    const histogram = this.histogram(gray);
    const vf = this.cumulative(histogram, gray.length);
    const vfg = this.idealCumulative();
    const difference = this.difference(vf, vfg);

    const extrema = this.markEdgeExtrema(this.findExtrema(difference));
    const chosen = this.selectExtremum(extrema, target);
    if (chosen) chosen.selected = true;
    const sOpt = chosen ? chosen.s : this.medianLevel(vf);

    const binary = this.threshold(gray, sOpt, target);
    const mask = this.fitToFrame(binary, width, height);
    if (!mask) throw new Error('Після сегментації не лишилося жодного пікселя фігури.');

    const figureArea = this.area(mask);

    return {
      mask,
      size: this.size,
      sOpt,
      target,
      curves: { levels: this.levelAxis(), vf, vfg, difference, extrema, sOpt },
      metrics: {
        'Обраний поріг яскравості': sOpt,
        'Екстремум на цьому порозі': chosen ? (chosen.type === 'max' ? 'максимум' : 'мінімум') : 'не знайдено',
        'Відхилення від ідеальної кривої': chosen ? chosen.value.toFixed(4) : '-',
        'Усього екстремумів': extrema.length,
        'Відкинуто як крайові': extrema.filter(point => point.ignored).length,
        'Обʼєкт': target === 'dark' ? 'темний на світлому' : 'світлий на темному',
        'Розмір кадру': `${this.size} × ${this.size}`,
        'Площа фігури, px': figureArea,
        'Заповнення кадру': (figureArea / (this.size * this.size) * 100).toFixed(2) + ' %',
      },
    };
  }

  toGray(imageData) {
    const { data } = imageData;
    const gray = new Uint8ClampedArray(data.length / 4);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      gray[i] = (data[p] * 0.299 + data[p + 1] * 0.587 + data[p + 2] * 0.114) | 0;
    }
    return gray;
  }

  histogram(gray) {
    const hist = new Float64Array(this.levels);
    for (const value of gray) hist[value]++;
    return hist;
  }

  cumulative(histogram, total) {
    const vf = new Float64Array(this.levels);
    let acc = 0;
    for (let s = 0; s < this.levels; s++) {
      acc += histogram[s];
      vf[s] = acc / total;
    }
    return vf;
  }

  idealCumulative() {
    const vfg = new Float64Array(this.levels);
    for (let s = 0; s < this.levels; s++) vfg[s] = (s + 1) / this.levels;
    return vfg;
  }

  levelAxis() {
    return Array.from({ length: this.levels }, (_, s) => s);
  }

  difference(vf, vfg) {
    const d = new Float64Array(this.levels);
    for (let s = 0; s < this.levels; s++) d[s] = vf[s] - vfg[s];
    return d;
  }

  findExtrema(d) {
    const last = d.length - 1;
    const found = [];
    const add = (s, type) => found.push({ s, value: d[s], type });

    if (d[0] !== d[1]) add(0, d[0] > d[1] ? 'max' : 'min');
    for (let s = 1; s < last; s++) {
      const prev = d[s - 1], cur = d[s], next = d[s + 1];
      if (cur > prev && cur >= next) add(s, 'max');
      else if (cur < prev && cur <= next) add(s, 'min');
    }
    if (d[last] !== d[last - 1]) add(last, d[last] > d[last - 1] ? 'max' : 'min');

    return this.filterProminence(found);
  }

  filterProminence(extrema) {
    const kept = [];
    for (const point of extrema) {
      const previous = kept[kept.length - 1];
      if (!previous) { kept.push(point); continue; }
      if (previous.type === point.type) {
        if (Math.abs(point.value) > Math.abs(previous.value)) kept[kept.length - 1] = point;
        continue;
      }
      if (Math.abs(point.value - previous.value) >= this.minProminence) kept.push(point);
    }
    return kept;
  }

  markEdgeExtrema(extrema) {
    const marked = extrema.map(point => ({ ...point, ignored: false, selected: false }));
    const first = marked[0];
    const last = marked[marked.length - 1];
    if (first && first.type === 'min') first.ignored = true;
    if (last && last !== first && last.type === 'max') last.ignored = true;
    return marked;
  }

  selectExtremum(extrema, target) {
    const candidates = extrema.filter(point => !point.ignored);
    const pool = candidates.length ? candidates : extrema;
    if (!pool.length) return null;

    const strongest = Math.max(...pool.map(point => Math.abs(point.value)));
    const significant = pool.filter(
      point => Math.abs(point.value) >= strongest * this.significance);

    return target === 'light' ? significant[significant.length - 1] : significant[0];
  }

  medianLevel(vf) {
    for (let s = 0; s < this.levels; s++) if (vf[s] >= 0.5) return s;
    return this.levels >> 1;
  }

  resolveTarget(gray, w, h) {
    if (this.target !== 'auto') return this.target;
    let sum = 0, n = 0;
    for (let x = 0; x < w; x++) { sum += gray[x] + gray[(h - 1) * w + x]; n += 2; }
    for (let y = 0; y < h; y++) { sum += gray[y * w] + gray[y * w + w - 1]; n += 2; }
    return sum / n > this.levels / 2 ? 'dark' : 'light';
  }

  threshold(gray, sOpt, target) {
    const mask = new Uint8Array(gray.length);
    for (let i = 0; i < gray.length; i++) {
      mask[i] = (target === 'dark' ? gray[i] <= sOpt : gray[i] > sOpt) ? 1 : 0;
    }
    return mask;
  }

  fitToFrame(mask, w, h) {
    const box = this.boundingBox(mask, w, h);
    if (!box) return null;

    const size = this.size;
    const scale = size / Math.max(box.w, box.h);
    const dw = Math.max(1, Math.round(box.w * scale));
    const dh = Math.max(1, Math.round(box.h * scale));
    const ox = Math.floor((size - dw) / 2);
    const oy = Math.floor((size - dh) / 2);
    const out = new Uint8Array(size * size);

    for (let dy = 0; dy < dh; dy++) {
      const y0 = Math.floor(box.y0 + dy * box.h / dh);
      const y1 = Math.max(y0 + 1, Math.ceil(box.y0 + (dy + 1) * box.h / dh));
      for (let dx = 0; dx < dw; dx++) {
        const x0 = Math.floor(box.x0 + dx * box.w / dw);
        const x1 = Math.max(x0 + 1, Math.ceil(box.x0 + (dx + 1) * box.w / dw));
        let sum = 0, n = 0;
        for (let y = y0; y < y1 && y < h; y++) {
          for (let x = x0; x < x1 && x < w; x++) { sum += mask[y * w + x]; n++; }
        }
        if (n && sum / n >= 0.5) out[(oy + dy) * size + ox + dx] = 1;
      }
    }
    return out;
  }

  boundingBox(mask, w, h) {
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!mask[y * w + x]) continue;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
    return x1 < 0 ? null : { x0, y0, x1, y1, w: x1 - x0 + 1, h: y1 - y0 + 1 };
  }

  area(mask) {
    let total = 0;
    for (const value of mask) total += value;
    return total;
  }
}
