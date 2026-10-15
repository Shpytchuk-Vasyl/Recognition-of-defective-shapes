export class CumulativeHistogramSegmenter {
  constructor({
    levels = 256,          // кількість градацій яскравості
    minProminence = 0.002, // мінімальна виразність екстремуму
    significance = 0.3,    // частка від найбільшого |D|, щоб екстремум був вагомим
    target = 'auto',       // 'dark' | 'light' | 'auto' - де шукати обʼєкт
    manualLevel = null,    // поріг, заданий вручну; null - шукати автоматично
    cleanup = true,        // постобробка маски для фото
    size = 128,            // сторона кадру на виході - вхід нейромережі
  } = {}) {
    this.levels = levels;
    this.minProminence = minProminence;
    this.significance = significance;
    this.target = target;
    this.manualLevel = manualLevel;
    this.cleanup = cleanup;
    this.size = size;
  }

  get name() { return 'Кумулятивна гістограма'; }


  segment(imageData) {
    const { width, height } = imageData;
    const analysis = this.analyze(imageData);
    const { gray, target, curves, sOpt, chosen, manual } = analysis;

    const thresholded = this.threshold(gray, sOpt, target);
    const binary = this.cleanup ? this.mainObject(thresholded, width, height) : thresholded;
    const mask = this.fitToFrame(binary, width, height);
    if (!mask) throw new Error('Після сегментації не лишилося жодного пікселя фігури.');

    return {
      mask,
      size: this.size,
      binary,
      width,
      height,
      sOpt,
      target,
      curves,
      metrics: {
        'Обраний поріг яскравості': sOpt,
        'Вибір порога': manual === null ? 'автоматично' : 'вручну, по графіку',
        'Екстремум на цьому порозі': chosen
          ? (chosen.type === 'max' ? 'максимум' : 'мінімум')
          : 'не екстремум',
        'Відхилення від ідеальної кривої': curves.difference[sOpt].toFixed(4),
        'Усього екстремумів': curves.extrema.length,
        'Відкинуто як крайові': curves.extrema.filter(point => point.ignored).length,
        'Обʼєкт': target === 'dark' ? 'темний на світлому' : 'світлий на темному',
        'Постобробка маски': this.cleanup ? 'так' : 'ні',
        ...this.frameMetrics(mask),
      },
    };
  }

  /** Криві методу й обраний поріг - без бінаризації. */
  analyze(imageData) {
    const { width, height } = imageData;
    const gray = this.toGray(imageData);
    const target = this.resolveTarget(gray, width, height);

    const histogram = this.histogram(gray);
    const vf = this.cumulative(histogram, gray.length);
    const vfg = this.idealCumulative();
    const difference = this.difference(vf, vfg);

    const extrema = this.markEdgeExtrema(this.findExtrema(difference));
    const manual = this.pickedLevel();
    const auto = this.selectExtremum(extrema, target);
    const sOpt = manual ?? (auto ? auto.s : this.medianLevel(vf));

    // поріг міг потрапити і повз екстремуми - тоді просто нема що підсвічувати
    const chosen = extrema.find(point => point.s === sOpt) ?? null;
    if (chosen) chosen.selected = true;

    return {
      gray, target, sOpt, chosen, manual,
      curves: { levels: this.levelAxis(), vf, vfg, difference, extrema, sOpt },
    };
  }

  frameMetrics(mask) {
    const figureArea = this.area(mask);
    return {
      'Розмір кадру': `${this.size} × ${this.size}`,
      'Площа фігури, px': figureArea,
      'Заповнення кадру': (figureArea / (this.size * this.size) * 100).toFixed(2) + ' %',
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

  pickedLevel() {
    if (!Number.isFinite(this.manualLevel)) return null;
    const level = Math.round(this.manualLevel);
    return Math.min(this.levels - 1, Math.max(0, level));
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

  /**
   * Постобробка для фото: на нерівно освітленому папері поріг лягає на початок моди
   * паперу, і темніші ділянки та шум потрапляють у маску. Лишаємо найбільшу
   * 8-зв'язну область, що не торкається краю кадру, і заповнюємо в ній порожнини.
   */
  mainObject(mask, w, h) {
    const labels = new Int32Array(mask.length);
    const sizes = [0];
    const touchesEdge = [false];
    const stack = [];

    for (let start = 0; start < mask.length; start++) {
      if (!mask[start] || labels[start]) continue;
      const label = sizes.length;
      let size = 0, edge = false;
      labels[start] = label;
      stack.push(start);
      while (stack.length) {
        const i = stack.pop();
        const x = i % w, y = (i - x) / w;
        size++;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1) edge = true;
        for (let ny = Math.max(0, y - 1); ny <= Math.min(h - 1, y + 1); ny++) {
          for (let nx = Math.max(0, x - 1); nx <= Math.min(w - 1, x + 1); nx++) {
            const j = ny * w + nx;
            if (mask[j] && !labels[j]) { labels[j] = label; stack.push(j); }
          }
        }
      }
      sizes.push(size);
      touchesEdge.push(edge);
    }
    if (sizes.length === 1) return mask;

    const largest = allowEdge => {
      let best = 0;
      for (let label = 1; label < sizes.length; label++) {
        if ((allowEdge || !touchesEdge[label]) && sizes[label] > sizes[best]) best = label;
      }
      return best;
    };
    // фігура на весь кадр торкається краю - тоді беремо просто найбільшу область
    const keep = largest(false) || largest(true);

    // фон, досяжний від краю кадру (4-зв'язність), лишається фоном; решта - фігура з порожнинами
    const out = new Uint8Array(mask.length).fill(1);
    const visit = i => {
      if (out[i] && labels[i] !== keep) { out[i] = 0; stack.push(i); }
    };
    for (let x = 0; x < w; x++) { visit(x); visit((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { visit(y * w); visit(y * w + w - 1); }
    while (stack.length) {
      const i = stack.pop();
      const x = i % w;
      if (x > 0) visit(i - 1);
      if (x < w - 1) visit(i + 1);
      if (i >= w) visit(i - w);
      if (i + w < mask.length) visit(i + w);
    }
    return out;
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
