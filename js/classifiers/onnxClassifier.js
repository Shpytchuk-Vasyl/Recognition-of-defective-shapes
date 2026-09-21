import { maskToCanvas } from '../core/imageUtils.js';

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';

export class OnnxDefectClassifier {
  constructor({ modelUrl = 'models/defect.onnx', input = 128, decision = 0.8, canvas = null } = {}) {
    this.modelUrl = modelUrl;
    this.input = input;
    this.decision = decision;
    // полотно з розмітки: воно ж показує користувачу те, що бачить модель
    this.canvas = canvas ?? document.createElement('canvas');
    this.session = null;
  }

  get name() { return 'Нейромережа (ONNX)'; }
  get ready() { return this.session !== null; }

  async init() {
    if (this.session) return;
    if (!window.ort) await loadScript(ORT_URL);

    const head = await fetch(this.modelUrl, { method: 'HEAD' }).catch(() => null);
    if (!head?.ok) {
      throw new Error(`Модель ${this.modelUrl} не знайдена — розпізнавання дефекту недоступне.`);
    }
    this.session = await window.ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ['wasm'],
    });
  }

  async classify(mask, size) {
    await this.init();

    const output = await this.session.run({
      [this.session.inputNames[0]]: this.toTensor(mask, size),
    });
    const raw = Array.from(output[this.session.outputNames[0]].data);
    const probability = raw.length >= 2 ? softmax(raw)[1] : raw[0];

    return {
      defect: probability > this.decision,
      probability,
      metrics: {
        'Імовірність дефекту': (probability * 100).toFixed(2) + ' %',
        'Поріг рішення': this.decision,
        'Вихід моделі': raw.map(value => value.toFixed(4)).join(', '),
        'Модель': `${this.label} (${this.modelUrl})`,
      },
    };
  }

  toTensor(mask, size) {
    const side = this.input;

    if (size === side) {
      maskToCanvas(mask, size, size, this.canvas);
      const data = new Float32Array(side * side * 3);
      for (let i = 0, p = 0; i < mask.length; i++, p += 3) {
        data[p] = data[p + 1] = data[p + 2] = mask[i] ? 0 : 255;
      }
      return new window.ort.Tensor('float32', data, [1, side, side, 3]);
    }

    const canvas = this.canvas;
    canvas.width = side;
    canvas.height = side;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, side, side);
    ctx.drawImage(maskToCanvas(mask, size, size), 0, 0, side, side);

    const pixels = ctx.getImageData(0, 0, side, side).data;
    const data = new Float32Array(side * side * 3);
    for (let i = 0, p = 0; i < side * side; i++, p += 3) {
      data[p] = pixels[i * 4];
      data[p + 1] = pixels[i * 4 + 1];
      data[p + 2] = pixels[i * 4 + 2];
    }
    return new window.ort.Tensor('float32', data, [1, side, side, 3]);
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error('Не вдалося завантажити onnxruntime-web'));
    document.head.appendChild(script);
  });
}

function softmax(values) {
  const max = Math.max(...values);
  const exp = values.map(value => Math.exp(value - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(value => value / sum);
}
