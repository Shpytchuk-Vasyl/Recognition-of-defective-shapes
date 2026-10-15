import { CumulativeHistogramSegmenter } from './cumulativeHistogramSegmenter.js';

export class DrawnSilhouette {
  constructor({ example = new CumulativeHistogramSegmenter() } = {}) {
    this.example = example;
  }

  segment(imageData) {
    const { width, height } = imageData;
    const binary = this.binarize(imageData);
    const mask = this.example.fitToFrame(binary, width, height);
    if (!mask) throw new Error('На полотні немає фігури.');

    const { curves } = this.example.analyze(imageData);

    return {
      mask,
      size: this.example.size,
      binary,
      width,
      height,
      direct: true,
      curves,
      metrics: {
        'Джерело': 'редактор, без сегментації',
        ...this.example.frameMetrics(mask),
      },
    };
  }

  binarize({ data }) {
    const ink = new Uint8Array(data.length / 4);
    let strongest = 0;
    for (let i = 0, p = 0; i < ink.length; i++, p += 4) {
      ink[i] = 255 - Math.min(data[p], data[p + 1], data[p + 2]);
      if (ink[i] > strongest) strongest = ink[i];
    }
    const level = Math.max(16, strongest / 2);
    const binary = new Uint8Array(ink.length);
    for (let i = 0; i < ink.length; i++) binary[i] = ink[i] >= level ? 1 : 0;
    return binary;
  }
}
