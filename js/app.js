import { SourcePanel } from './ui/sourcePanel.js';
import { Chart } from './ui/chart.js';
import { ResultView } from './ui/resultView.js';
import { CumulativeHistogramSegmenter } from './segmentation/cumulativeHistogramSegmenter.js';
import { DrawnSilhouette } from './segmentation/drawnSilhouette.js';
import { OnnxDefectClassifier } from './classifiers/onnxClassifier.js';
import { toImageData } from './core/imageUtils.js';

const $ = sel => document.querySelector(sel);

const source = new SourcePanel({
  root: $('#sourcePanel'),
  brushSize: () => Number($('#brushSize').value),
});

const view = new ResultView({
  verdict: $('#verdict'),
  preview: $('#maskPreview'),
  metrics: $('#metrics tbody'),
  charts: {
    cumulative: new Chart($('#chartCumulative')),
    difference: new Chart($('#chartDifference')),
  },
  chartNote: $('#chartNote'),
  maskCaption: $('#maskCaption'),
  directNote: $('#directNote'),
  onPickLevel: level => run(level),
});

const loaded = new Map();
const classifier = () => {
  const option = $('#pModel').selectedOptions[0];
  const url = option.value;
  if (!loaded.has(url)) {
    loaded.set(url, new OnnxDefectClassifier({
      modelUrl: url,
      label: option.textContent.trim(),
      canvas: $('#modelPreview'),
    }));
  }
  return loaded.get(url);
};

const segmenter = manualLevel => new CumulativeHistogramSegmenter({
  minProminence: Number($('#pProminence').value),
  target: $('#pTarget').value,
  cleanup: $('#pCleanup').checked,
  manualLevel,
});

// фігура з редактора йде в модель напряму, фото й файли - через сегментацію
const prepare = manualLevel => current.drawn
  ? new DrawnSilhouette({ example: segmenter(manualLevel) })
  : segmenter(manualLevel);

let current = null;

$('#process').addEventListener('click', () => {
  const image = source.current();
  if (!image) {
    return view.setVerdict('idle',
      'Немає зображення: намалюйте фігуру, завантажте файл або оберіть із бібліотеки.');
  }
  current = { image: toImageData(image), drawn: source.fromEditor };
  run(null);
});

async function run(manualLevel) {
  if (!current) return;
  view.setVerdict('idle', current.drawn ? 'Підготовка силуету…' : 'Сегментація…');

  let segmentation;
  try {
    segmentation = prepare(manualLevel).segment(current.image);
    view.showSegmentation(segmentation);
  } catch (err) {
    return view.setVerdict('idle', 'Сегментація не вдалася: ' + err.message);
  }

  view.setVerdict('idle', 'Розпізнавання…');
  try {
    view.showClassification(await classifier().classify(segmentation.mask, segmentation.size));
  } catch (err) {
    view.setVerdict('idle', 'Сегментацію виконано. ' + err.message);
  }
}
