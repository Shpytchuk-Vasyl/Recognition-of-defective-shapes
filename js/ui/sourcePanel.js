import { Board } from './board.js';
import { Tabs } from './tabs.js';
import { Gallery } from './gallery.js';
import { loadImage } from '../services/library.js';
import { ImageFolder } from '../services/imageFolder.js';
import { SampleGenerator } from '../data/sampleGenerator.js';

export class SourcePanel {
  constructor({ root, brushSize }) {
    this.root = root;
    this.uploaded = null;

    this.board = new Board(root.querySelector('#draw'), {
      tool: () => root.querySelector('#tool').value,
      brushSize,
    });

    this.gallery = new Gallery({
      root: root.querySelector('#gallery'),
      status: root.querySelector('#libraryStatus'),
    });

    this.tabs = new Tabs({
      root,
      onChange: name => {
        if (name === 'library') this.gallery.refresh();
      },
    });

    this.folder = new ImageFolder();

    this.bindDrawTab();
    this.bindUploadTab();
    this.root.querySelector('#generateSamples')
      .addEventListener('click', () => this.generateSamples());

    this.tabs.select('draw');
    this.board.demo(true);
    this.gallery.refresh();
  }

  /** Малює набір тестових фігур і кладе їх у папку зі сталими іменами. */
  async generateSamples() {
    const button = this.root.querySelector('#generateSamples');
    const status = this.root.querySelector('#libraryStatus');
    button.disabled = true;
    status.textContent = 'Малюю зразки…';

    try {
      const samples = await new SampleGenerator().build();
      status.textContent = `Записую ${samples.length} файлів…`;
      const { written, mode } = await this.folder.writeAll(samples);
      if (mode === 'folder') await this.gallery.refresh();
      status.textContent = mode === 'folder'
        ? `Записано ${written} файлів у вибрану папку.`
        : `Браузер не вміє писати в папку — ${written} файлів завантажено.`;
    } catch (err) {
      status.textContent = err.name === 'AbortError'
        ? 'Генерацію скасовано: папку не вибрано.'
        : 'Не вдалося згенерувати зразки: ' + err.message;
    } finally {
      button.disabled = false;
    }
  }

  current() {
    if (this.tabs.current === 'upload') return this.uploaded;
    if (this.tabs.current === 'library') return this.gallery.selected?.img || null;
    return this.board.isEmpty ? null : this.board.snapshot();
  }

  bindDrawTab() {
    const q = sel => this.root.querySelector(sel);
    q('#tool').addEventListener('change', () => this.board.syncTool());
    q('#brushSize').addEventListener('input', () => this.board.syncTool());
    q('#clearCanvas').addEventListener('click', () => this.board.clear());
    q('#demoClean').addEventListener('click', () => this.board.demo(false));
    q('#demoDefect').addEventListener('click', () => this.board.demo(true));

    window.addEventListener('keydown', event => {
      const onBoard = this.tabs.current === 'draw' && document.activeElement === document.body;
      if (onBoard && (event.key === 'Delete' || event.key === 'Backspace')
          && this.board.deleteSelection()) {
        event.preventDefault();
      }
    });
  }

  bindUploadTab() {
    const dropzone = this.root.querySelector('#dropzone');
    const preview = this.root.querySelector('#uploadPreview');

    this.root.querySelector('#fileInput').addEventListener('change', e => {
      if (e.target.files[0]) this.acceptFile(e.target.files[0], preview);
    });
    for (const type of ['dragenter', 'dragover', 'dragleave', 'drop']) {
      dropzone.addEventListener(type, e => {
        e.preventDefault();
        dropzone.classList.toggle('is-over', type === 'dragenter' || type === 'dragover');
      });
    }
    dropzone.addEventListener('drop', e => {
      if (e.dataTransfer.files[0]) this.acceptFile(e.dataTransfer.files[0], preview);
    });
  }

  async acceptFile(file, preview) {
    const url = URL.createObjectURL(file);
    try {
      this.uploaded = await loadImage(url);
      drawFitted(preview, this.uploaded);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

function drawFitted(canvas, img) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
  const w = img.width * scale, h = img.height * scale;
  ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
}
