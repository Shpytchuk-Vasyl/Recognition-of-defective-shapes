import { listImages } from '../services/library.js';

export class Gallery {
  constructor({ root, status }) {
    this.root = root;
    this.status = status;
    this.selected = null;
  }

  async refresh() {
    this.status.textContent = 'Читаю папку img/…';
    const { files, source } = await listImages();

    this.root.replaceChildren();
    this.selected = null;

    if (!files.length) {
      this.status.textContent = 'Папка img/ порожня - покладіть туди зображення і натисніть «Оновити».';
      return;
    }
    this.status.textContent = `${files.length} файл(ів), джерело: ${source}`;

    const stamp = Date.now();   // обходимо кеш, якщо файл замінили тим самим імʼям
    for (const name of files) this.root.append(this.createTile(name, stamp));
  }

  createTile(name, stamp) {
    const figure = document.createElement('figure');
    const img = new Image();
    img.src = `img/${encodeURIComponent(name)}?v=${stamp}`;
    img.alt = name;

    const caption = document.createElement('figcaption');
    caption.textContent = name;

    figure.append(img, caption);
    figure.addEventListener('click', () => {
      this.selected = { name, img };
      [...this.root.children].forEach(tile => tile.classList.toggle('is-selected', tile === figure));
    });
    return figure;
  }
}
