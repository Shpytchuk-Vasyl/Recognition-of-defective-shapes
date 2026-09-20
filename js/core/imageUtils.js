/** Дрібні операції з canvas: джерело -> пікселі, маска -> картинка. */

/** Будь-яке джерело (canvas або img) -> ImageData; прозорий фон вважаємо білим. */
export function toImageData(source) {
  const w = source.naturalWidth || source.width;
  const h = source.naturalHeight || source.height;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/** Бінарна маска -> canvas: чорна фігура на білому. */
export function maskToCanvas(mask, w, h, canvas = document.createElement('canvas')) {
  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(w, h);
  for (let i = 0; i < mask.length; i++) {
    const value = mask[i] ? 0 : 255;
    image.data[i * 4] = image.data[i * 4 + 1] = image.data[i * 4 + 2] = value;
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}
