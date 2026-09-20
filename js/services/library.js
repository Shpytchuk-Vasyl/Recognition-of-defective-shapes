const IMAGE_RE = /\.(png|jpe?g|gif|bmp|webp)$/i;

export async function listImages(dir = 'img/') {
  const api = await readApi();
  if (api) return { files: api, source: 'serve.py' };

  const index = await readDirectoryIndex(dir);
  if (index) return { files: index, source: 'листинг каталогу' };

  return { files: [], source: null };
}

async function readApi() {
  try {
    const res = await fetch('api/images', { cache: 'no-store' });
    if (!res.ok) return null;
    const json = await res.json();
    return (json.files || []).filter(name => IMAGE_RE.test(name));
  } catch {
    return null;
  }
}

async function readDirectoryIndex(dir) {
  try {
    const res = await fetch(dir, { cache: 'no-store' });
    if (!res.ok) return null;
    const doc = new DOMParser().parseFromString(await res.text(), 'text/html');
    const names = [...doc.querySelectorAll('a[href]')]
      .map(a => decodeURIComponent(a.getAttribute('href').split('/').pop().split('?')[0]))
      .filter(name => IMAGE_RE.test(name));
    return [...new Set(names)];
  } catch {
    return null;
  }
}

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Не вдалося завантажити ' + url));
    img.src = url;
  });
}
