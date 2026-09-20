export class ImageFolder {
  constructor() {
    this.handle = null;
  }

  get supported() {
    return typeof window.showDirectoryPicker === 'function';
  }

  async ensureAccess() {
    if (!this.supported) return false;
    if (!this.handle) {
      this.handle = await window.showDirectoryPicker({ id: 'samples', mode: 'readwrite' });
    }
    const options = { mode: 'readwrite' };
    const granted = await this.handle.queryPermission(options) === 'granted'
      || await this.handle.requestPermission(options) === 'granted';
    if (!granted) this.handle = null;
    return granted;
  }

  async writeAll(samples) {
    if (!await this.ensureAccess()) return this.downloadAll(samples);

    for (const { name, blob } of samples) {
      const file = await this.handle.getFileHandle(name, { create: true });
      const stream = await file.createWritable();
      await stream.write(blob);
      await stream.close();
    }
    return { written: samples.length, mode: 'folder' };
  }

  downloadAll(samples) {
    for (const { name, blob } of samples) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = name;
      link.click();
      URL.revokeObjectURL(url);
    }
    return { written: samples.length, mode: 'download' };
  }
}
