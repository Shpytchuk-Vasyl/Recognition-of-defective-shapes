import { readdirSync, writeFileSync } from 'node:fs';

const DIR = 'img';
const IMAGE_RE = /\.(png|jpe?g|gif|bmp|webp)$/i;

const files = readdirSync(DIR).filter(name => IMAGE_RE.test(name)).sort();
writeFileSync(`${DIR}/index.json`, JSON.stringify({ files }, null, 2) + '\n');

console.log(`${DIR}/index.json: ${files.length} файл(ів)`);
