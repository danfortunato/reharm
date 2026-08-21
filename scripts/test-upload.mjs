/** Headless check of the model dropdown's upload flow: build first, then
 *  `node scripts/test-upload.mjs`. Drives the hidden fallback input (the OS
 *  dialog of the picker path cannot be scripted), which shares the commit
 *  path: parse -> named dropdown entry -> chart -> fit. */
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { tmpdir } from 'node:os';
import puppeteer from 'puppeteer-core';

const DIST = new URL('../dist/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  try { const p = req.url === '/' ? '/index.html' : req.url.split('?')[0]; const d = await readFile(join(DIST, p));
    res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' }); res.end(d); }
  catch { res.writeHead(404); res.end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));

// a closed genus-0 mesh: the cube, quads fan-triangulated by the OBJ reader
const V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const Q = [[1, 4, 3, 2], [5, 6, 7, 8], [1, 2, 6, 5], [3, 4, 8, 7], [2, 3, 7, 6], [4, 1, 5, 8]];
const objPath = join(tmpdir(), `reharm-test-upload-${process.pid}.obj`);
await writeFile(objPath, [...V.map((v) => `v ${v.join(' ')}`), ...Q.map((q) => `f ${q.join(' ')}`)].join('\n'));

const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html`, { waitUntil: 'load' });
  const settled = "(() => { const t = document.getElementById('metrics')?.textContent ?? ''; return /filter:/.test(t) && /output:/.test(t) && !t.includes('checking…'); })()";
  await page.waitForFunction(settled, { timeout: 300_000 });
  const input = await page.$('#upload-fallback');
  const before = await page.$eval('#metrics', (el) => el.textContent);
  await input.uploadFile(objPath);
  await page.waitForFunction((b) => { const t = document.getElementById('metrics')?.textContent ?? ''; return t !== b && /output:/.test(t) && !t.includes('checking…'); }, { timeout: 300_000 }, before);
  const state = await page.evaluate(() => ({
    value: document.getElementById('geometry').value,
    label: document.getElementById('geometry').selectedOptions[0]?.textContent,
    options: [...document.getElementById('geometry').options].map((o) => o.value),
    status: document.getElementById('status').textContent,
  }));
  if (state.value !== 'uploaded' || !state.label.endsWith('.obj')) throw new Error(`dropdown state wrong: ${JSON.stringify(state)}`);
  if (state.options.indexOf('uploaded') !== state.options.length - 2 || state.options[state.options.length - 1] !== 'custom') {
    throw new Error(`option order wrong: ${state.options.join(', ')}`);
  }
  if (!/vertices/.test(state.status)) throw new Error(`no summary line after upload: '${state.status}'`);
  console.log(`upload ok: dropdown shows '${state.label}' (value uploaded), custom… stays last; status: ${state.status}`);

  // a POINT CLOUD upload: spot's vertices as a face-less PLY — the app must
  // map it (Choi-Ho-Lui) and mesh it by spherical Delaunay, then fit
  const shm = await readFile(new URL('../public/presets/spot.mesh', import.meta.url));
  const nvS = shm.readUInt32LE(4);
  const verts = new Float32Array(shm.buffer.slice(shm.byteOffset + 12, shm.byteOffset + 12 + nvS * 12));
  const plyPath = join(tmpdir(), `reharm-test-cloud-${process.pid}.ply`);
  const lines = ['ply', 'format ascii 1.0', `element vertex ${nvS}`, 'property float x', 'property float y', 'property float z', 'end_header'];
  for (let i = 0; i < nvS; i++) lines.push(`${verts[3 * i]} ${verts[3 * i + 1]} ${verts[3 * i + 2]}`);
  await writeFile(plyPath, lines.join('\n'));
  // pin the map to conformal: under 'auto' the crowded cloud map falls back to
  // the area chart and the map row would not say 'point cloud'
  await page.select('#chart', 'conformal');
  await page.waitForFunction(() => { const t = document.getElementById('metrics')?.textContent ?? ''; return /output:/.test(t) && !t.includes('checking…'); }, { timeout: 120_000 });
  const before2 = await page.$eval('#metrics', (el) => el.textContent);
  await input.uploadFile(plyPath);
  await page.waitForFunction((b) => { const t = document.getElementById('metrics')?.textContent ?? ''; return t !== b && /output:/.test(t) && /point cloud/.test(t) && !t.includes('checking…'); }, { timeout: 300_000 }, before2);
  const metrics = await page.$eval('#metrics', (el) => el.textContent);
  const status2 = await page.$eval('#status', (el) => el.textContent);
  if (!/map:\s+point cloud/.test(metrics)) throw new Error(`metrics lack the point-cloud map row:\n${metrics}`);
  if (!/vertices/.test(status2)) throw new Error(`no summary after cloud upload: '${status2}'`);   // a few near-coincident points may be compacted away
  console.log(`point-cloud upload ok: ${status2}`);
  console.log(metrics.split('\n').slice(0, 4).join('\n'));
} finally { await browser.close(); server.close(); }
