/** node scripts/shot.mjs out.png geometry chart lmax N [subdiv] [filter] [color]
 *  e.g. node scripts/shot.mjs bunny.png bunny area 127 127 limit hard mean */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';
const [out, geometry, chart, lmax, N, subdiv = 'limit', filter = 'hard', color = 'solid'] = process.argv.slice(2);
const DIST = new URL('../dist/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  try { const p = req.url === '/' ? '/index.html' : req.url.split('?')[0]; const d = await readFile(join(DIST, p));
    res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' }); res.end(d); }
  catch { res.writeHead(404); res.end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 1000 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => /filter:/.test(document.getElementById('metrics')?.textContent ?? ''), { timeout: 120_000 });
  await page.select('#subdiv', subdiv);
  await page.select('#lmax', lmax);
  await page.select('#chart', chart);
  await page.select('#filter', filter);
  await page.select('#color', color);
  if (process.env.WIRE) await page.select('#wire', process.env.WIRE);   // off | in | out | both
  if (process.env.SAMPLING) await page.select('#sampling', process.env.SAMPLING);   // uniform | adaptive
  if (process.env.CLOUD) await page.select('#inputform', 'cloud');   // ingest the model as a point cloud
  if (process.env.PLOTOS) await page.select('#plotos', process.env.PLOTOS);   // display oversampling (1 = off)
  if (process.env.POINTS) await page.evaluate(() => { const c = document.getElementById('showpoints'); c.checked = true; c.dispatchEvent(new Event('change')); });
  if (process.env.CHARTSMOOTH) await page.evaluate((v) => { document.getElementById('chartsmooth').value = v; }, process.env.CHARTSMOOTH);
  await page.select('#geometry', geometry);
  // synthetic shapes have no chart block in the metrics — do not wait for one.
  // 'output:' resolves asynchronously (worker); wait for it to settle so the
  // change-watch below cannot fire on its checking… -> counts transition
  await page.waitForFunction((g) => { const t = document.getElementById('metrics')?.textContent ?? ''; return (g.startsWith('synthetic:') || t.includes('map:')) && t.includes('filter:') && t.includes('transform:') && t.includes('output:') && !t.includes('checking…'); }, { timeout: 600_000 }, geometry);
  if (filter === 'hard' || filter === 'trapeziform') {
    // the metrics no longer echo N; wait for the refit by watching them change
    const before = await page.$eval('#metrics', (el) => el.textContent);
    await page.evaluate((n) => { const el = document.getElementById('N'); el.value = n; el.dispatchEvent(new Event('input', { bubbles: true })); }, N);
    await page.waitForFunction((b) => { const t = document.getElementById('metrics')?.textContent ?? ''; return t !== b && !t.includes('checking…'); }, { timeout: 60_000 }, before);
  }
  if (process.env.MORPH) await page.click('#morph');   // toggle to the sphere view
  if (process.env.POLE) await page.click('#polemode'); // arm pole editing (marker appears)
  await new Promise((r) => setTimeout(r, 600));
  console.log(await page.$eval('#metrics', (el) => el.textContent));
  await page.screenshot({ path: out });
} finally { await browser.close(); server.close(); }
