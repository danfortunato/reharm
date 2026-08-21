/** Headless smoke test: build first, then `node scripts/smoke.mjs out.png`.
 *  Loads the page in Chrome with WebGPU (SwiftShader), waits for the default
 *  preset fit, moves the N slider, and screenshots. */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const out = process.argv[2] ?? 'smoke.png';
const DIST = new URL('../dist/', import.meta.url).pathname;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/index.html' : req.url.split('?')[0];
    const data = await readFile(join(DIST, path));
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(data);
  } catch { console.log('  [404]', req.url); res.writeHead(404); res.end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 1000 });
  page.on('console', (m) => console.log('  [page]', m.text()));
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => /failed|not available/.test(document.getElementById('status')?.textContent ?? '')
    || (() => { const t = document.getElementById('metrics')?.textContent ?? ''; return /filter:/.test(t) && /output:/.test(t) && !t.includes('checking…'); })(), { timeout: 180_000 });
  console.log('status:', await page.$eval('#status', (el) => el.textContent));
  console.log('metrics:\n' + (await page.$eval('#metrics', (el) => el.textContent)));
  // slide N down and wait for the metrics to reflect it (the default filter is
  // Gaussian, which has no N control)
  await page.select('#filter', 'hard');
  // the metrics no longer echo N; wait for the refit by watching them change
  let before = await page.$eval('#metrics', (el) => el.textContent);
  await page.evaluate(() => {
    const n = document.getElementById('N');
    n.value = '8'; n.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForFunction((b) => { const t = document.getElementById('metrics')?.textContent ?? ''; return t !== b && !t.includes('checking…'); }, { timeout: 60_000 }, before);
  await new Promise((r) => setTimeout(r, 500));
  console.log('after N=8:\n' + (await page.$eval('#metrics', (el) => el.textContent)));
  // a mesh preset: chart in the browser (conformal: fast), resample, fit
  await page.select('#chart', 'conformal');
  await page.select('#geometry', 'spot');
  await page.waitForFunction(() => { const t = document.getElementById('metrics')?.textContent ?? ''; return /map: /.test(t) && /filter:/.test(t) && /output:/.test(t) && !t.includes('checking…'); }, { timeout: 180_000 });
  await new Promise((r) => setTimeout(r, 500));
  console.log('spot:\n' + (await page.$eval('#metrics', (el) => el.textContent)));
  before = await page.$eval('#metrics', (el) => el.textContent);
  await page.evaluate(() => { const n = document.getElementById('N'); n.value = '63'; n.dispatchEvent(new Event('input', { bubbles: true })); });
  await page.waitForFunction((b) => { const t = document.getElementById('metrics')?.textContent ?? ''; return t !== b && !t.includes('checking…'); }, { timeout: 60_000 }, before);
  await new Promise((r) => setTimeout(r, 500));
  console.log('spot at N=63:\n' + (await page.$eval('#metrics', (el) => el.textContent)).split('\n').slice(-3).join('\n'));
  await page.screenshot({ path: out });
  console.log('screenshot:', out);
} finally {
  await browser.close();
  server.close();
}
