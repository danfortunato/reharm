/** One-off: verify the 1x-display curvature dealias path completes. */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';
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
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => /output:/.test(document.getElementById('metrics')?.textContent ?? ''), { timeout: 300_000 });
  await page.select('#plotos', '1');
  await page.select('#color', 'mean');
  const t0 = Date.now();
  await page.waitForFunction(() => !!globalThis.__curv, { timeout: 300_000 });
  const n = await page.evaluate(() => globalThis.__curv.length);
  console.log(`1x dealiased curvature completed in ${((Date.now() - t0) / 1000).toFixed(1)} s: field ${n} points (display grid)`);
  await page.screenshot({ path: process.argv[2] ?? '/tmp/check-1x.png' });
} finally { await browser.close(); server.close(); }
