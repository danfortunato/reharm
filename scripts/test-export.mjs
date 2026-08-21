/** Headless check of the .h5 geometry export: build first, then
 *  `node scripts/test-export.mjs <downloadDir>`. Loads the page, waits for the
 *  default fit, clicks export, and verifies a well-formed HDF5 file lands. */
import { createServer } from 'node:http';
import { readFile, readdir, stat } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const dir = process.argv[2] ?? '.';
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
  await page.waitForFunction(() => { const t = document.getElementById('metrics')?.textContent ?? ''; return /filter:/.test(t) && /output:/.test(t) && !t.includes('checking…'); }, { timeout: 300_000 });
  const client = await page.createCDPSession();
  await client.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: dir });
  await page.click('#exporth5');
  let file = null;
  for (let i = 0; i < 120 && !file; i++) {
    await new Promise((r) => setTimeout(r, 500));
    const names = await readdir(dir);
    file = names.find((n) => n.endsWith('.h5')) ?? null;
  }
  if (!file) throw new Error('no .h5 download appeared (status: ' + (await page.$eval('#status', (el) => el.textContent)) + ')');
  const bytes = await readFile(join(dir, file));
  if (!(bytes[0] === 0x89 && bytes[1] === 0x48 && bytes[2] === 0x44 && bytes[3] === 0x46)) {
    throw new Error(`${file} is not HDF5 (magic ${bytes.subarray(0, 4).toString('hex')})`);
  }
  console.log(`export ok: ${file}, ${(await stat(join(dir, file))).size} bytes, HDF5 magic verified`);
} finally { await browser.close(); server.close(); }
