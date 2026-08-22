/** Headless end-to-end check of "Export to turing-surface": both apps built
 *  and served on two local ports (two origins, so the postMessage handshake is
 *  the real cross-origin one), reharm's default fit exported, and the imported
 *  surface verified installed in the turing-surface tab.
 *
 *    npm run build                      (here)
 *    npm run build                      (in ../turing-surface)
 *    node scripts/test-turing-e2e.mjs
 */
import { createServer } from 'node:http';
import { readFile, access } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';

const REHARM_DIST = new URL('../dist/', import.meta.url).pathname;
const TS_DIST = new URL('../../turing-surface/dist/', import.meta.url).pathname;
for (const d of [REHARM_DIST, TS_DIST]) {
  await access(join(d, 'index.html')).catch(() => {
    throw new Error(`${d} has no build — run npm run build there first`);
  });
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const serve = async (root) => {
  const server = createServer(async (req, res) => {
    // The import URL carries ?import=reharm, so strip the query BEFORE the
    // index fallback or '/?import=reharm' reads as a 404.
    try { let p = req.url.split('?')[0]; if (p === '/') p = '/index.html'; const d = await readFile(join(root, p));
      res.writeHead(200, { 'content-type': MIME[extname(p)] ?? 'application/octet-stream' }); res.end(d); }
    catch { res.writeHead(404); res.end(); }
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  return server;
};
const reharmSrv = await serve(REHARM_DIST);
const tsSrv = await serve(TS_DIST);
const tsUrl = `http://127.0.0.1:${tsSrv.address().port}/`;

const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('  [reharm pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${reharmSrv.address().port}/index.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => { const t = document.getElementById('metrics')?.textContent ?? ''; return /filter:/.test(t) && /output:/.test(t) && !t.includes('checking…'); }, { timeout: 300_000 });
  console.log('reharm: default fit ready');

  await page.evaluate((url) => localStorage.setItem('reharm-turing-surface-url', url), tsUrl);
  const popupPromise = new Promise((r) => page.once('popup', r));
  await page.click('#exportts');
  const popup = await popupPromise;
  popup.on('pageerror', (e) => console.log('  [turing-surface pageerror]', e.message));
  console.log(`turing-surface tab opened at ${popup.url()}`);

  // The handshake waits for the tab's boot (WebGPU compile — slow on
  // SwiftShader), then the surface must land as the selected 'imported' entry.
  await popup.waitForFunction(() => document.getElementById('geometry')?.value === 'imported', { timeout: 300_000 });
  console.log('turing-surface: imported entry selected; waiting for the rebuild');
  // The dropdown flips at install; the note follows only once the rebuild at
  // the adopted lmax finishes, so wait for it rather than asserting.
  await popup.waitForFunction(() => (document.getElementById('geomnote')?.textContent ?? '').includes('Handed over by reharm'), { timeout: 300_000 });
  const note = await popup.$eval('#geomnote', (el) => el.textContent);
  const err = await popup.$eval('#err', (el) => el.textContent);
  const cmd = await popup.$eval('#cmd', (el) => el.textContent);
  if (err) throw new Error(`turing-surface reported: ${err}`);
  if (!note.includes('Handed over by reharm')) throw new Error(`geometry note is not the import's: '${note}'`);
  if (!cmd.includes('imported surface')) throw new Error(`command line not guarded: '${cmd}'`);

  // The normalized surface has rms radius 1, so the note's radius range must
  // bracket O(1) — proof a full Geometry was built from the coefficients.
  const m = note.match(/Radius (\d+\.\d+)–(\d+\.\d+)/);
  if (!m) throw new Error(`no radius range in the note: '${note}'`);
  const [lo, hi] = [Number(m[1]), Number(m[2])];
  if (!(lo > 0.05 && lo < 1 && hi > 1 && hi < 20)) throw new Error(`radius range ${lo}–${hi} is not a unit-rms surface`);

  // And the exporter must have seen the ack: no failure in reharm's status.
  const rstatus = await page.$eval('#status', (el) => el.textContent);
  if (/failed/.test(rstatus)) throw new Error(`reharm status shows a failure: '${rstatus}'`);

  console.log(`import ok: '${note.trim()}' — ${cmd}`);
} finally { await browser.close(); reharmSrv.close(); tsSrv.close(); }
