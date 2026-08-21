import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import puppeteer from 'puppeteer-core';
const DIST = '/Users/dfortunato/Research/Melia/reharm/dist/';
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
  const settled = () => page.waitForFunction(() => { const t = document.getElementById('metrics')?.textContent ?? ''; return /filter:/.test(t) && /output:/.test(t) && !t.includes('checking…'); }, { timeout: 300_000 });
  await settled();
  const stats = () => page.evaluate(() => {
    const h = globalThis.__curv;
    if (!h) return null;
    // display grid of the sphere at lmax 127: auto oversampling gives nlat >= 512
    const n = h.length;
    let s2 = 0, mx = 0, cnt = 0;
    for (let i = 0; i < n; i++) { const e = Math.abs(h[i] - 1); if (!Number.isFinite(e)) continue; s2 += e * e; if (e > mx) mx = e; cnt++; }
    return { rms: Math.sqrt(s2 / cnt), max: mx, n: cnt };
  });
  const run = async (noRefine) => {
    await page.evaluate((v) => { globalThis.__noRefine = v; globalThis.__curv = null; }, noRefine);
    // reselect to force a full refit; __curv (nulled above) is set only after
    // the new fit's curvature synthesis — the metrics text may stay byte-identical
    await page.select('#geometry', 'synthetic:sphere');
    await page.waitForFunction(() => !!globalThis.__curv, { timeout: 300_000 });
    console.log(`  sphere refit done (noRefine=${noRefine})`);
    return stats();
  };
  await page.select('#lmax', '127');
  await page.select('#filter', 'none');   // the Gaussian filter shrinks l=1 by ~5.8e-4: a 0.06 % H bias that is not noise
  await settled();
  await page.select('#color', 'mean');   // triggers curvature synthesis
  await page.waitForFunction(() => !!globalThis.__curv, { timeout: 120_000 });
  const unref = await run(true);
  const ref = await run(false);
  console.log(`sphere, lmax 127, mean curvature |H-1| over the display grid (${ref.n} pts):`);
  console.log(`  WITHOUT refinement: rms ${(100 * unref.rms).toPrecision(3)} % · max ${(100 * unref.max).toPrecision(3)} %`);
  console.log(`  WITH refinement:    rms ${(100 * ref.rms).toPrecision(3)} % · max ${(100 * ref.max).toPrecision(3)} %`);
  console.log(`  improvement: rms ${(unref.rms / ref.rms).toFixed(0)}x · max ${(unref.max / ref.max).toFixed(0)}x`);

  // spot: no analytic truth, so measure the NOISE the refinement removes —
  // the refined field (relative-accurate by construction) is the reference
  const runSpot = async (noRefine) => {
    await page.evaluate((v) => { globalThis.__noRefine = v; globalThis.__curv = null; }, noRefine);
    await page.select('#geometry', 'spot');
    await page.waitForFunction(() => !!globalThis.__curv, { timeout: 300_000 });
    console.log(`  spot refit done (noRefine=${noRefine})`);
  };
  await runSpot(false);
  await page.evaluate(() => { globalThis.__curvRef = globalThis.__curv; });
  await runSpot(true);
  const spot = await page.evaluate(() => {
    const a = globalThis.__curv, r = globalThis.__curvRef;
    let s2 = 0, mx = 0, cnt = 0, habs = [];
    for (let i = 0; i < a.length; i++) {
      if (!Number.isFinite(a[i]) || !Number.isFinite(r[i])) continue;
      const e = Math.abs(a[i] - r[i]);
      s2 += e * e; if (e > mx) mx = e; cnt++;
      habs.push(Math.abs(r[i]));
    }
    habs.sort((x, y) => x - y);
    return { rms: Math.sqrt(s2 / cnt), max: mx, med: habs[cnt >> 1], p99: habs[Math.floor(0.99 * cnt)], n: cnt };
  });
  console.log(`spot, lmax 127, unfiltered: noise removed by refinement (|H_unref - H_ref|, ${spot.n} pts):`);
  console.log(`  rms ${spot.rms.toPrecision(3)} · max ${spot.max.toPrecision(3)}  (median |H| ${spot.med.toPrecision(3)}, p99 |H| ${spot.p99.toPrecision(3)})`);
  console.log(`  i.e. rms noise = ${(100 * spot.rms / spot.med).toPrecision(3)} % of the median |H|`);
} finally { await browser.close(); server.close(); }
