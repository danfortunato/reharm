// screenshot the page mid-chart (bunny, area) to check the progress overlay
import { createServer } from 'node:http'; import { readFile } from 'node:fs/promises'; import { extname, join } from 'node:path'; import puppeteer from 'puppeteer-core';
const out = process.argv[2]; const DIST = new URL('../dist/', import.meta.url).pathname;
const server = createServer(async (req, res) => { try { const p = req.url === '/' ? '/index.html' : req.url.split('?')[0]; const d = await readFile(join(DIST, p)); res.writeHead(200, { 'content-type': { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' }[extname(p)] ?? 'application/octet-stream' }); res.end(d); } catch { res.writeHead(404); res.end(); } });
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--headless=new', '--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] });
try { const page = await browser.newPage(); await page.setViewport({ width: 1200, height: 1000 });
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => /filter:/.test(document.getElementById('metrics')?.textContent ?? ''), { timeout: 120_000 });
  await page.select('#chart', 'area'); await page.select('#geometry', 'bunny');
  await page.waitForFunction(() => /SDEM step (4|8|12)\//.test(document.getElementById('busy-text')?.textContent ?? ''), { timeout: 120_000 });
  await new Promise((r) => setTimeout(r, 300));
  console.log('busy:', await page.$eval('#busy-text', (el) => el.textContent), '| status:', await page.$eval('#status', (el) => el.textContent));
  await page.screenshot({ path: out });
} finally { await browser.close(); server.close(); }
