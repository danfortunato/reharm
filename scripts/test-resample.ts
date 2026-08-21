// node scripts/test-resample.ts -- resampling correctness on a synthetic mesh (exact chart) and timing on a preset
import { readFileSync } from 'node:fs';
import { parseSHM } from '../src/mesh/loaders.ts';
import { syntheticMesh, syntheticShapes } from '../src/mesh/presets.ts';
import { SphereLocator } from '../src/fit/resample.ts';
import { sphericalChart } from '../src/chart/policy.ts';

// 1) synthetic: interpolating the positions at the chart points of a FINER lat-long set must approximate the shape to O(h^2)
const { mesh, chart } = syntheticMesh(syntheticShapes.bumpy.fn, 96, 192);
const loc = new SphereLocator(mesh, chart);
const nq = 20000, Q = new Float64Array(3 * nq), exact = new Float64Array(3 * nq);
let s = 12345; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
for (let q = 0; q < nq; q++) {
  const th = Math.acos(2 * rnd() - 1), ph = 2 * Math.PI * rnd();
  Q.set([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)], 3 * q);
  exact.set(syntheticShapes.bumpy.fn(th, ph), 3 * q);
}
let t = performance.now();
const { values, fallbacks } = loc.interpolate(mesh.positions, 3, Q);
let mx = 0; for (let q = 0; q < nq; q++) mx = Math.max(mx, Math.hypot(values[3 * q] - exact[3 * q], values[3 * q + 1] - exact[3 * q + 1], values[3 * q + 2] - exact[3 * q + 2]));
console.log(`synthetic 96x192 -> ${nq} random queries: max PL interpolation error ${mx.toExponential(2)} (O(h^2) ~ ${(Math.PI / 97) ** 2 * 0.3 | 0}e-3 expected), fallbacks ${fallbacks}, ${(performance.now() - t).toFixed(0)} ms`);
if (mx > 5e-3 || fallbacks > nq * 0.01) throw new Error('resample check failed');

// 2) preset: chart + resample onto a 256 x 512 grid, timing
for (const name of ['spot', 'brain']) {
  const m = parseSHM(readFileSync(new URL(`../public/presets/${name}.mesh`, import.meta.url)).buffer.slice(0));
  t = performance.now();
  const { S, info } = await sphericalChart(m, 'auto');
  const tc = performance.now() - t;
  t = performance.now();
  const L = new SphereLocator(m, S);
  const nlat = 256, nphi = 512, G = new Float64Array(3 * nlat * nphi);
  for (let i = 0; i < nlat; i++) for (let j = 0; j < nphi; j++) {
    const th = (Math.PI * (i + 0.5)) / nlat, ph = (2 * Math.PI * j) / nphi, p = 3 * (i * nphi + j);
    G[p] = Math.sin(th) * Math.cos(ph); G[p + 1] = Math.sin(th) * Math.sin(ph); G[p + 2] = Math.cos(th);
  }
  const r = L.interpolate(m.positions, 3, G);
  console.log(`${name}: chart ${tc.toFixed(0)} ms (${info.type}, folds ${info.foldsRepaired}->${info.foldsLeft}, lambda max/min ${info.lambdaRatio.toExponential(2)}), resample 256x512 ${(performance.now() - t).toFixed(0)} ms, fallbacks ${r.fallbacks}`);
}
