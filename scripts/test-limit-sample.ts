import { readFileSync } from 'node:fs';
import { parseSHM } from '../src/mesh/loaders.ts';
import { sphericalChart } from '../src/chart/policy.ts';
import { LoopLimitSampler } from '../src/fit/limitSample.ts';
const m = parseSHM(readFileSync(new URL('../public/presets/spot.mesh', import.meta.url)).buffer.slice(0));
const { S } = await sphericalChart(m, 'conformal');
let t = performance.now();
const L = new LoopLimitSampler(m, S);
console.log(`sampler build: ${(performance.now() - t).toFixed(0)} ms (control mesh ${L.controlMesh.nv} v)`);
const nlat = 256, nphi = 512, G = new Float64Array(3 * nlat * nphi);
for (let i = 0; i < nlat; i++) for (let j = 0; j < nphi; j++) { const th = (Math.PI * (i + 0.5)) / nlat, ph = (2 * Math.PI * j) / nphi, p = 3 * (i * nphi + j); G[p] = Math.sin(th) * Math.cos(ph); G[p + 1] = Math.sin(th) * Math.sin(ph); G[p + 2] = Math.cos(th); }
t = performance.now();
const r = L.sample(G);
console.log(`sample 256x512: ${(performance.now() - t).toFixed(0)} ms, fallbacks ${r.fallbacks}, max Newton residual ${r.maxResidual.toExponential(2)}`);
