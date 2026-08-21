// node scripts/test-zhou.ts -- Zhou adaptive-sampling warp: the grid mesh's
// density ratio against the chart's density must tighten substantially.
import { readFileSync } from 'node:fs';
import { parseSHM } from '../src/mesh/loaders.ts';
import { sphericalChart } from '../src/chart/policy.ts';
import { SphereLocator } from '../src/fit/resample.ts';
import { adaptiveWarp, vertexDensity } from '../src/fit/zhouWarp.ts';
import { buildTopology } from '../src/render/sphereMesh.ts';
import { gaussNodesWeights } from '../src/sht/gauss.ts';

const check = (cond: boolean, msg: string) => { if (!cond) throw new Error(msg); };

const mesh = parseSHM(readFileSync(new URL('../public/presets/spot.mesh', import.meta.url)).buffer.slice(0));
const { S } = await sphericalChart(mesh, 'conformal');
const locator = new SphereLocator(mesh, S);

// the 64x128 tensor grid mesh with pole caps, as the fit uses
const { x } = gaussNodesWeights(64);
const phi = new Float64Array(128);
for (let j = 0; j < 128; j++) phi[j] = (2 * Math.PI * j) / 128;
const topo = buildTopology(x, phi);
const dirs = Float64Array.from(topo.sphereRef);

/** spread (p99/p1) of the density ratio D_chart / D_grid over grid vertices */
const ratioSpread = (P: Float64Array): number => {
  const D0 = vertexDensity(S, mesh.faces, mesh.nv);
  const D2 = vertexDensity(P, topo.indices, P.length / 3);
  const D0at = locator.interpolate(D0, 1, P).values;
  const r = Array.from(D2, (d, i) => D0at[i] / d).sort((a, b) => a - b);
  return r[Math.floor(0.99 * (r.length - 1))] / r[Math.floor(0.01 * (r.length - 1))];
};

const before = ratioSpread(dirs);
const t = performance.now();
const P2 = adaptiveWarp(mesh, S, locator, dirs, topo.indices, 30);
const ms = performance.now() - t;
for (let i = 0; i < P2.length; i += 3) {
  const n = Math.hypot(P2[i], P2[i + 1], P2[i + 2]);
  check(Number.isFinite(n) && Math.abs(n - 1) < 1e-12, `warped vertex ${i / 3} is not a unit direction (|p| = ${n})`);
}
const after = ratioSpread(P2);
console.log(`spot conformal, 64x128 grid: density-ratio p99/p1 ${before.toFixed(1)} -> ${after.toFixed(1)} after 30 iterations (${ms.toFixed(0)} ms)`);
check(after < before / 3, `warp did not equalize density (spread ${before.toFixed(1)} -> ${after.toFixed(1)})`);
