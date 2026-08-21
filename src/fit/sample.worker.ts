/** Web Worker: surface sampling off the main thread. Holds the charted mesh
 *  and its sampler — the exact Loop limit evaluator (mode 'limit'; its
 *  construction and Newton evaluation block for seconds at high lmax) or the
 *  PL locator (mode 'pl') — and serves per-fit Sample requests. A request may
 *  ask for Zhou adaptive sampling: the grid mesh is warped toward the chart's
 *  vertex density (src/fit/zhouWarp.ts, 30 density-ratio Laplacian
 *  iterations) and the surface is sampled at the warped directions, so the
 *  transformed signal is f∘w. */
import { LoopLimitSampler } from './limitSample.ts';
import { SphereLocator } from './resample.ts';
import { adaptiveWarp } from './zhouWarp.ts';
import { makeMesh, type Mesh } from '../mesh/types.ts';

export interface SampleInit {
  init: true;
  positions: Float32Array;
  faces: Uint32Array;
  chart: Float64Array;
  mode: 'limit' | 'pl';
}
export interface SampleRequest {
  id: number;
  /** unit grid directions to sample at, interleaved xyz */
  queries: Float64Array;
  /** Zhou adaptive sampling: warp the closed grid mesh (queries ++ capDirs,
   *  triangulated by gridFaces) first, and sample at the warped directions. */
  warp?: { capDirs: Float64Array; gridFaces: Uint32Array; iters: number };
}
export interface SampleResult { id: number; values?: Float64Array; fallbacks?: number; maxResidual?: number; error?: string }

let mesh: Mesh | null = null;
let chart: Float64Array | null = null;
let limit: LoopLimitSampler | null = null;
let locator: SphereLocator | null = null;      // PL sampler, and the warp's D0 interpolant
let initError: string | null = null;

self.onmessage = (e: MessageEvent<SampleInit | SampleRequest>) => {
  if ('init' in e.data) {
    try {
      mesh = makeMesh(e.data.positions, e.data.faces);
      chart = e.data.chart;
      limit = e.data.mode === 'limit' ? new LoopLimitSampler(mesh, chart) : null;
      locator = null;
      initError = null;
    } catch (err) {
      mesh = null; chart = null; limit = null; locator = null;
      initError = (err as Error).message ?? String(err);
    }
    return;
  }
  const { id, queries, warp } = e.data;
  if (!mesh || !chart) { self.postMessage({ id, error: initError ?? 'sampler not initialized' }); return; }
  try {
    let q = queries;
    if (warp) {
      const dirs = new Float64Array(queries.length + warp.capDirs.length);
      dirs.set(queries); dirs.set(warp.capDirs, queries.length);
      locator ??= new SphereLocator(mesh, chart);
      q = adaptiveWarp(mesh, chart, locator, dirs, warp.gridFaces, warp.iters).subarray(0, queries.length) as Float64Array;
    }
    if (limit) {
      const { values, fallbacks, maxResidual } = limit.sample(q);
      (self as unknown as Worker).postMessage({ id, values, fallbacks, maxResidual }, [values.buffer]);
    } else {
      locator ??= new SphereLocator(mesh, chart);
      const { values, fallbacks } = locator.interpolate(mesh.positions, 3, q);
      (self as unknown as Worker).postMessage({ id, values, fallbacks, maxResidual: 0 }, [values.buffer]);
    }
  } catch (err) {
    self.postMessage({ id, error: (err as Error).message ?? String(err) });
  }
};
