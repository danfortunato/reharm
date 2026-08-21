/**
 * The chart policy of the MATLAB module (spherical_chart.m): conformal +
 * Möbius, unless the conformal factor's max/min exceeds a threshold -- then
 * the shape has features the uniform grid cannot resolve through that chart
 * and an area-equalized chart is called for (Tutte -> SDEM -> smooth -> repair).
 * Every result goes through fold repair.
 */
import { makeMesh, type Mesh } from '../mesh/types.ts';
import { convexHull } from '../mesh/hull.ts';
import { sphericalConformalMap, sphericalTutteMap } from './conformal.ts';
import { pcLaplacian, pcSphericalMap } from './pointcloud.ts';
import { mobiusAreaCorrection } from './mobius.ts';
import { repairSphericalFolds } from './repair.ts';
import { countFolds, lambdaStats, chartRoughness } from './meshops.ts';
import { sdem } from './sdem.ts';
import { smoothChart } from './smooth.ts';

export type ChartKind = 'auto' | 'conformal' | 'tutte' | 'area';

export interface ChartInfo {
  type: 'conformal + Möbius' | 'Tutte' | 'area-equalized (Tutte → SDEM → repair)' | 'point cloud + Möbius (Choi–Ho–Lui)';
  lambdaSpread: number;
  lambdaRatio: number;
  foldsRepaired: number;
  foldsLeft: number;
  crowded: boolean;   // conformal lambda ratio above the threshold
  timeMs: number;
  mobiusEvals?: number;
  sdemSteps?: number;
  sdemSpread?: number;
  sdemStopped?: 'converged' | 'stalled' | 'max steps';
  /** adjacent-face |Δ log λ|: mean and 99th percentile (smaller = smoother chart) */
  roughnessMean: number;
  roughnessP99: number;
}

/** `sdemMaxSteps` may be a callback: it is re-read before every SDEM step, so a
 *  UI can change the cap while the chart is running (pass `yieldStep` so the
 *  worker can pump its message queue between steps). */
export async function sphericalChart(m: Mesh, kind: ChartKind = 'auto', maxLambdaRatio = 1e4, onProgress?: (msg: string) => void, sdemMaxSteps: number | (() => number) = 300, chartSmoothing = 2, sdemCorrection: 'repair' | 'lbs' = 'repair', yieldStep?: () => Promise<void>): Promise<{ S: Float64Array; info: ChartInfo }> {
  const t0 = performance.now();
  let S: Float64Array, type: ChartInfo['type'], mobiusEvals: number | undefined, sdemSteps: number | undefined, sdemSpread: number | undefined;
  let sdemStopped: 'converged' | 'stalled' | 'max steps' | undefined;
  let crowded = false;
  const areaChart = async () => {
    onProgress?.('Tutte map…');
    const t = sphericalTutteMap(m);
    onProgress?.('SDEM…');
    const capOf = typeof sdemMaxSteps === 'function' ? sdemMaxSteps : () => sdemMaxSteps as number;
    const d = await sdem(m, t, { maxIterLive: capOf, correction: sdemCorrection, yieldStep, onStep: (k, e) => { if (k % 4 === 0) onProgress?.(`SDEM step ${k}/${capOf()}, density spread ${e.toFixed(3)}`); } });
    S = d.S; sdemSteps = d.steps; sdemSpread = d.spread; sdemStopped = d.stopped; type = 'area-equalized (Tutte → SDEM → repair)';
    if (chartSmoothing > 0) {
      onProgress?.('smoothing the chart…');
      smoothChart(m, S, chartSmoothing);   // take the repair kinks out; folds (if any) are repaired below
    }
  };
  if (kind === 'tutte') {
    S = sphericalTutteMap(m); type = 'Tutte';
  } else if (kind === 'area') {
    await areaChart();
  } else {
    onProgress?.('conformal map…');
    const conf = sphericalConformalMap(m);
    onProgress?.('Möbius area correction…');
    const mob = mobiusAreaCorrection(m, conf);
    S = mob.S; mobiusEvals = mob.iterations; type = 'conformal + Möbius';
    crowded = lambdaStats(m, S).ratio > maxLambdaRatio;
    if (kind === 'auto' && crowded) await areaChart();
  }
  const foldsBefore = countFolds(m.faces, S!);
  const rep = repairSphericalFolds(m, S!);
  const { spread, ratio } = lambdaStats(m, S!);
  const rough = chartRoughness(m, S!);
  return {
    S: S!,
    info: { type: type!, lambdaSpread: spread, lambdaRatio: ratio, foldsRepaired: foldsBefore, foldsLeft: rep.folds,
            crowded, timeMs: performance.now() - t0, mobiusEvals, sdemSteps, sdemSpread, sdemStopped,
            roughnessMean: rough.mean, roughnessP99: rough.p99 },
  };
}

/**
 * Chart a face-less point cloud: the MLS conformal map of Choi-Ho-Lui 2016
 * (src/chart/pointcloud.ts) followed by the spherical Delaunay triangulation
 * of the mapped points (the convex hull — identical for points on a sphere),
 * which induces a genus-0 closed mesh on the cloud. The result is a normal
 * charted mesh: every downstream stage (smoothing, refits with other maps,
 * metrics, export) applies unchanged. Hull faces are outward-oriented by
 * construction, so the chart is fold-free.
 */
export function pointCloudChart(positions: Float32Array, onProgress?: (msg: string) => void): { S: Float64Array; faces: Uint32Array; positions: Float32Array; dropped: number; info: ChartInfo } {
  const t0 = performance.now();
  const nv = positions.length / 3;
  onProgress?.('point cloud: MLS Laplacian…');
  const pc = pcLaplacian(positions, nv);
  const { S, info: mapInfo } = pcSphericalMap(positions, nv, pc, onProgress);
  onProgress?.('point cloud: spherical Delaunay…');
  let faces = convexHull(S, nv);
  // points whose sliver height fell below the hull's tolerance (near-coincident
  // map positions in heavily crowded regions) are not hull vertices; compact
  // them away so the induced mesh is closed (Euler characteristic 2)
  const used = new Int32Array(nv).fill(-1);
  let kept = 0;
  for (let i = 0; i < faces.length; i++) if (used[faces[i]] < 0) used[faces[i]] = kept++;
  let outPos = positions, outS = S;
  if (kept < nv) {
    outPos = new Float32Array(3 * kept);
    outS = new Float64Array(3 * kept);
    for (let v = 0; v < nv; v++)
      if (used[v] >= 0)
        for (let c = 0; c < 3; c++) { outPos[3 * used[v] + c] = positions[3 * v + c]; outS[3 * used[v] + c] = S[3 * v + c]; }
    faces = Uint32Array.from(faces, (i) => used[i]);
  }
  const m = makeMesh(outPos, faces);
  // the paper's pole balancing is a crude Möbius; the module's actual area
  // correction (a sphere automorphism, so folds and conformality are
  // untouched) tames the crowding by orders of magnitude
  onProgress?.('point cloud: Möbius area correction…');
  const mob = mobiusAreaCorrection(m, outS);
  outS = mob.S;
  const foldsBefore = countFolds(faces, outS);
  const rep = repairSphericalFolds(m, outS);
  const { spread, ratio } = lambdaStats(m, outS);
  const rough = chartRoughness(m, outS);
  return {
    S: outS, faces, positions: outPos, dropped: nv - kept,
    info: {
      type: 'point cloud + Möbius (Choi–Ho–Lui)', lambdaSpread: spread, lambdaRatio: ratio,
      foldsRepaired: foldsBefore, foldsLeft: rep.folds, crowded: ratio > 1e4,
      timeMs: performance.now() - t0, mobiusEvals: mob.iterations, sdemSteps: undefined,
      roughnessMean: rough.mean, roughnessP99: rough.p99,
    },
  };
}
