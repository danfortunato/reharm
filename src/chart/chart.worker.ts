/** Web Worker: computes a spherical chart off the main thread, reporting progress. */
import { pointCloudChart, sphericalChart, type ChartKind } from './policy.ts';
import { makeMesh } from '../mesh/types.ts';

export interface ChartRequest { id: number; positions: Float32Array; faces: Uint32Array; kind: ChartKind; maxLambdaRatio: number; sdemMaxSteps?: number; chartSmoothing?: number; sdemCorrection?: 'repair' | 'lbs' }
/** Live update to the in-flight request's SDEM step cap: applied at the next
 *  step (SDEM yields to the message queue between steps). */
export interface ChartUpdate { id: number; sdemMaxSteps: number }

let liveMaxSteps = 300;

self.onmessage = async (e: MessageEvent<ChartRequest | ChartUpdate>) => {
  if (!('positions' in e.data)) { liveMaxSteps = e.data.sdemMaxSteps; return; }
  const { id, positions, faces, kind, maxLambdaRatio, sdemMaxSteps, chartSmoothing, sdemCorrection } = e.data;
  liveMaxSteps = sdemMaxSteps ?? 300;
  try {
    if (faces.length === 0) {
      // a point cloud: MLS conformal map + spherical Delaunay induces a mesh.
      // The map kind then applies to the INDUCED mesh, mirroring the mesh
      // policy: auto falls back to the area chart when the cloud map crowds,
      // and an explicit area/Tutte request recharts the induced mesh outright.
      const r = pointCloudChart(positions, (progress) => self.postMessage({ id, progress }));
      let S2 = r.S, info2 = r.info;
      if (kind === 'area' || kind === 'balanced' || kind === 'tutte' || (kind === 'auto' && r.info.crowded)) {
        const im = makeMesh(r.positions, r.faces);
        const rechart = await sphericalChart(
          im, kind === 'auto' ? 'area' : kind, maxLambdaRatio,
          (progress) => self.postMessage({ id, progress }),
          () => liveMaxSteps, chartSmoothing, sdemCorrection,
          () => new Promise((res) => setTimeout(res)),
        );
        S2 = rechart.S; info2 = rechart.info;
      }
      (self as unknown as Worker).postMessage({ id, S: S2, info: info2, faces: r.faces, positions: r.positions }, [S2.buffer, r.faces.buffer]);
      return;
    }
    const m = makeMesh(positions, faces);
    const { S, info } = await sphericalChart(
      m, kind, maxLambdaRatio,
      (progress) => self.postMessage({ id, progress }),
      () => liveMaxSteps, chartSmoothing, sdemCorrection,
      () => new Promise((r) => setTimeout(r)),   // pump the queue between SDEM steps
    );
    (self as unknown as Worker).postMessage({ id, S, info }, [S.buffer]);
  } catch (err) {
    self.postMessage({ id, error: (err as Error).message ?? String(err) });
  }
};
