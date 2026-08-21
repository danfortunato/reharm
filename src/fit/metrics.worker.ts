/** Web Worker: embeddedness metrics of the fitted surface, off the main
 *  thread (the checker visits millions of candidate pairs on fine grids).
 *  The mesh is the analysis grid's tensor mesh with the poles merged into
 *  cap vertices -- the same construction the renderer uses -- built here for
 *  the filtered surface and for the input surface sampled at the same grid
 *  points; the latter supplies the reference areas for the collapsed count. */
import { buildTopology, fillPositions } from '../render/sphereMesh.ts';
import { faceAreas, collapsedCount, selfIntersections } from '../mesh/intersect.ts';

export interface GridMetricsRequest {
  id: number;
  cosTheta: Float64Array;
  phi: Float64Array;
  /** filtered surface on the grid, interleaved xyz */
  coords: Float32Array;
  /** input surface at the same grid points */
  exact: Float32Array;
}
export interface GridMetricsResult { id: number; selfIntersections: number; collapsed: number; nf: number }

self.onmessage = (e: MessageEvent<GridMetricsRequest>) => {
  const { id, cosTheta, phi, coords, exact } = e.data;
  const topo = buildTopology(cosTheta, phi);
  const pos = new Float32Array(topo.numVertices * 3);
  const ref = new Float32Array(topo.numVertices * 3);
  fillPositions(pos, coords, topo, 1);
  fillPositions(ref, exact, topo, 1);
  const result: GridMetricsResult = {
    id,
    selfIntersections: selfIntersections(pos, topo.indices),
    collapsed: collapsedCount(faceAreas(pos, topo.indices), faceAreas(ref, topo.indices)),
    nf: topo.indices.length / 3,
  };
  self.postMessage(result);
};
