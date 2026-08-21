/**
 * Embeddedness metrics of a triangle mesh: exact self-intersection counting
 * and the collapsed-face count. Port of the MATLAB module's fixed
 * check_self_intersections.m.
 *
 * A closed mesh is embedded (non-self-intersecting) iff no two triangles
 * that do not share a vertex intersect. selfIntersections tests exactly that:
 *   broad phase  : uniform-grid binning of the triangles' bounding boxes;
 *   narrow phase : Moller's (1997) triangle-triangle overlap test.
 * Signed plane distances use UNIT normals, so the tolerance is a true
 * distance -- unnormalized normals scale with triangle area, which made
 * sliver triangles (e.g. around a tensor grid's poles) look like grazing
 * contacts and fed the interval test garbage. Zero-area triangles have no
 * plane and are skipped; count those separately (collapsed faces). Two
 * measure-zero cases are deliberately NOT counted: exactly coplanar
 * overlapping triangles, and a vertex lying exactly on the other triangle's
 * plane (treated as a grazing contact).
 */

/** Areas of all faces. */
export function faceAreas(pos: ArrayLike<number>, faces: ArrayLike<number>): Float64Array {
  const nf = (faces.length / 3) | 0;
  const A = new Float64Array(nf);
  for (let t = 0; t < nf; t++) {
    const a = 3 * faces[3 * t], b = 3 * faces[3 * t + 1], c = 3 * faces[3 * t + 2];
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    A[t] = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
  }
  return A;
}

/** Faces whose area fell below `frac` of the same face's reference area (the
 *  MATLAB benchmark's 'collapsed' count). A zero-area reference face has no
 *  meaningful ratio and is skipped. */
export function collapsedCount(areas: Float64Array, ref: Float64Array, frac = 0.1): number {
  let n = 0;
  for (let t = 0; t < areas.length; t++) if (ref[t] > 0 && areas[t] < frac * ref[t]) n++;
  return n;
}

/** Count intersecting non-adjacent triangle pairs (0 = embedded).
 *  `relTol` is the plane-distance tolerance relative to the bbox diagonal. */
export function selfIntersections(pos: ArrayLike<number>, faces: ArrayLike<number>, relTol = 1e-10): number {
  const nf = (faces.length / 3) | 0;
  if (nf < 2) return 0;

  // triangle and global bounding boxes
  const lo = new Float64Array(3 * nf), hi = new Float64Array(3 * nf);
  const gmin = [Infinity, Infinity, Infinity], gmax = [-Infinity, -Infinity, -Infinity];
  for (let t = 0; t < nf; t++) {
    const a = 3 * faces[3 * t], b = 3 * faces[3 * t + 1], c = 3 * faces[3 * t + 2];
    for (let k = 0; k < 3; k++) {
      const l = Math.min(pos[a + k], pos[b + k], pos[c + k]);
      const h = Math.max(pos[a + k], pos[b + k], pos[c + k]);
      lo[3 * t + k] = l; hi[3 * t + k] = h;
      if (l < gmin[k]) gmin[k] = l;
      if (h > gmax[k]) gmax[k] = h;
    }
  }
  const tol = relTol * Math.hypot(gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]);

  // broad-phase cell size: 2x the median edge length (a stride sample -- the
  // median only sizes the cells), raised so no triangle spans more than ~9
  // cells per axis
  const step = Math.max(1, Math.floor(nf / 30000));
  const sample: number[] = [];
  for (let t = 0; t < nf; t += step) {
    const a = 3 * faces[3 * t], b = 3 * faces[3 * t + 1], c = 3 * faces[3 * t + 2];
    sample.push(
      Math.hypot(pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]),
      Math.hypot(pos[c] - pos[b], pos[c + 1] - pos[b + 1], pos[c + 2] - pos[b + 2]),
      Math.hypot(pos[a] - pos[c], pos[a + 1] - pos[c + 1], pos[a + 2] - pos[c + 2]),
    );
  }
  sample.sort((x, y) => x - y);
  const med = sample[sample.length >> 1];
  let maxExt = 0;
  for (let k = 0; k < 3 * nf; k++) if (hi[k] - lo[k] > maxExt) maxExt = hi[k] - lo[k];
  const h = Math.max(2 * med, maxExt / 8) || 1;

  // integer cell ranges of each triangle's box
  const nx = Math.floor((gmax[0] - gmin[0]) / h) + 1;
  const ny = Math.floor((gmax[1] - gmin[1]) / h) + 1;
  const nz = Math.floor((gmax[2] - gmin[2]) / h) + 1;
  const ncmax = [nx, ny, nz];
  const i0 = new Int32Array(3 * nf), i1 = new Int32Array(3 * nf);
  for (let k = 0; k < 3 * nf; k++) {
    const ax = k % 3;
    i0[k] = Math.max(0, Math.floor((lo[k] - gmin[ax]) / h));
    i1[k] = Math.min(ncmax[ax] - 1, Math.floor((hi[k] - gmin[ax]) / h));
  }

  // bucket the (cell, triangle) incidences: compact the occupied cell ids,
  // then CSR (two passes keep the memory flat for meshes of millions of faces)
  const cellOf = new Map<number, number>();
  const cellIds: number[] = [], counts: number[] = [];
  const eachCell = (t: number, fn: (id: number) => void) => {
    for (let iz = i0[3 * t + 2]; iz <= i1[3 * t + 2]; iz++)
      for (let iy = i0[3 * t + 1]; iy <= i1[3 * t + 1]; iy++)
        for (let ix = i0[3 * t]; ix <= i1[3 * t]; ix++) fn(ix + nx * (iy + ny * iz));
  };
  for (let t = 0; t < nf; t++)
    eachCell(t, (id) => {
      let c = cellOf.get(id);
      if (c === undefined) { c = counts.length; cellOf.set(id, c); cellIds.push(id); counts.push(0); }
      counts[c]++;
    });
  const ncell = counts.length;
  const starts = new Uint32Array(ncell + 1);
  for (let c = 0; c < ncell; c++) starts[c + 1] = starts[c] + counts[c];
  const entries = new Uint32Array(starts[ncell]);
  const cursor = starts.slice(0, ncell);
  for (let t = 0; t < nf; t++) eachCell(t, (id) => { entries[cursor[cellOf.get(id)!]++] = t; });

  // candidate pairs = triangles sharing a cell, each pair examined only in
  // its canonical cell (the first cell both boxes cover -- always shared, so
  // no pair list to deduplicate), then vertex-sharing and box-overlap culls,
  // then the Moller test
  let n = 0;
  for (let c = 0; c < ncell; c++) {
    const s = starts[c], e = starts[c + 1];
    if (e - s < 2) continue;
    const id = cellIds[c];
    const cz = Math.floor(id / (nx * ny)), rem = id - cz * nx * ny;
    const cy = Math.floor(rem / nx), cx = rem - cy * nx;
    for (let i = s; i < e; i++)
      for (let j = i + 1; j < e; j++) {
        const a = entries[i], b = entries[j];
        if (Math.max(i0[3 * a], i0[3 * b]) !== cx || Math.max(i0[3 * a + 1], i0[3 * b + 1]) !== cy
          || Math.max(i0[3 * a + 2], i0[3 * b + 2]) !== cz) continue;
        const a0 = faces[3 * a], a1 = faces[3 * a + 1], a2 = faces[3 * a + 2];
        const b0 = faces[3 * b], b1 = faces[3 * b + 1], b2 = faces[3 * b + 2];
        if (a0 === b0 || a0 === b1 || a0 === b2 || a1 === b0 || a1 === b1 || a1 === b2
          || a2 === b0 || a2 === b1 || a2 === b2) continue;   // adjacent triangles always touch
        if (lo[3 * a] > hi[3 * b] + tol || lo[3 * b] > hi[3 * a] + tol
          || lo[3 * a + 1] > hi[3 * b + 1] + tol || lo[3 * b + 1] > hi[3 * a + 1] + tol
          || lo[3 * a + 2] > hi[3 * b + 2] + tol || lo[3 * b + 2] > hi[3 * a + 2] + tol) continue;
        if (triTriIntersects(pos, faces, a, b, tol)) n++;
      }
  }
  return n;
}

// scratch for the narrow phase (single-threaded; avoids per-call allocation)
const VA = new Float64Array(9), VB = new Float64Array(9);
const DA = new Float64Array(3), DB = new Float64Array(3);
const PA = new Float64Array(3), PB = new Float64Array(3);
const TA = new Float64Array(2), TB = new Float64Array(2);

/** Moller (1997) test of triangles `ta`, `tb`: the intervals of the two
 *  triangles along the line of intersection of their planes overlap iff the
 *  triangles intersect. Exposed for the brute-force parity check in the tests. */
export function triTriIntersects(pos: ArrayLike<number>, faces: ArrayLike<number>, ta: number, tb: number, tol: number): boolean {
  for (let v = 0; v < 3; v++)
    for (let k = 0; k < 3; k++) {
      VA[3 * v + k] = pos[3 * faces[3 * ta + v] + k];
      VB[3 * v + k] = pos[3 * faces[3 * tb + v] + k];
    }
  // unit normals; zero-area slivers have no plane: skip (see module comment)
  if (!unitNormal(VA, N1) || !unitNormal(VB, N2)) return false;
  const d1 = -(N1[0] * VA[0] + N1[1] * VA[1] + N1[2] * VA[2]);
  const d2 = -(N2[0] * VB[0] + N2[1] * VB[1] + N2[2] * VB[2]);
  for (let v = 0; v < 3; v++) {
    DA[v] = N2[0] * VA[3 * v] + N2[1] * VA[3 * v + 1] + N2[2] * VA[3 * v + 2] + d2;
    DB[v] = N1[0] * VB[3 * v] + N1[1] * VB[3 * v + 1] + N1[2] * VB[3 * v + 2] + d1;
  }
  // coplanar overlap is not counted; grazing contacts (a vertex exactly on
  // the other plane) are pushed to the positive side
  if ((Math.abs(DA[0]) < tol && Math.abs(DA[1]) < tol && Math.abs(DA[2]) < tol)
    || (Math.abs(DB[0]) < tol && Math.abs(DB[1]) < tol && Math.abs(DB[2]) < tol)) return false;
  for (let v = 0; v < 3; v++) {
    if (Math.abs(DA[v]) < tol) DA[v] = tol;
    if (Math.abs(DB[v]) < tol) DB[v] = tol;
  }
  if ((DA[0] > 0 && DA[1] > 0 && DA[2] > 0) || (DA[0] < 0 && DA[1] < 0 && DA[2] < 0)
    || (DB[0] > 0 && DB[1] > 0 && DB[2] > 0) || (DB[0] < 0 && DB[1] < 0 && DB[2] < 0)) return false;

  // project the vertices onto the dominant axis of the intersection line
  const Dx = N1[1] * N2[2] - N1[2] * N2[1], Dy = N1[2] * N2[0] - N1[0] * N2[2], Dz = N1[0] * N2[1] - N1[1] * N2[0];
  const ax = Math.abs(Dx) >= Math.abs(Dy) ? (Math.abs(Dx) >= Math.abs(Dz) ? 0 : 2) : Math.abs(Dy) >= Math.abs(Dz) ? 1 : 2;
  for (let v = 0; v < 3; v++) { PA[v] = VA[3 * v + ax]; PB[v] = VB[3 * v + ax]; }
  interval(PA, DA, TA);
  interval(PB, DB, TB);
  return Math.max(TA[0], TA[1]) >= Math.min(TB[0], TB[1]) - tol
    && Math.max(TB[0], TB[1]) >= Math.min(TA[0], TA[1]) - tol;
}

const N1 = new Float64Array(3), N2 = new Float64Array(3);
/** Unit normal of the triangle in `V` into `out`; false for a zero-area sliver. */
function unitNormal(V: Float64Array, out: Float64Array): boolean {
  const ux = V[3] - V[0], uy = V[4] - V[1], uz = V[5] - V[2];
  const vx = V[6] - V[0], vy = V[7] - V[1], vz = V[8] - V[2];
  const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
  const n = Math.hypot(cx, cy, cz);
  const scale = Math.max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz,
    (vx - ux) ** 2 + (vy - uy) ** 2 + (vz - uz) ** 2);
  if (n < 1e-12 * scale) return false;
  out[0] = cx / n; out[1] = cy / n; out[2] = cz / n;
  return true;
}

/** Parametric interval of a triangle along the intersection line: the edges
 *  from the "lone" vertex (alone on its side of the other plane) to the other
 *  two vertices cross the plane at out[0], out[1]. */
function interval(p: Float64Array, d: Float64Array, out: Float64Array): void {
  let lone = 0;
  if (d[0] > 0 === d[2] > 0) lone = 1;
  if (d[0] > 0 === d[1] > 0) lone = 2;
  const o1 = (lone + 1) % 3, o2 = (lone + 2) % 3;
  out[0] = p[o1] + (p[lone] - p[o1]) * d[o1] / (d[o1] - d[lone]);
  out[1] = p[o2] + (p[lone] - p[o2]) * d[o2] / (d[o2] - d[lone]);
}
