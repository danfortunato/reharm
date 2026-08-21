/**
 * Loop subdivision of a closed triangle mesh, with its spherical chart refined
 * alongside. Positions follow Loop's rules (the limit surface is the standard
 * smooth interpretation of a triangle mesh: C2 away from extraordinary
 * vertices). The chart gets the SAME Loop rules (then renormalized to the
 * sphere), so the parameterization sphere -> surface is itself a smooth map
 * rather than a piecewise-linear one: a smooth surface seen through a kinked
 * chart is a kinked function of (theta, phi), and kinks are what rings. Loop
 * smoothing of the chart can in principle fold; callers run
 * repairSphericalFolds on the result (see applyChart).
 */
import { makeMesh, type Mesh } from './types.ts';

export function loopSubdivide(mesh: Mesh, chart: Float32Array | Float64Array): { mesh: Mesh; chart: Float32Array } {
  const { positions: P, faces: F, nv, nf } = mesh;
  // edges: key -> new vertex index; opposite vertices for the odd-vertex rule
  const edgeId = new Map<number, number>();
  const edgeEnds: number[] = [], edgeOpp: number[][] = [];
  const key = (a: number, b: number) => (a < b ? a * nv + b : b * nv + a);
  for (let t = 0; t < nf; t++)
    for (let k = 0; k < 3; k++) {
      const a = F[3 * t + k], b = F[3 * t + ((k + 1) % 3)], c = F[3 * t + ((k + 2) % 3)];
      const kk = key(a, b);
      let e = edgeId.get(kk);
      if (e === undefined) { e = edgeEnds.length / 2; edgeId.set(kk, e); edgeEnds.push(a, b); edgeOpp.push([c]); }
      else edgeOpp[e].push(c);
    }
  const ne = edgeEnds.length / 2;
  const nv2 = nv + ne;
  const pos = new Float32Array(3 * nv2), ch = new Float32Array(3 * nv2);
  // vertex adjacency for the even rule
  const nbr: number[][] = Array.from({ length: nv }, () => []);
  for (let e = 0; e < ne; e++) { nbr[edgeEnds[2 * e]].push(edgeEnds[2 * e + 1]); nbr[edgeEnds[2 * e + 1]].push(edgeEnds[2 * e]); }
  for (let i = 0; i < nv; i++) {
    const n = nbr[i].length;
    const beta = n === 3 ? 3 / 16 : (1 / n) * (5 / 8 - (3 / 8 + 0.25 * Math.cos((2 * Math.PI) / n)) ** 2);
    let x = (1 - n * beta) * P[3 * i], y = (1 - n * beta) * P[3 * i + 1], z = (1 - n * beta) * P[3 * i + 2];
    for (const j of nbr[i]) { x += beta * P[3 * j]; y += beta * P[3 * j + 1]; z += beta * P[3 * j + 2]; }
    pos[3 * i] = x; pos[3 * i + 1] = y; pos[3 * i + 2] = z;
    let cx = (1 - n * beta) * chart[3 * i], cy = (1 - n * beta) * chart[3 * i + 1], cz = (1 - n * beta) * chart[3 * i + 2];
    for (const j of nbr[i]) { cx += beta * chart[3 * j]; cy += beta * chart[3 * j + 1]; cz += beta * chart[3 * j + 2]; }
    const cr = Math.hypot(cx, cy, cz) || 1;
    ch[3 * i] = cx / cr; ch[3 * i + 1] = cy / cr; ch[3 * i + 2] = cz / cr;
  }
  for (let e = 0; e < ne; e++) {
    const a = edgeEnds[2 * e], b = edgeEnds[2 * e + 1], opp = edgeOpp[e], o = nv + e;
    for (let c = 0; c < 3; c++) {
      let v = (3 / 8) * (P[3 * a + c] + P[3 * b + c]);
      if (opp.length === 2) v += (1 / 8) * (P[3 * opp[0] + c] + P[3 * opp[1] + c]);
      else v = 0.5 * (P[3 * a + c] + P[3 * b + c]); // boundary edge (should not occur on a closed mesh)
      pos[3 * o + c] = v;
      let cv = (3 / 8) * (chart[3 * a + c] + chart[3 * b + c]);
      if (opp.length === 2) cv += (1 / 8) * (chart[3 * opp[0] + c] + chart[3 * opp[1] + c]);
      else cv = 0.5 * (chart[3 * a + c] + chart[3 * b + c]);
      ch[3 * o + c] = cv;
    }
    const r = Math.hypot(ch[3 * o], ch[3 * o + 1], ch[3 * o + 2]) || 1;
    ch[3 * o] /= r; ch[3 * o + 1] /= r; ch[3 * o + 2] /= r;
  }
  const faces = new Uint32Array(3 * 4 * nf);
  let q = 0;
  for (let t = 0; t < nf; t++) {
    const a = F[3 * t], b = F[3 * t + 1], c = F[3 * t + 2];
    const ab = nv + edgeId.get(key(a, b))!, bc = nv + edgeId.get(key(b, c))!, ca = nv + edgeId.get(key(c, a))!;
    faces.set([a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca], q); q += 12;
  }
  return { mesh: makeMesh(pos, faces), chart: ch };
}
