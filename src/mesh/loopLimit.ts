/**
 * Exact evaluation of the Loop limit surface (Stam 1998, without the tables).
 *
 * After one global Loop step every triangle has at most one extraordinary
 * vertex (valence != 6). A triangle's limit patch is determined by a stencil
 * of control points -- 12 for a regular triangle, N+6 when corner 0 has
 * valence N -- in a canonical order built by walking the oriented vertex
 * rings. Subdividing a patch maps its stencil linearly onto the stencils of
 * its four children; those matrices are obtained NUMERICALLY here, once per
 * valence, by running the generic Loop rules on a canonical patch mesh with
 * one-hot fields (no hand-coded subdivision matrices). Evaluating at a point
 * means descending into the child containing it DEPTH times (constant-size
 * products) and finishing with the exact limit mask at the corners: the
 * result is the limit surface to ~4^-DEPTH relative.
 *
 * Fields are d-dimensional, so the surface positions and the spherical chart
 * are evaluated together: both become smooth functions over the control
 * mesh, which is what keeps the parameterization crease-free.
 */
import { makeMesh, type Mesh } from './types.ts';

const DEPTH = 11;   // 4^-11 ~ 2e-7 relative: beyond fp32

/** Loop's beta(n) (Warren's form for n = 3 keeps it simple: 3/16). */
const loopBeta = (n: number): number => (n === 3 ? 3 / 16 : (1 / n) * (5 / 8 - (3 / 8 + 0.25 * Math.cos((2 * Math.PI) / n)) ** 2));
/** limit-position weight: v_inf = (1 - n w) v + w sum(ring), w = 1 / (n + 3/(8 beta)) */
const limitOmega = (n: number): number => 1 / (n + 3 / (8 * loopBeta(n)));

/** Oriented vertex rings of a closed, consistently oriented triangle mesh. */
class Rings {
  readonly mesh: Mesh;
  readonly valence: Int32Array;
  private readonly next: Map<number, number>[]; // per vertex: ring successor map (n_i -> n_{i+1}, face (v, n_i, n_{i+1}))
  constructor(mesh: Mesh) {
    this.mesh = mesh;
    const { faces: f, nv, nf } = mesh;
    this.next = Array.from({ length: nv }, () => new Map());
    this.valence = new Int32Array(nv);
    for (let t = 0; t < nf; t++)
      for (let k = 0; k < 3; k++) {
        const v = f[3 * t + k], n1 = f[3 * t + ((k + 1) % 3)], n2 = f[3 * t + ((k + 2) % 3)];
        this.next[v].set(n1, n2);
      }
    for (let v = 0; v < nv; v++) this.valence[v] = this.next[v].size;
  }
  /** ring of v starting at `from`, in orientation order */
  ring(v: number, from: number): number[] {
    const out = [from];
    const nx = this.next[v];
    for (let k = 1; k < nx.size; k++) out.push(nx.get(out[k - 1])!);
    return out;
  }
}

/**
 * Canonical stencil of face (a, b, c): [a, b, c, x2..x_{N-1}, y2, y3, y4, z2, z3]
 * with ring(a) = [b, c, x2, ..., x_{N-1}], ring(b) = [c, a, x_{N-1}, y2, y3, y4],
 * ring(c) = [a, b, y4, z2, z3, x2]. Corner 0 may be extraordinary (valence N);
 * corners 1 and 2 must be regular.
 */
function stencilOf(r: Rings, a: number, b: number, c: number): Int32Array {
  const ra = r.ring(a, b), rb = r.ring(b, c), rc = r.ring(c, a);
  if (rb.length !== 6 || rc.length !== 6) throw new Error('loopLimit: corners 1 and 2 of a patch must have valence 6');
  const N = ra.length;
  const s = new Int32Array(N + 6);
  s[0] = a; s[1] = b; s[2] = c;
  for (let k = 2; k < N; k++) s[1 + k] = ra[k];     // x2..x_{N-1} at 3..N
  s[N + 1] = rb[3]; s[N + 2] = rb[4]; s[N + 3] = rb[5];   // y2, y3, y4
  s[N + 4] = rc[3]; s[N + 5] = rc[4];                    // z2, z3
  return s;
}

/** Generic Loop subdivision of a d-dimensional vertex field (f64), closed mesh assumed (boundary edges get midpoints). */
export function loopSubdivideField(mesh: Mesh, field: Float64Array, d: number): { mesh: Mesh; field: Float64Array } {
  const { faces: F, nv, nf } = mesh;
  const edgeId = new Map<number, number>();
  const ends: number[] = [], opp: number[][] = [];
  const key = (a: number, b: number) => (a < b ? a * nv + b : b * nv + a);
  for (let t = 0; t < nf; t++)
    for (let k = 0; k < 3; k++) {
      const a = F[3 * t + k], b = F[3 * t + ((k + 1) % 3)], c = F[3 * t + ((k + 2) % 3)];
      const kk = key(a, b);
      let e = edgeId.get(kk);
      if (e === undefined) { e = ends.length / 2; edgeId.set(kk, e); ends.push(a, b); opp.push([c]); } else opp[e].push(c);
    }
  const ne = ends.length / 2, out = new Float64Array((nv + ne) * d);
  const nbr: number[][] = Array.from({ length: nv }, () => []);
  for (let e = 0; e < ne; e++) { nbr[ends[2 * e]].push(ends[2 * e + 1]); nbr[ends[2 * e + 1]].push(ends[2 * e]); }
  for (let i = 0; i < nv; i++) {
    const n = nbr[i].length, beta = loopBeta(n);
    for (let c = 0; c < d; c++) {
      let v = (1 - n * beta) * field[i * d + c];
      for (const j of nbr[i]) v += beta * field[j * d + c];
      out[i * d + c] = v;
    }
  }
  for (let e = 0; e < ne; e++) {
    const a = ends[2 * e], b = ends[2 * e + 1], o = nv + e;
    for (let c = 0; c < d; c++)
      out[o * d + c] = opp[e].length === 2
        ? (3 / 8) * (field[a * d + c] + field[b * d + c]) + (1 / 8) * (field[opp[e][0] * d + c] + field[opp[e][1] * d + c])
        : 0.5 * (field[a * d + c] + field[b * d + c]);
  }
  const faces = new Uint32Array(12 * nf);
  for (let t = 0, q = 0; t < nf; t++) {
    const a = F[3 * t], b = F[3 * t + 1], c = F[3 * t + 2];
    const ab = nv + edgeId.get(key(a, b))!, bc = nv + edgeId.get(key(b, c))!, ca = nv + edgeId.get(key(c, a))!;
    faces.set([a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca], q); q += 12;
  }
  return { mesh: makeMesh(new Float32Array((nv + ne) * 3), faces), field: out };
}

/** The canonical patch mesh for valence N at corner 0: N+6 vertices in stencil order. */
function canonicalPatch(N: number): Mesh {
  // vertices: 0 = a, 1 = b (x0), 2 = c (x1), 3..N = x2..x_{N-1}, N+1..N+3 = y2,y3,y4, N+4,N+5 = z2,z3
  const x = (i: number) => (i === 0 ? 1 : i === 1 ? 2 : 1 + i); // x_i index
  const y2 = N + 1, y3 = N + 2, y4 = N + 3, z2 = N + 4, z3 = N + 5;
  const faces: number[] = [];
  for (let i = 0; i < N; i++) faces.push(0, x(i), x((i + 1) % N));          // fan (a, x_i, x_{i+1})
  const b = 1, c = 2, xNm1 = x(N - 1), x2 = x(2 % N);
  faces.push(b, xNm1, y2, b, y2, y3, b, y3, y4, b, y4, c);                  // around b
  faces.push(c, y4, z2, c, z2, z3, c, z3, x2);                              // around c
  return makeMesh(new Float32Array(3 * (N + 6)), Uint32Array.from(faces));
}

interface PatchOps {
  /** child 0 (at corner 0): (N+6) x (N+6); children 1..3: 12 x (N+6) */
  child: Float64Array[];
  childSize: number[];
}

const opsCache = new Map<number, PatchOps>();

/** Subdivision operators of a patch of valence N, built numerically from the canonical patch. */
function patchOps(N: number): PatchOps {
  const cached = opsCache.get(N);
  if (cached) return cached;
  const pm = canonicalPatch(N), M = N + 6;
  const onehot = new Float64Array(M * M);
  for (let i = 0; i < M; i++) onehot[i * M + i] = 1;                       // field j of vertex i = delta_ij
  const sub = loopSubdivideField(pm, onehot, M);
  const r = new Rings(sub.mesh);
  // children of face 0 = (a, b, c) in loopSubdivideField's order: (a,ab,ca), (ab,b,bc), (ca,bc,c), (ab,bc,ca)
  const child: Float64Array[] = [], childSize: number[] = [];
  for (let k = 0; k < 4; k++) {
    const fa = sub.mesh.faces[3 * k], fb = sub.mesh.faces[3 * k + 1], fc = sub.mesh.faces[3 * k + 2];
    const st = stencilOf(r, fa, fb, fc);
    const C = new Float64Array(st.length * M);
    for (let i = 0; i < st.length; i++) for (let j = 0; j < M; j++) C[i * M + j] = sub.field[st[i] * M + j];
    child.push(C); childSize.push(st.length);
  }
  const ops = { child, childSize };
  opsCache.set(N, ops);
  return ops;
}

/**
 * Quartic box-spline basis of a regular patch: 12 polynomials on the triangle,
 * stored in the degree-4 BERNSTEIN basis B_{ijk}(u,v,w) = 4!/(i!j!k!) u^i v^j w^k
 * (15 functions; well conditioned, unlike monomials). Fitted once to machine
 * precision from a deep descent on the canonical regular patch with one-hot
 * control values. Derivatives along (b - a) and (c - a) use the standard
 * Bernstein difference formula with degree-3 polynomials.
 */
interface RegularBasis { coef: Float64Array }  // 12 x 15 Bernstein coefficients
let regularBasisCache: RegularBasis | null = null;
const BERN4: [number, number, number][] = [];   // (i, j, k) exponents of u, v, w with i + j + k = 4
for (let i = 4; i >= 0; i--) for (let j = 4 - i; j >= 0; j--) BERN4.push([i, j, 4 - i - j]);
const BERN3: [number, number, number][] = [];
for (let i = 3; i >= 0; i--) for (let j = 3 - i; j >= 0; j--) BERN3.push([i, j, 3 - i - j]);
const fact = [1, 1, 2, 6, 24];
const bern = (n: number, e: [number, number, number], u: number, v: number, w: number) =>
  (fact[n] / (fact[e[0]] * fact[e[1]] * fact[e[2]])) * u ** e[0] * v ** e[1] * w ** e[2];
const bern3Index = new Map<string, number>(); BERN3.forEach((e, i) => bern3Index.set(e.join(','), i));
// derivative tables: for BERN4[k] = (i,j,l): index in BERN3 of (i-1,j,l), (i,j-1,l), (i,j,l-1), or -1
const D4U = new Int32Array(15), D4V = new Int32Array(15), D4W = new Int32Array(15);
BERN4.forEach(([i, j, l], k) => {
  D4U[k] = i > 0 ? bern3Index.get(`${i - 1},${j},${l}`)! : -1;
  D4V[k] = j > 0 ? bern3Index.get(`${i},${j - 1},${l}`)! : -1;
  D4W[k] = l > 0 ? bern3Index.get(`${i},${j},${l - 1}`)! : -1;
});
const C4 = new Float64Array(15), C3 = new Float64Array(10);   // multinomial coefficients
BERN4.forEach((e, k) => { C4[k] = fact[4] / (fact[e[0]] * fact[e[1]] * fact[e[2]]); });
BERN3.forEach((e, k) => { C3[k] = fact[3] / (fact[e[0]] * fact[e[1]] * fact[e[2]]); });

export function descend(vals0: Float64Array, M0: number, d: number, N0: number, u: number, v: number, w: number, depth: number): Float64Array {
  // pure subdivision descent to `depth` levels, limit mask at the end (reference evaluator)
  let N = N0, M = M0, vals = vals0.slice(), next = new Float64Array(vals0.length);
  for (let k0 = 0; k0 < depth; k0++) {
    let k: number;
    if (u > 0.5) { k = 0; u = 2 * u - 1; v *= 2; w *= 2; }
    else if (v > 0.5) { k = 1; u *= 2; v = 2 * v - 1; w *= 2; }
    else if (w > 0.5) { k = 2; u *= 2; v *= 2; w = 2 * w - 1; }
    else { k = 3; const nu = 1 - 2 * w, nv = 1 - 2 * u, nw = 1 - 2 * v; u = nu; v = nv; w = nw; }
    const ops = patchOps(N), C = ops.child[k], Mc = ops.childSize[k];
    next.fill(0, 0, Mc * d);
    for (let i = 0; i < Mc; i++) for (let j = 0; j < M; j++) { const cij = C[i * M + j]; if (cij === 0) continue; for (let c = 0; c < d; c++) next[i * d + c] += cij * vals[j * d + c]; }
    const tmp = vals; vals = next; next = tmp; M = Mc;
    if (k !== 0) N = 6;
  }
  const ringA: number[] = [1, 2]; for (let k = 3; k <= N; k++) ringA.push(k);
  const ringB = [2, 0, N, N + 1, N + 2, N + 3], ringC = [0, 1, N + 3, N + 4, N + 5, 3];
  const out = new Float64Array(d);
  const lim = (i: number, ring: number[], c: number) => { const n = ring.length, om = limitOmega(n); let s = (1 - n * om) * vals[i * d + c]; for (const j of ring) s += om * vals[j * d + c]; return s; };
  for (let c = 0; c < d; c++) out[c] = u * lim(0, ringA, c) + v * lim(1, ringB, c) + w * lim(2, ringC, c);
  return out;
}

function regularBasis(): RegularBasis {
  if (regularBasisCache) return regularBasisCache;
  const onehot = new Float64Array(12 * 12); for (let i = 0; i < 12; i++) onehot[i * 12 + i] = 1;
  // interpolate on the degree-4 principal lattice (15 points, unisolvent, well conditioned): square solve
  const A = new Float64Array(15 * 15), B = new Float64Array(15 * 12);
  let p = 0;
  for (const [ei, ej, ek] of BERN4) {
    const u = ei / 4, v = ej / 4, w = ek / 4;
    // depth 16: the descent's accuracy is ~4^-depth (interpolation) + 2^depth eps (roundoff), best ~1e-10 here
    const val = descend(onehot, 12, 12, 6, u, v, w, 16);
    for (let k = 0; k < 15; k++) A[p * 15 + k] = bern(4, BERN4[k], u, v, w);
    for (let i = 0; i < 12; i++) B[p * 12 + i] = val[i];
    p++;
  }
  const x = solveDense(A, B, 15, 12);
  const coef = new Float64Array(12 * 15);
  for (let i = 0; i < 12; i++) for (let k = 0; k < 15; k++) coef[i * 15 + k] = x[k * 12 + i];
  // verify at independent interior points: the limit over a regular patch IS this quartic
  let worst = 0;
  for (const [v, w] of [[0.21, 0.33], [0.05, 0.7], [0.6, 0.1], [0.3333, 0.3333], [0.12, 0.12]]) {
    const u = 1 - v - w, val = descend(onehot, 12, 12, 6, u, v, w, 16);
    for (let i = 0; i < 12; i++) { let s0 = 0; for (let k = 0; k < 15; k++) s0 += coef[i * 15 + k] * bern(4, BERN4[k], u, v, w); worst = Math.max(worst, Math.abs(s0 - val[i])); }
  }
  if (!(worst <= 1e-8)) throw new Error(`loopLimit: regular basis verification residual ${worst}`);
  regularBasisCache = { coef };
  return regularBasisCache;
}

function solveDense(G: Float64Array, R: Float64Array, n: number, m: number): Float64Array {
  const a = G.slice(), b = R.slice();
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(a[r * n + c]) > Math.abs(a[piv * n + c])) piv = r;
    if (piv !== c) { for (let k = 0; k < n; k++) { const t = a[c * n + k]; a[c * n + k] = a[piv * n + k]; a[piv * n + k] = t; } for (let k = 0; k < m; k++) { const t = b[c * m + k]; b[c * m + k] = b[piv * m + k]; b[piv * m + k] = t; } }
    for (let r = c + 1; r < n; r++) { const f = a[r * n + c] / a[c * n + c]; if (f === 0) continue; for (let k = c; k < n; k++) a[r * n + k] -= f * a[c * n + k]; for (let k = 0; k < m; k++) b[r * m + k] -= f * b[c * m + k]; }
  }
  const x = new Float64Array(n * m);
  for (let r = n - 1; r >= 0; r--) for (let k = 0; k < m; k++) { let t = b[r * m + k]; for (let c = r + 1; c < n; c++) t -= a[r * n + c] * x[c * m + k]; x[r * m + k] = t / a[r * n + r]; }
  return x;
}

/** Evaluates the Loop limit of a d-dimensional field over a control mesh (after one isolating subdivision). */
export class LoopLimit {
  readonly mesh: Mesh;          // level-1 control mesh
  readonly field: Float64Array; // nv x d
  readonly d: number;
  private readonly rings: Rings;
  private readonly patchEV: Int8Array;      // per face: corner index (0..2) of the extraordinary vertex, or 0 if regular
  private readonly stencils: (Int32Array | null)[];
  private bufA = new Float64Array(0);
  private bufB = new Float64Array(0);
  private mono = new Float64Array(15);
  private monoV = new Float64Array(10);
  private dvr = new Float64Array(0);
  private dwr = new Float64Array(0);

  constructor(control: Mesh, field: Float64Array, d: number) {
    const sub = loopSubdivideField(control, field, d);   // isolate extraordinary vertices
    this.mesh = sub.mesh; this.field = sub.field; this.d = d;
    this.rings = new Rings(this.mesh);
    this.patchEV = new Int8Array(this.mesh.nf);
    for (let t = 0; t < this.mesh.nf; t++) {
      let ev = 0, count = 0;
      for (let k = 0; k < 3; k++) if (this.rings.valence[this.mesh.faces[3 * t + k]] !== 6) { ev = k; count++; }
      if (count > 1) throw new Error('loopLimit: a face with two extraordinary vertices after isolation');
      this.patchEV[t] = ev;
    }
    this.stencils = new Array(this.mesh.nf).fill(null);
    regularBasis();
  }

  private stencil(t: number): Int32Array {
    let s = this.stencils[t];
    if (!s) {
      const f = this.mesh.faces, rot = this.patchEV[t];
      s = stencilOf(this.rings, f[3 * t + rot], f[3 * t + ((rot + 1) % 3)], f[3 * t + ((rot + 2) % 3)]);
      this.stencils[t] = s;
    }
    return s;
  }

  /**
   * Limit value at barycentric (u, v, w) of face t (in the face's own vertex
   * order). If `gv`/`gw` are given, also the derivatives along the face's own
   * directions (corner1 - corner0) and (corner2 - corner0).
   */
  evaluate(t: number, u: number, v: number, w: number, out = new Float64Array(this.d), gv?: Float64Array, gw?: Float64Array): Float64Array {
    const d = this.d, f = this.mesh.faces, rot = this.patchEV[t];
    // rotate so the extraordinary corner is corner 0
    const bc = [u, v, w];
    u = bc[rot]; v = bc[(rot + 1) % 3]; w = bc[(rot + 2) % 3];
    const st = this.stencil(t);
    let N = this.rings.valence[f[3 * t + rot]];
    let M = st.length;
    const need = Math.max(M, 12) * d;   // children of an irregular patch are 12-point regular patches
    if (this.bufA.length < need) { this.bufA = new Float64Array(need); this.bufB = new Float64Array(need); }
    let vals = this.bufA, next = this.bufB;
    for (let i = 0; i < M; i++) for (let c = 0; c < d; c++) vals[i * d + c] = this.field[st[i] * d + c];
    // Jacobian of the current (v, w) w.r.t. the rotated face's (v, w), accumulated through the descent
    let j11 = 1, j12 = 0, j21 = 0, j22 = 1;
    let depth = 0;
    while (N !== 6 && depth < 16) {   // deeper than ~16 gains nothing (roundoff ~ 2^depth eps)
      let k: number;
      if (u > 0.5) { k = 0; u = 2 * u - 1; v *= 2; w *= 2; }
      else if (v > 0.5) { k = 1; u *= 2; v = 2 * v - 1; w *= 2; }
      else if (w > 0.5) { k = 2; u *= 2; v *= 2; w = 2 * w - 1; }
      else { k = 3; const nu = 1 - 2 * w, nv = 1 - 2 * u, nw = 1 - 2 * v; u = nu; v = nv; w = nw; }
      if (k === 3) { const a = j11, b = j12, c = j21, e = j22; j11 = 2 * a + 2 * c; j12 = 2 * b + 2 * e; j21 = -2 * a; j22 = -2 * b; }   // [[2,2],[-2,0]] * J
      else { j11 *= 2; j12 *= 2; j21 *= 2; j22 *= 2; }
      const ops = patchOps(N), C = ops.child[k], Mc = ops.childSize[k];
      next.fill(0, 0, Mc * d);
      for (let i = 0; i < Mc; i++) {
        const row = i * M, o = i * d;
        for (let j = 0; j < M; j++) { const cij = C[row + j]; if (cij === 0) continue; const jd = j * d; for (let c = 0; c < d; c++) next[o + c] += cij * vals[jd + c]; }
      }
      const tmp = vals; vals = next; next = tmp; M = Mc;
      if (k !== 0) N = 6;
      depth++;
    }
    let dvr: Float64Array | null = null, dwr: Float64Array | null = null;
    if (N === 6) {
      // analytic quartic box spline on the 12-point regular stencil (Bernstein form)
      const Bc = regularBasis().coef, bern4 = this.mono, bern3 = this.monoV;
      const u2 = u * u, v2 = v * v, w2 = w * w;
      const pu = [1, u, u2, u2 * u, u2 * u2], pv = [1, v, v2, v2 * v, v2 * v2], pw = [1, w, w2, w2 * w, w2 * w2];
      for (let k = 0; k < 15; k++) { const e = BERN4[k]; bern4[k] = C4[k] * pu[e[0]] * pv[e[1]] * pw[e[2]]; }
      out.fill(0);
      for (let i = 0; i < 12; i++) {
        let bi = 0; const row = i * 15;
        for (let k = 0; k < 15; k++) bi += Bc[row + k] * bern4[k];
        if (bi !== 0) for (let c = 0; c < d; c++) out[c] += bi * vals[i * d + c];
      }
      if (gv && gw) {
        // d/dv B^4_{ijl} (u = 1 - v - w dependent) = 4 (B^3_{i,j-1,l} - B^3_{i-1,j,l}); d/dw likewise with l
        for (let k = 0; k < 10; k++) { const e = BERN3[k]; bern3[k] = C3[k] * pu[e[0]] * pv[e[1]] * pw[e[2]]; }
        if (this.dvr.length < d) { this.dvr = new Float64Array(d); this.dwr = new Float64Array(d); }
        dvr = this.dvr; dwr = this.dwr; dvr.fill(0); dwr.fill(0);
        for (let i = 0; i < 12; i++) {
          let bv = 0, bw = 0; const row = i * 15;
          for (let k = 0; k < 15; k++) {
            const c4 = Bc[row + k]; if (c4 === 0) continue;
            const du = D4U[k] >= 0 ? bern3[D4U[k]] : 0;
            bv += 4 * c4 * ((D4V[k] >= 0 ? bern3[D4V[k]] : 0) - du);
            bw += 4 * c4 * ((D4W[k] >= 0 ? bern3[D4W[k]] : 0) - du);
          }
          for (let c = 0; c < d; c++) { dvr[c] += bv * vals[i * d + c]; dwr[c] += bw * vals[i * d + c]; }
        }
        // chain through the descent: d/d(v_rot, w_rot) = d/d(v_c, w_c) * J
        for (let c = 0; c < d; c++) { const a = dvr[c], b = dwr[c]; dvr[c] = a * j11 + b * j21; dwr[c] = a * j12 + b * j22; }
      }
    } else {
      // at the extraordinary vertex itself (within 4^-16): its limit position; derivative from the regular neighbourhood is not needed there
      const ringA: number[] = [1, 2]; for (let k = 3; k <= N; k++) ringA.push(k);
      const om = limitOmega(N);
      for (let c = 0; c < d; c++) { let s0 = (1 - N * om) * vals[c]; for (const j of ringA) s0 += om * vals[j * d + c]; out[c] = s0; }
      if (gv && gw) { if (this.dvr.length < d) { this.dvr = new Float64Array(d); this.dwr = new Float64Array(d); } dvr = this.dvr; dwr = this.dwr; dvr.fill(0); dwr.fill(0); }
    }
    if (gv && gw && dvr && dwr) {
      // derivatives along the face's OWN directions (b - a) and (c - a): express those directions
      // in the rotated frame as dv'*(∂/∂v') + dw'*(∂/∂w')
      const own = [[-1, 1, 0], [-1, 0, 1]];
      for (let q = 0; q < 2; q++) {
        const o = own[q], rv = o[(rot + 1) % 3], rw = o[(rot + 2) % 3];  // rotated components of the direction
        const target = q === 0 ? gv : gw;
        for (let c = 0; c < d; c++) target[c] = rv * dvr[c] + rw * dwr[c];
      }
    }
    return out;
  }

  /** Exact limit position of control vertex i (limit mask on the level-1 mesh). */
  limitOfVertex(i: number, out = new Float64Array(this.d)): Float64Array {
    const n = this.rings.valence[i], om = limitOmega(n);
    const f = this.mesh.faces;
    let from = -1;
    for (let t = 0; t < this.mesh.nf && from < 0; t++) for (let k = 0; k < 3; k++) if (f[3 * t + k] === i) { from = f[3 * t + ((k + 1) % 3)]; break; }
    const ring = this.rings.ring(i, from);
    for (let c = 0; c < this.d; c++) {
      let s0 = (1 - n * om) * this.field[i * this.d + c];
      for (const j of ring) s0 += om * this.field[j * this.d + c];
      out[c] = s0;
    }
    return out;
  }
}
