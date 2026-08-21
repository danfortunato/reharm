// Angle distortion of a chart per face: how far the spherical triangle's angles are from the mesh triangle's.
import { readFileSync } from 'node:fs';
import { parseSHM } from '../src/mesh/loaders.ts';
import { sphericalChart } from '../src/chart/policy.ts';
const m = parseSHM(readFileSync(new URL(`../public/presets/${process.argv[2] ?? 'spot'}.mesh`, import.meta.url)).buffer.slice(0));
const angles = (P: ArrayLike<number>, t: number): number[] => {
  const f = m.faces, v = [f[3*t], f[3*t+1], f[3*t+2]].map((i) => [P[3*i], P[3*i+1], P[3*i+2]]);
  const ang = (a: number[], b: number[], c: number[]) => { const u = b.map((x,k)=>x-a[k]), w = c.map((x,k)=>x-a[k]);
    const d = u[0]*w[0]+u[1]*w[1]+u[2]*w[2]; return Math.acos(Math.max(-1, Math.min(1, d / (Math.hypot(...u)*Math.hypot(...w)) ))); };
  return [ang(v[0],v[1],v[2]), ang(v[1],v[2],v[0]), ang(v[2],v[0],v[1])];
};
for (const kind of ['conformal', 'area'] as const) {
  const { S, info } = await sphericalChart(m, kind, 1e4, undefined, 200);
  const err: number[] = []; let slivers = 0;
  for (let t = 0; t < m.nf; t++) {
    const a = angles(m.positions, t), b = angles(S, t);
    err.push(Math.max(...a.map((x, k) => Math.abs(x - b[k]))));
    if (Math.min(...b) < 0.05) slivers++;   // spherical triangle with an angle < 3 degrees
  }
  err.sort((x, y) => x - y);
  const q = (p: number) => (err[Math.floor(p * (err.length - 1))] * 180 / Math.PI).toFixed(1);
  console.log(`${kind.padEnd(9)} ${info.type.padEnd(40)} max angle error per face: median ${q(0.5)}°, 90% ${q(0.9)}°, 99% ${q(0.99)}°, max ${q(1)}°;  sliver faces (angle < 3°): ${slivers} of ${m.nf}`);
}
