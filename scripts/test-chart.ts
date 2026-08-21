// node scripts/test-chart.ts [preset]  -- charts a preset mesh and reports the MATLAB-comparable numbers
import { readFileSync } from 'node:fs';
import { parseSHM } from '../src/mesh/loaders.ts';
import { eulerCharacteristic } from '../src/mesh/types.ts';
import { sphericalChart } from '../src/chart/policy.ts';
import { countFolds, lambdaStats } from '../src/chart/meshops.ts';
import { sphericalConformalMap } from '../src/chart/conformal.ts';

const names = process.argv.slice(2).length ? process.argv.slice(2) : ['spot'];
for (const name of names) {
  const m = parseSHM(readFileSync(new URL(`../public/presets/${name}.mesh`, import.meta.url)).buffer.slice(0));
  console.log(`\n== ${name}: ${m.nv} v, ${m.nf} f, chi ${eulerCharacteristic(m)}`);
  let t = performance.now();
  const conf = sphericalConformalMap(m);
  const lc = lambdaStats(m, conf);
  console.log(`conformal          : ${(performance.now() - t).toFixed(0)} ms, folds ${countFolds(m.faces, conf)}, lambda std/mean ${lc.spread.toFixed(3)}, max/min ${lc.ratio.toExponential(2)}`);
  t = performance.now();
  const { info } = await sphericalChart(m, 'conformal');
  console.log(`conformal + Mobius : ${info.timeMs.toFixed(0)} ms (NM evals ${info.mobiusEvals}), folds ${info.foldsRepaired} -> ${info.foldsLeft}, lambda std/mean ${info.lambdaSpread.toFixed(3)}, max/min ${info.lambdaRatio.toExponential(2)}${info.crowded ? '  [crowded]' : ''}`);
  const tu = await sphericalChart(m, 'tutte');
  console.log(`tutte              : ${tu.info.timeMs.toFixed(0)} ms, folds ${tu.info.foldsRepaired} -> ${tu.info.foldsLeft}, lambda std/mean ${tu.info.lambdaSpread.toFixed(3)}, max/min ${tu.info.lambdaRatio.toExponential(2)}`);
  if (process.env.SDEM) {
    const ar = await sphericalChart(m, 'area');
    console.log(`area-equalized     : ${ar.info.timeMs.toFixed(0)} ms (SDEM ${ar.info.sdemSteps} steps, density spread ${ar.info.sdemSpread?.toFixed(3)}), folds ${ar.info.foldsRepaired} -> ${ar.info.foldsLeft}, lambda std/mean ${ar.info.lambdaSpread.toFixed(3)}, max/min ${ar.info.lambdaRatio.toExponential(2)}`);
  }
}
