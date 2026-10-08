// gridworld: the Python reference (pipeline/gridworld/reference.py) on every fixture, not only on the default input
// that `atlas parity` checks: each source that reads the learner's input, frame by frame, its draws replayed (AK204).
// `atlas check` runs it in the "op tests" gate.   node --test src/lib/explainers/gridworld/reference.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Issues, openPackage } from '../../../../scripts/atlas/lib/pkg.ts';
import { processParity } from '../../../../scripts/atlas/commands/parity.ts';
import { pythonChoice, pythonCommand } from '../../../../scripts/atlas/config.ts';
import { toRaw } from './world.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
/** skip (not fail) where the configured Python is missing, e.g. a machine set up for kinds B and C only */
const noPython = spawnSync(pythonCommand(pythonChoice()), ['-c', 'import numpy'], { encoding: 'utf8' }).status !== 0;

test('the Python reference repeats the input-dependent sources on every fixture', { timeout: 400_000, skip: noPython && 'no configured Python with numpy (npm run atlas -- doctor)' }, async () => {
	const o = await openPackage('gridworld', SITE);
	const lines: string[] = [];
	for (const fx of await o.fixtures())
		for (const name of ['you', 'q', 'vi05', 'vi95']) {
			if (name === 'you' && !toRaw(fx.raw).pulls.length) continue; // no pulls: one init frame
			const s = o.sources[name] as { process: string; opt?: Record<string, unknown>; seed?: number };
			const is = new Issues();
			const r = await processParity(o, name, s.process, { opt: s.opt, seed: s.seed, input: () => fx.raw }, is, false);
			lines.push(`${fx.id} ${name}: ${r.frames} frames, max float difference ${r.maxFloat}`);
			assert.ok(r.pass && r.frames > 0 && r.frames === r.refFrames, `${fx.id} ${name}: ${is.list.map((i) => i.message).join('; ')}`);
		}
	console.log(lines.join('\n'));
});
