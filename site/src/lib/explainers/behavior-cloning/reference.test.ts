// behavior-cloning: the Python reference (pipeline/behavior-cloning/reference.py) on every fixture, not only on the
// default input that `atlas parity` checks (there: both roll sources). Here the cloned policy's rollouts (source roll,
// after its train source has run) on every fixture with demonstrations, frame by frame (AK204); droll runs the same
// process on other weights, so one source per fixture keeps the Python starts (several seconds each) to six.
// `atlas check` runs it in the "op tests" gate.
//   node --test src/lib/explainers/behavior-cloning/reference.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Issues, openPackage } from '../../../../scripts/atlas/lib/pkg.ts';
import { processParity } from '../../../../scripts/atlas/commands/parity.ts';
import { pythonChoice, pythonCommand } from '../../../../scripts/atlas/config.ts';
import { pairsOf, toRaw } from './world.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
/** skip (not fail) where the configured Python is missing, e.g. a machine set up for kinds B and C only */
const noPython = spawnSync(pythonCommand(pythonChoice()), ['-c', 'import numpy'], { encoding: 'utf8' }).status !== 0;

test('the Python reference repeats the roll source on every fixture', { timeout: 900_000, skip: noPython && 'no configured Python with numpy (npm run atlas -- doctor)' }, async () => {
	const o = await openPackage('behavior-cloning', SITE);
	const lines: string[] = [];
	for (const fx of await o.fixtures()) {
		if (!pairsOf(toRaw(fx.raw)).s.length) continue; // no demonstrations: nothing is trained, roll yields no frames
		const s = o.sources.roll as { process: string; opt?: Record<string, unknown>; seed?: number; after?: string; input?: unknown };
		const is = new Issues();
		const r = await processParity(o, 'roll', s.process, s, is, false, fx);
		lines.push(`${fx.id} roll: ${r.frames} frames, max float difference ${r.maxFloat}`);
		assert.ok(r.pass && r.frames > 0 && r.frames === r.refFrames, `${fx.id} roll: ${is.list.map((i) => i.message).join('; ')}`);
	}
	console.log(lines.join('\n'));
});
