// Deep links straight to a lesson step (#lesson-<id>), the way the Lesson Lab and a shared link open a page: every step
// of the kit's example packages must work when it is the FIRST step entered, and give the same numbers as stepping in
// from the start. The afternoon test found that hello-xor's and hello-xor-2's `live` step (a live-model source) failed:
// "source 'live' has no frames (status error: source 'train' is not a train source with frames)".
// Run from site/: node --test src/lib/atlas/integration/deeplink.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fsFetch } from '../core/io.ts';
import { createRuntime } from '../runtime/runtime.ts';
import { stepAt } from '../runtime/sources.ts';
import { compileLesson } from '../lesson/compile.ts';
import { renderI18n } from '../lesson/narrate.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const opts = { fetch: fsFetch, base: join(SITE, 'static') };

for (const slug of ['hello-xor', 'hello-xor-2']) {
	test(`${slug}: every lesson step works as the first step entered (deep link) and matches stepping in`, { skip: !existsSync(join(SITE, 'src', 'lib', 'explainers', slug, 'lesson.ts')) }, async () => {
		const { pkg } = await loadPackage(slug, SITE);
		const probe = await createRuntime(pkg, opts);
		const c = compileLesson(pkg.lesson, { spec: probe.spec, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes });
		// stepping in: one runtime, every step in order
		const seq: Record<string, { en: string; index: number }> = {};
		for (const step of c.steps) {
			const pos = await stepAt(probe, step);
			assert.equal(pos.error, undefined, `${step.id} (stepping in): ${pos.error}`);
			seq[step.id] = { en: step.narration(probe, pos.at).en, index: pos.index };
		}
		probe.close();
		const live = c.steps.filter((s) => pkg.manifest.sources?.[s.source]?.kind === 'live-model');
		assert.ok(live.length > 0, `${slug} has a live-model step`);
		for (const step of c.steps) {
			const rt = await createRuntime(pkg, opts);
			const pos = await stepAt(rt, step);
			assert.equal(pos.error, undefined, `${step.id} (deep link): ${pos.error}`);
			assert.ok(pos.found, `${step.id}: the cursor is found`);
			const text = renderI18n(step.src.narration, rt, pos.at).en;
			assert.ok(!text.includes('…'), `${step.id}: the narration resolves: ${text}`);
			// live-model and train steps read a train source run to its end: the same numbers either way
			assert.equal(step.narration(rt, pos.at).en, seq[step.id].en, `${step.id}: deep link and stepping in say the same`);
			rt.close();
		}
	});
}
