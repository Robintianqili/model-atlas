// W1-C acceptance 5: a full Node dry run of the 6-step fixture lesson (stub ops) in en and zh.
// createRuntime -> compileLesson -> for every fixture (lesson, adapter edges, presets) and every step: select, cursor,
// ensure (the implicit scan and the process run for real), At, narration and look in both languages, focus, props, task;
// then lintLesson over the same matrix. Prints the narration table (`node --test` diagnostics).
// Run from site/:  node --test src/lib/atlas/lesson/dryrun.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { compileLesson } from './compile.ts';
import { fixtureCases, formatIssues, lintLesson } from './lint.ts';
import { lesson6 } from './fixtures/lesson6.ts';
import { SEQ_SPEC, seqFetch, seqPackage } from './fixtures/seq.ts';
import { createRuntime } from '../runtime/runtime.ts';
import { effectiveProvenance, stepAt } from '../runtime/sources.ts';

test('acceptance 5: Node dry run of the 6-step fixture lesson in en and zh', async (t) => {
	const pkg = seqPackage(lesson6);
	const compiled = compileLesson(lesson6, { spec: SEQ_SPEC, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes });
	assert.equal(compiled.steps.length, 6);
	assert.deepEqual(compiled.issues, []);
	const probe = await createRuntime(pkg, { fetch: seqFetch, base: '' });
	assert.deepEqual(probe.warnings, []);
	const fixtures = fixtureCases(pkg, probe.presets);
	assert.deepEqual(fixtures.map((f) => f.id), ['mixed', 'edge:empty', 'edge:one', 'edge:long', 'preset:hello', 'preset:zz']);
	let rendered = 0;
	for (const fx of fixtures) {
		const rt = await createRuntime(pkg, { fetch: seqFetch, base: '', input: fx.raw });
		const rows: string[] = [];
		for (const step of compiled.steps) {
			const pos = await stepAt(rt, step);
			assert.equal(pos.error, undefined, `${fx.id}/${step.id}: ${pos.error}`);
			if (rt.pre?.empty) {
				assert.equal(pos.skipped, true, `${fx.id}/${step.id}: empty input skips the stage`);
				rows.push(`  ${step.id.padEnd(10)} (empty input: ${rt.pre.notes?.[0]?.en})`);
				continue;
			}
			assert.equal(pos.found, true, `${fx.id}/${step.id}: cursor ${JSON.stringify(pos.cursor)} found`);
			const n = step.narration(rt, pos.at);
			const look = step.look?.(rt, pos.at);
			for (const lang of ['en', 'zh'] as const) {
				const text = n[lang]!;
				assert.ok(text && text.trim(), `${fx.id}/${step.id}/${lang}: narration`);
				assert.ok(!text.includes('…'), `${fx.id}/${step.id}/${lang}: unresolved placeholder in "${text}"`);
				assert.ok(!/\{[A-Za-z_]/.test(text), `${fx.id}/${step.id}/${lang}: raw template in "${text}"`);
				if (look) assert.ok(look[lang] && !look[lang]!.includes('…'));
			}
			assert.ok(/[一-鿿]/.test(n.zh!), `${fx.id}/${step.id}: zh narration is Chinese, not an English fallback`);
			step.focus(rt, pos.at);
			step.props(rt, pos.at);
			step.task?.done(rt, pos.at);
			const badge = effectiveProvenance(rt, step);
			assert.ok(['live', 'precomputed', 'illustrative'].includes(badge.mode));
			rows.push(`  ${step.id.padEnd(10)} [${badge.mode}] #${pos.index}\n    en: ${n.en}\n    zh: ${n.zh}`);
			rendered += 2;
		}
		t.diagnostic(`fixture ${fx.id} (${JSON.stringify(fx.raw)}):\n${rows.join('\n')}`);
		rt.close();
	}
	// 5 non-empty fixtures × 6 steps × 2 languages
	assert.equal(rendered, 5 * 6 * 2);
	const issues = await lintLesson(compiled, (fx) => createRuntime(pkg, { fetch: seqFetch, base: '', input: fx.raw }), fixtures);
	assert.deepEqual(issues, [], formatIssues(issues));
	t.diagnostic(`lint: ${fixtures.length} fixtures × ${compiled.steps.length} steps × en/zh: 0 issues`);
});
