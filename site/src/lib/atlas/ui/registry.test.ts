// The view registry's pure part (registry-entries.ts; redesign wave F, per-package metas) and the rule that keeps view
// metas cheap to load (AK507): a meta.ts never imports a Svelte component or another package's code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { kitEntry, packageMetaPaths, packageOfKey, pkgEntry } from './registry-entries.ts';
import { metaImportProblems } from '../../../../scripts/atlas/lib/metalint.ts';
import type { ViewMeta } from '../types/view.ts';

const SITE = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const meta = (key: string) => ({ key, title: { en: 'T', zh: 'T' }, kinds: ['tensor'] }) as unknown as ViewMeta;
const comp = async () => (() => null) as never;

test('kit entries: key from the folder, the component found, problems named', () => {
	const e = kitEntry('./views/loss-curve/meta.ts', meta('loss-curve'), { './views/loss-curve/View.svelte': comp });
	assert.ok(e && e.key === 'loss-curve' && e.load && !e.problem);
	assert.match(kitEntry('./views/x/meta.ts', meta('y'), { './views/x/View.svelte': comp })!.problem!, /differs from its folder/);
	assert.match(kitEntry('./views/x/meta.ts', meta('x'), {})!.problem!, /View\.svelte is missing/);
	assert.equal(kitEntry('../../explainers/a/views/b/meta.ts', meta('a/b'), {}), null);
});

test('package entries register as <slug>/<key>', () => {
	const views = { '../../explainers/lora/views/matrices/View.svelte': comp };
	const e = pkgEntry('../../explainers/lora/views/matrices/meta.ts', meta('lora/matrices'), views);
	assert.ok(e && e.key === 'lora/matrices' && !e.problem);
	assert.ok(!pkgEntry('../../explainers/lora/views/matrices/meta.ts', meta('matrices'), views)!.problem, 'the bare key is accepted');
	assert.match(pkgEntry('../../explainers/lora/views/matrices/meta.ts', meta('other'), views)!.problem!, /should be 'lora\/matrices'/);
	assert.equal(packageOfKey('lora/matrices'), 'lora');
	assert.equal(packageOfKey('loss-curve'), null);
});

test('prepareViews picks one package: no prefix collisions (lora vs lora-x)', () => {
	const paths = ['../../explainers/lora/views/b/meta.ts', '../../explainers/lora-x/views/a/meta.ts', '../../explainers/lora/views/a/meta.ts', '../../explainers/lora/views/a/helper.ts'];
	assert.deepEqual(packageMetaPaths(paths, 'lora'), ['../../explainers/lora/views/a/meta.ts', '../../explainers/lora/views/b/meta.ts']);
	assert.deepEqual(packageMetaPaths(paths, 'none'), []);
});

test('AK507: every view meta.ts (kit and packages) imports no Svelte component and no other package', () => {
	const problems: string[] = [];
	const kit = join(SITE, 'src', 'lib', 'atlas', 'ui', 'views');
	for (const k of readdirSync(kit)) if (existsSync(join(kit, k, 'meta.ts'))) problems.push(...metaImportProblems(join(kit, k, 'meta.ts'), null));
	const pk = join(SITE, 'src', 'lib', 'explainers');
	for (const slug of readdirSync(pk)) {
		const v = join(pk, slug, 'views');
		if (!existsSync(v)) continue;
		for (const k of readdirSync(v)) if (existsSync(join(v, k, 'meta.ts'))) problems.push(...metaImportProblems(join(v, k, 'meta.ts'), slug));
	}
	assert.deepEqual(problems, []);
});

test('AK507 finds a Svelte import and another package', () => {
	const src = "import type { ViewMeta } from '../../../../atlas/types/view.ts';\nimport V from './View.svelte';\nimport { x } from '../../../lora/data.ts';\nimport { y } from '../../data.ts';\n";
	const p = metaImportProblems('X:/site/src/lib/explainers/demo/views/a/meta.ts', 'demo', src);
	assert.equal(p.length, 2);
	assert.match(p[0], /Svelte/);
	assert.match(p[1], /another package \(lora\)/);
});
