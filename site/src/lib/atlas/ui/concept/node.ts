// Node only (tests, never bundled): lint a concept package's page parts with the CLI's package loader, the way a
// package's concept.test.ts does it, so the `atlas check` "op tests" gate covers concept.ts (atlas check runs every
// *.test.ts of a package). One call per package:
//   test('concept page', async () => assertConceptClean(await lintConceptPackage('loss-gradient-descent', SITE)));
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { LintIssue } from '../../types/lesson.ts';
import { openPackage } from '../../../../../scripts/atlas/lib/pkg.ts';
import { glossaryIds, viewKeys } from '../../../../../scripts/atlas/commands/lesson.ts';
import { lintConcept } from './lint.ts';

export async function lintConceptPackage(slug: string, site: string): Promise<LintIssue[]> {
	const o = await openPackage(slug, site);
	const views = viewKeys(site);
	const terms = await glossaryIds(site);
	const probe = await o.runtime();
	const spec = probe.spec;
	probe.close?.();
	const concept = (await import(pathToFileURL(join(o.l.dir, 'concept.ts')).href)) as Record<string, unknown>;
	return lintConcept({
		pkg: o.l.pkg,
		concept,
		spec,
		makeRuntime: (raw) => o.runtime(raw),
		fixtures: await o.fixtures(),
		...(views.size ? { views: (k: string) => views.has(k) } : {}),
		...(terms ? { terms: (id: string) => terms.has(id) } : {})
	});
}

/** Fails on any error; prints warnings (AK402 literal numbers, AK410, AK412) without failing. */
export function assertConceptClean(issues: LintIssue[]): void {
	const errors = issues.filter((i) => i.level === 'error');
	for (const w of issues.filter((i) => i.level === 'warn')) console.log(`(warn) ${w.code} ${w.step ?? ''} ${w.fixture ?? ''} ${w.message}`);
	assert.deepEqual(
		errors.map((e) => `${e.code} ${e.step ?? ''} ${e.fixture ?? ''} ${e.lang ?? ''}: ${e.message}`),
		[],
		'the concept page has lint errors'
	);
}
