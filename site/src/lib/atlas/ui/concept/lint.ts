// The lint of a concept page's own parts (pure; Node and tests). The lesson is linted by `atlas lesson lint` like any
// package's; this adds what only a concept page has:
//   - the manifest says type 'concept' (AK301);
//   - the stage panels (hook, toy, real, compare) compiled and run exactly like lesson steps, on every fixture, in en and
//     zh: placeholders (AK401), views (AK405), provenance (AK502), zh (AK406), terms (AK411), literal numbers (AK402),
//     cursors (AK404). The lesson-shape rules (AK407, AK409) do not apply to panels;
//   - the plain text: the hook, pitfalls and further reading need zh (AK406); a pitfall's `step` must be a lesson step
//     (AK404); a further link is an absolute http(s) URL or a site path (AK301).
import type { I18n } from '../../types/core.ts';
import type { FixtureDef, LintIssue } from '../../types/lesson.ts';
import type { ExplainerRuntime, PackageModules } from '../../types/runtime.ts';
import type { AtlasSpec } from '../../types/spec.ts';
import { compileLesson } from '../../lesson/compile.ts';
import { lintLesson } from '../../lesson/lint.ts';
import { conceptOf, isConceptManifest, panelLesson, pitfallsOf, SHAPE_ONLY } from './panels.ts';

export interface ConceptLintCtx {
	pkg: PackageModules;
	/** the concept.ts module (its exports) */
	concept: Record<string, unknown>;
	makeRuntime: (raw: unknown) => Promise<ExplainerRuntime> | ExplainerRuntime;
	fixtures: FixtureDef[];
	spec?: AtlasSpec | null;
	/** registered view keys; when given, unknown views are AK405 */
	views?: (key: string) => boolean;
	/** site glossary ids; when given, unknown terms are AK411 */
	terms?: (id: string) => boolean;
	langs?: ('en' | 'zh')[];
}

const hasZh = (t: I18n | undefined) => !!t && typeof t.zh === 'string' && t.zh.trim().length > 0;

export async function lintConcept(c: ConceptLintCtx): Promise<LintIssue[]> {
	const m = c.pkg.manifest;
	const out: LintIssue[] = [];
	const err = (code: string, message: string, step?: string) => out.push({ code, level: 'error', message, ...(step ? { step } : {}) });
	if (!isConceptManifest(m)) err('AK301', `explainer.ts says type '${String((m as { type?: unknown }).type)}': a concept package says type: 'concept'`);
	const def = conceptOf(c.concept);
	const ctx = { spec: c.spec ?? null, ops: c.pkg.ops, manifest: m, adapters: c.pkg.adapters, processes: c.pkg.processes, views: c.views, terms: c.terms, plan: [] };
	// the plain text
	if (def.hook && !('view' in def.hook) && !hasZh(def.hook as I18n)) err('AK406', 'concept.ts hook has no zh', 'hook');
	const lessonIds = new Set(compileLesson(c.pkg.lesson, ctx).steps.map((s) => s.id));
	for (const [i, p] of pitfallsOf(def, m).entries()) {
		if (!hasZh(p.claim) || !hasZh(p.truth)) err('AK406', `pitfall ${i} has no zh`);
		if (p.step && !lessonIds.has(p.step)) err('AK404', `pitfall ${i} links lesson step '${p.step}', which the lesson does not have (${[...lessonIds].join(', ')})`);
	}
	for (const [i, f] of (def.further ?? []).entries()) {
		if (!hasZh(f.label)) err('AK406', `further[${i}] has no zh`);
		if (!/^(https?:\/\/|\/)/.test(f.href)) err('AK301', `further[${i}] href '${f.href}' is neither an http(s) URL nor a site path starting with '/'`);
	}
	// the stage panels, as lesson steps
	const pl = panelLesson(def, m);
	if (!pl.steps.length) return out;
	const compiled = compileLesson(pl, ctx);
	compiled.issues = compiled.issues.filter((i) => !SHAPE_ONLY.has(i.code));
	const issues = await lintLesson(compiled, (fx) => c.makeRuntime(fx.raw), c.fixtures.length ? c.fixtures : [{ id: 'default', label: { en: 'the default input', zh: '默认输入' }, raw: undefined }], { langs: c.langs });
	return [...out, ...issues.filter((i) => !SHAPE_ONLY.has(i.code))];
}
