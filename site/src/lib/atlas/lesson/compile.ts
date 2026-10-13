// compileLesson(def, { spec, ops, manifest }) (design §4.6): expands auto()/autoAll() against the planner, turns every
// StepDef into a CompiledStep (closures over the runtime) and runs the static lint rules. Runtime rules (placeholders
// on every fixture, cursors) are in lint.ts.
//
// Static rules here: AK401 (unknown |fmt), AK402 (literal numbers, warn), AK403 (en/zh placeholder sets), AK404
// (undeclared source), AK405 (view not registered, when a view list is given), AK406 (missing zh), AK407 (lesson shape),
// AK409 (a 'reviewed' lesson made only of auto steps), AK411 (a step term in no glossary, when a glossary is given),
// AK412 (a predict step's narration gives the answer away, warning), AK501 (TODO markers), AK502 (provenance).
// Redesign W2 (R1, all warnings): AK413 (chapters: 2-4, 1-6 steps each, every step in exactly one, in order), AK414 (a
// look line over 25 words), AK415 (a narration over 60 words; the gate stays AK407 at 90), AK416 (an index typed as
// "W0": use <sub> or W₀), AK417 (a Why answer over 120 words, or a Why summary that is not a question), AK418
// (page.poster names no step of the lesson).
import type { I18n, PredictQuestion, Provenance } from '../types/core.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { OpDef, OpRegistry } from '../types/op.ts';
import type { Cursor, ProcessDef } from '../types/process.ts';
import type { At, AutoRef, CompiledStep, LessonDef, LintIssue, Narration, StepDef } from '../types/lesson.ts';
import type { ExplainerManifest, InputAdapter } from '../types/explainer.ts';
import type { ExplainerRuntime } from '../types/runtime.ts';
import { isAutoRef, sourceDef, sourcesOf, stepSlug } from './define.ts';
import { FORMATS, hasRefs, literalNumbers, placeholderNames, placeholders, renderI18n, sentenceCount, wordCount } from './narrate.ts';
import { planLesson } from './plan.ts';

export interface CompileCtx {
	spec: AtlasSpec | null;
	ops: OpDef[] | OpRegistry;
	manifest: ExplainerManifest;
	adapters?: Record<string, InputAdapter>;
	processes?: Record<string, ProcessDef<any, any, any>>;
	/** registered view keys; when given, unknown views are AK405 */
	views?: (key: string) => boolean;
	/** a precomputed plan (otherwise planLesson runs) */
	plan?: StepDef[];
	/** ids of the site glossary; when given, a step term in neither it nor manifest.glossary is AK411 (K0) */
	terms?: (id: string) => boolean;
}

/** Lowercase words of a text without placeholders. */
const wordsOf = (s: string | undefined) => (s ?? '').replace(/\{[^}]*\}/g, ' ').toLowerCase().match(/[\p{L}\p{N}'’]+/gu) ?? [];

/** Why a predict step's narration (shown before the learner answers) gives the answer away, or null (AK412, K0): it
 *  quotes the right option (8+ characters), or shares a run of 6 words with the explanation shown after the answer. */
export function giveaway(narration: string | undefined, q: PredictQuestion): string | null {
	const text = wordsOf(narration).join(' ');
	if (!text) return null;
	const right = wordsOf(q.options[q.answer]?.en).join(' ');
	if (right.length >= 8 && ` ${text} `.includes(` ${right} `)) return `it quotes the right option "${q.options[q.answer].en}"`;
	const ex = wordsOf(q.explain.en);
	const n = 6;
	for (let i = 0; i + n <= ex.length; i++) {
		const run = ex.slice(i, i + n).join(' ');
		if (` ${text} `.includes(` ${run} `)) return `it repeats the explanation ("… ${run} …")`;
	}
	return null;
}

/** A CompiledStep that keeps the StepDef it came from (lint reads its templates). */
export interface AtlasCompiledStep extends CompiledStep {
	src: StepDef<any>;
	/** auto steps whose narration the author overrode count as authored work for AK409 */
	edited: boolean;
}

export interface CompiledLesson {
	id: string;
	title: I18n;
	summary: I18n;
	steps: AtlasCompiledStep[];
	issues: LintIssue[];
	plan: StepDef[];
	/** R1: LessonDef.chapters (the steps that exist, in lesson order) */
	chapters?: { id: string; title: I18n; steps: string[] }[];
	/** R1: LessonDef.recap, rendered over a step's moment */
	recap?: ((rt: ExplainerRuntime, at: At) => I18n)[];
}

/** An index typed into a word ("W0", "x1"): AK416 asks for a subscript (placeholders are not text). */
export function typedIndices(text: string | undefined): string[] {
	const plain = (text ?? '').replace(/\{[^}]*\}/g, ' ');
	return [...new Set(plain.match(/\b[A-Za-z]0\b/g) ?? [])];
}

const PROVENANCE = new Set<Provenance>(['live', 'precomputed', 'illustrative']);
const strict = (q: ExplainerManifest['quality']) => q === 'reviewed' || q === 'flagship';

function hasZh(t: I18n | undefined): boolean {
	return !!t && typeof t.zh === 'string' && t.zh.trim().length > 0;
}

function provenanceOf(s: StepDef<any>): { mode: Provenance; note?: I18n } | null {
	const p = s.provenance as unknown;
	if (typeof p === 'string' && PROVENANCE.has(p as Provenance)) return { mode: p as Provenance };
	if (p && typeof p === 'object' && PROVENANCE.has((p as { mode: Provenance }).mode)) return { mode: (p as { mode: Provenance }).mode, note: (p as { note?: I18n }).note };
	return null;
}

/** The step's At with its source name (K1: {opt.<name>} resolves from the source before it has run). */
export const withSource = (at: At | undefined, source: string): At => (at?.source ? at : { ...(at ?? { index: -1 }), source });

export function compileStep(s: StepDef<any>, o: { origin: 'auto' | 'author'; autoId?: string; edited?: boolean; defaultSource: string }): AtlasCompiledStep {
	const prov = provenanceOf(s) ?? { mode: 'illustrative' as Provenance, note: { en: 'This step declares no provenance.', zh: '这一步没有声明数据来源。' } };
	const view = typeof s.view === 'string' ? s.view : (s.view?.key ?? '');
	const props = typeof s.view === 'object' && s.view?.props ? s.view.props : null;
	const source = s.source ?? o.defaultSource;
	const step: AtlasCompiledStep = {
		id: stepSlug(s.id),
		origin: o.origin,
		title: s.title,
		narration: (rt: ExplainerRuntime, at: At) => renderI18n(s.narration as Narration<ExplainerRuntime>, rt, withSource(at, source)),
		source,
		at: (rt: ExplainerRuntime): Cursor => {
			const a = typeof s.at === 'function' ? s.at(rt) : s.at;
			if (a !== undefined) return a;
			return rt.sel.layer && rt.sources[source]?.kind !== 'process' && rt.sources[source]?.kind !== 'train' ? { layer: rt.sel.layer } : 'first';
		},
		view,
		props: (rt: ExplainerRuntime, at: At) => (props ? props(rt, at) : {}),
		focus: (rt: ExplainerRuntime, at: At) => s.focus?.(rt, at) ?? [],
		provenance: prov,
		terms: s.terms ? [...s.terms] : [],
		waitForDone: s.waitForDone ?? Boolean(s.predict || s.task || s.play === 'view'),
		src: s,
		edited: o.edited ?? o.origin === 'author'
	};
	if (o.autoId !== undefined) step.autoId = o.autoId;
	// K2: a skipped step's look line drops the sentences without numbers silently (the narration says why)
	if (s.look) step.look = (rt: ExplainerRuntime, at: At) => renderI18n(s.look as Narration<ExplainerRuntime>, rt, withSource(at, source), { emptyLine: false });
	if (s.play !== undefined) step.play = s.play;
	if (s.select) step.select = s.select;
	if (s.predict) step.predict = s.predict;
	if (s.predictAt) step.predictAt = (rt: ExplainerRuntime, at: At) => s.predictAt!(rt, at);
	if (s.task) step.task = s.task;
	if (s.compare) step.compare = s.compare;
	if (s.kindOf) step.kindOf = s.kindOf;
	if (s.linger !== undefined) step.linger = s.linger;
	if (s.duration !== undefined) step.duration = s.duration;
	if (s.why) {
		const why = s.why;
		step.why = { q: why.q, a: (rt: ExplainerRuntime, at: At) => renderI18n(why.a as Narration<ExplainerRuntime>, rt, withSource(at, source)) };
	}
	if (s.refs) step.refs = (rt: ExplainerRuntime, at: At) => s.refs!(rt, at);
	if (hasRefs(s.narration) || hasRefs(s.look)) {
		step.chips = {};
		if (hasRefs(s.narration)) step.chips.narration = (rt: ExplainerRuntime, at: At) => renderI18n(s.narration as Narration<ExplainerRuntime>, rt, withSource(at, source), { chips: true });
		if (s.look && hasRefs(s.look)) step.chips.look = (rt: ExplainerRuntime, at: At) => renderI18n(s.look as Narration<ExplainerRuntime>, rt, withSource(at, source), { emptyLine: false, chips: true });
	}
	return step;
}

export function compileLesson(def: LessonDef<any> | null, ctx: CompileCtx): CompiledLesson {
	const m = ctx.manifest;
	const issues: LintIssue[] = [];
	const plan = ctx.plan ?? planLesson(ctx.spec, ctx.ops, m, { adapters: ctx.adapters, processes: ctx.processes });
	const lesson: LessonDef<any> = def ?? { title: { en: m.slug, zh: m.slug }, summary: { en: '', zh: '' }, steps: [{ autoAll: {} }] };
	const sources = Object.keys(sourcesOf(m));
	const defaultSource = sources[0] ?? '';
	const planIds = plan.map((s) => s.id);
	const explicit = new Set(lesson.steps.filter((s): s is { auto: string } => isAutoRef(s) && 'auto' in s).map((s) => s.auto));
	const err = (code: string, message: string, extra: Partial<LintIssue> = {}): void => void issues.push({ code, level: 'error', message, ...extra });
	const warn = (code: string, message: string, extra: Partial<LintIssue> = {}): void => void issues.push({ code, level: 'warn', message, ...extra });

	const steps: AtlasCompiledStep[] = [];
	for (const item of lesson.steps as (StepDef<any> | AutoRef<any>)[]) {
		if (isAutoRef(item)) {
			if ('autoAll' in item) {
				const { from, until, skip } = item.autoAll;
				let a = 0;
				let b = plan.length;
				if (from !== undefined) {
					a = planIds.indexOf(from);
					if (a < 0) err('AK407', `autoAll: 'from' step '${from}' is not in the plan`, { fix: `planned ids: ${planIds.join(', ')}` });
					a = Math.max(0, a);
				}
				if (until !== undefined) {
					b = planIds.indexOf(until);
					if (b < 0) err('AK407', `autoAll: 'until' step '${until}' is not in the plan`, { fix: `planned ids: ${planIds.join(', ')}` });
					if (b < 0) b = plan.length;
				}
				for (const s of plan.slice(a, b)) if (!skip?.includes(s.id) && !explicit.has(s.id)) steps.push(compileStep(s, { origin: 'auto', autoId: s.id, defaultSource }));
			} else {
				const s = plan.find((p) => p.id === item.auto);
				if (!s) {
					err('AK407', `auto('${item.auto}') is not in the plan`, { step: item.auto, fix: `planned ids: ${planIds.join(', ') || 'none'}` });
					continue;
				}
				const merged = { ...s, ...(item.override ?? {}) } as StepDef<any>;
				steps.push(compileStep(merged, { origin: 'auto', autoId: s.id, edited: !!(item.override?.narration || item.override?.title), defaultSource }));
			}
		} else steps.push(compileStep(item, { origin: 'author', defaultSource }));
	}

	// ------------------------------------------------------------------ static checks
	const seen = new Map<string, number>();
	for (const s of steps) seen.set(s.id, (seen.get(s.id) ?? 0) + 1);
	for (const [id, n] of seen) if (n > 1) err('AK407', `step id '${id}' is used ${n} times (deep links must be unique)`, { step: id });
	if (steps.length < 5 || steps.length > 12) warn('AK407', `the lesson has ${steps.length} steps (aim for 5-12)`);
	if (!steps.some((s) => s.predict)) (strict(m.quality) ? err : warn)('AK407', 'no step asks the learner to predict first (predict)');
	if (!steps.some((s) => s.task)) (strict(m.quality) ? err : warn)('AK407', 'no step gives the learner a task (task)');
	if (!steps.some((s) => s.kindOf)) (strict(m.quality) ? err : warn)('AK407', "no honest-limits step (kindOf: 'limits' | 'breakIt')");
	if (strict(m.quality) && steps.length && steps.every((s) => s.origin === 'auto' && !s.edited)) err('AK409', `quality '${m.quality}' needs authored steps: every step is an unedited auto step`);
	if (!hasZh(lesson.title)) err('AK406', 'the lesson title has no zh');

	for (const s of steps) {
		const src = s.src;
		const at = { step: s.id };
		if (!src.title?.en) err('AK406', 'step has no en title', at);
		else if (!hasZh(src.title)) err('AK406', 'title has no zh', { ...at, lang: 'zh' });
		if (!provenanceOf(src)) err('AK502', 'step has no provenance: its stage would show no badge', { ...at, fix: "add provenance: 'live' | 'precomputed' | { mode: 'illustrative', note }" });
		else if (s.provenance.mode === 'illustrative' && !s.provenance.note) err('AK502', "provenance 'illustrative' needs a note saying what is made up", at);
		if (!sourceDef(m, s.source)) err('AK404', `source '${s.source}' is not declared (declared: ${sources.join(', ') || 'none'})`, at);
		if (!s.view) err('AK405', 'step has no view', at);
		else if (ctx.views && !ctx.views(s.view)) err('AK405', `view '${s.view}' is not registered`, at);
		for (const [field, tm] of [
			['narration', src.narration],
			['look', src.look]
		] as const) {
			if (!tm) continue;
			if (!tm.en?.trim()) err('AK406', `${field} has no en text`, at);
			if (!hasZh(tm)) (strict(m.quality) ? err : warn)('AK406', `${field} has no zh`, { ...at, lang: 'zh' });
			else {
				const en = placeholderNames(tm.en).sort();
				const zh = placeholderNames(tm.zh).sort();
				if (en.join('|') !== zh.join('|')) err('AK403', `${field}: en uses {${en.join('}, {')}} but zh uses {${zh.join('}, {')}}`, at);
			}
			for (const lang of ['en', 'zh'] as const) {
				const text = tm[lang];
				if (!text) continue;
				for (const p of placeholders(text)) if (p.fmt && !(FORMATS as readonly string[]).includes(p.fmt)) err('AK401', `${field}: {${p.name}|${p.fmt}} uses an unknown format (${FORMATS.join(', ')})`, { ...at, lang });
				const lits = literalNumbers(text);
				if (lits.length) warn('AK402', `${field} quotes the number${lits.length > 1 ? 's' : ''} ${lits.join(', ')} as text: use a placeholder bound to the live value or fact:<key>`, { ...at, lang });
				if (/\bTODO\b/.test(text)) err('AK501', `${field} still has a TODO marker`, { ...at, lang });
			}
			if (field === 'narration' && tm.en) {
				const w = wordCount(tm.en);
				const n = sentenceCount(tm.en, 'en');
				if (w > 90) warn('AK407', `narration has ${w} words (at most 90)`, at);
				if (n < 2 || n > 4) warn('AK407', `narration has ${n} sentence${n === 1 ? '' : 's'} (aim for 2-4)`, at);
			}
		}
		const glossary = new Set((m.glossary ?? []).map((g) => g.id));
		for (const id of s.terms) if (ctx.terms && !ctx.terms(id) && !glossary.has(id)) err('AK411', `term '${id}' is in neither the site glossary nor manifest.glossary`, { ...at, fix: 'use a glossary id (src/lib/catalog/glossary.ts), or add { id, term, def } to manifest.glossary' });
		const q: PredictQuestion | undefined = src.predict;
		// K4: predictAt gives the live options, answer and explanation; the question itself stays in predict
		if (src.predictAt && !q) err('AK407', 'predictAt needs predict (the question, and the options and answer it falls back to)', at);
		if (q) {
			// K1: the title and the look hint show before the answer too
			for (const [field, text] of [
				['narration', src.narration?.en],
				['look', src.look?.en],
				['title', src.title?.en]
			] as const) {
				const why = giveaway(text, q);
				if (why) warn('AK412', `the ${field} shows before the learner answers, and ${why}`, { ...at, fix: `set the question up in the ${field}; put the answer in predict.explain (shown after the answer)` });
			}
			const texts = [q.question, q.explain, ...q.options];
			if (texts.some((t) => !hasZh(t))) err('AK406', 'the predict question needs zh for its question, options and explanation', { ...at, lang: 'zh' });
			if (!(q.answer >= 0 && q.answer < q.options.length)) err('AK407', `predict answer ${q.answer} is not one of the ${q.options.length} options`, at);
		}
		if (src.task) {
			if (!hasZh(src.task.prompt)) err('AK406', 'the task prompt has no zh', { ...at, lang: 'zh' });
			if (src.task.hint && !hasZh(src.task.hint)) err('AK406', 'the task hint has no zh', { ...at, lang: 'zh' });
		}
		if (src.compare && src.compare.label.some((l) => !hasZh(l))) err('AK406', 'compare labels need zh', { ...at, lang: 'zh' });
		if (src.compare) for (const k of [src.compare.a, src.compare.b]) if (!sourceDef(m, k)) err('AK404', `compare source '${k}' is not declared`, at);
		if (src.provenance && typeof src.provenance === 'object' && src.provenance.note && !hasZh(src.provenance.note)) err('AK406', 'the provenance note has no zh', { ...at, lang: 'zh' });
	}
	// ------------------------------------------------------------------ R1 (redesign W2): chapters, recap, caps (warnings)
	const ids = steps.map((s) => s.id);
	let chapters: CompiledLesson['chapters'];
	if (lesson.chapters?.length) {
		const cs = lesson.chapters;
		const flat = cs.flatMap((c) => c.steps.map(stepSlug));
		if (cs.length < 2 || cs.length > 4) warn('AK413', `the lesson has ${cs.length} chapters (aim for 2-4)`);
		for (const c of cs) {
			if (!hasZh(c.title)) warn('AK413', `chapter '${c.id}' has no zh title`, { lang: 'zh' });
			if (c.steps.length < 1 || c.steps.length > 6) warn('AK413', `chapter '${c.id}' has ${c.steps.length} steps (1-6)`);
			for (const sid of c.steps) if (!ids.includes(stepSlug(sid))) warn('AK413', `chapter '${c.id}' names step '${sid}', which the lesson does not have`, { fix: `step ids: ${ids.join(', ')}` });
		}
		const missing = ids.filter((id) => !flat.includes(id));
		const twice = flat.filter((id, i) => flat.indexOf(id) !== i);
		if (missing.length) warn('AK413', `steps in no chapter: ${missing.join(', ')}`);
		if (twice.length) warn('AK413', `steps in two chapters: ${[...new Set(twice)].join(', ')}`);
		const order = flat.filter((id) => ids.includes(id)).map((id) => ids.indexOf(id));
		if (order.some((v, i) => i > 0 && v < order[i - 1])) warn('AK413', 'the chapters list the steps out of lesson order');
		chapters = cs.map((c) => ({ id: c.id, title: c.title, steps: c.steps.map(stepSlug).filter((id) => ids.includes(id)) }));
	}
	let recap: CompiledLesson['recap'];
	if (lesson.recap?.length) {
		if (lesson.recap.length > 3) warn('AK413', `the recap has ${lesson.recap.length} lines (at most 3)`);
		recap = lesson.recap.slice(0, 3).map((r) => (rt: ExplainerRuntime, at: At) => renderI18n(r as Narration<ExplainerRuntime>, rt, at));
	}
	for (const s of steps) {
		const src = s.src;
		const at = { step: s.id };
		const w = src.narration?.en ? wordCount(src.narration.en) : 0;
		if (w > 60 && w <= 90) warn('AK415', `narration has ${w} words (the content pass caps it at 60; move the rest into why)`, at);
		const lw = src.look?.en ? wordCount(src.look.en) : 0;
		if (lw > 25) warn('AK414', `the look line has ${lw} words (at most 25, naming one thing in the stage)`, at);
		for (const [field, text] of [
			['title', src.title?.en],
			['narration', src.narration?.en],
			['look', src.look?.en]
		] as const) {
			const typed = typedIndices(text);
			if (typed.length) warn('AK416', `the ${field} types the index ${typed.join(', ')} as a word: write W₀ (or <sub>0</sub>)`, at);
		}
		if (src.why) {
			const aw = src.why.a?.en ? wordCount(src.why.a.en) : 0;
			if (aw > 120) warn('AK417', `the Why answer has ${aw} words (at most 120)`, at);
			if (!/[?？]\s*$/.test(src.why.q?.en ?? '')) warn('AK417', 'the Why summary should be a question', at);
			if (!hasZh(src.why.q) || !hasZh(src.why.a as I18n)) warn('AK406', 'the Why needs zh', { ...at, lang: 'zh' });
		}
	}
	const poster = (m.page as { poster?: { step?: string } } | undefined)?.poster?.step;
	if (poster && !ids.includes(stepSlug(poster))) warn('AK418', `page.poster names step '${poster}', which the lesson does not have`, { fix: `step ids: ${ids.join(', ')}` });
	return { id: lesson.id ?? 'lesson', title: lesson.title, summary: lesson.summary, steps, issues, plan, ...(chapters ? { chapters } : {}), ...(recap ? { recap } : {}) };
}
