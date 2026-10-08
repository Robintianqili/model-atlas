// lintLesson(compiled, makeRuntime, fixtures) (design §4.6, §6.2, §6.3): every step × every fixture × en/zh, on a real
// runtime. Used by `atlas lesson lint`, `atlas check` and the Lesson Lab. Pure TS (no Node imports).
//
// Runtime rules: AK401 (a placeholder is unresolved or non-finite on some fixture, or a step function threw), AK404 (the
// cursor matches no frame on some fixture, or the step cannot be entered), AK410 (warning: two different numbers of one
// narration print the same, as "the loss falls from 0.699 to 0.699"; narrate the change itself or use |n4; K1: not for
// numbers that print as a saturated ±1.00, 0% or 100%), AK407 (K1, warning: the narration has more than 90 words once
// its placeholders are filled in). The static rules of compileLesson are included in the result.
// K2: AK401 also for a narration or look that prints "…" from a resolved value (a vars() fallback string), and on a step
// the empty input skipped (rendered as the page renders it: the sentences without numbers left out); a warning for a
// narration that prints "NaN", "undefined" or "null" from a vars() string (rawGap).
import type { I18n } from '../types/core.ts';
import type { FixtureDef, LintIssue, Narration } from '../types/lesson.ts';
import type { ExplainerRuntime, PackageModules } from '../types/runtime.ts';
import type { AtlasCompiledStep, CompiledLesson } from './compile.ts';
import { extraEllipses, formatValue, MISSING, PLACEHOLDER_RE, render, wordCount, type Lang, type Rendered } from './narrate.ts';
import { stepAt } from '../runtime/sources.ts';
import { withSource } from './compile.ts';

export interface FixtureCase extends FixtureDef {
	/** K2: 'start' is the page's own start input (manifest.input.start) when no other fixture is the same input */
	kind: 'lesson' | 'edge' | 'preset' | 'start';
}

const sameRaw = (a: unknown, b: unknown): boolean => {
	try {
		return JSON.stringify(a) === JSON.stringify(b);
	} catch {
		return false;
	}
};

/** Every fixture a lesson must survive: the lesson's own, the input adapter's edge fixtures, the presets, and (K2) the
 *  input the page starts with when none of those is the same input (a lesson had to repeat it as a fixture by hand). */
export function fixtureCases(pkg: PackageModules, presets: { id: string; label: I18n; raw: unknown }[] = []): FixtureCase[] {
	const out: FixtureCase[] = [];
	const ids = new Set<string>();
	const add = (f: FixtureCase) => {
		let id = f.id;
		for (let k = 2; ids.has(id); k++) id = `${f.id}#${k}`;
		ids.add(id);
		out.push({ ...f, id });
	};
	for (const f of pkg.lesson?.fixtures ?? []) add({ ...f, kind: 'lesson' });
	const adapters = pkg.manifest.inputs?.map((i) => pkg.adapters[i.adapter]) ?? (pkg.manifest.input ? [pkg.adapters[pkg.manifest.input.adapter]] : []);
	if (!pkg.manifest.inputs?.length) for (const a of adapters) for (const e of a?.edgeFixtures() ?? []) add({ id: `edge:${e.id}`, label: e.label, raw: e.raw, kind: 'edge' });
	for (const p of presets) add({ id: `preset:${p.id}`, label: p.label, raw: p.raw, kind: 'preset' });
	const start = pkg.manifest.input?.start;
	if (start !== undefined && !out.some((f) => sameRaw(f.raw, start))) add({ id: 'start', label: { en: 'the page’s start input', zh: '页面一开始的输入' }, raw: start, kind: 'start' });
	return out;
}

export interface LintOpts {
	langs?: Lang[];
	/** stop after this many issues (default 500) */
	max?: number;
}

/** Run every step on every fixture. `makeRuntime` builds a fresh runtime with the fixture's raw input. */
export async function lintLesson(
	compiled: CompiledLesson | AtlasCompiledStep[],
	makeRuntime: (fx: FixtureDef) => Promise<ExplainerRuntime> | ExplainerRuntime,
	fixtures: FixtureDef[],
	o: LintOpts = {}
): Promise<LintIssue[]> {
	const steps = Array.isArray(compiled) ? compiled : compiled.steps;
	const issues: LintIssue[] = Array.isArray(compiled) ? [] : [...compiled.issues];
	const langs = o.langs ?? ['en', 'zh'];
	const max = o.max ?? 500;
	const keyOf = (i: LintIssue) => `${i.code}|${i.step}|${i.fixture}|${i.lang}|${i.message}`;
	const have = new Set(issues.map(keyOf));
	const push = (i: LintIssue) => {
		const k = keyOf(i);
		if (have.has(k) || issues.length >= max) return;
		have.add(k);
		issues.push(i);
	};
	for (const fx of fixtures) {
		let rt: ExplainerRuntime;
		try {
			rt = await makeRuntime(fx);
		} catch (e) {
			push({ code: 'AK404', level: 'error', fixture: fx.id, message: `the runtime could not start on this fixture: ${e instanceof Error ? e.message : String(e)}` });
			continue;
		}
		if (fx.expect?.empty !== undefined && rt.pre && rt.pre.empty !== fx.expect.empty)
			push({ code: 'AK404', level: 'error', fixture: fx.id, message: `the fixture expects empty = ${fx.expect.empty}, the adapter says ${rt.pre.empty}` });
		for (const step of steps) {
			const pos = await stepAt(rt, step);
			const where = { step: step.id, fixture: fx.id };
			if (pos.error) {
				push({ code: 'AK404', level: 'error', ...where, message: `the step cannot be entered: ${pos.error}` });
				continue;
			}
			if (pos.skipped) {
				// empty input: the stage shows the adapter's empty note. K2: the narration above it must not print "…"
				// either (its sentences without numbers are left out; a vars() string or a literal can still say "…")
				for (const lang of langs)
					for (const [field, tm] of [
						['narration', step.src.narration],
						['look', step.src.look]
					] as const) {
						if (!tm) continue;
						const r = render(tm as Narration<ExplainerRuntime>, rt, withSource(pos.at, step.source), lang, { emptyLine: field === 'narration' });
						if (extraEllipses(r.text, (lang === 'zh' ? tm.zh : undefined) ?? tm.en) > 0)
							push({ code: 'AK401', level: 'error', ...where, lang, message: `${field} on the empty input prints "…": ${JSON.stringify(r.text.slice(0, 120))}`, fix: 'return a value or an I18n text that reads well without numbers, not "…", from vars() on an empty input' });
					}
				continue;
			}
			if (!pos.found) {
				const len = rt.sources[step.source]?.tape?.length ?? 0;
				push({ code: 'AK404', level: 'error', ...where, message: `cursor ${JSON.stringify(pos.cursor)} matches no frame of '${step.source}' (${len} frames): the step would silently show frame ${pos.index}`, fix: 'use a cursor query ({ phase, step, where, last }) instead of a fixed index, or clamp it in at(rt)' });
			}
			for (const lang of langs) {
				for (const [field, tm] of [
					['narration', step.src.narration],
					['look', step.src.look]
				] as const) {
					if (!tm) continue;
					const r = render(tm as Narration<ExplainerRuntime>, rt, withSource(pos.at, step.source), lang);
					if (r.error) push({ code: 'AK401', level: 'error', ...where, lang, message: `${field}: vars() threw: ${r.error}` });
					if (lang === 'en' && field === 'narration' && !r.missing.length) {
						// K1: the 90-word limit on what the learner reads (compile counts a placeholder as one word)
						const w = wordCount(r.text);
						if (w > 90 && wordCount(tm.en) <= 90) push({ code: 'AK407', level: 'warn', step: step.id, message: `narration has ${w} words once its numbers are filled in (at most 90)` });
					}
					if (lang === 'en') {
						const same = samePrint(r);
						if (same)
							push({
								code: 'AK410',
								level: 'warn',
								...where,
								message: `${field}: {${same.a}} = ${same.va} and {${same.b}} = ${same.vb} both print as "${same.text}", so the learner sees no difference`,
								fix: 'narrate the change itself (a vars() entry such as drop: before - after), or give both more digits with {name|n4}'
							});
					}
					for (const name of r.missing) {
						const ph = r.placeholders.find((p) => p.name === name);
						const v = ph?.value;
						const why = v === undefined || v === null ? 'unresolved' : typeof v === 'number' ? `non-finite (${v})` : `not displayable (${JSON.stringify(v).slice(0, 40)})`;
						push({ code: 'AK401', level: 'error', ...where, lang, message: `${field}: {${name}} is ${why}, so the learner would read "…"` });
					}
					// K2: a value that resolved but prints "…" itself (a vars() fallback string) reads exactly like a missing one
					if (!r.missing.length && extraEllipses(r.text, (lang === 'zh' ? tm.zh : undefined) ?? tm.en) > 0) {
						const ph = r.placeholders.find((p) => typeof p.value === 'string' ? p.value.includes(MISSING) : !!p.value && typeof p.value === 'object' && JSON.stringify(p.value).includes(MISSING));
						push({ code: 'AK401', level: 'error', ...where, lang, message: `${field}: ${ph ? `{${ph.name}}` : 'a value'} prints "…", so the learner reads a gap: ${JSON.stringify(r.text.slice(0, 120))}`, fix: 'compute the value for this fixture, or narrate the case without it (pick() in vars())' });
					}
					// K2 (warning): a vars() string built with a raw number prints "NaN" or "undefined" (the formats would
					// have printed "…" and failed)
					const bad = rawGap(r.text, (lang === 'zh' ? tm.zh : undefined) ?? tm.en);
					if (!r.missing.length && bad)
						push({ code: 'AK401', level: 'warn', ...where, lang, message: `${field} prints ${bad}: ${JSON.stringify(r.text.slice(0, 120))}`, fix: 'format numbers with num()/formatValue() in vars() (never `${x}`), and narrate a missing value with pick()' });
				}
			}
			for (const [what, fn] of [
				['focus', () => step.focus(rt, pos.at)],
				['props', () => step.props(rt, pos.at)],
				['task.done', () => step.task?.done(rt, pos.at)]
			] as const) {
				try {
					fn();
				} catch (e) {
					push({ code: 'AK401', level: 'error', ...where, message: `${what}() threw: ${e instanceof Error ? e.message : String(e)}` });
				}
			}
		}
		(rt as { close?: () => void }).close?.();
	}
	return issues;
}

/** K2: what a rendered narration prints that its template does not and no learner should read: "NaN", "undefined" or
 *  "null" (a raw number or a missing field in a vars() string). Null when there is none. */
export function rawGap(rendered: string, template: string | undefined): string | null {
	const tpl = (template ?? '').replace(PLACEHOLDER_RE, ' ');
	for (const w of ['NaN', 'undefined', 'null']) {
		const re = new RegExp(`(?<![A-Za-z])${w}(?![A-Za-z])`, 'g');
		if ((rendered.match(re) ?? []).length > (tpl.match(re) ?? []).length) return `"${w}"`;
	}
	return null;
}

/** Two placeholders of one rendered narration whose numbers differ but print the same text (rounded formats only:
 *  `|int` rounds on purpose), unless the narration also states their difference (a `drop` placeholder: "falls by
 *  3.76e−4 to 0.699" is the documented remedy). Returns the first such pair. */
export function samePrint(r: Rendered): { a: string; b: string; va: number; vb: number; text: string } | null {
	const nums: { name: string; v: number; text: string | null; rounded: boolean }[] = [];
	const names = new Set<string>();
	for (const p of r.placeholders) {
		if (!p.ok || typeof p.value !== 'number' || !Number.isFinite(p.value) || names.has(p.name)) continue;
		names.add(p.name);
		const exact = p.fmt === 'int' || (Number.isInteger(p.value) && p.fmt === undefined); // printed exactly, or rounded on purpose
		nums.push({ name: p.name, v: p.value, text: exact ? null : formatValue(p.value, p.fmt, r.lang), rounded: !exact });
	}
	const close = (x: number, y: number) => Math.abs(x - y) <= 1e-9 * Math.max(Math.abs(x), Math.abs(y), 1e-300);
	for (let i = 0; i < nums.length; i++)
		for (let j = i + 1; j < nums.length; j++) {
			const a = nums[i];
			const b = nums[j];
			if (!a.rounded || !b.rounded || a.text === null || a.text !== b.text || close(a.v, b.v)) continue;
			if (saturated(a.text)) continue; // K1: two saturated gates (i and g both 1.00) are not a change that was hidden
			const d = a.v - b.v;
			if (nums.some((c, k) => k !== i && k !== j && (close(c.v, d) || close(c.v, -d)))) continue; // the change is narrated
			return { a: a.name, b: b.name, va: a.v, vb: b.v, text: a.text };
		}
	return null;
}

/** A printed value at a saturation bound: ±1.00 (a gate, a tanh), 0% or 100% (a share), "<0.1%". Two different
 *  quantities that both saturate print the same on purpose (K1: AK410 used to flag them). */
export function saturated(text: string): boolean {
	const t = text.replace(/−/g, '-').trim();
	return /^-?1(\.0+)?$/.test(t) || /^(100(\.0+)?|0(\.0+)?|<0\.1|>99\.9)\s?%$/.test(t);
}

export function errorsOf(issues: LintIssue[]): LintIssue[] {
	return issues.filter((i) => i.level === 'error');
}

/** One line per issue, in the CLI's error format (design §4.12). */
export function formatIssues(issues: LintIssue[]): string {
	return issues
		.map((i) => {
			const where = [i.step && `step "${i.step}"`, i.fixture && `fixture "${i.fixture}"`, i.lang].filter(Boolean).join(', ');
			return `${i.code}  ${i.level === 'warn' ? '(warn) ' : ''}${i.message}${where ? `\n  where   ${where}` : ''}${i.fix ? `\n  fix     ${i.fix}` : ''}\n  docs    site/DEVKIT.md#${i.code.toLowerCase()}`;
		})
		.join('\n');
}
