// Concept packages (K2, LOG.md "Continue completing the atlas"): the pure part, shared by the page (AutoConceptPage), the
// Node lint of the panels (lint.ts) and the tests. Nothing here imports Svelte, the kit or a Vite alias, so Node loads it.
//
// A concept package is an ordinary package folder, src/lib/explainers/<concept-slug>/ (the same loaders, sources, lesson,
// views, workers, lint and gates as a model), with two differences:
//   explainer.ts  declares type: 'concept' (the slug is a catalog CONCEPT slug, src/lib/catalog/concepts.ts);
//   concept.ts    the page around the lesson, in ConceptPageShell order: hook, toy, real, compare, pitfalls, further.
//                 Each of hook (or a plain I18n text), toy, real and compare is a StepDef: the lesson's own format, played
//                 on the same runtime and sources as the lesson, but as a free-standing stage (no Walkthrough, no Play).
// The route /train/[slug] renders AutoConceptPage for an included concept package and the Planned page otherwise.
//
// Until a types revision adds them to the frozen types (proposed in notes/k2-concepts.md), the concept types live here and
// a package's explainer.ts spells its manifest type as Omit<ExplainerManifest, 'type'> & { type: 'concept' }.
import type { I18n } from '../../types/core.ts';
import type { ExplainerManifest } from '../../types/explainer.ts';
import type { LessonDef, StepDef } from '../../types/lesson.ts';
import type { Cursor } from '../../types/process.ts';
import type { ExplainerRuntime } from '../../types/runtime.ts';
import type { AtlasCompiledStep } from '../../lesson/compile.ts';
import { atFor, type StepPosition } from '../../runtime/sources.ts';
import { resolveCursor } from '../../runtime/selection.ts';

/** The stage panels of a concept page, in page order. */
export const PANELS = ['hook', 'toy', 'real', 'compare'] as const;
export type PanelKey = (typeof PANELS)[number];

/** A misconception and what is true instead; `step` names the lesson step that shows it (a link). */
export interface Pitfall {
	claim: I18n;
	truth: I18n;
	step?: string;
}

/** concept.ts (default export, or the same names as named exports). Every field is optional. */
export interface ConceptDef {
	/** the question that opens the page: a short text, or a stage (a StepDef) */
	hook?: I18n | StepDef<any>;
	/** every variable visible: the learner's input widget (the manifest's input adapter) is shown above this stage */
	toy?: StepDef<any>;
	/** the same idea on a real model (a live model's assets, or a bigger network trained live) */
	real?: StepDef<any>;
	/** two runs side by side (a step `compare`, or a view with props.compare) */
	compare?: StepDef<any>;
	/** misconceptions; manifest.page.misconception comes first */
	pitfalls?: Pitfall[];
	/** further reading under the links (absolute URLs, or site paths starting with '/') */
	further?: { label: I18n; href: string }[];
}

/** The page's plain text, embedded in the page data at build time (serializable: no functions), so the prerendered HTML
 *  already shows the goals, the predict-first question, the hook text, the pitfalls, the checkpoint and the links. */
export interface ConceptText {
	page: ExplainerManifest['page'];
	credits: ExplainerManifest['credits'];
	hook: I18n | null;
	pitfalls: Pitfall[];
	further: NonNullable<ConceptDef['further']>;
	/** the stage panels concept.ts has (so the prerendered page lays out the same sections) */
	panels: PanelKey[];
	/** the manifest has an input adapter (the toy section shows its widget) */
	input: boolean;
}

/** The manifest of a concept package (until the frozen ExplainerManifest.type admits 'concept'). */
export type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

export function isConceptManifest(m: { type?: unknown } | null | undefined): boolean {
	return !!m && m.type === 'concept';
}

const isStep = (v: unknown): v is StepDef<any> => !!v && typeof v === 'object' && 'view' in (v as object) && 'narration' in (v as object);

/** The ConceptDef of a loaded concept.ts module: its default export, else its named exports. */
export function conceptOf(mod: Record<string, unknown> | null | undefined): ConceptDef {
	if (!mod) return {};
	const d = (mod.default ?? mod.concept ?? mod) as Record<string, unknown>;
	const out: ConceptDef = {};
	if (d.hook && (isStep(d.hook) || typeof (d.hook as I18n).en === 'string')) out.hook = d.hook as ConceptDef['hook'];
	for (const k of ['toy', 'real', 'compare'] as const) if (isStep(d[k])) out[k] = d[k] as StepDef<any>;
	if (Array.isArray(d.pitfalls)) out.pitfalls = d.pitfalls as Pitfall[];
	if (Array.isArray(d.further)) out.further = d.further as ConceptDef['further'];
	return out;
}

/** The stage panels as one LessonDef (ids forced to the panel keys), so compileLesson and lintLesson treat them exactly
 *  like lesson steps: placeholders, views, provenance, zh, terms. Its id 'concept' keeps them out of the lesson's deep
 *  links. A hook given as plain text is not a stage and is left out. */
export function panelLesson(def: ConceptDef, manifest: { slug: string }): LessonDef<any> & { keys: PanelKey[] } {
	const keys: PanelKey[] = [];
	const steps: StepDef<any>[] = [];
	for (const k of PANELS) {
		const s = def[k];
		if (!isStep(s)) continue;
		keys.push(k);
		steps.push({ ...s, id: k });
	}
	return { id: 'concept', title: { en: manifest.slug, zh: manifest.slug }, summary: { en: '', zh: '' }, steps, keys };
}

/** Lint codes that judge a lesson's shape (step count, predict, task, limits steps, auto steps): they do not apply to the
 *  page's free-standing panels. */
export const SHAPE_ONLY = new Set(['AK407', 'AK409']);

/** The stage text of a hook given as plain text, else null. */
export function hookText(def: ConceptDef): I18n | null {
	return def.hook && !isStep(def.hook) ? (def.hook as I18n) : null;
}

/** The page's pitfalls: the manifest's misconception first, then concept.ts's. */
export function pitfallsOf(def: ConceptDef, manifest: Pick<ExplainerManifest, 'page'> | null): Pitfall[] {
	const mc = manifest?.page.misconception;
	return [...(mc ? [mc] : []), ...(def.pitfalls ?? [])];
}

/** Resolve a panel's moment WITHOUT touching the lesson selection (stepAt() replaces rt.sel, which belongs to the lesson
 *  step on screen): run its source (and a compare's second source) to the cursor and build the At. A panel has no
 *  `select`; one that needs a unit explanation uses a scan source. */
export async function panelPosition(rt: ExplainerRuntime, step: Pick<AtlasCompiledStep, 'at' | 'source'> & Partial<Pick<AtlasCompiledStep, 'compare'>>): Promise<StepPosition> {
	let cursor: Cursor | null = null;
	try {
		cursor = step.at(rt);
		await rt.ensure(step.source, cursor);
		for (const s of step.compare ? [step.compare.a, step.compare.b] : []) {
			if (s === step.source) continue;
			try {
				await rt.ensure(s, cursor);
			} catch {
				/* the compared side shows its own error */
			}
		}
	} catch (e) {
		return { at: { index: -1 }, cursor, found: false, index: -1, skipped: false, error: e instanceof Error ? e.message : String(e) };
	}
	const st = rt.sources[step.source];
	if (!st?.tape || st.tape.length === 0) {
		if (rt.pre?.empty) return { at: { index: -1 }, cursor, found: false, index: -1, skipped: true };
		return { at: { index: -1 }, cursor, found: false, index: -1, skipped: false, error: `source '${step.source}' has no frames (status ${st?.status ?? 'missing'}${st?.error ? `: ${st.error}` : ''})` };
	}
	const r = resolveCursor(st.tape, cursor);
	return { at: atFor(rt, step.source, r.index), cursor, found: r.found, index: r.index, skipped: false };
}

/** A site path or an absolute URL for a `further` link (site paths get the base prefix). */
export function furtherHref(href: string, base: string): string {
	return href.startsWith('/') ? `${base}${href}` : href;
}

/** The catalog status a concept page needs to be routed: drafts only while they are included (the dev server, or
 *  ATLAS_INCLUDE_DRAFTS=1); devOnly packages never. */
export function conceptIncluded(m: { type?: unknown; devOnly?: boolean } | null | undefined, status: 'live' | 'planned' | undefined, drafts: boolean): boolean {
	if (!m || m.devOnly || !isConceptManifest(m) || !status) return false;
	return status === 'live' || drafts;
}

