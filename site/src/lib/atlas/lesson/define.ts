// Authoring helpers for explainer.ts and lesson.ts (design §4.6, §5.4). Identity functions that give contributors
// type inference, plus the manifest defaults every other module uses.
import type { AutoRef, LessonDef, StepDef } from '../types/lesson.ts';
import type { ExplainerManifest } from '../types/explainer.ts';
import type { ExplainerRuntime, SourceDef } from '../types/runtime.ts';

export function defineLesson<R extends ExplainerRuntime = ExplainerRuntime>(def: LessonDef<R>): LessonDef<R> {
	return def;
}

/** Use one planned step ('input', 'op:<layer>', 'group:<id>', 'output', 'train:<motif>@<source>',
 *  'proc:event.<phase>@<source>', 'proc:run@<source>'), optionally overriding any StepDef field. */
export function auto<R extends ExplainerRuntime = ExplainerRuntime>(id: string, override?: Partial<StepDef<R>>): AutoRef<R> {
	return override ? { auto: id, override } : { auto: id };
}

/** Every planned step from `from` (inclusive) up to `until` (exclusive), minus `skip` and minus any id the lesson
 *  also uses through auto(). */
export function autoAll<R extends ExplainerRuntime = ExplainerRuntime>(o: { from?: string; until?: string; skip?: string[] } = {}): AutoRef<R> {
	return { autoAll: o };
}

export function defineExplainer(m: ExplainerManifest): ExplainerManifest {
	return m;
}

export function isAutoRef<R extends ExplainerRuntime>(s: StepDef<R> | AutoRef<R>): s is AutoRef<R> {
	return 'auto' in s || 'autoAll' in s;
}

/** The manifest's sources, with the default { fwd: { kind: 'trace' } } when kinds has 'trace' and none are declared. */
export function sourcesOf(m: ExplainerManifest): Record<string, SourceDef> {
	if (m.sources && Object.keys(m.sources).length) return m.sources;
	return m.kinds.includes('trace') ? { fwd: { kind: 'trace' } } : {};
}

/** Implicit scan sources: a step may name 'scan:<layer>' without declaring it; it scans <layer> of the first trace source. */
export const SCAN_PREFIX = 'scan:';

export function sourceDef(m: ExplainerManifest, name: string): SourceDef | undefined {
	const all = sourcesOf(m);
	if (all[name]) return all[name];
	if (name.startsWith(SCAN_PREFIX)) {
		const of = Object.entries(all).find(([, d]) => d.kind === 'trace')?.[0];
		if (of) return { kind: 'scan', of, layer: name.slice(SCAN_PREFIX.length) };
	}
	return undefined;
}

/** Deep-link safe step id: 'op:L0.attn.q' -> 'op-L0-attn-q', 'train:update@sgd1' -> 'train-update-sgd1'. */
export function stepSlug(id: string): string {
	return id.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
}
