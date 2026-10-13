// The browser package loader (design §4.7): src/lib/explainers/<slug>/ -> PackageModules, built from Vite globs. Every glob
// is lazy, so each package (and each kit adapter) is its own chunk and only the visited package loads. The rules are the
// Node loader's (scripts/atlas/load.ts), so the browser and `atlas` CLI build the same object:
//   explainer.ts  default export (or `manifest`); its slug must equal the folder
//   lesson.ts     default export (or `lesson`); optional
//   ops.ts        default export OpDef[]; else CORE_OPS
//   process.ts    every ProcessDef export, registered under its `id`
//   trainer.ts    every TrainerDef export, wrapped with trainingProcess() and registered as 'train:<id>'
//   adapters      kit adapters (atlas/adapters/<id>/adapter.ts) + the package's adapters/<id>/adapter.ts (a local adapter
//                 with a kit id replaces the kit one), keyed by InputAdapter.id
//   worker.ts     the package's worker (Vite ?worker), exposed as PackageModules.worker
// Also: Capture components (kit + package adapters), section overrides (sections/<Snippet>.svelte) and the article
// (content/En.svelte, content/Zh.svelte), all lazy.
import type { Component } from 'svelte';
import type { ExplainerManifest, InputAdapter } from '../types/explainer.ts';
import type { LessonDef } from '../types/lesson.ts';
import type { OpDef } from '../types/op.ts';
import type { ProcessDef } from '../types/process.ts';
import type { TrainerDef } from '../types/train.ts';
import type { PackageModules, SourceRunner } from '../types/runtime.ts';
import { prepareViews } from './registry.ts';

type Mod = Record<string, unknown>;
type Loader<T = Mod> = () => Promise<T>;

const explainers = import.meta.glob<Mod>('../../explainers/*/explainer.ts');
const lessons = import.meta.glob<Mod>('../../explainers/*/lesson.ts');
const opsMods = import.meta.glob<Mod>('../../explainers/*/ops.ts');
const processMods = import.meta.glob<Mod>('../../explainers/*/process.ts');
const trainerMods = import.meta.glob<Mod>('../../explainers/*/trainer.ts');
const pkgAdapters = import.meta.glob<Mod>('../../explainers/*/adapters/*/adapter.ts');
const kitAdapters = import.meta.glob<Mod>('../adapters/*/adapter.ts');
const workers = import.meta.glob<new () => Worker>('../../explainers/*/worker.ts', { query: '?worker', import: 'default' });
const kitCaptures = import.meta.glob<Component<any>>('../adapters/*/Capture.svelte', { import: 'default' });
const pkgCaptures = import.meta.glob<Component<any>>('../../explainers/*/adapters/*/Capture.svelte', { import: 'default' });
const sectionMods = import.meta.glob<Component<any>>('../../explainers/*/sections/*.svelte', { import: 'default' });
const contentMods = import.meta.glob<Component<any>>('../../explainers/*/content/*.svelte', { import: 'default' });
// atlas parity --write's per-source record (K0): the stage badge of train and process steps reads it
const parityMods = import.meta.glob<PackageModules['parity']>('../../explainers/*/parity.json', { import: 'default' });
// W1-B2's engine runner, when this tree has it (optional seam: no import error before integration 1b)
const engineMods = import.meta.glob<Mod>('../engine/runner.ts');

const pkgPath = (slug: string, file: string) => `../../explainers/${slug}/${file}`;

/** Slugs with an explainer.ts (devOnly ones included; callers decide what to route). */
export function packageSlugs(): string[] {
	return Object.keys(explainers)
		.map((p) => /^\.\.\/\.\.\/explainers\/([^/]+)\/explainer\.ts$/.exec(p)?.[1])
		.filter((s): s is string => !!s)
		.sort();
}

export function hasPackage(slug: string): boolean {
	return !!explainers[pkgPath(slug, 'explainer.ts')];
}

const isProcessDef = (v: unknown): v is ProcessDef<any, any, any> =>
	!!v && typeof v === 'object' && typeof (v as ProcessDef).run === 'function' && typeof (v as ProcessDef).id === 'string' && !!(v as ProcessDef).phases && !!(v as ProcessDef).schema;
const isTrainerDef = (v: unknown): v is TrainerDef<any, any, any, any> =>
	!!v && typeof v === 'object' && typeof (v as TrainerDef).id === 'string' && typeof (v as TrainerDef).model === 'function' && typeof (v as TrainerDef).batch === 'function' && typeof (v as TrainerDef).loss === 'function';
const isAdapter = (v: unknown): v is InputAdapter => !!v && typeof v === 'object' && typeof (v as InputAdapter).id === 'string' && typeof (v as InputAdapter).preprocess === 'function';

async function adaptersFrom(globbed: Record<string, Loader>, prefix: string, notes: string[]): Promise<Record<string, InputAdapter>> {
	const out: Record<string, InputAdapter> = {};
	const paths = Object.keys(globbed).filter((p) => p.startsWith(prefix)).sort();
	const mods = await Promise.all(paths.map((p) => globbed[p]()));
	paths.forEach((p, i) => {
		const a = (mods[i].default ?? mods[i].adapter) as unknown;
		if (!isAdapter(a)) {
			notes.push(`${p}: no InputAdapter default export`);
			return;
		}
		out[a.id] = a;
	});
	return out;
}

/** The kit adapters (drawing28, text-bpe, text-char, and W1-B2's points2d / series once merged). */
export async function kitAdapterSet(notes: string[] = []): Promise<Record<string, InputAdapter>> {
	return adaptersFrom(kitAdapters, '../adapters/', notes);
}

let coreOps: Promise<OpDef[]> | null = null;
export function loadCoreOps(): Promise<OpDef[]> {
	coreOps ??= import('../ops/index.ts').then((m) => m.CORE_OPS);
	return coreOps;
}

export interface LoadedPackage {
	pkg: PackageModules;
	notes: string[];
}

/** The package for `slug`, or null when there is none. Throws when a package file fails to load. Its view metas are in the
 *  registry when it resolves (prepareViews: hasView / viewMeta / loadView stay synchronous for every caller). */
export async function loadPackage(slug: string): Promise<LoadedPackage | null> {
	const em = explainers[pkgPath(slug, 'explainer.ts')];
	if (!em) return null;
	const notes: string[] = [];
	const emod = await em();
	const manifest = (emod.default ?? emod.manifest) as ExplainerManifest | undefined;
	if (!manifest || typeof manifest !== 'object' || !('slug' in manifest)) throw new Error(`${slug}: explainer.ts must default-export its ExplainerManifest`);
	if (manifest.slug !== slug) throw new Error(`${slug}: manifest slug '${manifest.slug}' differs from the folder`);
	const get = async (globbed: Record<string, Loader>, file: string): Promise<Mod | null> => {
		const l = globbed[pkgPath(slug, file)];
		return l ? l() : null;
	};
	const [lm, om, pm, tm] = await Promise.all([get(lessons, 'lesson.ts'), get(opsMods, 'ops.ts'), get(processMods, 'process.ts'), get(trainerMods, 'trainer.ts'), prepareViews(slug)]);
	const lesson = lm ? (((lm.default ?? lm.lesson) as LessonDef<any> | undefined) ?? null) : null;
	if (lm && !lesson) notes.push('lesson.ts has no default export');
	let ops: OpDef[];
	if (om) {
		ops = (om.default ?? om.ops) as OpDef[];
		if (!Array.isArray(ops)) throw new Error(`${slug}: ops.ts must default-export an OpDef[] ([...CORE_OPS, ...local])`);
	} else ops = await loadCoreOps();
	const processes: Record<string, ProcessDef<any, any, any>> = {};
	const add = (key: string, p: ProcessDef<any, any, any>) => {
		if (processes[key]) throw new Error(`${slug}: two processes are registered as '${key}'`);
		processes[key] = p;
	};
	// an export name that names its id (K1; the same rule as scripts/atlas/load.ts sameName)
	const norm = (s: string) => s.replace(/[^A-Za-z0-9]/g, '').toLowerCase();
	const sameName = (name: string, id: string) => name === id || name === 'default' || norm(name) === norm(id) || norm(name) === norm(id.replace(/^train:/, ''));
	if (pm) for (const [name, v] of Object.entries(pm)) if (isProcessDef(v)) {
		if (!sameName(name, v.id)) notes.push(`process.ts: export '${name}' has id '${v.id}'`);
		add(v.id, v);
	}
	if (tm) {
		const trainers = Object.entries(tm).filter(([, v]) => isTrainerDef(v)) as [string, TrainerDef<any, any, any, any>][];
		if (trainers.length) {
			const { trainingProcess } = await import('../train/trainer.ts');
			for (const [name, t] of trainers) {
				if (!sameName(name, t.id)) notes.push(`trainer.ts: export '${name}' has id '${t.id}'`);
				add(`train:${t.id}`, trainingProcess(t) as ProcessDef<any, any, any>);
			}
		}
	}
	const adapters = { ...(await kitAdapterSet(notes)), ...(await adaptersFrom(pkgAdapters, pkgPath(slug, 'adapters/'), notes)) };
	const pkg: PackageModules = { slug, manifest, lesson, ops, processes, adapters };
	const pl = parityMods[pkgPath(slug, 'parity.json')];
	if (pl) {
		const pr = await pl();
		if (pr && pr.format === 'atlas-parity/1') pkg.parity = pr;
	}
	const wl = workers[pkgPath(slug, 'worker.ts')];
	if (wl) {
		try {
			const W = await wl();
			pkg.worker = () => new W();
		} catch (e) {
			notes.push(`worker.ts did not load: ${e instanceof Error ? e.message : String(e)}`);
		}
	}
	return { pkg, notes };
}

/** A package built around a manifest that has no folder (the Lesson Lab's ?spec= on shipped assets). */
export async function syntheticPackage(manifest: ExplainerManifest): Promise<PackageModules> {
	return { slug: manifest.slug, manifest, lesson: null, ops: await loadCoreOps(), processes: {}, adapters: await kitAdapterSet() };
}

/** The Capture component for an adapter id: the package's own first, then the kit's. */
export async function loadCapture(slug: string, adapterId: string): Promise<Component<any> | null> {
	const l = pkgCaptures[pkgPath(slug, `adapters/${adapterId}/Capture.svelte`)] ?? kitCaptures[`../adapters/${adapterId}/Capture.svelte`];
	return l ? l() : null;
}

export const SECTION_NAMES = ['Facts', 'Input', 'Lesson', 'Overview', 'Inspector', 'Unit', 'Output', 'TryThis', 'Training', 'Article', 'Code', 'Links', 'MoreLinks', 'Checkpoint', 'Credits'] as const;
export type SectionName = (typeof SECTION_NAMES)[number];

/** Which sections/<Snippet>.svelte overrides the package has (names only, nothing loaded). */
export function sectionOverrides(slug: string): SectionName[] {
	return SECTION_NAMES.filter((n) => !!sectionMods[pkgPath(slug, `sections/${n}.svelte`)]);
}

export function loadSection(slug: string, name: SectionName): Promise<Component<any>> | null {
	const l = sectionMods[pkgPath(slug, `sections/${name}.svelte`)];
	return l ? l() : null;
}

/** The article components (content/En.svelte, content/Zh.svelte), when the package has them. */
export async function loadArticle(slug: string): Promise<{ en: Component<any>; zh?: Component<any> } | null> {
	const en = contentMods[pkgPath(slug, 'content/En.svelte')];
	if (!en) return null;
	const zh = contentMods[pkgPath(slug, 'content/Zh.svelte')];
	return { en: await en(), zh: zh ? await zh() : undefined };
}

export function hasArticle(slug: string): boolean {
	return !!contentMods[pkgPath(slug, 'content/En.svelte')];
}

/** W1-B2's engineRunner when this tree has engine/runner.ts and the package has a worker; else undefined, and
 *  createRuntime falls back to localRunner (main thread). The export may be a SourceRunner or a factory of one. */
export async function engineRunnerFor(pkg: PackageModules): Promise<SourceRunner | undefined> {
	const l = engineMods['../engine/runner.ts'];
	if (!l || !pkg.worker) return undefined;
	try {
		const m = await l();
		const r = m.engineRunner ?? m.default;
		if (r && typeof (r as SourceRunner).open === 'function') return r as SourceRunner;
		if (typeof r === 'function') {
			const made = (r as () => unknown)();
			if (made && typeof (made as SourceRunner).open === 'function') return made as SourceRunner;
		}
	} catch (e) {
		console.warn('atlas: engine runner unavailable, training runs on the main thread', e);
	}
	return undefined;
}
