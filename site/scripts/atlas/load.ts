// The Node package loader (design §4.7): src/lib/explainers/<slug>/ -> PackageModules, by dynamic import of paths.
// The browser loader (ui/packages.ts, W1-D1) builds the same object from Vite globs and must follow the same rules:
//
//   explainer.ts  default export (or `manifest`): the ExplainerManifest. Required.
//   lesson.ts     default export (or `lesson`): the LessonDef. Optional (null).
//   ops.ts        default export: OpDef[]. Optional; else CORE_OPS from atlas/ops/index.ts (W1-A).
//   process.ts    every export that is a ProcessDef, registered under its `id`.
//   trainer.ts    every export that is a TrainerDef, wrapped with trainingProcess() (atlas/train/trainer.ts, W1-B1)
//                 and registered as process 'train:<id>'. A source { kind: 'train', trainer: 'x' } runs 'train:x'.
//   adapters      kit adapters (atlas/adapters/<id>/adapter.ts) + the package's adapters/<id>/adapter.ts, default export
//                 (or `adapter`), keyed by InputAdapter.id; a local adapter with a kit id replaces the kit one.
//   worker.ts     browser only (never imported in Node).
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { ExplainerManifest, InputAdapter, LessonDef, OpDef, PackageModules, ProcessDef, TrainerDef } from '../../src/lib/atlas/types/index.ts';
import { AtlasError } from './errors.ts';
import { posix } from './lib/io.ts';
import { roots } from './root.ts';

export interface LoadedPackage {
	pkg: PackageModules;
	dir: string;
	/** package files found, by role */
	files: string[];
	/** things the loader could not do (a kit module missing in this tree, an export name that differs from its id, ...) */
	notes: string[];
}

type Mod = Record<string, unknown>;

const isDir = (p: string) => existsSync(p) && statSync(p).isDirectory();

export function explainersDir(site = roots().site): string {
	return join(site, 'src', 'lib', 'explainers');
}

/** Slugs of every package folder (a folder with explainer.ts). */
export function listPackages(site = roots().site): string[] {
	const d = explainersDir(site);
	if (!isDir(d)) return [];
	return readdirSync(d)
		.filter((n) => existsSync(join(d, n, 'explainer.ts')))
		.sort();
}

export async function importTs(file: string, role: string): Promise<Mod> {
	try {
		return (await import(pathToFileURL(file).href)) as Mod;
	} catch (e) {
		const err = e as NodeJS.ErrnoException;
		const msg = (err.message ?? String(e)).split('\n')[0];
		const typeSyntax = err.code === 'ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX' || /TypeScript|strip/i.test(msg);
		const missing = err.code === 'ERR_MODULE_NOT_FOUND';
		throw new AtlasError(missing || typeSyntax ? 'AK503' : 'AK301', `${role} does not load in Node: ${msg}`, {
			where: posix(file),
			likely: missing
				? 'an import that only Vite resolves ($lib/..., $app/..., a .svelte file) or a kit module this tree does not have yet'
				: typeSyntax
					? 'non-erasable TypeScript (enum, namespace, parameter properties, import =)'
					: 'the module throws while loading',
			fix: 'pure package files import only src/lib/atlas/** (outside ui/) and src/lib/i18n/types.ts, with relative paths and .ts extensions'
		});
	}
}

/** An export name that names its id (K1): the id itself, 'default', or the id without punctuation and case ('train:flow'
 *  exported as trainFlow or flow, 'walk-2' as walk2), so a process registered as 'train:<id>' is not a note. */
export function sameName(name: string, id: string): boolean {
	if (name === id || name === 'default') return true;
	const norm = (s: string) => s.replace(/[^A-Za-z0-9]/g, '').toLowerCase();
	return norm(name) === norm(id) || norm(name) === norm(id.replace(/^train:/, ''));
}

export const isProcessDef = (v: unknown): v is ProcessDef<any, any, any> =>
	!!v && typeof v === 'object' && typeof (v as ProcessDef).run === 'function' && typeof (v as ProcessDef).id === 'string' && !!(v as ProcessDef).phases && !!(v as ProcessDef).schema;

export const isTrainerDef = (v: unknown): v is TrainerDef<any, any, any, any> =>
	!!v &&
	typeof v === 'object' &&
	typeof (v as TrainerDef).id === 'string' &&
	typeof (v as TrainerDef).model === 'function' &&
	typeof (v as TrainerDef).batch === 'function' &&
	typeof (v as TrainerDef).loss === 'function';

const isAdapter = (v: unknown): v is InputAdapter =>
	!!v && typeof v === 'object' && typeof (v as InputAdapter).id === 'string' && typeof (v as InputAdapter).preprocess === 'function';

async function adaptersIn(dir: string, notes: string[]): Promise<Record<string, InputAdapter>> {
	const out: Record<string, InputAdapter> = {};
	if (!isDir(dir)) return out;
	for (const id of readdirSync(dir).sort()) {
		const f = join(dir, id, 'adapter.ts');
		if (!existsSync(f)) continue;
		const m = await importTs(f, `adapter ${id}`);
		const a = (m.default ?? m.adapter) as unknown;
		if (!isAdapter(a)) {
			notes.push(`${posix(f)}: no InputAdapter default export`);
			continue;
		}
		if (a.id !== id) notes.push(`${posix(f)}: adapter id '${a.id}' differs from its folder '${id}'`);
		out[a.id] = a;
	}
	return out;
}

export async function loadPackage(slug: string, site = roots().site): Promise<LoadedPackage> {
	const dir = join(explainersDir(site), slug);
	const atlas = join(site, 'src', 'lib', 'atlas');
	const notes: string[] = [];
	const files: string[] = [];
	const f = (name: string) => join(dir, name);
	if (!existsSync(f('explainer.ts')))
		throw new AtlasError('AK301', `no package '${slug}' (missing ${posix(f('explainer.ts'))})`, {
			fix: `create it with: npm run atlas -- new model ${slug} --kind ... (packages: ${listPackages(site).join(', ') || 'none yet'})`
		});

	files.push('explainer.ts');
	const em = await importTs(f('explainer.ts'), 'explainer.ts');
	const manifest = (em.default ?? em.manifest) as ExplainerManifest | undefined;
	if (!manifest || typeof manifest !== 'object' || !('slug' in manifest))
		throw new AtlasError('AK301', 'explainer.ts must default-export its ExplainerManifest', { where: posix(f('explainer.ts')) });
	if (manifest.slug !== slug)
		throw new AtlasError('AK301', `manifest slug '${manifest.slug}' differs from the folder '${slug}'`, { where: posix(f('explainer.ts')) });

	let lesson: LessonDef<any> | null = null;
	if (existsSync(f('lesson.ts'))) {
		files.push('lesson.ts');
		const lm = await importTs(f('lesson.ts'), 'lesson.ts');
		lesson = ((lm.default ?? lm.lesson) as LessonDef<any> | undefined) ?? null;
		if (!lesson) notes.push('lesson.ts has no default export');
	}

	let ops: OpDef[] = [];
	if (existsSync(f('ops.ts'))) {
		files.push('ops.ts');
		const om = await importTs(f('ops.ts'), 'ops.ts');
		ops = (om.default ?? om.ops) as OpDef[];
		if (!Array.isArray(ops)) throw new AtlasError('AK301', 'ops.ts must default-export an OpDef[] ([...CORE_OPS, ...local])', { where: posix(f('ops.ts')) });
	} else if (existsSync(join(atlas, 'ops', 'index.ts'))) {
		const core = await importTs(join(atlas, 'ops', 'index.ts'), 'atlas/ops/index.ts');
		ops = (core.CORE_OPS ?? core.default ?? []) as OpDef[];
	} else {
		notes.push('atlas/ops/index.ts (W1-A) is not in this tree: ops = []');
	}

	const processes: Record<string, ProcessDef<any, any, any>> = {};
	const add = (key: string, p: ProcessDef<any, any, any>, from: string) => {
		if (processes[key]) throw new AtlasError('AK301', `two processes are registered as '${key}'`, { where: from });
		processes[key] = p;
	};
	if (existsSync(f('process.ts'))) {
		files.push('process.ts');
		const pm = await importTs(f('process.ts'), 'process.ts');
		for (const [name, v] of Object.entries(pm)) {
			if (!isProcessDef(v)) continue;
			if (!sameName(name, v.id)) notes.push(`process.ts: export '${name}' has id '${v.id}'; it is registered as '${v.id}'`);
			add(v.id, v, posix(f('process.ts')));
		}
	}
	if (existsSync(f('trainer.ts'))) {
		files.push('trainer.ts');
		const tm = await importTs(f('trainer.ts'), 'trainer.ts');
		const trainers = Object.entries(tm).filter(([, v]) => isTrainerDef(v)) as [string, TrainerDef<any, any, any, any>][];
		const tp = join(atlas, 'train', 'trainer.ts');
		if (trainers.length && !existsSync(tp)) {
			notes.push(`atlas/train/trainer.ts (W1-B1) is not in this tree: trainers ${trainers.map(([, t]) => t.id).join(', ')} are not registered`);
		} else if (trainers.length) {
			const { trainingProcess } = (await importTs(tp, 'atlas/train/trainer.ts')) as { trainingProcess: (d: TrainerDef<any, any, any, any>) => ProcessDef<any, any, any> };
			for (const [name, t] of trainers) {
				if (!sameName(name, t.id)) notes.push(`trainer.ts: export '${name}' has id '${t.id}'; it is registered as 'train:${t.id}'`);
				add(`train:${t.id}`, trainingProcess(t), posix(f('trainer.ts')));
			}
		}
	}

	const adapters = { ...(await adaptersIn(join(atlas, 'adapters'), notes)), ...(await adaptersIn(join(dir, 'adapters'), notes)) };
	for (const extra of ['worker.ts', 'claims.test.ts']) if (existsSync(f(extra))) files.push(extra);
	const pkg: PackageModules = { slug, manifest, lesson, ops, processes, adapters };
	// atlas parity --write's per-source record (K0)
	if (existsSync(f('parity.json'))) {
		files.push('parity.json');
		try {
			const pr = JSON.parse(readFileSync(f('parity.json'), 'utf8').replace(/^\uFEFF/, '')) as PackageModules['parity'];
			if (pr && pr.format === 'atlas-parity/1') pkg.parity = pr;
			else notes.push('parity.json is not atlas-parity/1: ignored');
		} catch (e) {
			notes.push(`parity.json does not parse: ${(e as Error).message}`);
		}
	}

	return { pkg, dir, files, notes };
}
