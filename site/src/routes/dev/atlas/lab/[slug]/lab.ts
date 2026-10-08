// Lesson Lab helpers (browser): open a package, or a package synthesized from a spec folder (?spec=models/<dir>), and
// lint its lesson over every fixture with progress.
import type { ExplainerManifest } from '$lib/atlas/types/explainer';
import type { LintIssue } from '$lib/atlas/types/lesson';
import type { PackageModules } from '$lib/atlas/types/runtime';
import type { AtlasSpec } from '$lib/atlas/types/spec';
import { normalizeSpec } from '$lib/atlas/graph/legacy';
import { compileLesson, type CompiledLesson } from '$lib/atlas/lesson/compile';
import { fixtureCases, lintLesson, type FixtureCase } from '$lib/atlas/lesson/lint';
import { runtimeMaker } from '$lib/atlas/lesson/fixtures';
import { synthManifest } from '$lib/atlas/ui/synth';
import { hasPackage, loadPackage, syntheticPackage } from '$lib/atlas/ui/packages';
import { hasView } from '$lib/atlas/ui/registry';
import { openAtlas, staticBase, type AtlasStore } from '$lib/atlas/ui/runtime.svelte';
import { textInput } from '$lib/atlas/adapters/text-bpe/adapter';

export interface Lab {
	pkg: PackageModules;
	store: AtlasStore;
	lesson: CompiledLesson;
	fixtures: FixtureCase[];
	/** asset folder relative to static/ */
	dir: string;
	/** no package: the manifest was made from the spec */
	synthetic: boolean;
	notes: string[];
}

const join = (base: string, ...p: string[]) => [base.replace(/\/+$/, ''), ...p.map((x) => x.replace(/^\/+|\/+$/g, ''))].join('/');

async function exists(url: string): Promise<boolean> {
	try {
		const r = await fetch(url, { method: 'GET' });
		return r.ok;
	} catch {
		return false;
	}
}

/** A first text for a text model (the lab's start input): real tokens from the model's tokenizer. */
const START_TEXT = 'Once upon a time, there was a little girl named Lily. She';

async function synthesize(slug: string, dir: string, base: string, known?: string[]): Promise<{ pkg: PackageModules; notes: string[] }> {
	const has = async (name: string) => (known ? known.includes(name) : exists(join(base, dir, name)));
	const r = await fetch(join(base, dir, 'spec.json'));
	if (!r.ok) throw new Error(`no spec.json in static/${dir} (${r.status})`);
	const spec: AtlasSpec = normalizeSpec(await r.json());
	const notes = [`no explainer package '${slug}': the manifest is synthesized from static/${dir}/spec.json`];
	const adapter = spec.inputs?.[0]?.adapter;
	const presets = (await has('presets.json')) ? 'presets.json' : null;
	const auto = ['spec.json', 'weights.bin'];
	let options: Record<string, unknown> | undefined;
	let start: unknown;
	if (adapter === 'text-bpe') {
		const t = (await has('tokenizer.json')) ? await fetch(join(base, dir, 'tokenizer.json')) : null;
		if (t?.ok) {
			const tok = await t.json();
			options = { tokenizer: tok };
			start = textInput(tok, START_TEXT);
			auto.push('tokenizer.json');
		} else notes.push('text-bpe without tokenizer.json: only token-id fixtures work');
	}
	const manifest: ExplainerManifest = synthManifest(spec, { slug, dir, presets, auto, options, start });
	return { pkg: await syntheticPackage(manifest), notes };
}

/** files: the server's listing of static/models/<dir> folders (lab +page.server.ts), so nothing is probed by URL */
export async function openLab(slug: string, specParam: string | null, base: string, files: Record<string, string[]> = {}): Promise<Lab> {
	let pkg: PackageModules;
	let notes: string[] = [];
	let synthetic = false;
	const cleanSpec = specParam?.replace(/^\/+|\/+$/g, '') || null;
	if (hasPackage(slug)) {
		const loaded = await loadPackage(slug);
		if (!loaded) throw new Error(`package '${slug}' did not load`);
		pkg = loaded.pkg;
		notes = loaded.notes;
		if (cleanSpec) {
			pkg = { ...pkg, manifest: { ...pkg.manifest, assets: { auto: pkg.manifest.assets?.auto ?? ['spec.json', 'weights.bin'], ...pkg.manifest.assets, dir: cleanSpec } } };
			notes.push(`assets read from static/${cleanSpec} (?spec=)`);
		}
	} else {
		const dir = cleanSpec ?? `models/${slug}`;
		const s = await synthesize(slug, dir, base, files[dir]);
		pkg = s.pkg;
		notes = s.notes;
		synthetic = true;
	}
	const store = await openAtlas(pkg, { base });
	notes.push(...store.rt.warnings);
	const lesson = compileLesson(pkg.lesson, { spec: store.rt.spec, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes, views: hasView });
	const fixtures = fixtureCases(pkg, store.rt.presets);
	return { pkg, store, lesson, fixtures, dir: pkg.manifest.assets?.dir ?? `models/${slug}`, synthetic, notes };
}

/** Lint every step on every fixture (en and zh), one fixture at a time so the page stays responsive. */
export async function lintAll(lab: Lab, base: string, onProgress: (done: number, total: number) => void): Promise<LintIssue[]> {
	const issues: LintIssue[] = [...lab.lesson.issues];
	const make = runtimeMaker(lab.pkg, { fetch: (u: string) => fetch(u), base: staticBase(base) });
	const n = lab.fixtures.length;
	onProgress(0, n);
	for (let k = 0; k < n; k++) {
		issues.push(...(await lintLesson(lab.lesson.steps, make, [lab.fixtures[k]])));
		onProgress(k + 1, n);
		await new Promise((r) => setTimeout(r, 0));
	}
	return issues;
}
