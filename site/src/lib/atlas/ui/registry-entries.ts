// The pure part of the view registry (registry.ts): turning globbed meta paths into entries, and picking one package's
// metas. Kept apart from the Vite globs so Node tests can run it (registry.test.ts).
import type { Component } from 'svelte';
import type { ViewMeta } from '../types/view.ts';

export interface ViewEntry {
	key: string;
	meta: ViewMeta;
	load: (() => Promise<Component<any>>) | null;
	/** registry problems (key differs from its folder, no View.svelte) */
	problem?: string;
}

type ViewLoaders = Record<string, () => Promise<Component<any>>>;

export const KIT_META_RE = /^\.\/views\/([^/]+)\/meta\.ts$/;
export const PKG_META_RE = /^\.\.\/\.\.\/explainers\/([^/]+)\/views\/([^/]+)\/meta\.ts$/;

/** A kit view's entry from its globbed meta path ('./views/<key>/meta.ts'), or null for another path. */
export function kitEntry(path: string, meta: ViewMeta | undefined, views: ViewLoaders): ViewEntry | null {
	const m = KIT_META_RE.exec(path);
	if (!m || !meta) return null;
	const key = m[1];
	const view = `./views/${key}/View.svelte`;
	const e: ViewEntry = { key, meta, load: views[view] ?? null };
	if (meta.key !== key) e.problem = `meta.key '${meta.key}' differs from its folder '${key}'`;
	if (!e.load) e.problem = `${view} is missing`;
	return e;
}

/** A package view's entry ('<slug>/<key>') from its globbed meta path, or null for another path. */
export function pkgEntry(path: string, meta: ViewMeta | undefined, views: ViewLoaders): ViewEntry | null {
	const m = PKG_META_RE.exec(path);
	if (!m || !meta) return null;
	const key = `${m[1]}/${m[2]}`;
	const view = `../../explainers/${m[1]}/views/${m[2]}/View.svelte`;
	const e: ViewEntry = { key, meta, load: views[view] ?? null };
	if (meta.key !== key && meta.key !== m[2]) e.problem = `meta.key '${meta.key}' should be '${key}'`;
	if (!e.load) e.problem = `${view} is missing`;
	return e;
}

/** The globbed meta paths of one package's views. */
export function packageMetaPaths(paths: string[], slug: string): string[] {
	const prefix = `../../explainers/${slug}/views/`;
	return paths.filter((p) => p.startsWith(prefix) && PKG_META_RE.test(p)).sort();
}

/** The package slug a view key belongs to ('<slug>/<key>'), or null for a kit view. */
export function packageOfKey(key: string): string | null {
	const i = key.indexOf('/');
	return i > 0 ? key.slice(0, i) : null;
}
