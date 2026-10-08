// The view registry (design §4.8). Metas are read with import.meta.glob (eager: they are small and pure, and lint needs a
// synchronous "is this view registered?"); components are lazy, one chunk per view. Kit views live in
// atlas/ui/views/<key>/{meta.ts, View.svelte}; package views in explainers/<slug>/views/<key>/ and register as '<slug>/<key>'.
// Only lazy code imports this module (AutoModelPage, LessonPlayer, StageHost, the Lesson Lab), so planned pages and the
// hand-built model pages never carry the metas (design §5.1, page-shell budget).
import type { Component } from 'svelte';
import type { ViewMeta } from '../types/view.ts';

const kitMetas = import.meta.glob<ViewMeta>('./views/*/meta.ts', { eager: true, import: 'default' });
const pkgMetas = import.meta.glob<ViewMeta>('../../explainers/*/views/*/meta.ts', { eager: true, import: 'default' });
const kitViews = import.meta.glob<Component<any>>('./views/*/View.svelte', { import: 'default' });
const pkgViews = import.meta.glob<Component<any>>('../../explainers/*/views/*/View.svelte', { import: 'default' });

interface Entry {
	key: string;
	meta: ViewMeta;
	load: (() => Promise<Component<any>>) | null;
	/** registry problems (key differs from its folder, no View.svelte) */
	problem?: string;
}

const KIT_RE = /^\.\/views\/([^/]+)\/meta\.ts$/;
const PKG_RE = /^\.\.\/\.\.\/explainers\/([^/]+)\/views\/([^/]+)\/meta\.ts$/;

function build(): Map<string, Entry> {
	const out = new Map<string, Entry>();
	for (const [path, meta] of Object.entries(kitMetas)) {
		const m = KIT_RE.exec(path);
		if (!m || !meta) continue;
		const key = m[1];
		const view = `./views/${key}/View.svelte`;
		const e: Entry = { key, meta, load: kitViews[view] ?? null };
		if (meta.key !== key) e.problem = `meta.key '${meta.key}' differs from its folder '${key}'`;
		if (!e.load) e.problem = `${view} is missing`;
		out.set(key, e);
	}
	for (const [path, meta] of Object.entries(pkgMetas)) {
		const m = PKG_RE.exec(path);
		if (!m || !meta) continue;
		const key = `${m[1]}/${m[2]}`;
		const view = `../../explainers/${m[1]}/views/${m[2]}/View.svelte`;
		const e: Entry = { key, meta, load: pkgViews[view] ?? null };
		if (meta.key !== key && meta.key !== m[2]) e.problem = `meta.key '${meta.key}' should be '${key}'`;
		if (!e.load) e.problem = `${view} is missing`;
		out.set(key, e);
	}
	return out;
}

const REG = build();
const loaded = new Map<string, Promise<Component<any> | null>>();

/** Every registered view key (kit views first, then package views). */
export function viewKeys(): string[] {
	return [...REG.keys()];
}

/** Is `key` a registered view with a component? (lint AK405 in the browser) */
export function hasView(key: string): boolean {
	const e = REG.get(key);
	return !!e && !!e.load;
}

export function viewMeta(key: string): ViewMeta | null {
	return REG.get(key)?.meta ?? null;
}

export function viewProblems(): { key: string; problem: string }[] {
	return [...REG.values()].filter((e) => e.problem).map((e) => ({ key: e.key, problem: e.problem! }));
}

/** The view component (lazy, cached), or null when the key is not registered or fails to load. */
export function loadView(key: string): Promise<Component<any> | null> {
	let p = loaded.get(key);
	if (!p) {
		const e = REG.get(key);
		p = e?.load
			? e.load().catch((err) => {
					console.error(`atlas view '${key}' failed to load`, err);
					return null;
				})
			: Promise.resolve(null);
		loaded.set(key, p);
	}
	return p;
}
