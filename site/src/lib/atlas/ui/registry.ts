// The view registry (design §4.8; redesign wave F: per package). Kit metas are read eagerly with import.meta.glob (they
// are few and every lesson uses them); package metas are lazy, and prepareViews(slug) loads one package's metas, so a page
// carries the metas of its own package only (before, every package's metas were in two shared 316-338 KB chunks).
// loadPackage(slug) (packages.ts) awaits prepareViews(slug), so every caller that has its package (AutoModelPage,
// AutoConceptPage, the Lesson Lab) keeps the synchronous hasView / viewMeta / loadView with no API change. Components are
// lazy, one chunk per view. Kit views live in atlas/ui/views/<key>/{meta.ts, View.svelte}; package views in
// explainers/<slug>/views/<key>/ and register as '<slug>/<key>'. Only lazy code imports this module (AutoModelPage,
// LessonPlayer, StageHost, the Lesson Lab), so planned pages and the hand-built model pages never carry the metas.
import type { Component } from 'svelte';
import type { ViewMeta } from '../types/view.ts';
import { kitEntry, packageMetaPaths, pkgEntry, type ViewEntry } from './registry-entries.ts';

const kitMetas = import.meta.glob<ViewMeta>('./views/*/meta.ts', { eager: true, import: 'default' });
const pkgMetas = import.meta.glob<ViewMeta>('../../explainers/*/views/*/meta.ts', { import: 'default' });
const kitViews = import.meta.glob<Component<any>>('./views/*/View.svelte', { import: 'default' });
const pkgViews = import.meta.glob<Component<any>>('../../explainers/*/views/*/View.svelte', { import: 'default' });

const REG = new Map<string, ViewEntry>();
for (const [path, meta] of Object.entries(kitMetas)) {
	const e = kitEntry(path, meta, kitViews);
	if (e) REG.set(e.key, e);
}

const prepared = new Map<string, Promise<void>>();
const done = new Set<string>();
const loaded = new Map<string, Promise<Component<any> | null>>();

/** Load one package's view metas into the registry (once; later calls share the promise). Throws when a meta fails to
 *  load, like any other package file (loadPackage). */
export function prepareViews(slug: string): Promise<void> {
	let p = prepared.get(slug);
	if (!p) {
		const paths = packageMetaPaths(Object.keys(pkgMetas), slug);
		p = Promise.all(paths.map(async (path) => [path, await pkgMetas[path]()] as const)).then((metas) => {
			for (const [path, meta] of metas) {
				const e = pkgEntry(path, meta, pkgViews);
				if (e) REG.set(e.key, e);
			}
			done.add(slug);
		});
		p.catch(() => prepared.delete(slug));
		prepared.set(slug, p);
	}
	return p;
}

/** Has prepareViews(slug) finished? */
export function viewsPrepared(slug: string): boolean {
	return done.has(slug);
}

/** Every registered view key: the kit views, then the package views prepared so far. */
export function viewKeys(): string[] {
	return [...REG.keys()];
}

/** Is `key` a registered view with a component? (lint AK405 in the browser; a package view once its package is prepared) */
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
		if (!e) return Promise.resolve(null); // not cached: the key may register once its package is prepared
		p = e.load
			? e.load().catch((err) => {
					console.error(`atlas view '${key}' failed to load`, err);
					return null;
				})
			: Promise.resolve(null);
		loaded.set(key, p);
	}
	return p;
}
