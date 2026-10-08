// A package opened for a command: the PackageModules, the kit of the same site, and runtime helpers.
import { existsSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type { ExplainerRuntime, FixtureDef, I18n, SourceDef, SourceRunner } from '../../../src/lib/atlas/types/index.ts';
import { formatIssue, type AkCode, type AtlasErrorInfo } from '../errors.ts';
import { loadPackage, type LoadedPackage } from '../load.ts';
import { roots } from '../root.ts';
import { assetsDir, kit, type Kit } from './kit.ts';

export const LEGACY = ['tinycnn', 'tinygpt'] as const;
export const isLegacy = (slug: string) => (LEGACY as readonly string[]).includes(slug);

export interface Opened {
	slug: string;
	site: string;
	kit: Kit;
	l: LoadedPackage;
	/** static/<assets.dir> */
	dir: string;
	sources: Record<string, SourceDef>;
	/** a runtime on `raw` (default: the runtime's own first input); `runner` (K1): e.g. the kit's sharedRunner() */
	runtime(raw?: unknown, o?: { runner?: SourceRunner }): Promise<ExplainerRuntime & { close?: () => void; presets?: { id: string; label: I18n; raw: unknown }[]; warnings?: string[] }>;
	/** lesson fixtures + adapter edge fixtures + presets (lint order) */
	fixtures(): Promise<(FixtureDef & { kind: string })[]>;
}

export async function openPackage(slug: string, site = roots().site): Promise<Opened> {
	const l = await loadPackage(slug, site);
	const k = await kit(site);
	const m = l.pkg.manifest;
	const dir = assetsDir(site, slug, m.assets?.dir);
	const sources = m.sources ?? (m.kinds.includes('trace') ? { fwd: { kind: 'trace' } as SourceDef } : {});
	const base = join(site, 'static');
	const runtime = async (raw?: unknown, ro?: { runner?: SourceRunner }) =>
		(await k.runtime.createRuntime(l.pkg, { fetch: k.io.fsFetch, base, ...(raw !== undefined ? { input: raw } : {}), ...(ro?.runner ? { runner: ro.runner } : {}) })) as Awaited<ReturnType<Opened['runtime']>>;
	const fixtures = async () => {
		const probe = await runtime();
		const presets = probe.presets ?? [];
		probe.close?.();
		return k.lint.fixtureCases(l.pkg, presets);
	};
	return { slug, site, kit: k, l, dir, sources, runtime, fixtures };
}

/** A collected problem: printed in the §4.12 block format. */
export interface Issue {
	code: AkCode | string;
	level: 'error' | 'warn';
	message: string;
	info?: AtlasErrorInfo;
}

export class Issues {
	list: Issue[] = [];
	error(code: AkCode, message: string, info?: AtlasErrorInfo) {
		this.list.push({ code, level: 'error', message, info });
	}
	warn(code: AkCode, message: string, info?: AtlasErrorInfo) {
		this.list.push({ code, level: 'warn', message, info });
	}
	get errors() {
		return this.list.filter((i) => i.level === 'error');
	}
	get warnings() {
		return this.list.filter((i) => i.level === 'warn');
	}
	format(): string {
		return this.list.map((i) => formatIssue(i.code, `${i.level === 'warn' ? '(warn) ' : ''}${i.message}`, i.info ?? {})).join('\n');
	}
}

/** Every file under a folder (relative POSIX paths), skipping the given folder names. */
export function walk(dir: string, skip: string[] = [], rel = ''): string[] {
	if (!existsSync(dir)) return [];
	const out: string[] = [];
	for (const name of readdirSync(dir).sort()) {
		const p = join(dir, name);
		const r = rel ? `${rel}/${name}` : name;
		const st = statSync(p);
		if (st.isDirectory()) {
			if (!skip.includes(name)) out.push(...walk(p, skip, r));
		} else out.push(r);
	}
	return out;
}

export const today = () => new Date().toLocaleDateString('sv-SE');

/** 3 significant digits for reports (exponent form below 1e-3). */
export function sig(v: number | null | undefined): string {
	if (v === null || v === undefined) return '-';
	if (!Number.isFinite(v)) return String(v);
	if (v === 0) return '0';
	const a = Math.abs(v);
	return a < 1e-3 || a >= 1e6 ? v.toExponential(2) : String(Number(v.toPrecision(3)));
}
