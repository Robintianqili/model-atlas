// svelte-check, the whole site or one package (K2 speed pack).
//   whole site   npm run check (svelte-kit sync && svelte-check): every file the site's tsconfig includes (~1,200)
//   packages     svelte-check --tsconfig .atlas/<name>/tsconfig.check.json: a tsconfig that extends the site's and
//                includes only src/lib/explainers/<slug>/** (with the ambient SvelteKit types). svelte-check diagnoses only
//                the files a --tsconfig includes, so the kit and other packages are type-checked only as far as the
//                package imports them, and their own errors are not reported here. The bar is the same: 0 errors,
//                0 warnings.
import { existsSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, relative } from 'node:path';
import { runNode, runNpm, type RunResult } from './child.ts';
import { posix, writeJson } from './io.ts';

export interface TypeCheck {
	ok: boolean;
	errors: number | null;
	warnings: number | null;
	files: number | null;
	ms: number;
	scope: string;
	out: string;
	res: RunResult;
}

function bin(site: string, pkg: string, rel: string): string {
	const req = createRequire(join(site, 'package.json'));
	return join(dirname(req.resolve(`${pkg}/package.json`)), rel);
}

/** svelte-check's summary line ("svelte-check found 0 errors and 0 warnings in 12 files"). */
export function parseSvelteCheck(text: string): { errors: number; warnings: number; files: number | null } | null {
	const m = /svelte-check found (\d+) errors? and (\d+) warnings?(?: in (\d+) files?)?/.exec(text);
	if (m) return { errors: Number(m[1]), warnings: Number(m[2]), files: m[3] ? Number(m[3]) : null };
	// the machine output (svelte-check without a terminal): "COMPLETED 1193 FILES 0 ERRORS 0 WARNINGS ..."
	const c = /COMPLETED (\d+) FILES (\d+) ERRORS (\d+) WARNINGS/.exec(text);
	return c ? { errors: Number(c[2]), warnings: Number(c[3]), files: Number(c[1]) } : null;
}

/** The scoped tsconfig for some packages (written under .atlas/<name>/). */
export function scopedTsconfig(site: string, slugs: string[], name: string): { file: string; config: Record<string, unknown> } {
	const dir = join(site, '.atlas', name);
	const rel = (p: string) => posix(relative(dir, join(site, p)));
	const ambient = ['.svelte-kit/ambient.d.ts', '.svelte-kit/non-ambient.d.ts', 'src/app.d.ts'].filter((p) => existsSync(join(site, p)));
	const include = [...ambient.map(rel), ...slugs.flatMap((s) => ['ts', 'js', 'svelte'].map((x) => `${rel(`src/lib/explainers/${s}`)}/**/*.${x}`))];
	const config = { extends: rel('tsconfig.json'), include, exclude: [`${rel('node_modules')}/**`] };
	return { file: join(dir, 'tsconfig.check.json'), config };
}

/** svelte-kit sync when the generated tsconfig is missing (a fresh lane); it is otherwise left alone, so a running dev
 *  server sees no change. */
export async function ensureSync(site: string): Promise<RunResult | null> {
	const t = join(site, '.svelte-kit', 'tsconfig.json');
	if (existsSync(t) && statSync(t).size > 0) return null;
	return runNode([bin(site, '@sveltejs/kit', 'svelte-kit.js'), 'sync'], { cwd: site, timeoutMs: 120_000 });
}

/** svelte-check over some packages only. */
export async function checkPackages(site: string, slugs: string[], name: string, o: { echo?: boolean | string } = {}): Promise<TypeCheck> {
	const t0 = performance.now();
	await ensureSync(site);
	const { file, config } = scopedTsconfig(site, slugs, name);
	writeJson(file, config, 1);
	const res = await runNode([bin(site, 'svelte-check', 'bin/svelte-check'), '--workspace', site, '--tsconfig', file, '--output', 'human'], { cwd: site, echo: o.echo, timeoutMs: 540_000 });
	const p = parseSvelteCheck(res.stdout + res.stderr);
	const ok = res.code === 0 && !!p && p.errors === 0 && p.warnings === 0;
	return { ok, errors: p?.errors ?? null, warnings: p?.warnings ?? null, files: p?.files ?? null, ms: performance.now() - t0, scope: `${slugs.join(', ')} (${posix(relative(site, file))})`, out: res.stdout + res.stderr, res };
}

/** npm run check (the whole site). */
export async function checkSite(site: string, o: { echo?: boolean | string } = {}): Promise<TypeCheck> {
	const t0 = performance.now();
	const res = await runNpm('check', [], { cwd: site, echo: o.echo, timeoutMs: 540_000 });
	const p = parseSvelteCheck(res.stdout + res.stderr);
	const ok = res.code === 0 && (!p || (p.errors === 0 && p.warnings === 0));
	return { ok, errors: p?.errors ?? null, warnings: p?.warnings ?? null, files: p?.files ?? null, ms: performance.now() - t0, scope: 'the whole site (npm run check)', out: res.stdout + res.stderr, res };
}

/** The first diagnostics of a failed check, for the gate message. */
export function firstProblems(out: string, n = 4): string {
	return out
		.split('\n')
		.filter((l) => /\b(Error|Warn(ing)?)\b/.test(l) && !/svelte-check found/.test(l))
		.slice(0, n)
		.map((l) => l.trim().slice(0, 200))
		.join(' | ');
}
