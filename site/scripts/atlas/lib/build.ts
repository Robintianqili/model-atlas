// The site gate of `atlas check` and `atlas build` (K2 speed pack): a type check, check:catalog, a build into
// .atlas/build-<name>/ and check:build.
//   packages   (atlas check <slug>, atlas build <slug>...): svelte-check scoped to the packages (lib/typecheck.ts) and a
//              package-only build (ATLAS_ONLY=<slugs>: the home page and the packages' pages; svelte.config.js)
//   site       (atlas check --all, atlas build --all): npm run check and the full build (every page)
// Every build includes drafts (ATLAS_INCLUDE_DRAFTS=1). A failed step keeps its whole output in
// .atlas/<name>.<step>.log (K1). A build that fails on a locked file (EBUSY/EPERM, exFAT) while this site's kept dev
// server runs is retried once after stopping that server.
import { join, relative } from 'node:path';
import { runNpm, type RunResult } from './child.ts';
import { readRecord, stopKept } from './devserver.ts';
import { posix, writeText } from './io.ts';
import type { Issues } from './pkg.ts';
import { checkPackages, checkSite, firstProblems } from './typecheck.ts';

export interface SiteOpts {
	site: string;
	/** build folder name under .atlas/ (build-<slug>, build-all) */
	name: string;
	/** package-only (ATLAS_ONLY) and a scoped type check; null = the whole site */
	slugs: string[] | null;
	echo?: boolean;
	/** skip the type check (it already ran, e.g. as the quick gate's) */
	noTypes?: boolean;
}

export interface SiteResult {
	summary: string;
	dir: string;
	ok: boolean;
	ms: Record<string, number>;
}

/** The environment of a build (and of check:build on it). */
export function buildEnv(name: string, slugs: string[] | null): Record<string, string> {
	return { ATLAS_BUILD_DIR: posix(join('.atlas', name)), ATLAS_INCLUDE_DRAFTS: '1', ...(slugs?.length ? { ATLAS_ONLY: slugs.join(',') } : { ATLAS_ONLY: '' }) };
}

const strip = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, '');

export async function siteSteps(is: Issues, o: SiteOpts): Promise<SiteResult> {
	const out: string[] = [];
	const ms: Record<string, number> = {};
	const keepLog = (label: string, cmd: string, res: RunResult) => {
		const f = join(o.site, '.atlas', `${o.name}.${label.replace(/[^A-Za-z0-9_.-]+/g, '-')}.log`);
		writeText(f, `$ ${cmd}\n${strip(res.stdout)}\n${strip(res.stderr)}\n`);
		return posix(relative(o.site, f));
	};
	const note = (label: string, ok: boolean, t: number, log = '') => {
		ms[label] = t;
		out.push(`${label} ${ok ? 'ok' : 'FAILED'} (${(t / 1000).toFixed(0)} s${log ? `, log in ${log}` : ''})`);
	};
	const step = async (label: string, script: string, env: Record<string, string> = {}) => {
		const res = await runNpm(script, [], { cwd: o.site, echo: o.echo ? `    [${label}] ` : false, env, timeoutMs: 540_000 });
		note(label, res.code === 0, res.ms, res.code === 0 ? '' : keepLog(label, `npm run ${script}`, res));
		return res;
	};
	if (!o.noTypes) {
		const tc = o.slugs?.length ? await checkPackages(o.site, o.slugs, o.name, { echo: o.echo ? '    [svelte-check] ' : false }) : await checkSite(o.site, { echo: o.echo ? '    [svelte-check] ' : false });
		const label = o.slugs?.length ? 'svelte-check (packages)' : 'svelte-check';
		note(label, tc.ok, tc.ms, tc.ok ? '' : keepLog('svelte-check', `svelte-check ${tc.scope}`, tc.res));
		if (!tc.ok) is.error('AK301', `svelte-check ${tc.scope}: ${tc.errors ?? '?'} errors, ${tc.warnings ?? '?'} warnings`, { likely: firstProblems(strip(tc.out)) || undefined, fix: 'fix every error and warning (the bar is 0/0)' });
	}
	const cat = await step('check:catalog', 'check:catalog');
	if (cat.code !== 0) is.error('AK301', 'npm run check:catalog failed', { likely: cat.stdout.split('\n').filter((l) => /FAIL|error/i.test(l)).slice(0, 3).join(' | ') });
	const env = buildEnv(o.name, o.slugs);
	let b = await step('build', 'build', env);
	const locked = (r: RunResult) => /EBUSY|EPERM|resource busy or locked/i.test(r.stdout + r.stderr);
	if (b.code !== 0 && locked(b) && readRecord(o.site)) {
		const s = await stopKept(o.site);
		out.push(`build hit a locked file: ${s.note}; retrying`);
		b = await step('build', 'build', env);
	}
	if (b.code !== 0)
		is.error('AK301', `build into ${env.ATLAS_BUILD_DIR}${o.slugs?.length ? ` (ATLAS_ONLY=${env.ATLAS_ONLY})` : ''} failed`, {
			likely: (b.stderr + '\n' + b.stdout).trim().split('\n').filter((l) => /error|Error|failed|EBUSY|EPERM/.test(l)).slice(-4).join(' | ') || (b.stderr || b.stdout).trim().split('\n').slice(-3).join(' | '),
			where: `.atlas/${o.name}.build.log`,
			fix: 'npm run build shows the error; on exFAT a dev server may lock files: npm run atlas -- e2e --stop-server, then build again'
		});
	else {
		const cb = await step('check:build', 'check:build', env);
		if (cb.code !== 0) is.error('AK301', 'check:build failed', { likely: cb.stdout.split('\n').concat(cb.stderr.split('\n')).filter((l) => l.startsWith('FAIL')).slice(0, 4).join(' | ') });
	}
	const scope = o.slugs?.length ? `package-only build (${o.slugs.join(', ')})` : 'full build';
	return { summary: `${scope}: ${out.join('; ')}`, dir: env.ATLAS_BUILD_DIR, ok: !is.errors.length, ms };
}
