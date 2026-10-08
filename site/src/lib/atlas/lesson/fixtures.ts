// withFixtures(pkg, fn) (design §4.6 "Claims are tests"): runs `fn` on a fresh runtime for every fixture and preset of a
// package. claims.test.ts files use it:
//
//   import { withFixtures } from '../../atlas/lesson/fixtures.ts';
//   test('bag of words gives both captions the same score', () => withFixtures(pkg, async (rt, fx) => { … }));
//
// In Node, `fetch` defaults to the file system under $ATLAS_SITE/static (else ./static of the working directory) and
// `base` to ''. Pass both explicitly anywhere else.
import type { FetchLike } from '../types/core.ts';
import type { ExplainerRuntime, PackageModules, SourceRunner } from '../types/runtime.ts';
import { createRuntime, type AtlasRuntime } from '../runtime/runtime.ts';
import { sharedRunner, type SharedRunner } from '../runtime/local.ts';
import { fixtureCases, type FixtureCase } from './lint.ts';

/** K1: the runner withFixtures uses by default: identical runs are shared between fixtures and between the claim tests of
 *  one test file (a source that ignores the learner's input, a preset equal to the start input, the same source asked for
 *  by two tests). Each runtime still gets its own view; a control gives it a private run. */
let shared: SharedRunner | null = null;
export function sharedClaimsRunner(): SharedRunner {
	shared ??= sharedRunner({ max: 96 });
	return shared;
}

async function nodeStaticFetch(): Promise<FetchLike> {
	// variable specifiers keep bundlers from resolving Node modules if this file is ever imported in a browser graph
	const fsName = 'node:fs/promises';
	const pathName = 'node:path';
	const fs = (await import(/* @vite-ignore */ fsName)) as typeof import('node:fs/promises');
	const path = (await import(/* @vite-ignore */ pathName)) as typeof import('node:path');
	const env = (globalThis as { process?: { env: Record<string, string | undefined>; cwd(): string } }).process;
	const root = path.join(env?.env.ATLAS_SITE ?? env?.cwd() ?? '.', 'static');
	return async (url: string) => {
		const p = path.join(root, url.replace(/^\/+/, ''));
		try {
			const buf = await fs.readFile(p);
			return {
				ok: true,
				status: 200,
				json: async () => JSON.parse(buf.toString('utf8').replace(/^﻿/, '')),
				arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
				text: async () => buf.toString('utf8')
			};
		} catch {
			return { ok: false, status: 404, json: async () => null, arrayBuffer: async () => new ArrayBuffer(0), text: async () => '' };
		}
	};
}

export interface WithFixturesOpts {
	fetch?: FetchLike;
	base?: string;
	/** default (K1): sharedClaimsRunner(), which records each distinct run once per test file */
	runner?: SourceRunner;
	/** only these fixture ids */
	only?: string[];
	/** skip the presets (they need the presets file) */
	noPresets?: boolean;
}

/** Calls fn(rt, fixture) for every fixture (lesson fixtures, adapter edge fixtures, presets), each on its own runtime.
 *  Rejects with every failing fixture listed. */
export async function withFixtures(pkg: PackageModules, fn: (rt: AtlasRuntime, fx: FixtureCase) => void | Promise<void>, o: WithFixturesOpts = {}): Promise<{ ran: string[] }> {
	const fetch = o.fetch ?? (await nodeStaticFetch());
	const base = o.base ?? '';
	const runner = o.runner ?? sharedClaimsRunner();
	const probe = await createRuntime(pkg, { fetch, base, runner });
	const cases = fixtureCases(pkg, o.noPresets ? [] : probe.presets).filter((f) => !o.only || o.only.includes(f.id));
	probe.close();
	const failures: string[] = [];
	const ran: string[] = [];
	for (const fx of cases) {
		const rt = await createRuntime(pkg, { fetch, base, runner, input: fx.raw });
		try {
			await fn(rt, fx);
			ran.push(fx.id);
		} catch (e) {
			failures.push(`${fx.id}: ${e instanceof Error ? e.message : String(e)}`);
		} finally {
			rt.close();
		}
	}
	if (failures.length) throw new Error(`AK408 claim failed on ${failures.length} of ${cases.length} fixtures:\n  ${failures.join('\n  ')}`);
	return { ran };
}

/** A makeRuntime for lintLesson. */
export function runtimeMaker(pkg: PackageModules, o: { fetch: FetchLike; base: string; runner?: SourceRunner }): (fx: { raw: unknown }) => Promise<ExplainerRuntime> {
	return (fx) => createRuntime(pkg, { ...o, input: fx.raw });
}
