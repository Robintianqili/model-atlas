// Acceptance 4 (W1-E1): ATLAS_ROOT resolves correctly from the main tree and from a .work lane, in TS and in Python.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { findRoot, laneName, resolveRoots } from '../root.ts';
import { PYTHON, ROOT, SITE, python, tempRepo } from './helpers.ts';

const MAIN_SITE = join(ROOT, 'site');

test('main tree: ATLAS_SITE = site/, ATLAS_ROOT = the repo root', () => {
	const r = resolveRoots({ ATLAS_SITE: MAIN_SITE });
	assert.equal(resolve(r.site), resolve(MAIN_SITE));
	assert.equal(resolve(r.root), resolve(ROOT));
	assert.equal(r.lane, null);
	assert.ok(existsSync(join(r.root, 'CONTRACTS.md')) && existsSync(join(r.root, 'pipeline')));
});

test('lane: site/.work/<name>/site resolves to the same repo root (site/.work/<name> and site/ have no pipeline/)', () => {
	const lanes = existsSync(join(MAIN_SITE, '.work'))
		? readdirSync(join(MAIN_SITE, '.work')).filter((n) => existsSync(join(MAIN_SITE, '.work', n, 'site', 'svelte.config.js')))
		: [];
	if (!lanes.length) return; // no lane on this machine
	for (const name of lanes) {
		const r = resolveRoots({ ATLAS_SITE: join(MAIN_SITE, '.work', name, 'site') });
		assert.equal(resolve(r.root), resolve(ROOT), name);
		assert.equal(r.lane, name);
	}
});

test('the CLI defaults to the site it belongs to (this file runs in ' + (laneName(SITE) ?? 'the main tree') + ')', () => {
	const r = resolveRoots({});
	assert.equal(resolve(r.site), resolve(SITE));
	assert.equal(resolve(r.root), resolve(ROOT));
	assert.equal(r.siteFrom, 'cli');
	const out = spawnSync(process.execPath, [join(SITE, 'scripts', 'atlas', 'root.ts')], { encoding: 'utf8', env: { ...process.env, ATLAS_SITE: '', ATLAS_ROOT: '' } });
	assert.equal(out.status, 0, out.stderr);
	const j = JSON.parse(out.stdout);
	assert.equal(resolve(j.root), resolve(ROOT));
	assert.equal(j.lane, laneName(SITE));
});

test('synthetic layout: findRoot skips folders without pipeline/', () => {
	const t = tempRepo();
	try {
		const lane = join(t.site, '.work', 'x', 'site');
		mkdirSync(join(lane, 'src', 'lib'), { recursive: true });
		writeFileSync(join(lane, 'svelte.config.js'), '');
		writeFileSync(join(lane, 'package.json'), '{}');
		assert.equal(resolve(findRoot(lane)!), resolve(t.repo));
		assert.equal(laneName(lane), 'x');
		const r = resolveRoots({ ATLAS_SITE: lane });
		assert.equal(resolve(r.root), resolve(t.repo));
	} finally {
		t.cleanup();
	}
});

test('bad overrides fail with AK002 and a fix', () => {
	assert.throws(() => resolveRoots({ ATLAS_SITE: ROOT }), (e: Error & { code?: string }) => e.code === 'AK002');
	assert.throws(() => resolveRoots({ ATLAS_ROOT: SITE }), (e: Error & { code?: string }) => e.code === 'AK002');
});

test('Python resolves the same roots (main tree and lane)', { skip: !PYTHON && 'no configured Python with atlas_kit' }, () => {
	const code = 'from atlas_kit.paths import atlas_site, atlas_root; print(atlas_site()); print(atlas_root())';
	for (const site of [MAIN_SITE, SITE]) {
		// as the CLI calls it (env set) ...
		const a = python(code, { ATLAS_SITE: site, ATLAS_ROOT: ROOT });
		assert.equal(a.status, 0, a.stderr);
		const [s1, r1] = a.stdout.trim().split(/\r?\n/);
		assert.equal(resolve(s1), resolve(site));
		assert.equal(resolve(r1), resolve(ROOT));
		// ... and on its own, from a folder inside the site (env unset)
		const b = spawnSync(PYTHON!, ['-c', code], { cwd: join(site, 'src', 'lib'), env: { ...process.env, ATLAS_SITE: '', ATLAS_ROOT: '', PYTHONPATH: join(ROOT, 'pipeline') }, encoding: 'utf8' });
		assert.equal(b.status, 0, b.stderr);
		const [s2, r2] = b.stdout.trim().split(/\r?\n/);
		assert.equal(resolve(s2), resolve(site));
		assert.equal(resolve(r2), resolve(ROOT));
	}
});
