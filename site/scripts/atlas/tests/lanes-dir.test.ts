// Lanes outside the main site (ATLAS_LANES_DIR / "lanesDir"): the setting, lane.json's root and main, the roots a lane
// outside the repo resolves (TS and Python), its vite allow, worktree / --sync / merge / lanes --rm on it, lanes left in
// the default site/.work, and the shared node_modules check. Everything runs in throwaway folders.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { isInside, lanesChoice } from '../config.ts';
import { checkLanes } from '../commands/doctor.ts';
import { ALLOW_OUTSIDE, depsState, patchViteConfig } from '../lib/lanes.ts';
import { posix } from '../lib/io.ts';
import { findRoot, laneName, laneOf, resolveRoots } from '../root.ts';
import { PYTHON, SITE, cli, python } from './helpers.ts';

const VITE = `import { defineConfig } from 'vite';
export default defineConfig({
	server: {
		fs: {
			strict: true
		},
		watch: { ignored: ['**/.work/**', '**/.atlas/**'] }
	}
});
`;

function scratch() {
	const base = mkdtempSync(join(tmpdir(), 'atlas-lanesdir-'));
	const repo = join(base, 'repo');
	const site = join(repo, 'site');
	const out = join(base, 'elsewhere', 'lanes'); // another "drive": not inside the repo
	const put = (rel: string, text: string) => {
		const f = join(repo, ...rel.split('/'));
		mkdirSync(dirname(f), { recursive: true });
		writeFileSync(f, text);
	};
	put('CONTRACTS.md', '# temp\n');
	put('pipeline/README.md', 'pipeline\n');
	put('site/package.json', '{ "name": "scratch", "type": "module" }\n');
	put('site/package-lock.json', '{ "lockfileVersion": 3 }\n');
	put('site/svelte.config.js', 'export default {};\n');
	put('site/vite.config.ts', VITE);
	put('site/src/lib/kit/a.ts', 'export const a = 1;\n');
	put('site/src/lib/explainers/scratch/explainer.ts', 'export default { slug: "scratch" };\n');
	put('site/static/models/scratch/spec.json', '{}\n');
	const read = (rel: string) => readFileSync(join(repo, ...rel.split('/')), 'utf8');
	return {
		base,
		repo,
		site,
		out,
		put,
		read,
		env: { ATLAS_ROOT: repo, ATLAS_SITE: site, ATLAS_LANES_DIR: out },
		cleanup: () => rmSync(base, { recursive: true, force: true })
	};
}

test('lanesChoice: ATLAS_LANES_DIR, then the local and the shared config, then site/.work; relative to the main site', () => {
	const t = scratch();
	try {
		const cfg = join(t.site, 'scripts', 'atlas');
		mkdirSync(cfg, { recursive: true });
		let c = lanesChoice(t.site, {});
		assert.equal(c.from, 'default');
		assert.equal(resolve(c.dir), resolve(t.site, '.work'));
		assert.equal(c.outside, false);
		assert.equal(resolve(c.deps), resolve(t.site, 'node_modules'));
		writeFileSync(join(cfg, 'atlas.config.json'), '{ "python": "python", "lanesDir": "../../lanes-shared" }\n');
		c = lanesChoice(t.site, {});
		assert.equal(c.from, 'atlas.config.json');
		assert.equal(resolve(c.dir), resolve(t.base, 'lanes-shared'));
		assert.equal(c.outside, true);
		assert.equal(resolve(c.deps), resolve(t.base, 'lanes-shared', 'node_modules'));
		writeFileSync(join(cfg, 'atlas.config.local.json'), JSON.stringify({ python: 'py', lanesDir: posix(t.out) }));
		c = lanesChoice(t.site, {});
		assert.equal(c.from, 'atlas.config.local.json');
		assert.equal(resolve(c.dir), resolve(t.out));
		c = lanesChoice(t.site, { ATLAS_LANES_DIR: join(t.base, 'env-lanes') });
		assert.equal(c.from, 'ATLAS_LANES_DIR');
		assert.equal(resolve(c.dir), resolve(t.base, 'env-lanes'));
		c = lanesChoice(t.site, { ATLAS_LANES_DIR: '' });
		assert.equal(c.from, 'default', 'set but empty: the default, whatever the config files say');
		c = lanesChoice(t.site, { ATLAS_LANES_DIR: '.work/inner' });
		assert.equal(c.outside, false, 'a folder inside the main site still resolves its node_modules');
		assert.ok(isInside(join(t.site, 'a', 'b'), t.site) && !isInside(t.base, t.site) && isInside(t.site, t.site));
	} finally {
		t.cleanup();
	}
});

test('vite allow of a lane outside the main site: the lane and the shared node_modules, idempotent', () => {
	const p = patchViteConfig(VITE, ALLOW_OUTSIDE);
	assert.ok(p.patched);
	assert.match(p.text, /strict: true,\n\t\t\tallow: \['\.', '\.\.\/\.\.\/node_modules'\]\n/);
	assert.doesNotMatch(p.text, /\.work/);
	assert.equal(patchViteConfig(p.text, ALLOW_OUTSIDE).text, p.text);
	assert.doesNotMatch(patchViteConfig(VITE).text, /node_modules/, 'the default stays the in-repo allow');
});

test('a lane outside the repo: worktree, roots from lane.json (TS and Python), merge, --sync, lanes, lanes --rm', () => {
	const t = scratch();
	try {
		const w = cli(['worktree', 'scratch'], t.env);
		assert.equal(w.status, 0, w.stderr + w.stdout);
		const laneDir = join(t.out, 'scratch');
		const lane = join(laneDir, 'site');
		const L = (rel: string) => join(lane, ...rel.split('/'));
		assert.ok(existsSync(L('src/lib/kit/a.ts')), 'the lane is in the lanes folder');
		assert.ok(!existsSync(join(t.site, '.work')), 'nothing in site/.work');
		assert.match(w.stdout, /lanes folder .*\(ATLAS_LANES_DIR\); outside the main site/);
		assert.match(w.stdout, /WARN .*shared node_modules .*not installed: run npm run atlas -- lanes --deps/);
		assert.match(readFileSync(L('vite.config.ts'), 'utf8'), /allow: \['\.', '\.\.\/\.\.\/node_modules'\]/);
		const lj = JSON.parse(readFileSync(join(laneDir, 'lane.json'), 'utf8'));
		assert.equal(lj.root, posix(t.repo));
		assert.equal(lj.main, posix(t.site));

		// roots from inside the lane: no walk up finds the repo, lane.json does
		mkdirSync(L('src/lib'), { recursive: true });
		const r = resolveRoots({ ATLAS_SITE: lane });
		assert.equal(resolve(r.root), resolve(t.repo));
		assert.equal(r.rootFrom, 'lane');
		assert.equal(r.lane, 'scratch');
		assert.equal(resolve(r.laneDir!), resolve(laneDir));
		assert.equal(resolve(r.main), resolve(t.site));
		assert.equal(resolve(findRoot(lane)!), resolve(t.repo));
		assert.equal(laneName(lane), 'scratch');
		assert.equal(laneOf(t.site), null, 'the main tree is no lane');
		const rootTs = spawnSync(process.execPath, [join(SITE, 'scripts', 'atlas', 'root.ts')], { encoding: 'utf8', env: { ...process.env, ATLAS_SITE: lane, ATLAS_ROOT: '' } });
		assert.equal(rootTs.status, 0, rootTs.stderr);
		assert.equal(resolve(JSON.parse(rootTs.stdout).root), resolve(t.repo));
		if (PYTHON) {
			const py = python('from atlas_kit.paths import atlas_site, atlas_root, lane_name; print(atlas_site()); print(atlas_root()); print(lane_name())', { ATLAS_SITE: lane, ATLAS_ROOT: '' });
			assert.equal(py.status, 0, py.stderr);
			const [s, root, name] = py.stdout.trim().split(/\r?\n/);
			assert.equal(resolve(s), resolve(lane));
			assert.equal(resolve(root), resolve(t.repo));
			assert.equal(name, 'scratch');
		}

		// a lane with owned work: listed, merged from main, and not deletable until merged
		writeFileSync(L('src/lib/explainers/scratch/explainer.ts'), 'export default { slug: "scratch", v: 2 };\n');
		let d = cli(['merge', 'scratch', '--dry-run', '--json'], t.env);
		assert.equal(d.status, 0, d.stderr);
		assert.deepEqual(JSON.parse(d.stdout).copy, ['src/lib/explainers/scratch/explainer.ts']);
		const ls = cli(['lanes', '--json'], t.env);
		assert.equal(ls.status, 0, ls.stderr);
		const lj2 = JSON.parse(ls.stdout);
		assert.equal(lj2.outside, true);
		assert.equal(lj2.deps.ok, false);
		assert.deepEqual(lj2.lanes.map((x: { name: string }) => x.name), ['scratch']);
		const rm1 = cli(['lanes', '--rm', 'scratch'], t.env);
		assert.equal(rm1.status, 1);
		assert.match(rm1.stderr, /^AK702 {2}lane 'scratch' holds 1 owned change\(s\) that main does not have/m);
		assert.ok(existsSync(lane));

		// main moves on: --sync re-patches the lane's vite.config.ts with the outside allow
		t.put('site/src/lib/kit/a.ts', 'export const a = 3; // kit fix\n');
		t.put('site/vite.config.ts', VITE.replace('strict: true', 'strict: true // main edit'));
		const s = cli(['worktree', 'scratch', '--sync'], t.env);
		assert.equal(s.status, 0, s.stderr + s.stdout);
		const vite = readFileSync(L('vite.config.ts'), 'utf8');
		assert.ok(/main edit/.test(vite) && /allow: \['\.', '\.\.\/\.\.\/node_modules'\]/.test(vite), vite);
		assert.match(readFileSync(L('src/lib/kit/a.ts'), 'utf8'), /kit fix/);

		const m = cli(['merge', 'scratch'], t.env);
		assert.equal(m.status, 0, m.stderr + m.stdout);
		assert.match(t.read('site/src/lib/explainers/scratch/explainer.ts'), /v: 2/);
		assert.doesNotMatch(t.read('site/vite.config.ts'), /allow/);
		const rm2 = cli(['lanes', '--rm', 'scratch'], t.env);
		assert.equal(rm2.status, 0, rm2.stderr + rm2.stdout);
		assert.match(rm2.stdout, /atlas lanes --rm scratch: deleted .*scratch \(\d+ files/);
		assert.ok(!existsSync(laneDir));
	} finally {
		t.cleanup();
	}
});

test('lanes made in site/.work before a lanesDir was set are still found; a name is never used twice', () => {
	const t = scratch();
	try {
		const inRepo = { ...t.env, ATLAS_LANES_DIR: '' };
		assert.equal(cli(['worktree', 'old'], inRepo).status, 0);
		const oldLane = join(t.site, '.work', 'old', 'site');
		assert.match(readFileSync(join(oldLane, 'vite.config.ts'), 'utf8'), /allow: \['\.', '\.\.\/\.\.\/\.\.'\]/);
		// the setting changes: the old lane stays where it is and is still found
		writeFileSync(join(oldLane, 'static', 'models', 'scratch', 'facts.json'), '{}\n');
		const d = cli(['merge', 'old', '--dry-run', '--json'], { ...t.env, ATLAS_LANES_DIR: t.out });
		assert.equal(d.status, 1, 'an owned path of another slug is outside lane old');
		assert.deepEqual(JSON.parse(d.stdout).outside, ['static/models/scratch/facts.json']);
		const again = cli(['worktree', 'old'], t.env);
		assert.equal(again.status, 1);
		assert.match(again.stderr, /lane 'old' already exists: .*\.work\/old\/site/);
		assert.equal(cli(['worktree', 'fresh'], t.env).status, 0);
		const ls = JSON.parse(cli(['lanes', '--json'], t.env).stdout) as { lanes: { name: string; dir: string; outside: boolean }[] };
		assert.deepEqual(ls.lanes.map((x) => [x.name, x.outside]), [['fresh', true], ['old', false]]);
		rmSync(join(oldLane, 'static', 'models', 'scratch', 'facts.json'));
		const rm = cli(['lanes', '--rm', 'old'], t.env);
		assert.equal(rm.status, 0, rm.stderr);
		assert.ok(!existsSync(join(t.site, '.work', 'old')));
	} finally {
		t.cleanup();
	}
});

test('shared node_modules: not installed, up to date, stale; doctor warns with the fix', () => {
	const t = scratch();
	try {
		const lanes = lanesChoice(t.site, { ATLAS_LANES_DIR: t.out });
		assert.equal(depsState(lanes, t.site).why, 'not installed');
		const r = resolveRoots({ ATLAS_SITE: t.site, ATLAS_ROOT: t.repo });
		const [c] = checkLanes(r, { ATLAS_LANES_DIR: t.out });
		assert.equal(c.status, 'warn');
		assert.match(c.issue!.fix!, /npm run atlas -- lanes --deps/);
		// what npm ci leaves behind, and the copy of the lock lanes --deps keeps
		mkdirSync(lanes.deps, { recursive: true });
		writeFileSync(join(lanes.deps, '.package-lock.json'), '{}\n');
		writeFileSync(join(t.out, 'package-lock.json'), t.read('site/package-lock.json'));
		assert.equal(depsState(lanes, t.site).ok, true);
		assert.equal(checkLanes(r, { ATLAS_LANES_DIR: t.out })[0].status, 'ok');
		t.put('site/package-lock.json', '{ "lockfileVersion": 3, "packages": { "": { "name": "x" } } }\n');
		const st = depsState(lanes, t.site);
		assert.equal(st.ok, false);
		assert.match(st.why, /^stale/);
		assert.equal(checkLanes(r, { ATLAS_LANES_DIR: '' })[0].status, 'ok', 'lanes in site/.work need nothing more');
	} finally {
		t.cleanup();
	}
});
