// W1-E2 acceptance 4: `worktree`, `merge --dry-run`, `merge` and `worktree --sync` on a scratch lane, in a throwaway
// repo (ATLAS_ROOT / ATLAS_SITE point there), so the real tree is never touched.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { matcher, patchViteConfig } from '../lib/lanes.ts';
import { cli } from './helpers.ts';

const VITE = `import { defineConfig } from 'vite';
export default defineConfig({
	server: {
		fs: {
			strict: true
		}
	}
});
`;

function scratchRepo() {
	const base = mkdtempSync(join(tmpdir(), 'atlas-lanes-'));
	const repo = join(base, 'repo');
	const site = join(repo, 'site');
	const put = (rel: string, text: string) => {
		const f = join(repo, ...rel.split('/'));
		mkdirSync(dirname(f), { recursive: true });
		writeFileSync(f, text);
	};
	put('CONTRACTS.md', '# temp\n');
	put('pipeline/README.md', 'pipeline\n');
	put('site/package.json', '{ "name": "scratch", "type": "module" }\n');
	put('site/svelte.config.js', 'export default {};\n');
	put('site/vite.config.ts', VITE);
	put('site/src/lib/kit/a.ts', 'export const a = 1;\n');
	put('site/src/lib/explainers/scratch/explainer.ts', 'export default { slug: "scratch" };\n');
	put('site/static/models/scratch/spec.json', '{}\n');
	put('site/static/models/other/spec.json', '{}\n');
	put('site/node_modules/junk/index.js', 'junk\n');
	put('site/.svelte-kit/junk.js', 'junk\n');
	put('site/.atlas/x/report.json', '{}\n');
	put('site/package-lock.json', '{}\n');
	const read = (rel: string) => readFileSync(join(repo, ...rel.split('/')), 'utf8');
	return { base, repo, site, put, read, env: { ATLAS_ROOT: repo, ATLAS_SITE: site }, cleanup: () => rmSync(base, { recursive: true, force: true }) };
}

test('lanes: globs and the vite.config.ts patch', () => {
	const m = matcher(['site/src/lib/atlas/{core,ops}/**', 'site/src/routes/m/[slug]/**', 'site/src/lib/atlas/ui/*.ts', 'notes/x.md']);
	assert.ok(m('site/src/lib/atlas/core/rng.ts'));
	assert.ok(m('site/src/lib/atlas/ops/a/b.ts'));
	assert.ok(!m('site/src/lib/atlas/graph/x.ts'));
	assert.ok(m('site/src/routes/m/[slug]/+page.ts'));
	assert.ok(m('site/src/lib/atlas/ui/registry.ts'));
	assert.ok(!m('site/src/lib/atlas/ui/views/x/meta.ts'));
	assert.ok(m('notes/x.md') && !m('notes/y.md'));
	const p = patchViteConfig(VITE);
	assert.ok(p.patched);
	assert.match(p.text, /strict: true,\n\t\t\tallow: \['\.', '\.\.\/\.\.\/\.\.'\]\n/);
	assert.equal(patchViteConfig(p.text).text, p.text, 'idempotent');
});

test('acceptance 4: worktree, merge --dry-run, merge and --sync on a scratch lane', () => {
	const t = scratchRepo();
	try {
		// worktree: a copy without node_modules/.svelte-kit/.atlas/package-lock, patched vite config, lane.json snapshot
		const w = cli(['worktree', 'scratch'], t.env);
		assert.equal(w.status, 0, w.stderr + w.stdout);
		const lane = join(t.site, '.work', 'scratch', 'site');
		const L = (rel: string) => join(lane, ...rel.split('/'));
		assert.ok(existsSync(L('src/lib/kit/a.ts')) && existsSync(L('static/models/scratch/spec.json')));
		for (const skipped of ['node_modules', '.svelte-kit', '.atlas', 'package-lock.json']) assert.ok(!existsSync(L(skipped)), `${skipped} is not copied`);
		assert.match(readFileSync(L('vite.config.ts'), 'utf8'), /allow: \['\.', '\.\.\/\.\.\/\.\.'\]/);
		const lj = JSON.parse(readFileSync(join(t.site, '.work', 'scratch', 'lane.json'), 'utf8'));
		assert.equal(lj.format, 'atlas-lane/1');
		assert.deepEqual(lj.owns, ['pipeline/scratch/**', 'site/src/lib/explainers/scratch/**', 'site/static/models/scratch/**', 'notes/scratch.md', 'notes/shots/scratch/**']);
		assert.equal(Object.keys(lj.snapshot).length, 7);
		assert.equal(cli(['worktree', 'scratch'], t.env).status, 1, 'an existing lane is never overwritten');

		// a fresh lane merges clean, and the patched vite.config.ts is never a change
		let d = cli(['merge', 'scratch', '--dry-run', '--json'], t.env);
		assert.equal(d.status, 0, d.stderr);
		assert.deepEqual(JSON.parse(d.stdout).copy, []);

		// owned edits: listed by --dry-run, main untouched
		writeFileSync(L('src/lib/explainers/scratch/explainer.ts'), 'export default { slug: "scratch", v: 2 };\n');
		writeFileSync(L('static/models/scratch/facts.json'), '{"format":"atlas-facts/1","facts":{}}\n');
		d = cli(['merge', 'scratch', '--dry-run', '--json'], t.env);
		assert.equal(d.status, 0, d.stderr);
		assert.deepEqual(JSON.parse(d.stdout).copy, ['src/lib/explainers/scratch/explainer.ts', 'static/models/scratch/facts.json']);
		assert.doesNotMatch(t.read('site/src/lib/explainers/scratch/explainer.ts'), /v: 2/);

		// an edit outside the owned paths: dry-run and merge refuse with AK702, nothing is copied
		writeFileSync(L('src/lib/kit/a.ts'), 'export const a = 2;\n');
		d = cli(['merge', 'scratch', '--dry-run'], t.env);
		assert.equal(d.status, 1);
		assert.match(d.stderr, /^AK702 {2}merge refused: lane 'scratch' changed 1 file\(s\) outside its owned paths/m);
		assert.match(d.stderr, /where {3}src\/lib\/kit\/a\.ts/);
		const m1 = cli(['merge', 'scratch'], t.env);
		assert.equal(m1.status, 1);
		assert.doesNotMatch(t.read('site/src/lib/explainers/scratch/explainer.ts'), /v: 2/, 'a refused merge copies nothing');

		// --sync refuses (AK703) while the lane has a non-owned change, and changes nothing
		const s1 = cli(['worktree', 'scratch', '--sync'], t.env);
		assert.equal(s1.status, 1);
		assert.match(s1.stderr, /^AK703 {2}sync refused/m);
		writeFileSync(L('src/lib/kit/a.ts'), 'export const a = 1;\n'); // undo the kit edit

		// main moves on (kit fix, new kit file, removed file); --sync brings it in and keeps the owned edits
		t.put('site/src/lib/kit/a.ts', 'export const a = 3; // kit fix\n');
		t.put('site/src/lib/kit/b.ts', 'export const b = 1;\n');
		rmSync(join(t.site, 'static', 'models', 'other', 'spec.json'));
		t.put('site/vite.config.ts', VITE.replace('strict: true', 'strict: true // main edit'));
		const s2 = cli(['worktree', 'scratch', '--sync'], t.env);
		assert.equal(s2.status, 0, s2.stderr + s2.stdout);
		assert.match(s2.stdout, /updated 2, added 1, removed 1 non-owned file\(s\)/);
		assert.match(readFileSync(L('src/lib/kit/a.ts'), 'utf8'), /kit fix/);
		assert.ok(existsSync(L('src/lib/kit/b.ts')));
		assert.ok(!existsSync(L('static/models/other/spec.json')));
		const vite = readFileSync(L('vite.config.ts'), 'utf8');
		assert.ok(/main edit/.test(vite) && /allow: \['\.', '\.\.\/\.\.\/\.\.'\]/.test(vite), `re-patched after the sync:\n${vite}`);
		assert.match(readFileSync(L('src/lib/explainers/scratch/explainer.ts'), 'utf8'), /v: 2/, 'owned edits survive a sync');

		// merge copies the owned changes only; main keeps its own kit fix and its unpatched vite config
		const m2 = cli(['merge', 'scratch'], t.env);
		assert.equal(m2.status, 0, m2.stderr + m2.stdout);
		assert.match(t.read('site/src/lib/explainers/scratch/explainer.ts'), /v: 2/);
		assert.ok(existsSync(join(t.site, 'static', 'models', 'scratch', 'facts.json')));
		assert.match(t.read('site/src/lib/kit/a.ts'), /kit fix/);
		assert.doesNotMatch(t.read('site/vite.config.ts'), /allow/);
		d = cli(['merge', 'scratch', '--dry-run', '--json'], t.env);
		assert.deepEqual(JSON.parse(d.stdout).copy, [], 'nothing left to merge');

		// an owned file changed in both trees since the snapshot: refused
		t.put('site/static/models/scratch/spec.json', '{"main": 1}\n');
		writeFileSync(L('static/models/scratch/spec.json'), '{"lane": 1}\n');
		d = cli(['merge', 'scratch', '--dry-run'], t.env);
		assert.equal(d.status, 1);
		assert.match(d.stderr, /changed both in lane 'scratch' and in main/);
	} finally {
		t.cleanup();
	}
});
