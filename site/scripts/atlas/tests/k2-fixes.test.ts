// K2 kit fixes (CLI side): `atlas worktree <name> --from-git <rev>` and `--sync --from-git <rev>` in a throwaway git repo
// (ATLAS_ROOT / ATLAS_SITE point there), so the real tree is never touched. Skipped when git is not on PATH.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { untar } from '../commands/worktree.ts';
import { cli } from './helpers.ts';

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

const hasGit = spawnSync('git', ['--version'], { encoding: 'utf8' }).status === 0;

function gitRepo() {
	const base = mkdtempSync(join(tmpdir(), 'atlas-k2-'));
	const repo = join(base, 'repo');
	const site = join(repo, 'site');
	const put = (rel: string, text: string) => {
		const f = join(repo, ...rel.split('/'));
		mkdirSync(dirname(f), { recursive: true });
		writeFileSync(f, text);
	};
	const git = (...args: string[]) => {
		const r = spawnSync('git', ['-c', 'user.name=atlas', '-c', 'user.email=atlas@example.invalid', '-c', 'commit.gpgsign=false', '-c', `safe.directory=${repo.replace(/\\/g, '/')}`, '-C', repo, ...args], { encoding: 'utf8' });
		assert.equal(r.status, 0, `git ${args.join(' ')}: ${r.stderr}`);
		return r.stdout.trim();
	};
	put('CONTRACTS.md', '# temp\n');
	put('pipeline/README.md', 'pipeline\n');
	put('site/package.json', '{ "name": "scratch", "type": "module" }\n');
	put('site/package-lock.json', '{}\n');
	put('site/svelte.config.js', 'export default {};\n');
	put('site/vite.config.ts', VITE);
	put('site/.gitignore', '/scripts/atlas/atlas.config.local.json\n');
	put('site/src/lib/kit/a.ts', 'export const a = 1;\n');
	put(`site/src/lib/explainers/scratch/${'long-folder-name-'.repeat(6)}x/explainer.ts`, 'export default { slug: "scratch" };\n');
	put('site/static/models/scratch/spec.json', '{}\n');
	git('init', '-q');
	git('config', 'core.autocrlf', 'false');
	git('add', '-A');
	git('commit', '-q', '-m', 'one');
	// machine-local, git-ignored: copied from main into a --from-git lane
	put('site/scripts/atlas/atlas.config.local.json', '{ "python": "py" }\n');
	const read = (rel: string) => readFileSync(join(repo, ...rel.split('/')), 'utf8');
	return { base, repo, site, put, git, read, env: { ATLAS_ROOT: repo, ATLAS_SITE: site }, cleanup: () => rmSync(base, { recursive: true, force: true }) };
}

test('untar reads ustar entries, pax long paths and skips folders', () => {
	const block = (name: string, size: number, type: string) => {
		const h = Buffer.alloc(512);
		h.write(name, 0, 'utf8');
		h.write(size.toString(8).padStart(11, '0') + '\0', 124, 'ascii');
		h.write(type, 156, 'ascii');
		h.write('ustar\u000000', 257, 'ascii');
		return h;
	};
	const body = (b: Buffer) => Buffer.concat([b, Buffer.alloc((512 - (b.length % 512)) % 512)]);
	const long = `site/${'d'.repeat(120)}/f.txt`;
	const rec = `path=${long}\n`;
	let len = rec.length + 3;
	if (String(len).length + 1 + rec.length !== len) len = String(len + 1).length + 1 + rec.length;
	const pax = Buffer.from(`${len} ${rec}`);
	const tar = Buffer.concat([
		block('pax_global_header', 52, 'g'),
		body(Buffer.from('52 comment=0123456789012345678901234567890123456789\n')),
		block('site/', 0, '5'),
		block('site/a.txt', 3, '0'),
		body(Buffer.from('abc')),
		block('PaxHeaders/x', pax.length, 'x'),
		body(pax),
		block('site/ddd/f.txt', 2, '0'),
		body(Buffer.from('hi')),
		Buffer.alloc(1024)
	]);
	const e = untar(tar).filter((x) => x.type === '0');
	assert.deepEqual(e.map((x) => x.path), ['site/a.txt', long]);
	assert.equal(e[1].data.toString(), 'hi');
});

test('worktree --from-git <rev>: the lane holds the commit, not the working tree; --sync --from-git takes a later commit', { skip: !hasGit }, () => {
	const t = gitRepo();
	try {
		const one = t.git('rev-parse', 'HEAD');
		// main's working tree moves on without a commit: the lane must not see it
		t.put('site/src/lib/kit/a.ts', 'export const a = 2; // uncommitted\n');
		t.put('site/src/lib/kit/wip.ts', 'export const wip = 1;\n');
		const w = cli(['worktree', 'scratch', '--from-git', 'HEAD'], t.env);
		assert.equal(w.status, 0, w.stderr + w.stdout);
		assert.match(w.stdout, new RegExp(`from commit ${one.slice(0, 7)} \\(HEAD\\)`));
		assert.match(w.stdout, /local config {4}scripts\/atlas\/atlas\.config\.local\.json copied from main/);
		const lane = join(t.site, '.work', 'scratch', 'site');
		const L = (rel: string) => join(lane, ...rel.split('/'));
		assert.equal(readFileSync(L('src/lib/kit/a.ts'), 'utf8'), 'export const a = 1;\n', 'the committed version');
		assert.ok(!existsSync(L('src/lib/kit/wip.ts')), 'an uncommitted file is not in the lane');
		assert.ok(existsSync(L(`src/lib/explainers/scratch/${'long-folder-name-'.repeat(6)}x/explainer.ts`)), 'a path over 100 characters');
		assert.ok(!existsSync(L('package-lock.json')), 'the lane skips apply');
		assert.ok(existsSync(L('scripts/atlas/atlas.config.local.json')));
		const vite = readFileSync(L('vite.config.ts'), 'utf8');
		assert.match(vite, /allow: \['\.', '\.\.\/\.\.\/\.\.'\]/);
		assert.doesNotMatch(vite, /\.work/);
		const lj = JSON.parse(readFileSync(join(t.site, '.work', 'scratch', 'lane.json'), 'utf8'));
		assert.equal(lj.rev, one);
		assert.equal(Object.keys(lj.snapshot).length, 7, Object.keys(lj.snapshot).join(', '));

		// a fresh lane from HEAD: ownership OK; main's uncommitted kit edit is "main changed", not a lane change
		let d = cli(['merge', 'scratch', '--dry-run', '--json'], t.env);
		assert.equal(d.status, 0, d.stderr);
		assert.deepEqual(JSON.parse(d.stdout).copy, []);

		// --sync (from main) would take the uncommitted edit; --sync --from-git HEAD keeps the committed state
		d = cli(['worktree', 'scratch', '--sync', '--dry-run'], t.env);
		assert.match(d.stdout, /from main\): update 1, add 1/);
		d = cli(['worktree', 'scratch', '--sync', '--dry-run', '--from-git', 'HEAD'], t.env);
		assert.match(d.stdout, /update 0, add 0, remove 0, conflicts 0/);

		// a later commit (kit fix, new and removed files); the lane edits an owned file meanwhile
		t.put('site/src/lib/kit/a.ts', 'export const a = 3; // kit fix\n');
		t.put('site/src/lib/kit/b.ts', 'export const b = 1;\n');
		rmSync(join(t.site, 'static', 'models', 'scratch', 'spec.json'));
		t.put('site/static/models/other/spec.json', '{}\n');
		t.git('add', 'site/src/lib/kit/a.ts', 'site/src/lib/kit/b.ts', 'site/static');
		t.git('commit', '-q', '-m', 'two');
		const two = t.git('rev-parse', 'HEAD');
		writeFileSync(L('src/lib/explainers/scratch/explainer.ts'), 'export default { v: 2 };\n');
		const s = cli(['worktree', 'scratch', '--sync', '--from-git', 'HEAD'], t.env);
		assert.equal(s.status, 0, s.stderr + s.stdout);
		assert.match(s.stdout, new RegExp(`from HEAD \\(${two.slice(0, 7)}\\)\\): updated 1, added 2, removed 0`));
		assert.match(readFileSync(L('src/lib/kit/a.ts'), 'utf8'), /kit fix/);
		assert.ok(existsSync(L('src/lib/kit/b.ts')) && existsSync(L('static/models/other/spec.json')));
		assert.ok(!existsSync(L('src/lib/kit/wip.ts')), 'still nothing uncommitted');
		assert.ok(existsSync(L('static/models/scratch/spec.json')), 'owned files are never touched by a sync');
		assert.equal(readFileSync(L('src/lib/explainers/scratch/explainer.ts'), 'utf8'), 'export default { v: 2 };\n');
		assert.ok(existsSync(L('scripts/atlas/atlas.config.local.json')), 'the local config stays');
		assert.equal(JSON.parse(readFileSync(join(t.site, '.work', 'scratch', 'lane.json'), 'utf8')).rev, two);

		// an unknown revision: AK301, nothing written
		const bad = cli(['worktree', 'other', '--from-git', 'no-such-branch'], t.env);
		assert.equal(bad.status, 1);
		assert.match(bad.stderr, /AK301 {2}--from-git: no commit 'no-such-branch'/);
		assert.ok(!existsSync(join(t.site, '.work', 'other')));
	} finally {
		t.cleanup();
	}
});
