// DevKit K0: the CLI fixes from wave 2a (LOG.md "K0"): the lane's vite watch patch, new op --local --like, promote's
// credit dedupe, the page-code count, the per-source parity record.
//   node --test scripts/atlas/tests/k0.test.ts   (from site/)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { patchViteConfig } from '../lib/lanes.ts';
import { appendCredits } from '../commands/promote.ts';
import { codeLines } from '../commands/check.ts';
import { sourceParityFile, type ParityReport } from '../commands/parity.ts';
import { cli, tempRepo } from './helpers.ts';

test('lane vite.config.ts: the .work watch pattern goes (it matched every file of the lane), idempotent', () => {
	const main = `export default defineConfig({
	server: {
		fs: {
			strict: true
		},
		watch: { ignored: ['**/.work/**', '**/.atlas/**', '**/build/**'] }
	}
});
`;
	const p = patchViteConfig(main);
	assert.ok(p.patched);
	assert.ok(!p.text.includes('.work'), p.text);
	assert.match(p.text, /ignored: \['\*\*\/\.atlas\/\*\*', '\*\*\/build\/\*\*'\]/);
	assert.match(p.text, /allow: \['\.', '\.\.\/\.\.\/\.\.'\]/);
	assert.equal(patchViteConfig(p.text).text, p.text, 'idempotent');
	assert.equal(patchViteConfig("watch: { ignored: ['**/build/**', '**/.work/**'] }").text, "watch: { ignored: ['**/build/**'] }");
});

test('new op --local --like: imports re-pointed, export renamed, op name changed, ops.ts written, mirror copied', async () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		assert.equal(cli(['new', 'model', 'demo-z', '--dev', '--kind', 'process'], env).status, 0);
		const r = cli(['new', 'op', 'gcn_conv', '--local', 'demo-z', '--like', 'linear'], env);
		assert.equal(r.status, 0, r.stderr + r.stdout);
		const pkg = join(t.site, 'src', 'lib', 'explainers', 'demo-z');
		const src = readFileSync(join(pkg, 'ops', 'gcn_conv.ts'), 'utf8');
		assert.match(src, /from '\.\.\/\.\.\/\.\.\/atlas\/types\/index\.ts'/, 'imports point at the kit from the package folder');
		assert.match(src, /export const gcnConv\b/);
		assert.match(src, /op: 'gcn_conv'/);
		const mod = (await import(pathToFileURL(join(pkg, 'ops', 'gcn_conv.ts')).href)) as { gcnConv: { op: string; forward: unknown } };
		assert.equal(mod.gcnConv.op, 'gcn_conv');
		const ops = (await import(pathToFileURL(join(pkg, 'ops.ts')).href)) as { default: { op: string }[] };
		assert.ok(ops.default.some((o) => o.op === 'gcn_conv') && ops.default.some((o) => o.op === 'linear'), 'ops.ts = CORE_OPS + the local op');
		const test = spawnSync(process.execPath, ['--test', join(pkg, 'ops', 'gcn_conv.test.ts')], { encoding: 'utf8' });
		assert.equal(test.status, 0, test.stdout + test.stderr);
		const py = readFileSync(join(t.repo, 'pipeline', 'demo-z', 'ops_gcn_conv.py'), 'utf8');
		assert.match(py, /^OP = "gcn_conv"$/m);
		assert.match(py, /def reference\(/);
	} finally {
		t.cleanup();
	}
});

test('promote: a credit whose url is already listed (under any name) is not added again', () => {
	const dir = mkdtempSync(join(tmpdir(), 'atlas-k0-'));
	try {
		const f = join(dir, 'licenses.json');
		writeFileSync(f, JSON.stringify([{ name: 'Quick, Draw! (TinyCNN)', url: 'https://github.com/googlecreativelab/quickdraw-dataset', license: 'CC BY 4.0', usedFor: 'x', attribution: 'y' }]));
		const c = { name: 'Quick, Draw! dataset', url: 'https://github.com/googlecreativelab/quickdraw-dataset', license: 'CC BY 4.0', usedFor: 'TinyViT', attribution: 'Google' };
		const r = appendCredits(f, [c, { ...c, name: 'Other', url: 'https://example.org/other' }], false);
		assert.deepEqual(r.added.map((x) => x.name), ['Other']);
		assert.deepEqual(r.sameUrl.map((x) => x.listedAs), ['Quick, Draw! (TinyCNN)']);
		assert.equal((JSON.parse(readFileSync(f, 'utf8')) as unknown[]).length, 2);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('page code count: comments, blank lines and <style> blocks are not code', () => {
	const ts = `// a comment\nconst a = 1;\n\n/** doc\n * more\n */\nconst b = 2; /* trailing */\n/* one */ const c = 3;\n`;
	assert.equal(codeLines(ts), 3);
	const sv = `<!--\n  header\n-->\n<script lang="ts">\n\tconst x = 1;\n</script>\n\n<p>{x}</p>\n<style>\n\tp { color: red; }\n\t.a { margin: 0; }\n</style>\n`;
	assert.equal(codeLines(sv, true), 4);
});

test('parity --write: the per-source record (atlas-parity/1) of twins and references', () => {
	const report = {
		slug: 'p',
		checked: '2026-10-06',
		train: [{ source: 'train', trainer: 'fit', twin: 'twin.py', pass: true, runs: [{ optimizer: 'sgd', steps: 50, losses: 1e-15, grads0: 1e-15, final: 7e-16, bar: 1e-10, pass: true }, { optimizer: 'adam', steps: 50, losses: 2e-15, grads0: 1e-15, final: 9e-16, bar: 1e-7, pass: true }] }],
		process: [
			{ source: 'q', process: 'q', reference: 'reference.py', frames: 400, refFrames: 400, firstDivergence: null, maxFloat: 0, pass: true },
			{ source: 'q', process: 'q', reference: 'reference.py', frames: 380, refFrames: 380, firstDivergence: null, maxFloat: 1e-17, pass: true, fixture: 'edge:wall' }
		],
		pass: true,
		wrote: null,
		issues: []
	} as unknown as ParityReport;
	const rec = sourceParityFile('p', report, { train: { kind: 'train', trainer: 'fit' }, hi: { kind: 'train', trainer: 'fit', opt: { lr: 3 } }, q: { kind: 'process', process: 'q' } })!;
	assert.equal(rec.format, 'atlas-parity/1');
	assert.deepEqual(Object.keys(rec.sources).sort(), ['hi', 'q', 'train']);
	assert.equal(rec.sources.train.scope, 'train');
	assert.equal(rec.sources.train.bar, 1e-7);
	assert.equal(rec.sources.q.fixtures, 1);
	assert.equal(rec.sources.q.max, 1e-17);
	assert.equal(sourceParityFile('p', { ...report, train: [], process: [] }, {}), null);
	assert.ok(existsSync(join(import.meta.dirname, '..', 'commands', 'parity.ts')));
});
