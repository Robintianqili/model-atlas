// CLI fixes before wave 2a (afternoon test + R0): DEVKIT.md's generated tables are part of the check:atlas gate,
// `atlas py`, `atlas new model <slug> --like <package>`, the dev-server port that `new` picks and records, the numbered
// "Next:" list, and the ports e2e uses.
// Run from site/: node --test scripts/atlas/tests/kitfix.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { ExplainerManifest } from '../../../src/lib/atlas/types/index.ts';
import { generate } from '../docs.ts';
import { loadPackage } from '../load.ts';
import { PORTS, claimedPorts, notesPort, packagePort, pickPort, portFree, suggestPort } from '../ports.ts';
import { renameSlug } from '../commands/new.ts';
import { resolveScript } from '../commands/py.ts';
import { PYTHON, ROOT, SITE, cli, tempRepo } from './helpers.ts';

test('docs gate: the generated tables of DEVKIT.md (commands, codes, ports, ops, views) match the code', async () => {
	const file = join(SITE, 'DEVKIT.md');
	const text = readFileSync(file, 'utf8');
	const r = await generate(SITE, text);
	assert.deepEqual(r.blocks.sort(), ['codes', 'commands', 'ops', 'ports', 'views']);
	assert.ok(r.text === text, 'DEVKIT.md is out of date: run node scripts/atlas/docs.ts --write (and review the diff)');
});

test('atlas py: resolves a script from the working folder or the repo root and runs it with the configured Python', { skip: !PYTHON && 'no configured Python' }, () => {
	assert.equal(resolveScript('pipeline/atlas_kit/__init__.py', SITE, ROOT), join(ROOT, 'pipeline', 'atlas_kit', '__init__.py'));
	// K2: relative to this site, which in a lane is site/.work/<name>/site (was a fixed '../pipeline', so the test failed in every lane)
	assert.equal(resolveScript(relative(SITE, join(ROOT, 'pipeline', 'atlas_kit', '__init__.py')).split('\\').join('/'), SITE, ROOT), join(ROOT, 'pipeline', 'atlas_kit', '__init__.py'));
	assert.equal(resolveScript('pipeline/nope.py', SITE, ROOT), null);
	const r = cli(['py', '-c', 'import os, sys, atlas_kit; print(sys.version_info >= (3, 10), os.environ["ATLAS_ROOT"] != "", atlas_kit.__name__)'], { INIT_CWD: SITE });
	assert.equal(r.status, 0, r.stderr);
	assert.equal(r.stdout.trim(), 'True True atlas_kit');
	const bad = cli(['py', 'pipeline/does-not-exist.py']);
	assert.equal(bad.status, 1);
	assert.match(bad.stderr, /AK301 {2}no script/);
	const code = cli(['py', '-c', 'import sys; sys.exit(3)']);
	assert.equal(code.status, 3, 'the exit code is the script’s');
});

test('ports: free-port check, the port notes record, and a pick that skips the table, claimed and busy ports', async () => {
	const t = tempRepo();
	try {
		writeFileSync(join(t.repo, 'notes', 'a-pkg.md'), '# a-pkg\n\n- Dev server: `npm run dev -- --port 5233`, then ...\n');
		assert.equal(notesPort(t.repo, 'a-pkg'), 5233);
		assert.equal(packagePort('a-pkg', t.repo), 5233);
		assert.equal(packagePort('tinyvit', t.repo), PORTS.tinyvit);
		assert.equal(packagePort('nobody', t.repo), null);
		const claimed = claimedPorts(t.repo, 'b-pkg');
		assert.ok(claimed.has(5233) && claimed.has(PORTS['hello-xor']));
		const s = suggestPort('b-pkg', t.repo);
		assert.ok(!claimed.has(s) && s >= 5210);
		const p = await pickPort({ slug: 'b-pkg', root: t.repo });
		assert.ok(!claimed.has(p) && p >= 5210 && (await portFree(p)));
		// a busy preferred port is skipped
		const { createServer } = await import('node:net');
		const srv = createServer();
		await new Promise<void>((res) => srv.listen({ port: p, host: '127.0.0.1' }, () => res()));
		try {
			assert.equal(await portFree(p), false);
			assert.notEqual(await pickPort({ preferred: p, slug: 'b-pkg', root: t.repo }), p);
		} finally {
			await new Promise<void>((res) => srv.close(() => res()));
		}
	} finally {
		t.cleanup();
	}
});

test('renameSlug: whole-word kebab, camelCase and snake_case forms', () => {
	const src = "slug: 'hello-xor', dir: 'models/hello-xor', id: 'hello-xor-pre', helloXor, hello_xor.py, not othello-xor or hello-xorz";
	const r = renameSlug(src, 'hello-xor', 'my-net');
	assert.equal(r.text, "slug: 'my-net', dir: 'models/my-net', id: 'my-net-pre', myNet, my_net.py, not othello-xor or hello-xorz");
	assert.equal(r.count, 5);
});

test('new model --like: a renamed copy of hello-xor (sources only), devOnly, the port it records, a numbered Next list', async () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		cpSync(join(SITE, 'src', 'lib', 'explainers', 'hello-xor'), join(t.site, 'src', 'lib', 'explainers', 'hello-xor'), { recursive: true });
		cpSync(join(ROOT, 'pipeline', 'hello-xor'), join(t.repo, 'pipeline', 'hello-xor'), { recursive: true });
		const r = cli(['new', 'model', 'my-xor', '--like', 'hello-xor', '--dev', '--port', '5333'], env);
		assert.equal(r.status, 0, r.stderr + r.stdout);
		const pkg = join(t.site, 'src', 'lib', 'explainers', 'my-xor');
		for (const f of ['explainer.ts', 'lesson.ts', 'trainer.ts', 'process.ts', 'claims.test.ts', 'adapters/xor-point/adapter.ts', 'adapters/xor-point/Capture.svelte'])
			assert.ok(existsSync(join(pkg, f)), f);
		for (const f of ['model.py', 'train.py', 'export.py', 'twin.py', 'reference.py', 'README.md']) assert.ok(existsSync(join(t.repo, 'pipeline', 'my-xor', f)), f);
		for (const f of ['model_final.pt', 'train_metrics.json']) assert.ok(!existsSync(join(t.repo, 'pipeline', 'my-xor', f)), `${f} is an output of the old model`);
		const m = ((await import(pathToFileURL(join(pkg, 'explainer.ts')).href)) as { default: ExplainerManifest }).default;
		assert.equal(m.slug, 'my-xor');
		assert.equal(m.devOnly, true);
		assert.equal(m.assets?.dir, 'models/my-xor');
		const loaded = await loadPackage('my-xor', t.site);
		assert.deepEqual(Object.keys(loaded.pkg.processes).sort(), ['sgdStep', 'train:xor'], 'ids are per package: kept');
		assert.match(readFileSync(join(t.repo, 'pipeline', 'my-xor', 'export.py'), 'utf8'), /my-xor/);
		assert.doesNotMatch(readFileSync(join(t.repo, 'pipeline', 'my-xor', 'export.py'), 'utf8'), /hello-xor/);
		const notes = readFileSync(join(t.repo, 'notes', 'my-xor.md'), 'utf8');
		assert.match(notes, /npm run dev -- --port 5333/);
		assert.match(notes, /atlas new model my-xor --like hello-xor/);
		// the Next list is numbered 1, 2, 3, ... and names the port it recorded
		const nums = [...r.stdout.matchAll(/^ {2}(\d+)\. /gm)].map((x) => Number(x[1]));
		assert.deepEqual(nums, nums.map((_, i) => i + 1));
		assert.ok(nums.length >= 4);
		assert.match(r.stdout, /npm run dev -- --port 5333/);
		assert.match(r.stdout, /claims\.test\.ts states facts about 'hello-xor'/);
		// nothing is overwritten
		const again = cli(['new', 'model', 'my-xor', '--like', 'hello-xor', '--dev'], env);
		assert.equal(again.status, 1);
		assert.match(again.stderr, /already exists/);
		// a new package without --port gets a port that no table row and no notes claim (here: not 5333)
		const y = cli(['new', 'model', 'demo-y', '--dev', '--kind', 'process'], env);
		assert.equal(y.status, 0, y.stderr + y.stdout);
		const py = notesPort(t.repo, 'demo-y')!;
		assert.ok(py >= 5210 && py !== 5333 && !Object.values(PORTS).includes(py), `demo-y got ${py}`);
		assert.match(y.stdout, new RegExp(`npm run dev -- --port ${py}`));
	} finally {
		t.cleanup();
	}
});
