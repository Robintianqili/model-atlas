// Acceptance 3 (W1-E1): doctor is green on this machine, and reports a readable failure when the Python path is unset
// or wrong. The Python checks are run directly (checkPython) for the failure cases, so the test stays fast.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pythonChoice } from '../config.ts';
import { checkPython } from '../commands/doctor.ts';
import { formatIssue } from '../errors.ts';
import { resolveRoots } from '../root.ts';
import { PYTHON, cli } from './helpers.ts';

const roots = resolveRoots({});

test('doctor --json is green here', { skip: !PYTHON && 'no configured Python on this machine' }, () => {
	const r = cli(['doctor', '--json']);
	const j = JSON.parse(r.stdout) as { ok: boolean; checks: { name: string; status: string; detail: string }[] };
	const bad = j.checks.filter((c) => c.status === 'fail');
	assert.equal(r.status, 0, JSON.stringify(bad));
	assert.equal(j.ok, true);
	for (const name of ['node', 'packages', 'python', 'py-packages', 'atlas_kit', 'git']) assert.ok(j.checks.some((c) => c.name === name && c.status === 'ok'), name);
});

test('Python unset: ATLAS_PYTHON empty -> AK002 with the config file and the --node-only hint', () => {
	const checks = checkPython(roots, { ...process.env, ATLAS_PYTHON: '' });
	assert.equal(checks.length, 1);
	const c = checks[0];
	assert.equal(c.status, 'fail');
	assert.equal(c.code, 'AK002');
	const text = formatIssue(c.code!, c.issue!.message, c.issue!);
	assert.match(text, /^AK002 {2}no Python configured/);
	assert.match(text, /atlas\.config\.local\.json/);
	assert.match(text, /--node-only/);
	assert.match(text, /docs {4}site\/DEVKIT\.md#ak002/);
});

test('Python path wrong: "Python not found" names the path and where it came from', () => {
	const [c] = checkPython(roots, { ...process.env, ATLAS_PYTHON: 'C:/no/such/python.exe' });
	assert.equal(c.status, 'fail');
	assert.match(c.detail, /Python not found: 'C:\/no\/such\/python\.exe' \(from ATLAS_PYTHON\)/);
});

test('no local config and nothing on PATH: the shared default "python" fails readably', () => {
	const dir = mkdtempSync(join(tmpdir(), 'atlas-cfg-'));
	try {
		writeFileSync(join(dir, 'atlas.config.json'), '{ "python": "python" }\n');
		const env = { ...process.env, PATH: dir, Path: dir } as NodeJS.ProcessEnv;
		delete env.ATLAS_PYTHON;
		const choice = pythonChoice(env, dir);
		assert.deepEqual(choice, { python: 'python', from: 'atlas.config.json' });
		// 'python' is either missing from this PATH or a Python without the packages: both must fail readably
		const c = checkPython(roots, env, choice).find((x) => x.status === 'fail');
		assert.ok(c, 'expected a failing Python check');
		assert.equal(c.code, 'AK002');
		assert.match(c.issue!.fix!, /atlas\.config\.local\.json|requirements\.txt/);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('the full command exits 1 and prints the AK block when Python is unset', () => {
	const r = cli(['doctor'], { ATLAS_PYTHON: '' });
	assert.equal(r.status, 1);
	assert.match(r.stdout, /fail {2}python {6}no Python configured/);
	assert.match(r.stdout, /^AK002 {2}no Python configured/m);
	assert.match(r.stdout, /doctor: 1 problem to fix/);
	const nodeOnly = cli(['doctor', '--node-only'], { ATLAS_PYTHON: '' });
	assert.equal(nodeOnly.status, 0, nodeOnly.stdout);
});
