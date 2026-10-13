// DevKit K4: the CLI fixes from batches 7 and 8 (LOG.md "K4"): promote --verify / --apply and the catalog patcher, the
// review packet's owned path, shots for lanes outside the repo, lanes --clean, the check gate changes, the port race.
// Run from site/:  node --test scripts/atlas/tests/k4.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:net';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { addRows, eolOf, removeRows, sameStyle, setFields } from '../lib/catalogpatch.ts';
import { catalogBytes, similarRow, similarRows, sizeBytes } from '../commands/promote.ts';
import { pageLines, testTimeoutMs, viewContract, withoutCatalog } from '../commands/check.ts';
import { cacheDirs } from '../commands/lanes.ts';
import { modelOwns, withReview } from '../lib/lanes.ts';
import { inside, shotsDir, type Roots } from '../root.ts';
import { pickPort } from '../ports.ts';
import { cli, SITE, tempRepo } from './helpers.ts';

const CRLF_SRC = [
	"import type { ModelEntry, I18n } from './types.ts';",
	'const f = (en: string, zh: string): I18n => ({ en, zh });',
	"const n = (name: string, zh?: string): I18n => (zh ? { en: name, zh } : { en: name, zh: name });",
	'export const MODELS: ModelEntry[] = [',
	'\t{',
	"\t\tslug: 'a', // the first",
	"\t\tname: n('A'),",
	"\t\thook: f(",
	"\t\t\t'An { odd } hook, with a comma, and a \\'quote\\'',",
	"\t\t\t'钩子'",
	'\t\t),',
	"\t\tstatus: 'planned',",
	"\t\tinputs: ['image', 'webcam'],",
	"\t\tprereqs: []",
	'\t},',
	'\t{',
	"\t\tslug: 'b',",
	"\t\tstatus: 'planned',",
	'\t\tlesson: { steps: 1, summary: f(\'x\', \'x\') }',
	'\t}',
	'];',
	''
].join('\r\n');

test('catalog patch: fields set or added in the right entry, CRLF kept; rows appended after a last row without a comma', () => {
	assert.equal(eolOf(CRLF_SRC), '\r\n');
	const r = setFields(CRLF_SRC, 'MODELS', 'a', [
		['status', { kind: 'string', v: 'live' }],
		['inputs', { kind: 'list', v: ['drawing'] }],
		['mode', { kind: 'string', v: 'live' }],
		['lesson', { kind: 'lesson', steps: 9, summary: { en: 'Nine steps.', zh: '九步。' } }]
	]);
	assert.deepEqual(r.set, ['status', 'inputs']);
	assert.deepEqual(r.added, ['mode', 'lesson']);
	const out = sameStyle(CRLF_SRC, r.text);
	assert.ok(!/[^\r]\n/.test(out), 'every line ends with CRLF');
	assert.match(out, /slug: 'a', \/\/ the first\r\n/);
	assert.match(out, /inputs: \['drawing'\],\r\n/);
	assert.match(out, /prereqs: \[\],\r\n\t\tmode: 'live',\r\n\t\tlesson: \{ steps: 9, summary: f\('Nine steps\.', '九步。'\) \}\r\n\t\},/);
	assert.match(out, /slug: 'b',\r\n\t\tstatus: 'planned'/, 'the other entry is untouched');
	// a field goes before an existing lesson
	const b = setFields(CRLF_SRC, 'MODELS', 'b', [['mode', { kind: 'string', v: 'both' }]]).text;
	assert.match(b, /status: 'planned',\r\n\t\tmode: 'both',\r\n\t\tlesson:/);
	assert.throws(() => setFields(CRLF_SRC, 'MODELS', 'zz', []).text.length && setFields(CRLF_SRC, 'MODELS', 'zz', [['status', { kind: 'string', v: 'live' }]]), /no entry with slug 'zz'/);

	const glo = "export const GLOSSARY: X[] = [\n\t// ---- one\n\t{ id: 'a', term: f('a', 'a'), def: f('d', 'd') }\n\t// a trailing note\n];\n";
	const g2 = addRows(glo, 'GLOSSARY', ["{ id: 'b', term: f('b', 'b'), def: f('e', 'e') }"], '// ---- two');
	assert.equal(g2, "export const GLOSSARY: X[] = [\n\t// ---- one\n\t{ id: 'a', term: f('a', 'a'), def: f('d', 'd') },\n\t// a trailing note\n\t// ---- two\n\t{ id: 'b', term: f('b', 'b'), def: f('e', 'e') },\n];\n");
	const tech = "export const TECH_ZH: Record<string, string> = {\n\t'A': 'a',\n\t'B': 'b'\n};\n";
	const t2 = removeRows(addRows(tech, 'TECH_ZH', ["'C': 'c'"]), 'TECH_ZH', ['A', 'nope']);
	assert.deepEqual(t2.removed, ['A']);
	assert.equal(t2.text, "export const TECH_ZH: Record<string, string> = {\n\t'B': 'b',\n\t'C': 'c',\n};\n");
});

test('promote hints: similar rows by shared words (no more "JS" for a long runtime); sizes in the catalog unit', () => {
	const rows = { JS: 'JS', Live: '实时', 'Own JS forward pass (checked against PyTorch); sampling in a worker': 'x', 'Own TypeScript, live in the browser (checked against a Python run)': 'y' };
	assert.equal(similarRow('Own JS forward pass (checked against PyTorch)', rows), 'Own JS forward pass (checked against PyTorch); sampling in a worker');
	assert.equal(similarRow('Own TypeScript, live in the browser (checked against a NumPy run)', rows), 'Own TypeScript, live in the browser (checked against a Python run)');
	assert.deepEqual(similarRows('Something else entirely', rows), []);
	assert.equal(catalogBytes(63972), '62 KB');
	assert.equal(catalogBytes(9492), '9.3 KB');
	assert.equal(catalogBytes(1530000), '1.46 MB');
	assert.equal(sizeBytes('64 KB of weights'), 65536);
	assert.equal(sizeBytes('<1 MB (design)'), null);
});

test("page code: the manifest's catalog block is not counted; a test file may raise its time limit", () => {
	const dir = mkdtempSync(join(tmpdir(), 'atlas-k4-'));
	try {
		const ex = "const m = {\n\tslug: 'x',\n\tcatalog: {\n\t\thook: L('a', 'b'),\n\t\tshows: L('c', 'd'),\n\t\tsize: L('e', 'f')\n\t},\n\tcredits: []\n};\nexport default m;\n";
		writeFileSync(join(dir, 'explainer.ts'), ex);
		assert.ok(!withoutCatalog(ex).includes('hook'));
		assert.equal(pageLines(dir).lines, 5);
		writeFileSync(join(dir, 'claims.test.ts'), '// atlas-timeout: 900\nimport x from "y";\n');
		writeFileSync(join(dir, 'b.test.ts'), '// atlas-timeout: 99999\n');
		assert.equal(testTimeoutMs([join(dir, 'claims.test.ts')]), 900_000);
		assert.equal(testTimeoutMs([join(dir, 'explainer.ts')]), 540_000);
		assert.equal(testTimeoutMs([join(dir, 'b.test.ts')]), 3_600_000);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test("check's view contract on a package's own views (tinymoe's demo passed every package gate and failed check:atlas)", async () => {
	const dir = mkdtempSync(join(tmpdir(), 'atlas-k4v-'));
	try {
		mkdirSync(join(dir, 'views', 'good'), { recursive: true });
		mkdirSync(join(dir, 'views', 'bad'), { recursive: true });
		writeFileSync(join(dir, 'views', 'good', 'meta.ts'), "export default { key: 'p/good', title: { en: 'G', zh: '好' }, kinds: ['trace'], accepts: () => false, describe: () => ({ en: 'g', zh: '好' }), demo: async () => ({ provenance: 'illustrative' }) };\n");
		writeFileSync(join(dir, 'views', 'bad', 'meta.ts'), "export default { key: 'p/bad', title: { en: 'B', zh: '坏' }, kinds: ['trace'], accepts: () => false, describe: () => ({ en: 'b', zh: '' }), demo: async () => { throw new Error('spec has no layers'); } };\n");
		const vc = await viewContract(dir, 'p');
		assert.equal(vc.views, 2);
		assert.deepEqual(vc.problems, ['p/bad: spec has no layers']);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('lanes: the review packet is owned (new lanes and older lane.json); shots of a lane outside the repo stay in the lane; --clean', () => {
	assert.ok(modelOwns('x').includes('notes/review-x.md'));
	assert.deepEqual(withReview(['pipeline/x/**', 'notes/x.md', 'notes/shots/x/**'], 'x'), ['pipeline/x/**', 'notes/x.md', 'notes/review-x.md', 'notes/shots/x/**']);
	assert.deepEqual(withReview(['site/src/lib/atlas/**'], 'k0'), ['site/src/lib/atlas/**'], 'kit lanes unchanged');
	const base: Roots = { site: 'X:/r/site', root: 'X:/r', siteFrom: 'cli', rootFrom: 'lane', lane: 'x', laneDir: 'D:/lanes/x', main: 'X:/r/site' };
	assert.equal(shotsDir('x', base).replace(/\\/g, '/'), 'D:/lanes/x/shots/x');
	assert.equal(shotsDir('x', { ...base, laneDir: 'X:/r/site/.work/x' }).replace(/\\/g, '/'), 'X:/r/notes/shots/x');
	assert.equal(shotsDir('x', { ...base, lane: null, laneDir: null }).replace(/\\/g, '/'), 'X:/r/notes/shots/x');
	assert.ok(inside('X:/r/site/.work/x', 'X:/r') && !inside('D:/lanes/x', 'X:/r'));
	const dir = mkdtempSync(join(tmpdir(), 'atlas-k4c-'));
	try {
		for (const d of ['build', '.svelte-kit', 'node_modules/.vite/deps', '.atlas/build-x', '.atlas/x']) mkdirSync(join(dir, d), { recursive: true });
		assert.deepEqual(cacheDirs(dir).map((d) => d.replace(/\\/g, '/')).sort(), ['.atlas/build-x', '.svelte-kit', 'build', 'node_modules/.vite']);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('pickPort skips the ports a start found answered by another server', async () => {
	const p = await pickPort({ preferred: null });
	const q = await pickPort({ preferred: p, exclude: [p] });
	assert.notEqual(q, p);
	// a busy preferred port is skipped too
	const srv = createServer();
	await new Promise<void>((res) => srv.listen({ port: q, host: '127.0.0.1' }, () => res()));
	try {
		assert.notEqual(await pickPort({ preferred: q }), q);
	} finally {
		srv.close();
	}
});

test('promote --verify applies the whole patch to a copy (the tree stays as it was); --apply writes it, keeping CRLF', () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		assert.equal(cli(['new', 'model', 'gridworld', '--template', 'process-blank'], env).status, 0);
		const exFile = join(t.site, 'src', 'lib', 'explainers', 'gridworld', 'explainer.ts');
		const catalog = "catalog: { name: { en: 'Gridworld (tabular RL)', zh: '网格世界（表格型强化学习）' }, inputs: ['game'], mode: 'live', runtime: { en: 'Own TypeScript gridworld, K4 test run', zh: '自研 TypeScript 网格世界（K4 测试）' } }";
		const ex0 = readFileSync(exFile, 'utf8');
		writeFileSync(exFile, ex0.includes('catalog: {}') ? ex0.replace('catalog: {}', catalog) : ex0.replace('credits: []', `credits: [], ${catalog}`));
		mkdirSync(join(t.site, 'static'), { recursive: true });
		writeFileSync(join(t.site, 'static', 'licenses.json'), '[]\n');
		mkdirSync(join(t.site, 'scripts'), { recursive: true });
		mkdirSync(join(t.site, 'src', 'lib', 'i18n'), { recursive: true });
		writeFileSync(join(t.site, 'src', 'lib', 'i18n', 'strings.ts'), readFileSync(join(SITE, 'src', 'lib', 'i18n', 'strings.ts')));
		writeFileSync(join(t.site, 'scripts', 'check-catalog.ts'), readFileSync(join(SITE, 'scripts', 'check-catalog.ts')));
		const models = join(t.site, 'src', 'lib', 'catalog', 'models.ts');
		const before = readFileSync(models);
		const v = cli(['promote', 'gridworld', '--verify'], env);
		assert.equal(v.status, 0, v.stdout + v.stderr);
		assert.match(v.stdout, /check:catalog ok; the entry reads back as patched/);
		assert.match(v.stdout, /inputs: \['game'\],/);
		assert.ok(readFileSync(models).equals(before), '--verify leaves the catalog as it was');
		const copy = readFileSync(join(t.site, '.atlas', 'gridworld', 'verify', 'src', 'lib', 'catalog', 'models.ts'), 'utf8');
		assert.equal(eolOf(copy), eolOf(before.toString('utf8')), 'the copy keeps the line endings');
		const a = cli(['promote', 'gridworld', '--apply'], env);
		assert.equal(a.status, 0, a.stdout + a.stderr);
		const after = readFileSync(models, 'utf8');
		assert.equal(eolOf(after), eolOf(before.toString('utf8')));
		assert.match(after, /name: f\('Gridworld \(tabular RL\)', '网格世界（表格型强化学习）'\)/);
		assert.match(readFileSync(join(t.site, 'src', 'lib', 'catalog', 'tech-zh.ts'), 'utf8'), /'Own TypeScript gridworld, K4 test run': '自研 TypeScript 网格世界（K4 测试）',/);
	} finally {
		t.cleanup();
	}
});
