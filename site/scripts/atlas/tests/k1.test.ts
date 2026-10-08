// DevKit K1: the CLI and atlas_kit fixes after wave 2b (LOG.md "K1"): export names of 'train:<id>' processes, one twin
// run per trainer and options, the process input for reference.py, similar tech-zh rows, a tracked tape field that starts
// late (TS and Python write the same file), ak.json_arg.
//   node --test scripts/atlas/tests/k1.test.ts   (from site/)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { isDeepStrictEqual } from 'node:util';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { sameName } from '../load.ts';
import { inputJson, sourceParityFile, twinKey, type ParityReport } from '../commands/parity.ts';
import { similarRow } from '../commands/promote.ts';
import { TapeStore } from '../../../src/lib/atlas/process/tape.ts';
import { saveTape } from '../../../src/lib/atlas/process/io.ts';
import { PYTHON, ROOT } from './helpers.ts';
import { FACTIONS, KARATE_EDGES } from '../../../src/lib/atlas/data/karate.ts';

test('export names: a process registered as train:<id> under its id without the prefix is not a loader note', () => {
	assert.ok(sameName('flow', 'train:flow'));
	assert.ok(sameName('trainFlow', 'train:flow'));
	assert.ok(sameName('walk2', 'walk-2'));
	assert.ok(sameName('default', 'anything'));
	assert.ok(!sameName('other', 'train:flow'));
});

test('parity: one twin run per trainer and options; every source of that pair gets the record, no other', () => {
	assert.equal(twinKey('pinn', { seed: 1, lam: 0, steps: 1000 }), twinKey('pinn', { lam: 0, seed: 1, steps: 50, snapshotEvery: 5 }));
	assert.notEqual(twinKey('pinn', { seed: 1, lam: 0 }), twinKey('pinn', { seed: 1, lam: 0.01 }));
	const run = (sources: string[], max: number) => ({ source: sources[0], trainer: 'pinn', twin: 'twin.py', runs: [{ optimizer: 'adam', steps: 50, bar: 1e-7, losses: max, grads0: 0, final: max, pass: true, fixture: 'f' }], pass: true, sources });
	const report = { slug: 's', checked: '2026-10-07', train: [run(['a', 'b'], 1e-9), run(['c'], 2e-9)], process: [], pass: true, wrote: null, issues: [] } as unknown as ParityReport;
	const srcs = { a: { kind: 'train', trainer: 'pinn', opt: { lam: 0 } }, b: { kind: 'train', trainer: 'pinn', opt: { lam: 0 } }, c: { kind: 'train', trainer: 'pinn', opt: { lam: 0.01 } } } as never;
	const f = sourceParityFile('s', report, srcs)!;
	assert.equal(f.sources.a.max, 1e-9);
	assert.equal(f.sources.b.max, 1e-9);
	assert.equal(f.sources.c.max, 2e-9, "c has its own run's record");
});

test('parity: the process input for reference.py is plain JSON (typed arrays as lists, a model object left out)', () => {
	class Model {
		w = 1;
	}
	const s = inputJson({ ids: Int32Array.of(1, 2), model: new Model(), f: () => 1, nested: { x: Float64Array.of(0.5) } });
	assert.deepEqual(JSON.parse(s), { ids: [1, 2], nested: { x: [0.5] } });
	assert.equal(inputJson(undefined), 'null');
});

test('promote: a tech-zh row that shares most words with an existing one is pointed out', () => {
	const rows = { 'Own autograd core in a worker': '自己的自动微分内核，在 worker 里运行', 'Static page': '静态页面' };
	assert.equal(similarRow('Own autograd core in a worker, with second derivatives', rows), 'Own autograd core in a worker');
	assert.equal(similarRow('A completely different runtime', rows), null);
});

const PY_LATE = `import sys
import numpy as np
import atlas_kit as ak
schema = {"Q": {"kind": "tensor", "label": {"en": "Q", "zh": "Q"}, "track": "changes"}, "pos": {"kind": "int", "label": {"en": "p", "zh": "p"}}}
phases = {"go": {"label": {"en": "Go", "zh": "走"}, "kind": "act"}}
w = ak.TapeWriter("late", schema, seed=1, opt={}, key_every=2, phases=phases)
w.frame("go", {"pos": 0}, step=0)
Q = np.zeros(3)
w.frame("go", {"pos": 1, "Q": Q.copy()}, step=1)
w.frame("go", {"pos": 2, "Q": Q.copy()}, step=2)
Q[1] = 5.0
w.frame("go", {"pos": 3, "Q": Q.copy()}, step=3, changes=[{"field": "Q", "index": [1], "from": 0.0, "to": 5.0}])
w.write(sys.argv[1])
assert ak.json_arg('[1, 2]') == [1, 2]
open(sys.argv[1] + ".arg.json", "w").write('{"a": 3}')
assert ak.json_arg("@" + sys.argv[1] + ".arg.json") == {"a": 3}
print("ok")
`;

test("the kit's karate club (atlas/data/karate.ts) is ak.data.karate(): the same 78 friendships and factions", { skip: !PYTHON && 'no configured Python' }, () => {
	const code = 'import json, atlas_kit as ak\nk = ak.data.karate()\nprint(json.dumps({"edges": [list(e) for e in k["edges"]], "labels": [int(x) for x in k["labels"]]}))';
	const res = spawnSync(PYTHON!, ['-c', code], { env: { ...process.env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1' }, encoding: 'utf8', timeout: 300_000 });
	if (res.status !== 0 && /networkx/.test(res.stderr)) return; // networkx missing: doctor reports it
	assert.equal(res.status, 0, res.stderr);
	const py = JSON.parse(res.stdout.trim().split('\n').pop()!);
	assert.deepEqual(KARATE_EDGES, py.edges);
	assert.deepEqual(FACTIONS, py.labels);
});

test('a tracked field that starts after frame 0: the Python TapeWriter accepts it and writes the bytes the TS writer writes', { skip: !PYTHON && 'no configured Python' }, () => {
	const dir = mkdtempSync(join(tmpdir(), 'atlas-k1-'));
	try {
		const schema = { Q: { kind: 'tensor' as const, label: { en: 'Q', zh: 'Q' }, track: 'changes' as const }, pos: { kind: 'int' as const, label: { en: 'p', zh: 'p' } } };
		const phases = { go: { label: { en: 'Go', zh: '走' }, kind: 'act' as const } };
		const store = new TapeStore({ process: 'late', seed: 1, opt: {}, schema, phases, keyEvery: 2, version: '1', provenance: 'live' });
		const Q = new Float64Array(3);
		store.push({ phase: 'go', step: 0, state: { pos: 0 } });
		store.push({ phase: 'go', step: 1, state: { pos: 1, Q: { shape: [3], data: Q.slice() } } });
		store.push({ phase: 'go', step: 2, state: { pos: 2, Q: { shape: [3], data: Q.slice() } } });
		Q[1] = 5;
		store.push({ phase: 'go', step: 3, state: { pos: 3, Q: { shape: [3], data: Q.slice() } }, changes: [{ field: 'Q', index: [1], from: 0, to: 5 }] });
		const ts = saveTape(store, { binName: 'py.tape.bin' });
		const res = spawnSync(PYTHON!, ['-c', PY_LATE, join(dir, 'py')], { env: { ...process.env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1' }, encoding: 'utf8', timeout: 300_000 });
		assert.equal(res.status, 0, res.stderr);
		const py = JSON.parse(readFileSync(join(dir, 'py.tape.json'), 'utf8'));
		const blob = readFileSync(join(dir, 'py.tape.bin'));
		const { bin: _a, version: _v1, provenance: _p1, ...h1 } = JSON.parse(JSON.stringify(ts.header));
		const { bin: _b, version: _v2, provenance: _p2, ...h2 } = py;
		assert.ok(isDeepStrictEqual(h1.frames, h2.frames), `frames differ:\n${JSON.stringify(h1.frames)}\n${JSON.stringify(h2.frames)}`);
		assert.ok(Buffer.from(ts.blob).equals(blob), 'bins differ');
		assert.equal('Q' in py.frames[0].state, false, 'no Q at the first keyframe: it starts later');
		writeFileSync(join(dir, 'done'), '');
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
