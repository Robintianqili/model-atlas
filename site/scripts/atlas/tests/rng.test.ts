// Acceptance 1 (W1-E1): Python ak.Rng is bit-identical to the TS mulberry32 for next(), int() and fork().
// The TS side here is scripts/atlas/lib/fallback.ts createRng (the §4.1 definition). When W1-A's core/rng.ts is in the
// tree, the same vectors are checked against it too (export name probed: createRng | rng | makeRng | mulberry32Rng | Rng).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { Rng } from '../../../src/lib/atlas/types/index.ts';
import { createRng, fnv1a32 } from '../lib/fallback.ts';
import { PYTHON, SITE, python } from './helpers.ts';

// Shared with pipeline/atlas_kit/tests/test_sdk.py (computed independently in both languages).
export const VECTORS = {
	next42: [0.6011037519, 0.448290559, 0.8524657935, 0.6697340414],
	next1: [0.6270739406, 0.0027357212, 0.52744704],
	int42x10: [6, 4, 8, 6, 1, 5, 2, 6],
	fnvInit: 0x16b1d373,
	fork42init: { seed: 380752729, first: 0.3591537452302873 },
	fork7batchInt100: [27, 91, 61, 27, 53]
};

function check(make: (seed: number) => Rng, label: string) {
	const r42 = make(42);
	assert.deepEqual([r42.next(), r42.next(), r42.next(), r42.next()].map((v) => v.toFixed(10)), VECTORS.next42.map((v) => v.toFixed(10)), label);
	const r1 = make(1);
	assert.deepEqual([r1.next(), r1.next(), r1.next()].map((v) => v.toFixed(10)), VECTORS.next1.map((v) => v.toFixed(10)), label);
	const ri = make(42);
	assert.deepEqual(Array.from({ length: 8 }, () => ri.int(10)), VECTORS.int42x10, label);
	const f = make(42).fork('init');
	assert.equal(f.seed, VECTORS.fork42init.seed, label);
	assert.equal(f.next(), VECTORS.fork42init.first, label);
	const g = make(7).fork('batch');
	assert.deepEqual(Array.from({ length: 5 }, () => g.int(100)), VECTORS.fork7batchInt100, label);
	// fork is independent of how much was consumed
	const a = make(123);
	const b = make(123);
	for (let i = 0; i < 17; i++) b.next();
	assert.equal(a.fork('x').next(), b.fork('x').next(), label);
	// state round trip
	const s = make(5);
	s.next();
	const st = s.getState();
	const x = s.next();
	s.setState(st);
	assert.equal(s.next(), x, label);
}

test('fallback createRng: doc vectors, int, fork, state', () => {
	assert.equal(fnv1a32('init'), VECTORS.fnvInit);
	assert.equal(fnv1a32(''), 0x811c9dc5);
	assert.equal(fnv1a32('foobar'), 0xbf9cf968); // published FNV-1a test value
	check((s) => createRng(s), 'fallback');
});

test('W1-A core/rng.ts (when present) gives the same vectors', async (t) => {
	const f = join(SITE, 'src', 'lib', 'atlas', 'core', 'rng.ts');
	if (!existsSync(f)) {
		t.skip('core/rng.ts is not in this tree yet (W1-A)');
		return;
	}
	const m = (await import(pathToFileURL(f).href)) as Record<string, unknown>;
	const name = ['createRng', 'rng', 'makeRng', 'mulberry32Rng'].find((n) => typeof m[n] === 'function');
	const make = name ? (m[name] as (s: number) => Rng) : typeof m.Rng === 'function' ? (s: number) => new (m.Rng as new (s: number) => Rng)(s) : null;
	if (!make) {
		t.skip(`core/rng.ts exports ${Object.keys(m).join(', ')}: no known factory name`);
		return;
	}
	check(make, `core/rng.ts ${name ?? 'Rng'}`);
});

test('cross-engine: Python ak.Rng equals TS for 300 seeds x labels (next, int, fork)', { skip: !PYTHON && 'no configured Python' }, () => {
	const seeds = [0, 1, 2, 42, 0x7fffffff, 0x80000000, 0xffffffff, -1, -12345, 2 ** 32 + 5];
	for (let i = 0; i < 290; i++) seeds.push(Math.floor(((i * 2654435761) % 4294967296) / 3) - (i % 7 === 0 ? 2 ** 31 : 0));
	const labels = ['init', 'batch', 'x', '', 'α-ε', '数据', 'emoji-🙂', 'L0.attn.q#h2'];
	const out = python(
		`import json, sys
import atlas_kit as ak
seeds = json.loads(sys.argv[1]); labels = json.loads(sys.argv[2])
res = []
for s in seeds:
    r = ak.Rng(s)
    nx = [r.next() for _ in range(5)]
    ints = [r.int(n) for n in (2, 3, 10, 1000, 2**31)]
    forks = [[ak.Rng(s).fork(l).seed, ak.Rng(s).fork(l).next()] for l in labels]
    ff = ak.Rng(s).fork('a').fork('b').next()
    res.append([nx, ints, forks, ff, r.get_state()])
print(json.dumps(res))`.replace('sys.argv[1]', JSON.stringify(JSON.stringify(seeds))).replace('sys.argv[2]', JSON.stringify(JSON.stringify(labels)))
	);
	assert.equal(out.status, 0, out.stderr);
	const py = JSON.parse(out.stdout) as [number[], number[], [number, number][], number, number][];
	seeds.forEach((s, k) => {
		const r = createRng(s);
		const nx = Array.from({ length: 5 }, () => r.next());
		const ints = [2, 3, 10, 1000, 2 ** 31].map((n) => r.int(n));
		const forks = labels.map((l) => [createRng(s).fork(l).seed, createRng(s).fork(l).next()]);
		const ff = createRng(s).fork('a').fork('b').next();
		assert.deepEqual([nx, ints, forks, ff, r.getState()], py[k], `seed ${s}`);
	});
});
