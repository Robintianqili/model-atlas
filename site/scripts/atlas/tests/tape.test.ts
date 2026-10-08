// The tape layout shared by the TS writer (lib/tapefile.ts) and pipeline/atlas_kit/tape.py, and the fallback recorder's
// semantics (keyframes + changes, state(i) pure in i, determinism, recordDraws).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import type { Frame, ProcessDef, Tensor } from '../../../src/lib/atlas/types/index.ts';
import { recordFallback } from '../lib/fallback.ts';
import { readTapeFiles, writeTapeFiles } from '../lib/tapefile.ts';
import { spawnSync } from 'node:child_process';
import { PYTHON, ROOT } from './helpers.ts';

type S = { Q: Tensor; pos: number; ids: { shape: number[]; data: Int32Array }; params: Record<string, Tensor>; note: string };
/** A walk on 4 cells: Q is tracked by changes; pos, ids (i32), params (record of f32 tensors), note are snapshots. */
const walk: ProcessDef<null, { n: number; alpha: number }, S> = {
	id: 'walk',
	kind: 'algorithm',
	title: { en: 'Walk', zh: '行走' },
	phases: {
		act: { label: { en: 'Act', zh: '行动' }, kind: 'act' },
		update: { label: { en: 'Update', zh: '更新' }, kind: 'update', formula: { tex: 'Q \\leftarrow {{q}}', values: (f) => ({ q: f.metrics!.q }) } }
	},
	schema: {
		Q: { kind: 'tensor', label: { en: 'Q', zh: 'Q' }, track: 'changes', tol: 1e-12 },
		pos: { kind: 'int', label: { en: 'Position', zh: '位置' }, tol: 'exact' },
		ids: { kind: 'tensor', label: { en: 'Visited', zh: '访问过' } },
		params: { kind: 'json', label: { en: 'Params', zh: '参数' } },
		note: { kind: 'text', label: { en: 'Note', zh: '说明' } },
		V: { kind: 'tensor', label: { en: 'V', zh: 'V' }, derive: (s: S) => s.Q }
	},
	defaults: { n: 30, alpha: 0.5 },
	*run(_i, ctx) {
		const Q: Tensor = { shape: [4, 2], data: new Float64Array(8) };
		const ids: Tensor = { shape: [4], data: new Float32Array(4) };
		let pos = 0;
		for (let t = 0; t < ctx.opt.n; t++) {
			const a = ctx.rng.int(2);
			const z = t % 5 === 0 ? ctx.rng.normal() : 0;
			yield { phase: 'act', step: t, state: { pos, Q, ...(t % 10 === 0 ? { note: `t=${t}`, params: { w: { shape: [2], data: Float32Array.from([t + 1, -(t + 1)]) } } } : {}) }, metrics: { a, z, loss: t === 3 ? NaN : t } } as Frame<S>;
			const k = pos * 2 + a;
			const from = Q.data[k];
			Q.data[k] = from + ctx.opt.alpha * (1 - from);
			pos = (pos + (a ? 1 : 3)) % 4;
			ids.data[pos] = 1;
			yield { phase: 'update', step: t, state: { Q, pos, ids: { shape: [4], data: Int32Array.from(ids.data) } }, changes: [{ field: 'Q', index: [k >> 1, k & 1], from, to: Q.data[k] }], metrics: { q: Q.data[k] } } as Frame<S>;
		}
	}
};

test('fallback: state(i) from keyframes + changes equals a straight replay; keyEvery 1 vs 64; same seed twice', () => {
	const a = recordFallback(walk, null, walk.defaults, { seed: 3, keyEvery: 64 });
	const b = recordFallback(walk, null, walk.defaults, { seed: 3, keyEvery: 1 });
	const c = recordFallback(walk, null, walk.defaults, { seed: 3, keyEvery: 7 });
	assert.equal(a.length, 60);
	assert.equal(a.frames.filter((f) => f.key).length, 1);
	assert.equal(b.frames.filter((f) => f.key).length, 60);
	for (let i = 0; i < a.length; i++) {
		assert.ok(isDeepStrictEqual(a.state(i), b.state(i)), `frame ${i}`);
		assert.ok(isDeepStrictEqual(a.state(i), c.state(i)), `frame ${i}`);
	}
	const hash = (t: typeof a) => createHash('sha256').update(JSON.stringify(t.frames, (_, v) => (ArrayBuffer.isView(v) ? Array.from(v as Float64Array) : v))).digest('hex');
	assert.equal(hash(a), hash(recordFallback(walk, null, walk.defaults, { seed: 3 })));
	assert.notEqual(hash(a), hash(recordFallback(walk, null, walk.defaults, { seed: 4 })));
	assert.equal((a.state(59) as S).Q.data[0] > 0, true);
	assert.equal(a.find({ phase: 'update', step: 2 }), 5);
	assert.equal(a.find({ phase: 'update', last: true }), 59);
	assert.equal(a.find({ phase: 'act', where: (f) => f.metrics!.a === 1 }) >= 0, true);
	assert.equal(a.metric('q').length, 60);
	assert.ok(a.notes.length === 0);
});

test('fallback: an edit not reported in changes fails (AK301)', () => {
	const bad: ProcessDef<null, Record<string, never>, { Q: Tensor }> = {
		id: 'bad',
		kind: 'algorithm',
		title: { en: 'Bad' },
		phases: { u: { label: { en: 'u' }, kind: 'update' } },
		schema: { Q: { kind: 'tensor', label: { en: 'Q' }, track: 'changes' } },
		defaults: {},
		*run() {
			const Q: Tensor = { shape: [2], data: new Float64Array(2) };
			yield { phase: 'u', state: { Q } };
			Q.data[0] = 1;
			Q.data[1] = 5; // not reported
			yield { phase: 'u', state: { Q }, changes: [{ field: 'Q', index: [0], from: 0, to: 1 }] };
		}
	};
	assert.throws(() => recordFallback(bad, null, {}), (e: Error & { code?: string }) => e.code === 'AK301' && /do not explain/.test(e.message));
});

test('recordDraws: the value of every top-level rng call, per frame', () => {
	const t = recordFallback(walk, null, { n: 6, alpha: 0.5 }, { seed: 9, recordDraws: true });
	const act0 = t.frame(0);
	assert.equal(act0.draws!.length, 2); // int + normal (t = 0)
	assert.equal(t.frame(1).draws!.length, 0);
	assert.equal(t.frame(2).draws!.length, 1); // int only
	assert.equal(act0.draws![0], act0.metrics!.a);
	assert.equal(act0.draws![1], act0.metrics!.z);
});

const PY_REPLAY = `import json, sys
import numpy as np
import atlas_kit as ak
DT = {"Float64Array": np.float64, "Float32Array": np.float32, "Int32Array": np.int32}
def dec(x):
    if isinstance(x, dict):
        if "$arr" in x:
            return np.array(x["v"], dtype=DT[x["$arr"]])
        if "$num" in x:
            return float(x["$num"].replace("Infinity", "inf"))
        if set(x) == {"shape", "data"}:
            d = dec(x["data"]) if isinstance(x["data"], dict) else np.array(x["data"], dtype=np.float64)
            return d.reshape(x["shape"])
        return {k: dec(v) for k, v in x.items()}
    if isinstance(x, list):
        return [dec(v) for v in x]
    return x
frames = dec(json.loads(sys.stdin.read()))
schema = {"Q": {"kind": "tensor", "label": {"en": "Q", "zh": "Q"}, "track": "changes", "tol": 1e-12},
          "pos": {"kind": "int", "label": {"en": "Position", "zh": "位置"}, "tol": "exact"},
          "ids": {"kind": "tensor", "label": {"en": "Visited", "zh": "访问过"}},
          "params": {"kind": "json", "label": {"en": "Params", "zh": "参数"}},
          "note": {"kind": "text", "label": {"en": "Note", "zh": "说明"}},
          "V": {"kind": "tensor", "label": {"en": "V", "zh": "V"}, "derived": True}}
phases = {"act": {"label": {"en": "Act", "zh": "行动"}, "kind": "act"},
          "update": {"label": {"en": "Update", "zh": "更新"}, "kind": "update", "tex": "Q \\\\leftarrow {{q}}"}}
w = ak.TapeWriter("walk", schema, seed=11, opt={"n": 30, "alpha": 0.5}, key_every=16, phases=phases, provenance="precomputed")
for f in frames:
    w.frame(f["phase"], f["state"], step=f.get("step"), sub=f.get("sub"), changes=f.get("changes"), metrics=f.get("metrics"), draws=f.get("draws"))
w.write(sys.argv[1])
print("ok", len(frames))
`;

test('TS writer and Python TapeWriter give identical files (f64, f32, i32, records, NaN, draws, keyframes)', { skip: !PYTHON && 'no configured Python' }, () => {
	const dir = mkdtempSync(join(tmpdir(), 'atlas-tape-'));
	try {
		const tape = recordFallback(walk, null, walk.defaults, { seed: 11, keyEvery: 16, recordDraws: true });
		writeTapeFiles(join(dir, 'ts'), tape, walk.phases, 'precomputed');
		// Python replays the frames through ak.TapeWriter, receiving the full Q on every frame (as a reference would)
		const frames = JSON.stringify(
			Array.from({ length: tape.length }, (_, i) => {
				const f = tape.frame(i);
				const given = { ...(f.state as Record<string, unknown>), Q: { shape: (tape.state(i) as S).Q.shape, data: Array.from((tape.state(i) as S).Q.data) } };
				const { key: _k, ...rest } = f;
				return { ...rest, state: given };
			}),
			(_, v) => (ArrayBuffer.isView(v) ? { $arr: v.constructor.name, v: Array.from(v as Float64Array) } : typeof v === 'number' && !Number.isFinite(v) ? { $num: String(v) } : v)
		);
		const res = spawnSync(PYTHON!, ['-c', PY_REPLAY, join(dir, 'py')], { input: frames, env: { ...process.env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1' }, encoding: 'utf8', timeout: 300_000 });
		assert.equal(res.status, 0, res.stderr);
		const ts = readTapeFiles(join(dir, 'ts'));
		const py = readTapeFiles(join(dir, 'py'));
		const { bin: _b1, ...h1 } = ts.header;
		const { bin: _b2, ...h2 } = py.header;
		assert.ok(isDeepStrictEqual(JSON.parse(JSON.stringify(h1)), JSON.parse(JSON.stringify(h2))), 'headers differ');
		assert.ok(Buffer.from(ts.blob).equals(Buffer.from(py.blob)), `bins differ (${ts.blob.length} vs ${py.blob.length} bytes)`);
		assert.ok(ts.blob.length > 0);
		assert.equal(ts.header.frames.filter((f) => (f as { key?: boolean }).key).length, 4);
		assert.ok(Number.isNaN(ts.frames[6].metrics!.loss));
		assert.deepEqual(Array.from((py.frames[1].state as S).ids.data), Array.from((ts.frames[1].state as S).ids.data));
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
