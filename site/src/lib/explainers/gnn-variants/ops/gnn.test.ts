// Tests for the gnn-variants local ops (node --test; `atlas check` runs them in the op-tests gate).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AtlasLayer, AtlasSpec, OpDef, Tensor } from '../../../atlas/types/index.ts';
import { fsFetch } from '../../../atlas/core/io.ts';
import { loadGraph } from '../../../atlas/graph/executor.ts';
import { checkExplain } from '../../../atlas/graph/explain.ts';
import ops from '../ops.ts';
import { gat_conv, gcn_conv, gin_agg, sage_conv } from './gnn.ts';

const SITE = fileURLToPath(new URL('../../../../../', import.meta.url));
const T = (shape: number[], v: number[]): Tensor => ({ shape, data: Float32Array.from(v) });
const L = (op: string, outShape: number[]): AtlasLayer => ({ id: 'L', op, label: { en: 'L', zh: 'L' }, in: ['h', 'a'], outShape });
// a path 0 - 1 - 2 and an isolated member 3; H has 2 numbers per member
const A = T([4, 4], [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
const H = T([4, 2], [1, 0, 0, 1, 2, -1, 0.5, 0.5]);
function run<P extends Record<string, unknown>>(op: OpDef<P>, w: Record<string, Tensor>, p = {} as P) {
	const subs: Record<string, Tensor> = {};
	const ctx = { layer: L(op.op, []), dims: {}, w: (s: string) => w[s], has: (s: string) => !!w[s], emit: (k: string, x: Tensor) => void (subs[k] = x) };
	const out = op.forward([H, A], p, ctx);
	// explain() equals the output (and every sub-step) on every unit
	for (const [name, x] of [['', out], ...Object.entries(subs)] as [string, Tensor][])
		for (let k = 0; k < x.data.length; k++) {
			const e = op.explain!({ layer: L(op.op, out.shape), unit: [Math.floor(k / x.shape[1]), k % x.shape[1]], ins: [H, A], out, w: (s) => w[s], p, trace: { order: [], t: {} }, spec: {} as AtlasSpec });
			assert.equal(name ? e.values[name] : e.result, x.data[k], `${op.op} ${name || 'out'} ${k}`);
		}
	return { out: Array.from(out.data), subs };
}
const near = (a: number[], b: number[]) => a.forEach((v, k) => assert.ok(Math.abs(v - b[k]) < 1e-6, `${k}: ${v} vs ${b[k]}`));

test('gcn_conv: Â = 1/√(dᵢ dⱼ) with self-loops; an isolated member keeps its own row with weight 1', () => {
	const r = run(gcn_conv, { weight: T([1, 2], [1, 0]), bias: T([1], [0.5]) });
	near(Array.from(r.subs.ahat.data.slice(0, 4)), [1 / 2, 1 / Math.sqrt(6), 0, 0]);
	near(r.out, [1 / 2 + 0 + 0.5, 1 / Math.sqrt(6) + 0 + 2 / Math.sqrt(6) + 0.5, 1 / Math.sqrt(6) * 0 + 2 / 2 + 0.5, 0.5 + 0.5]);
	assert.deepEqual(gcn_conv.inferShape([[34, 34], [34, 34]], {}, { weight: [4, 34], bias: [4] }), [34, 4]);
});

test('sage_conv: [own ‖ friends’ mean] Wᵀ + b; the mean of no friends is 0', () => {
	const r = run(sage_conv, { weight: T([1, 4], [1, 10, 100, 1000]), bias: T([1], [0]) });
	near(Array.from(r.subs.mean.data), [0, 1, 1.5, -0.5, 0, 1, 0, 0]);
	near(r.out, [1 + 1000, 10 + 150 - 500, 2 - 10 + 1000, 0.5 + 5]);
});

test('gin_agg: (1 + ε) h + the friends’ rows, no division', () => {
	const r = run(gin_agg, { eps: T([1], [0.5]) });
	near(r.out, [1.5, 1, 3, 0.5, 3, -0.5, 0.75, 0.75]);
});

test('gat_conv: α is a softmax of LeakyReLU scores over the senders (sums to 1, 0 off the edges); out = Σ α z + b', () => {
	const r = run(gat_conv, { weight: T([2, 2], [1, 0, 0, 1]), att_src: T([2], [1, -1]), att_dst: T([2], [0.5, 0.5]), bias: T([2], [0, 0]) }, { slope: 0.2 });
	const al = Array.from(r.subs.alpha.data);
	for (let i = 0; i < 4; i++) assert.ok(Math.abs(al.slice(i * 4, i * 4 + 4).reduce((s, v) => s + v, 0) - 1) < 1e-6);
	assert.deepEqual([al[2], al[3], al[15]], [0, 0, 1]);
	assert.equal(r.subs.e.data[2], -Infinity);
	// member 0: senders 0 (score LeakyReLU(0.5 + 1) = 1.5) and 1 (LeakyReLU(0.5 − 1) = −0.1)
	const e = [1.5, -0.1].map(Math.exp);
	near(al.slice(0, 2), e.map((v) => v / (e[0] + e[1])));
});

test('the exported model: explain() equals the trace on every unit of every layer and sub-step (Zachary’s club)', async () => {
	const model = await loadGraph(fsFetch, join(SITE, 'static/models/gnn-variants'), ops);
	const N = 34;
	const adj = new Float32Array(N * N);
	const k = JSON.parse(readFileSync(join(SITE, 'static/models/gnn-variants/karate.json'), 'utf8'));
	for (const [i, j] of k.edges) adj[i * N + j] = adj[j * N + i] = 1;
	const trace = model.forward({ adj: { shape: [N, N], data: adj }, x: { shape: [N, N], data: Float32Array.from({ length: N * N }, (_, q) => (q % (N + 1) ? 0 : 1)) } }, { record: 'all' });
	const res = checkExplain(model, trace);
	assert.ok(res.units > 10000 && res.max <= 1e-5, `${res.units} units, max ${res.max} at ${JSON.stringify(res.worst)}`);
});
