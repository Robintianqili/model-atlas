// Op goldens (design §6.3 "Op tests", W1-A acceptance 4): every CORE_OP against PyTorch float64 on the same float32
// inputs, |a − b| ≤ 1e-6 + 1e-6·|b|; explain() equals forward for every unit (≤ 1e-5); inferShape equals the output
// shape; receptive() agrees with the window explain() reads; narration placeholders resolve from vars() / explain().
// Fixtures: site/tests/atlas/ops/<op>.json, written by pipeline/atlas_fixtures/ops/make.py (torch, CPU, TF32 off).
// Run from site/: node --test src/lib/atlas/ops/goldens.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { AnyTensor, AtlasLayer, AtlasSpec, OpCtx, OpDef, Tensor, Trace, UnitExplanation } from '../types/index.ts';
import { CORE_OPS, coreRegistry, mergeParams } from './index.ts';
import { MissingWeightError } from './common.ts';
import { allclose } from '../core/compare.ts';
import { f32FromB64, f64FromB64, i32FromB64 } from '../core/io.ts';
import { numel, unravel } from '../core/tensor.ts';
import { texNames } from '../core/tex.ts';

const DIR = fileURLToPath(new URL('../../../../tests/atlas/ops/', import.meta.url));
const ATOL = 1e-6;
const RTOL = 1e-6;
const EXPLAIN_BAR = 1e-5;

interface B64 {
	shape: number[];
	dtype: string;
	b64: string;
}
interface GoldenCase {
	id: string;
	op: string;
	params: Record<string, unknown>;
	inputs: B64[];
	weights: Record<string, B64>;
	out: B64;
	subs?: Record<string, B64>;
}

function dec(t: B64): AnyTensor {
	if (t.dtype === 'int32') return { shape: t.shape, data: i32FromB64(t.b64), dtype: 'i32' };
	if (t.dtype === 'float64') return { shape: t.shape, data: f64FromB64(t.b64) };
	return { shape: t.shape, data: f32FromB64(t.b64) };
}

/** Everything one golden case needs: the op, its params, a layer, a mini spec, and a runner. */
export function setup(c: GoldenCase) {
	const r = coreRegistry.get(c.op);
	const def = r.def as OpDef;
	const params = mergeParams(def, r.params, c.params);
	const ins = c.inputs.map(dec);
	const weights: Record<string, Tensor> = {};
	for (const [k, v] of Object.entries(c.weights)) weights[k] = dec(v) as Tensor;
	const layer: AtlasLayer = { id: 'g', op: c.op, label: { en: 'golden', zh: '金标' }, in: ins.map((_, i) => `x${i}`), outShape: c.out.shape, params: c.params };
	const spec: AtlasSpec = {
		slug: 'golden',
		name: { en: 'golden', zh: '金标' },
		version: '1',
		params: 0,
		precision: 'fp32',
		tensors: Object.entries(weights).map(([k, v]) => ({ name: `g.${k}`, shape: v.shape, offset: 0, length: v.data.length })),
		layers: [...ins.map((x, i): AtlasLayer => ({ id: `x${i}`, op: 'input', label: { en: 'x' }, in: [], outShape: x.shape })), layer],
		inputs: ins.map((x, i) => ({ name: `in${i}`, adapter: 'none', kind: 'tensor', shape: x.shape })),
		parity: { against: 'torch-f64', max_abs_diff: null, checked: null }
	};
	const w = (slot: string): Tensor => {
		const t = weights[slot];
		if (!t) throw new MissingWeightError(slot, `g.${slot}`, 'g');
		return t;
	};
	const run = () => {
		const subs: Record<string, Tensor> = {};
		const ctx: OpCtx = { layer, dims: {}, w, emit: (s, t) => (subs[s] = t) };
		const out = def.forward(ins, params, ctx);
		return { out, subs };
	};
	const traceOf = (out: Tensor, subs: Record<string, Tensor>): Trace => {
		const t: Record<string, Tensor> = {};
		ins.forEach((x, i) => (t[`x${i}`] = { shape: x.shape, data: Float32Array.from(x.data) }));
		t.g = out;
		for (const [k, v] of Object.entries(subs)) t[`g#${k}`] = v;
		return { order: Object.keys(t), t };
	};
	const explain = (trace: Trace, unit: number[]): UnitExplanation =>
		def.explain!({ layer, unit, ins, out: trace.t.g, w, p: params, trace, spec });
	return { def, params, ins, weights, layer, spec, run, traceOf, explain };
}

export function loadGolden(op: string): { cases: GoldenCase[] } | null {
	const f = `${DIR}${op}.json`;
	return existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : null;
}

const PLACEHOLDER = /\{([A-Za-z_][\w.:#-]*)(?:\|([a-z0-9]+))?\}/g;
const names = (s: string) => [...s.matchAll(PLACEHOLDER)].map((m) => m[1]);

for (const def0 of CORE_OPS) {
	test(`golden ${def0.op}`, (t) => {
		const g = loadGolden(def0.op);
		assert.ok(g, `missing tests/atlas/ops/${def0.op}.json (run pipeline/atlas_fixtures/ops/make.py)`);
		assert.equal(g.cases.length, 8, `${def0.op}: 8 seeded cases`);
		let worstRatio = 0;
		let worstAbs = 0;
		let units = 0;
		let explainMax = 0;
		for (const c of g.cases) {
			const S = setup(c);
			const { out, subs } = S.run();
			const want = dec(c.out);
			assert.deepEqual(out.shape, want.shape, `${c.id}: shape`);
			const r = allclose(out.data, want.data, { atol: ATOL, rtol: RTOL });
			assert.ok(r.ok, `${c.id}: |a−b| ≤ 1e-6 + 1e-6|b| fails at ${JSON.stringify(r.worst)}`);
			worstRatio = Math.max(worstRatio, r.ratio);
			if (r.worst && Number.isFinite(r.worst.diff)) worstAbs = Math.max(worstAbs, r.worst.diff);
			for (const [k, v] of Object.entries(c.subs ?? {})) {
				assert.ok(subs[k], `${c.id}: sub-step ${k} not emitted`);
				const rs = allclose(subs[k].data, dec(v).data, { atol: ATOL, rtol: RTOL });
				assert.ok(rs.ok, `${c.id}#${k}: ${JSON.stringify(rs.worst)}`);
				worstRatio = Math.max(worstRatio, rs.ratio);
			}
			// inferShape
			const wShapes = Object.fromEntries(Object.entries(S.weights).map(([k, v]) => [k, v.shape]));
			assert.deepEqual(S.def.inferShape(S.ins.map((x) => x.shape), S.params, wShapes), out.shape, `${c.id}: inferShape`);
			// explain = forward on every unit
			const trace = S.traceOf(out, subs);
			const n = numel(out.shape);
			for (let f = 0; f < n; f++) {
				const unit = unravel(out.shape, f);
				const e = S.explain(trace, unit);
				units++;
				assert.ok(Object.is(e.traced, out.data[f]) || e.traced === out.data[f], `${c.id} ${unit}: traced is not the output value`);
				const d = e.result === e.traced ? 0 : Math.abs(e.result - e.traced);
				if (!(d <= EXPLAIN_BAR)) assert.fail(`${c.id} unit [${unit}]: explain ${e.result} vs traced ${e.traced}`);
				explainMax = Math.max(explainMax, d);
				for (const sub of Object.keys(subs)) {
					const sv = subs[sub].data[f];
					assert.ok(Math.abs(e.values[sub] - sv) <= EXPLAIN_BAR, `${c.id} unit [${unit}]: values.${sub} ${e.values[sub]} vs sub-step ${sv}`);
				}
				if (S.def.receptive && e.reads[0]?.box && out.shape.length === 3) {
					const [, y, x] = unit;
					const b = S.def.receptive({ y0: y, x0: x, y1: y, x1: x }, S.params);
					const [, H, W] = S.ins[0].shape;
					const clamped = { y0: Math.max(0, b.y0), x0: Math.max(0, b.x0), y1: Math.min(H - 1, b.y1), x1: Math.min(W - 1, b.x1) };
					assert.deepEqual(e.reads[0].box, clamped, `${c.id} unit [${unit}]: receptive() vs explain() window`);
				}
				if (f === 0) {
					// templates resolve: what/how/why/look/repeat from vars(); unit and formula also from explain().values
					const vars = S.def.vars ? S.def.vars(S.layer, S.spec) : {};
					for (const [field, txt] of Object.entries(S.def.narration)) {
						if (!txt) continue;
						for (const lang of ['en', 'zh'] as const) {
							for (const name of names(txt[lang] ?? '')) {
								const v = name in e.values && field === 'unit' ? e.values[name] : vars[name];
								assert.ok(v !== undefined, `${S.def.op}.narration.${field}.${lang}: {${name}} does not resolve`);
								if (typeof v === 'number') assert.ok(Number.isFinite(v), `${S.def.op}.narration.${field}: {${name}} = ${v}`);
								// an auto step merges explain().values over vars() (lesson/plan.ts opVars), so a vars() text or
								// phrase must not share its name with an explain value (R0: layernorm's ε printed as 1.00e−5)
								if (name in e.values && field !== 'unit') assert.equal(typeof vars[name], 'number', `${S.def.op}.narration.${field}: {${name}} is a vars() ${typeof vars[name]} that explain().values.${name} shadows`);
							}
						}
					}
					for (const name of texNames(S.def.formula ?? '')) assert.ok(name in e.values, `${S.def.op}.formula: {{${name}}} is not in explain().values`);
				}
			}
		}
		t.diagnostic(`${def0.op}: 8 cases; worst |a−b|/(1e-6+1e-6|b|) = ${worstRatio.toExponential(2)} (max |a−b| ${worstAbs.toExponential(2)}); explain = forward on ${units} units, max |diff| ${explainMax.toExponential(2)}`);
	});
}
