// W1-D2 view tests (Node; part of `npm run check:atlas`). Run alone from site/:
//   node --test src/lib/atlas/ui/views/kit.parts/views.test.ts
//
//  1. Every kit view's demo and variants: describe() is non-empty in en and zh and never prints NaN, undefined, '…'
//     or '[object'; zoom demos resolve a UnitExplanation whose result equals the traced value.
//  2. Real models: TinyCNN and TinyGPT loaded from static/models/ (their shipped reference inputs), every layer of
//     both models resolved the way the zoom views resolve a stage (the op's own unit pick), explained by W1-A's
//     explain(), equal to the trace (≤ 1e-5), and described by the zoom view its op asks for, in en and zh.
//     This is the data side of acceptance 3; the gallery's "On real models" section renders the same stages.
//  3. alignTex keeps every {{name}} of every CORE_OPS formula.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import type { I18n, GraphModel, Trace, ViewMeta } from '../../../types/index.ts';
import type { Props } from './stage.ts';
import { resolveUnit } from './stage.ts';
import { alignTex } from './tex.ts';
import { formulaOf } from '../zoom.formula/formula.ts';
import { fsFetch } from '../../../core/io.ts';
import { loadGraph } from '../../../graph/executor.ts';
import { readReference } from '../../../graph/refs.ts';
import { CORE_OPS } from '../../../ops/index.ts';
import { fakeRuntime } from '../../../runtime/fake.ts';
import { traceToTape } from '../../../process/adapters.ts';
import { atFor } from '../../../runtime/sources.ts';

const here = dirname(fileURLToPath(import.meta.url));
const VIEWS = join(here, '..');
const SITE = join(here, '..', '..', '..', '..', '..', '..');
const MODELS = join(SITE, 'static', 'models');

type MetaMod = { default: ViewMeta; variants?: () => Promise<{ label: I18n; stage: Props }[]> };

async function metas(): Promise<{ key: string; mod: MetaMod }[]> {
	const out: { key: string; mod: MetaMod }[] = [];
	for (const key of readdirSync(VIEWS).sort()) {
		const file = join(VIEWS, key, 'meta.ts');
		if (existsSync(file)) out.push({ key, mod: (await import(pathToFileURL(file).href)) as MetaMod });
	}
	return out;
}

const BAD = /NaN|undefined|…|\[object|\{\{/;
function checkText(d: I18n, what: string) {
	assert.ok(d.en?.trim(), `${what}: describe().en is empty`);
	assert.ok(d.zh?.trim(), `${what}: describe().zh is empty`);
	assert.ok(!BAD.test(d.en), `${what}: en text has a bad token: ${d.en}`);
	assert.ok(!BAD.test(d.zh!), `${what}: zh text has a bad token: ${d.zh}`);
	assert.ok(/[一-鿿]/.test(d.zh!), `${what}: zh text has no Chinese: ${d.zh}`);
}

test('every kit view: demo and variants describe themselves in en and zh; zoom demos equal the trace', async () => {
	const all = await metas();
	assert.ok(all.length >= 15, `expected the 15 W1-D2 views, found ${all.length}`);
	let stages = 0;
	for (const { key, mod } of all) {
		const list = [{ label: { en: 'demo' }, stage: (await mod.default.demo()) as Props }, ...((await mod.variants?.()) ?? [])];
		for (const v of list) {
			stages++;
			assert.equal(v.stage.provenance, 'illustrative', `${key} / ${v.label.en}: demo provenance`);
			checkText(mod.default.describe(v.stage as Parameters<ViewMeta['describe']>[0]), `${key} / ${v.label.en}`);
			if (key.startsWith('zoom.') && key !== 'zoom.formula') {
				const r = resolveUnit(v.stage);
				assert.ok(r.e, `${key} / ${v.label.en}: no unit explanation (${r.error})`);
				const same = r.e!.result === r.e!.traced || Math.abs(r.e!.result - r.e!.traced) <= 1e-5; // −∞ == −∞ (masks)
				assert.ok(same, `${key} / ${v.label.en}: result ${r.e!.result} vs traced ${r.e!.traced}`);
			}
		}
	}
	assert.ok(stages >= 40, `only ${stages} demo stages`);
});

function realStage(model: GraphModel, trace: Trace, layer: string, extra: Record<string, unknown> = {}): Props {
	const spec = model.spec;
	const rt = fakeRuntime({ slug: spec.slug, spec, model });
	rt.sources = { fwd: { kind: 'trace', status: 'ready', provenance: 'live', trace, tape: traceToTape(spec, trace), model } };
	rt.sel = { layer };
	const tape = rt.sources.fwd.tape!;
	const at = atFor(rt, 'fwd', tape.find({ layer }));
	at.layer = layer;
	return { rt, source: 'fwd', at, focus: [], sel: rt.sel, provenance: 'live', autoplay: false, onend: () => {}, props: extra };
}

test('real models: every TinyCNN and TinyGPT layer resolves through its zoom view, explain() equals the trace', async (t) => {
	const byKey = new Map((await metas()).map((m) => [m.key, m.mod.default]));
	const covered = new Map<string, string[]>();
	for (const slug of ['tinycnn', 'tinygpt']) {
		const dir = join(MODELS, slug);
		const model = await loadGraph(fsFetch, dir, CORE_OPS);
		const refs = await readReference(fsFetch, dir);
		const ref = refs[0];
		const trace = model.forward(ref.inputs);
		for (const layer of model.spec.layers) {
			const zoom = model.ops.get(layer.op).def.zoom;
			const key = zoom === 'none' ? 'layer.tensor' : `zoom.${zoom}`;
			const meta = byKey.get(key);
			assert.ok(meta, `${slug}/${layer.id}: view ${key} is not registered`);
			const stage = realStage(model, trace, layer.id);
			if (key !== 'layer.tensor') {
				const r = resolveUnit(stage);
				assert.ok(r.e, `${slug}/${layer.id}: ${r.error}`);
				const d = Math.abs(r.e!.result - r.e!.traced);
				assert.ok(r.e!.result === r.e!.traced || d <= 1e-5, `${slug}/${layer.id}: |result − traced| = ${d}`);
				(covered.get(key) ?? covered.set(key, []).get(key)!).push(`${slug}/${layer.id}`);
			}
			checkText(meta!.describe(stage as Parameters<ViewMeta['describe']>[0]), `${slug}/${layer.id} via ${key}`);
			// zoom.formula on the same real stage: the op formula of the unit, filled with explain() values
			const f = formulaOf(stage);
			assert.ok(f && f.from === 'op', `${slug}/${layer.id}: zoom.formula found no op formula`);
			const missing = [...f!.tex.matchAll(/\{\{\s*([\w.-]+)\s*\}\}/g)].map((m) => m[1]).filter((n) => !(n in f!.values));
			assert.deepEqual(missing, [], `${slug}/${layer.id}: formula placeholders without values`);
			(covered.get('zoom.formula') ?? covered.set('zoom.formula', []).get('zoom.formula')!).push(`${slug}/${layer.id}`);
		}
	}
	for (const [k, v] of [...covered].sort()) t.diagnostic(`${k}: ${v.length} layers (${v.slice(0, 4).join(', ')}${v.length > 4 ? ', …' : ''})`);
	// every zoom view except gate (no LSTM in either model) sees a real explanation; zoom.formula through the op path
	for (const k of ['zoom.weighted-sum', 'zoom.reduce', 'zoom.elementwise', 'zoom.normalize', 'zoom.softmax', 'zoom.lookup', 'zoom.index-map', 'zoom.formula']) {
		assert.ok(covered.get(k)?.length, `${k} is not exercised by TinyCNN or TinyGPT`);
	}
});

test('alignTex keeps every {{name}} of every op formula', () => {
	const names = (s: string) => [...s.matchAll(/\{\{\s*([\w.-]+)\s*\}\}/g)].map((m) => m[1]).sort();
	for (const op of CORE_OPS) {
		if (!op.formula) continue;
		const a = alignTex(op.formula);
		assert.deepEqual(names(a), names(op.formula), op.op);
		const open = (a.match(/\{/g) ?? []).length;
		const close = (a.match(/\}/g) ?? []).length;
		assert.equal(open, close, `${op.op}: unbalanced braces`);
	}
});
