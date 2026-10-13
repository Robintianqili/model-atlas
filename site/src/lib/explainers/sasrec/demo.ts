// sasrec: the view gallery's demo (illustrative): a made-up model with the page's layer names and shapes (random tables
// from createRng(5), no file, a crude attention without the learned weights), traced on the first preset.
import type { GraphModel, StageProps, Tensor, Trace } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';
import { N_ITEMS, PRESETS } from './world.ts';

const D = 16;
export async function demoStage(props: Record<string, unknown>): Promise<Omit<StageProps, 'ctx'>> {
	const rng = createRng(5), tab = (n: number) => ({ shape: [n, D], data: Float32Array.from({ length: n * D }, () => 0.5 * rng.normal()) });
	const w: Record<string, Tensor> = { 'item.weight': tab(N_ITEMS), 'pos.weight': tab(12) };
	const forward = (inp: Record<string, Tensor>): Trace => {
		const ids = Array.from(inp.ids.data), n = ids.length, E = w['item.weight'].data, Pw = w['pos.weight'].data;
		const item = Float32Array.from({ length: n * D }, (_, k) => E[ids[Math.floor(k / D)] * D + (k % D)]);
		const pos = Pw.slice(0, n * D), x = item.map((v, k) => v + pos[k]);
		const att = new Float32Array(n * n), lg = new Float32Array(n * N_ITEMS);
		for (let q = 0; q < n; q++) {
			const s = Array.from({ length: q + 1 }, (_, j) => Array.from({ length: D }, (_, d) => x[q * D + d] * x[j * D + d]).reduce((a, b) => a + b, 0) / 4);
			const m = Math.max(...s), e = s.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
			e.forEach((v, j) => (att[q * n + j] = v / z));
			for (let i = 0; i < N_ITEMS; i++) for (let d = 0; d < D; d++) lg[q * N_ITEMS + i] += x[q * D + d] * E[i * D + d];
		}
		const pr = new Float32Array(n * N_ITEMS);
		for (let q = 0; q < n; q++) {
			const r = lg.subarray(q * N_ITEMS, (q + 1) * N_ITEMS), m = Math.max(...r), z = r.reduce((a, v) => a + Math.exp(v - m), 0);
			r.forEach((v, i) => (pr[q * N_ITEMS + i] = Math.exp(v - m) / z));
		}
		const t: Record<string, Tensor> = { item: { shape: [n, D], data: item }, pos: { shape: [n, D], data: pos }, x: { shape: [n, D], data: x }, 'att.w': { shape: [1, n, n], data: att }, logits: { shape: [n, N_ITEMS], data: lg }, probs: { shape: [n, N_ITEMS], data: pr } };
		return { order: Object.keys(t), t };
	};
	const model = { weights: w, forward } as unknown as GraphModel;
	const raw = PRESETS[0].raw;
	const trace = forward({ ids: { shape: [raw.clicks.length], data: Float32Array.from(raw.clicks) } });
	const rt = fakeRuntime({ slug: 'sasrec', input: raw, trace, model, spec: { slug: 'sasrec', version: 'demo', layers: [], clicks: Array.from({ length: N_ITEMS }, (_, i) => (i === N_ITEMS - 1 ? 0 : 100 + 40 * (i % 7))), popTop5: [10, 15, 11, 0, 14] } as never });
	return { rt, source: 'fwd', at: { index: 0 } as never, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
