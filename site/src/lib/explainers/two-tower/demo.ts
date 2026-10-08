// two-tower: the views' gallery demos (illustrative): a made-up model with the page's layer names and shapes (random
// weights from createRng(5), no file), traced on the first preset, and the inbatch process recorded on it.
import type { GraphModel, StageProps, Tensor, Trace } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';
import { record } from '../../atlas/process/record.ts';
import { atFor } from '../../atlas/runtime/sources.ts';
import { MOVIES, PRESETS, itemFeatures, userFeatures } from './world.ts';
import { batchOf } from './recs.ts';
import { inbatch } from './process.ts';

export async function demoStage(props: Record<string, unknown>, source = 'fwd'): Promise<Omit<StageProps, 'ctx'>> {
	const rng = createRng(5), w: Record<string, Tensor> = {};
	const lin = (id: string, o: number, i: number) => ((w[`${id}.weight`] = { shape: [o, i], data: Float32Array.from({ length: o * i }, () => 0.5 * rng.normal()) }), (w[`${id}.bias`] = { shape: [o], data: new Float32Array(o) }));
	[['item', 8], ['user', 15]].forEach(([k, n]) => (lin(`${k}.fc1`, 16, n as number), lin(`${k}.fc2`, 8, 16)));
	const mv = (id: string, x: ArrayLike<number>) => Float32Array.from({ length: w[`${id}.bias`].shape[0] }, (_, o) => Array.from(x).reduce((s, v, i) => s + v * w[`${id}.weight`].data[o * x.length + i], 0));
	const tower = (k: string, x: Float32Array) => { const h = mv(`${k}.fc1`, x).map((v) => Math.max(0, v)); return { x, h, y: mv(`${k}.fc2`, h) }; };
	const T = (d: Float32Array): Tensor => ({ shape: [d.length], data: d });
	w['scores.weight'] = { shape: [MOVIES.length, 8], data: Float32Array.from(MOVIES.flatMap((m) => Array.from(tower('item', itemFeatures(m)).y))) };
	const forward = (inp: Record<string, Tensor>): Trace => {
		const a = tower('item', inp.item.data as Float32Array), b = tower('user', inp.user.data as Float32Array);
		const t = { item: T(a.x), 'item.relu': T(a.h), 'item.fc2': T(a.y), user: T(b.x), 'user.relu': T(b.h), 'user.fc2': T(b.y), scores: T(Float32Array.from(MOVIES, (_, i) => Array.from(b.y).reduce((s, v, d) => s + v * w['scores.weight'].data[i * 8 + d], 0))) };
		return { order: Object.keys(t), t };
	};
	const model = { weights: w, forward } as unknown as GraphModel;
	const raw = PRESETS[0].raw;
	const trace = forward({ user: T(userFeatures(raw.genres, raw.picks)), item: T(itemFeatures(raw.item)) });
	const rt = fakeRuntime({ slug: 'two-tower', input: raw, trace, model, spec: { slug: 'two-tower', version: 'demo', layers: [], likes: MOVIES.map((_, i) => 100 + 40 * (i % 7)), ranker: { w: [1, 0.5], b: -1, popMean: 5, popStd: 0.3 } } as never });
	if (source === 'inbatch') {
		const tape = record(inbatch, batchOf(rt), {}, { seed: 1 });
		const rt2 = fakeRuntime({ slug: 'two-tower', input: raw, tapes: { inbatch: tape } });
		return { rt: rt2, source, at: atFor(rt2, 'inbatch', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
	}
	return { rt, source, at: { index: 0 } as never, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
