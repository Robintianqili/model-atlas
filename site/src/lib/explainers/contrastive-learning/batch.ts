// contrastive-learning: the batch's vectors. Every pair of the learner's batch (adapters/batch) runs through a TinyCLIP
// graph (rt.sources[src].model: the shipped TinyCLIP of /m/tinyclip/, or a checkpoint of the re-run in
// static/models/contrastive-learning/ck/<update>/), giving one image vector and one caption vector of length 1 per pair.
// One forward per pair (both towers); vectors are memoized per model and per (drawing, caption), so a new temperature
// or one added pair costs nothing or one forward.
import type { ExplainerRuntime, GraphModel, IntTensor } from '../../atlas/types/index.ts';
import { render, tokenize, type Pt } from './scenes.ts';
import { drawingKey, pairsOf, tauOf } from './adapters/batch/adapter.ts';
import { scaleOf } from './ops/clip.ts';

/** the checkpoints of the re-run (training updates; static/models/contrastive-learning/ck/<update>/) */
export const CKPTS = [0, 10, 20, 30, 50, 100, 300, 4680] as const;
export const ckSource = (u: number) => `ck${u}`;

const memo = new WeakMap<GraphModel, Map<string, [number[], number[]]>>();
export function textTensor(ids: number[]): IntTensor {
	return { shape: [1, Math.max(1, ids.length)], data: Int32Array.from(ids.length ? ids : [0]), dtype: 'i32' };
}
/** [image vector, caption vector] of one pair through model m */
export function pairVecs(m: GraphModel, strokes: Pt[][], caption: string, image?: Float32Array): [number[], number[]] {
	if (!memo.has(m)) memo.set(m, new Map());
	const k = `${drawingKey(strokes)}#${caption}`;
	const hit = memo.get(m)!.get(k);
	if (hit) return hit;
	const tr = m.forward({ image: { shape: [1, 28, 28], data: image ?? render(strokes) }, text: textTensor(tokenize(caption).ids) }, { record: 'layers' });
	const out: [number[], number[]] = [Array.from(tr.t['img.norm'].data), Array.from(tr.t['txt.norm'].data)];
	memo.get(m)!.set(k, out);
	return out;
}
/** The image tower alone (SimCLR views): the vector of one 28×28 drawing. */
export function imageVec(m: GraphModel, image: Float32Array): number[] {
	const tr = m.forward({ image: { shape: [1, 28, 28], data: image }, text: textTensor([]) }, { record: 'layers' });
	return Array.from(tr.t['img.norm'].data);
}
export const scaleIn = (m: GraphModel | null | undefined) => (m ? scaleOf(m.weights['sim.logit_scale'].data[0]) : NaN);

/** nce input: the N image vectors and N caption vectors (row i of each is pair i), D numbers each, and τ. */
export type BatchIn = { I: number[]; T: number[]; N: number; D: number; tau: number };
/** The learner's batch through the model of source `src` (default the shipped TinyCLIP), or null before it is loaded. */
export function batchIn(rt: ExplainerRuntime, src = 'clip'): BatchIn | null {
	const m = rt.sources[src]?.model;
	const pairs = pairsOf(rt.input);
	if (!m || pairs.length < 2) return null;
	const imgs = rt.pre?.tensors.images?.data as Float32Array | undefined;
	const v = pairs.map((p, i) => pairVecs(m, p.strokes, p.caption, imgs?.subarray(i * 784, (i + 1) * 784)));
	return { I: v.flatMap((x) => x[0]), T: v.flatMap((x) => x[1]), N: pairs.length, D: v[0][0].length, tau: tauOf(rt.input) };
}
/** The batch through every checkpoint of the re-run (progress input): the models are passed by reference (run: 'main'). */
export function progressIn(rt: ExplainerRuntime) {
	const pairs = pairsOf(rt.input);
	const cks = CKPTS.map((u) => ({ update: u, model: rt.sources[ckSource(u)]?.model ?? null }));
	if (pairs.length < 2 || cks.some((c) => !c.model)) return null;
	return { pairs: pairs.map((p) => ({ strokes: p.strokes, caption: p.caption })), cks: cks as { update: number; model: GraphModel }[] };
}
/** SimCLR input: the drawings and the shipped TinyCLIP's image tower; τ as set. */
export function simclrIn(rt: ExplainerRuntime) {
	const m = rt.sources.clip?.model;
	const pairs = pairsOf(rt.input);
	if (!m || pairs.length < 2) return null;
	return { strokes: pairs.map((p) => p.strokes), model: m, tau: tauOf(rt.input) };
}
