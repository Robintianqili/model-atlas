// tinyclip: the page's batch of N = 6 pairs. Pair 0 is the learner's (their drawing and their first caption); pairs 1 to 5
// are fixed (scenes.ts PARTNERS). Each pair runs through the exported model (rt.model, the same graph as the `fwd` trace),
// giving N image vectors and N caption vectors of length 1: the input of the `nce` process. batchOf(rt, 'init') runs the
// same pairs through the starting weights (the `init` trace source, static/models/tinyclip/init): the batch before training.
// The partners' vectors do not depend on the learner's input, so they are computed once per model.
import type { ExplainerRuntime, GraphModel } from '../../atlas/types/index.ts';
import type { BatchIn } from './process.ts';
import { PARTNERS, render, strokesFor, tokenize } from './scenes.ts';
import { captionsOf, textTensor } from './adapters/shapes-captions/adapter.ts';
import { scaleOf } from './ops/clip.ts';

let partners: { image: Float32Array; caption: string }[] | null = null;
const fixedVecs = new WeakMap<GraphModel, number[][][]>();
const memo = new Map<string, WeakMap<object, { images: Float32Array[]; captions: string[]; batch: BatchIn | null }>>();
/** [image vector, caption vector] of one pair */
const vecs = (m: GraphModel, image: Float32Array, caption: string) => {
	const tr = m.forward({ image: { shape: [1, 28, 28], data: image }, text: textTensor([tokenize(caption).ids]) }, { record: 'layers' });
	return [Array.from(tr.t['img.norm'].data), Array.from(tr.t['txt.norm'].data)];
};

/** The N drawings (28×28 each) and captions of the batch, and the batch's vectors (null before the model is loaded). */
export function batchOf(rt: ExplainerRuntime, src = 'fwd') {
	const model = src === 'fwd' ? rt.model : (rt.sources[src]?.model ?? null);
	const key = rt.pre ?? rt;
	if (!memo.has(src)) memo.set(src, new WeakMap());
	const hit = memo.get(src)!.get(key);
	if (hit && (hit.batch || !model)) return hit;
	partners ??= PARTNERS.map((f) => ({ image: render(strokesFor(f)), caption: f.captions[0] }));
	const own = rt.pre?.tensors.image?.data as Float32Array | undefined;
	const images = [own ?? new Float32Array(784), ...partners.map((p) => p.image)];
	const captions = [captionsOf(rt.input)[0]?.text ?? '', ...partners.map((p) => p.caption)];
	let batch: BatchIn | null = null;
	if (model && own) {
		if (!fixedVecs.has(model)) fixedVecs.set(model, partners.map((p) => vecs(model, p.image, p.caption)));
		const all = [vecs(model, own, captions[0]), ...fixedVecs.get(model)!];
		batch = { I: all.flatMap((v) => v[0]), T: all.flatMap((v) => v[1]), N: images.length, D: all[0][0].length, scale: scaleOf(model.weights['sim.logit_scale'].data[0]) };
	}
	const out = { images, captions, batch };
	memo.get(src)!.set(key, out);
	return out;
}
