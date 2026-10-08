// tinyclip view 'match': what each mode draws, read from the step's source (pure TS: the view, its describe() and the
// lesson's narration use the same numbers).
//   pairs     the synth frame: training drawings and their captions
//   words     the text tower on the first caption: its word rows, their average, the caption vector
//   vectors   the image and caption vectors before and after dividing by their length
//   pair      the products of the two unit vectors, their sum (the cosine) and the score s · cos
//   matrix    the nce frame: scores L, row softmax P, column softmax Q or gradient G (props.show), with the batch's
//             drawings and captions
//   captions  the drawing against the learner's captions (the fwd trace), or against props.texts (run here on the
//             exported model with the learner's drawing)
import type { ExplainerRuntime, StageProps, Tensor } from '../../../../atlas/types/index.ts';
import { batchOf } from '../../batch.ts';
import { captionsOf, textTensor } from '../../adapters/shapes-captions/adapter.ts';
import { tokenize } from '../../scenes.ts';
import { scaleOf } from '../../ops/clip.ts';
import { num } from '../../../../atlas/lesson/say.ts';

type P = Pick<StageProps, 'rt' | 'at' | 'props'>;
const tr = (rt: ExplainerRuntime, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
const row = (t: Tensor | undefined, k: number, n: number) => (t ? Array.from(t.data.subarray(k * n, (k + 1) * n)) : []);
const len = (v: number[]) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
/** a share as a percentage, with more digits near 100% ("99.97%", "over 99.99%"; never a rounded "100.0%") */
export const pct = (v: number) => {
	if (v >= 0.9999 && v < 1) return { en: 'over 99.99%', zh: '99.99% 以上' };
	const n = num(v * 100, 'n4');
	return v > 0.995 && v < 1 && typeof n === 'object' ? { en: `${n.en}%`, zh: `${n.zh}%` } : num(v, 'pct');
};
export const scaleIn = (rt: ExplainerRuntime) => (rt.model ? scaleOf(rt.model.weights['sim.logit_scale'].data[0]) : NaN);

export function pairsData(p: P) {
	const st = (p.at.tape && p.at.index >= 0 ? p.at.tape.state(p.at.index) : {}) as { images?: Tensor; captions?: string[] };
	const caps = (st.captions ?? []) as string[];
	return caps.map((caption, i) => ({ caption, image: st.images!.data.subarray(i * 784, (i + 1) * 784) }));
}

export function wordsData(rt: ExplainerRuntime) {
	const e = tr(rt, 'txt.emb'), pool = tr(rt, 'txt.pool'), proj = tr(rt, 'txt.proj');
	const W = e ? e.shape[2] : 0, L = e ? e.shape[1] : 0;
	const cap = captionsOf(rt.input)[0];
	const words = (cap ? tokenize(cap.text).words.filter((w) => w.id > 0).slice(0, L) : []).map((w, t) => ({ w: w.w, row: row(e, t, W) }));
	return { caption: cap?.text ?? '', words, avg: row(pool, 0, W), vec: row(proj, 0, proj?.shape[1] ?? 0), width: W };
}

export function vectorsData(rt: ExplainerRuntime) {
	const D = tr(rt, 'img.proj')?.shape[0] ?? 0;
	const a = row(tr(rt, 'img.proj'), 0, D), b = row(tr(rt, 'txt.proj'), 0, D);
	const an = row(tr(rt, 'img.norm'), 0, D), bn = row(tr(rt, 'txt.norm'), 0, D);
	const prods = an.map((x, d) => x * bn[d]);
	const cos = prods.reduce((s, x) => s + x, 0);
	const scale = scaleIn(rt);
	return { D, a, b, an, bn, lenA: len(a), lenB: len(b), prods, cos, scale, score: tr(rt, 'sim')?.data[0] ?? NaN, caption: captionsOf(rt.input)[0]?.text ?? '' };
}

export type Show = 'L' | 'P' | 'Q' | 'G';
export function matrixData(p: P, show?: Show) {
	const b = batchOf(p.rt);
	const st = (p.at.tape && p.at.index >= 0 ? p.at.tape.state(p.at.index) : {}) as Partial<Record<Show, Tensor>>;
	const s = show ?? ((p.props.show as Show | undefined) ?? 'L');
	const M = st[s];
	return { N: b.images.length, show: s, M: M ? Array.from(M.data) : null, images: b.images, captions: b.captions };
}

/** Scores and shares of the learner's drawing against captions: the trace's, or `texts` run on the exported model. */
export function captionsData(rt: ExplainerRuntime, texts?: string[]) {
	if (!texts) {
		const sim = tr(rt, 'sim'), probs = tr(rt, 'probs');
		const caps = captionsOf(rt.input);
		const texts = caps.length ? caps.map((c) => c.text) : [''];
		return texts.map((text, k) => ({ text, score: sim?.data[k] ?? NaN, share: probs?.data[k] ?? NaN }));
	}
	if (!rt.model || !rt.pre) return texts.map((text) => ({ text, score: NaN, share: NaN }));
	const t = rt.model.forward({ image: rt.pre.tensors.image, text: textTensor(texts.map((x) => tokenize(x).ids)) }, { record: 'layers' });
	return texts.map((text, k) => ({ text, score: t.t.sim.data[k], share: t.t.probs.data[k] }));
}
