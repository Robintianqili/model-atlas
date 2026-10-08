// View 'ddpm-unet/pics': the pictures of a DDPM, by props.mode:
//   mix      (trace) your drawing x₀, the noise ε and the noisy x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε, with one pixel worked out
//   noise    (forward tape) your drawing at the frame's step t, and the two weights √ᾱ_t, √(1 − ᾱ_t) over all steps
//   predict  (trace) x_t, the noise that was mixed in and the network's guess ε̂ = UNet(x_t, t); buttons pick t
//   step     (sample tape) one denoising step x_t → ε̂ → x_{t−1}, with the formula for one pixel
//   chain    (sample tape) the picture at the frame, and every fifth of the run so far
//   gallery  (gallery tape) the last picture of 12 seeds; a click picks the seed (the raw input's `seed`)
//   final    (sample tape) the last picture of the run and its number of steps (the two panels of a compare)
//   unet     (trace) the U-Net: what each stage holds, its shape, and the two skip connections
//   pick     gallery + chain: the task
//   limits   (gallery tape + trace) the twelve samples next to the learner's own drawing
import type { AtlasSpec, I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { N, T, mse, tOf, tr } from '../../ddpm.ts';

export type Mode = 'mix' | 'noise' | 'predict' | 'step' | 'chain' | 'gallery' | 'final' | 'unet' | 'pick' | 'limits';
type P = Omit<StageProps, 'ctx'>;
const xOf = (s: unknown) => (s as { x?: Tensor } | undefined)?.x?.data;
export const modeOf = (p: P) => (p.props?.mode as Mode | undefined) ?? 'mix';

/** the first pixel with the most ink in the learner's drawing (where the noise is easiest to see) */
export function probeOf(x0: ArrayLike<number>): number {
	let b = 0;
	for (let i = 1; i < x0.length; i++) if (x0[i] > x0[b]) b = i;
	return b;
}

const TC = new WeakMap<object, ReturnType<typeof traceOne>>();
/** the traced x₀, ε, x_t, ε̂ and the mixing weights of the trace's step t (one object per trace, so views redraw only
 *  when the trace changes, not on every frame another source streams in) */
export function traceOf(p: Pick<P, 'rt'>) {
	const key = p.rt.sources.fwd?.trace;
	if (!key) return null;
	if (!TC.has(key)) TC.set(key, traceOne(p));
	return TC.get(key)!;
}
function traceOne(p: Pick<P, 'rt'>) {
	const x0 = tr(p.rt, 'input')?.data, eps = tr(p.rt, 'eps')?.data, xt = tr(p.rt, 'xt')?.data, out = tr(p.rt, 'out')?.data;
	const tab = p.rt.spec?.layers.find((l) => l.id === 'xt')?.params?.abar as number[] | undefined;
	if (!x0 || !eps || !xt || !out || !tab) return null;
	const t = tOf(p.rt.input), abar = tab[t - 1], i = probeOf(x0);
	return { x0, eps, xt, out, t, abar, a: Math.sqrt(abar), b: Math.sqrt(1 - abar), i, s: 2 * x0[i] - 1, err: mse(out, eps) };
}

/** frame i of a sampling tape shows x after i steps; the step frame holds ε̂ and the metrics of its formula */
export function stepOf(p: Pick<P, 'at'>) {
	const tape = p.at?.tape, i = p.at?.index ?? -1;
	if (!tape || i < 1) return null;
	const before = xOf(tape.state(i - 1)), s = tape.state(i) as { x?: Tensor; eps?: Tensor };
	return before && s.x && s.eps ? { before, eps: s.eps.data, after: s.x.data, m: tape.frame(i).metrics ?? {} } : null;
}

/** the frame's picture and the pictures at every fifth of the run so far (their steps t) */
export function filmOf(p: Pick<P, 'at'>) {
	const tape = p.at?.tape, i = p.at?.index ?? -1;
	if (!tape || i < 0) return null;
	const K = Number(tape.frame(0).metrics?.K ?? NaN), stride = T / K;
	const film = [0, 1, 2, 3, 4, 5].map((q) => Math.round((q * K) / 5)).filter((j, q, a) => j <= i && a.indexOf(j) === q);
	return { x: xOf(tape.state(i)), t: T - i * stride, K, film: film.map((j) => ({ t: T - j * stride, x: xOf(tape.state(j)) })) };
}

/** the gallery tape (precomputed): per seed, its pictures at t = 100, 80, …, 0, and the last one */
export function galleryOf(p: Pick<P, 'rt'>) {
	const tape = p.rt.sources.seeds?.tape, out: { seed: number; x?: ArrayLike<number>; film: { t: number; x?: ArrayLike<number> }[] }[] = [];
	for (let i = 0; i < (tape?.length ?? 0); i++) {
		const f = tape!.frame(i), q = (out[Number(f.step)] ??= { seed: Number(f.step), film: [] });
		q.film.push({ t: Number(f.metrics?.t), x: xOf(tape!.state(i)) });
		q.x = q.film[q.film.length - 1].x;
	}
	return out.filter(Boolean);
}

/** share of the 784 pixels with more than half ink (a DDPM picture runs from −1, paper, to 1, ink) */
export const inkShare = (x: ArrayLike<number> | undefined) => (x ? Array.from(x).filter((v) => v > 0).length / N : NaN);

export function describePics(p: P): I18n {
	const mode = modeOf(p);
	const tc = traceOf(p);
	if (['mix', 'predict', 'unet'].includes(mode)) {
		if (!tc) return { en: 'Nothing drawn yet: the pictures come from your drawing.', zh: '还没有画：这些图来自你的画。' };
		if (mode === 'mix') return { en: `Your drawing, the noise ε and the noisy drawing x_${tc.t} = ${fmt3(tc.a)} × drawing + ${fmt3(tc.b)} × noise.`, zh: `你的画、噪声 ε，以及加噪后的画 x_${tc.t} = ${fmt3(tc.a)} × 画 + ${fmt3(tc.b)} × 噪声。` };
		if (mode === 'predict') return { en: `The noisy drawing at t = ${tc.t}, the noise that was mixed in, and the network's guess of it; mean squared difference ${fmt3(tc.err)}.`, zh: `t = ${tc.t} 时加噪的画、混进去的噪声，以及网络对它的猜测；平均平方差 ${fmt3(tc.err)}。` };
		return { en: 'The U-Net: down from 28 × 28 to 7 × 7 pixels with more channels, back up to 28 × 28, with two skip connections that copy the maps of the way down to the way up.', zh: 'U-Net：从 28 × 28 像素往下到 7 × 7，通道变多；再往上回到 28 × 28；两条跳跃连接把下行路径的图复制到上行路径。' };
	}
	if (mode === 'noise') {
		const m = p.at?.frame?.metrics;
		return m ? { en: `Your drawing at step t = ${m.t}: ${fmt3(m.a)} × drawing + ${fmt3(m.b)} × noise.`, zh: `第 t = ${m.t} 步的画：${fmt3(m.a)} × 画 + ${fmt3(m.b)} × 噪声。` } : { en: 'The noising run has not started yet.', zh: '加噪还没有开始。' };
	}
	if (mode === 'step') {
		const s = stepOf(p);
		return s ? { en: `One step from t = ${s.m.t} to ${s.m.tNext}: the picture, the predicted noise, and the picture after the step.`, zh: `从 t = ${s.m.t} 到 ${s.m.tNext} 的一步：这一步之前的图、预测的噪声，以及这一步之后的图。` } : { en: 'The sampling run has no step yet.', zh: '采样还没有走出一步。' };
	}
	if (mode === 'limits') return { en: `The last picture of ${galleryOf(p).length} sampling runs, next to your own drawing.`, zh: `${galleryOf(p).length} 次采样的最后一幅图，旁边是你自己的画。` };
	if (mode === 'gallery') return { en: `The last picture of ${galleryOf(p).length} sampling runs, one per seed.`, zh: `${galleryOf(p).length} 次采样的最后一幅图，每个种子一幅。` };
	const f = filmOf(p);
	if (!f) return { en: 'The sampling run has not started yet.', zh: '采样还没有开始。' };
	return { en: `The picture at t = ${f.t} of a ${f.K}-step run, ${fmt3(inkShare(f.x) * 100)}% of its pixels more than half ink.`, zh: `${f.K} 步采样中 t = ${f.t} 时的图，${fmt3(inkShare(f.x) * 100)}% 的像素墨量超过一半。` };
}

const meta = {
	key: 'ddpm-unet/pics',
	title: { en: 'Drawings, noise and denoising', zh: '画、噪声与去噪' },
	kinds: ['trace', 'process'],
	describe: describePics,
	async demo() {
		// made-up data: a ring, noise from a fixed rng, and the ring mixed with it at ᾱ = 0.5
		const [{ fakeRuntime }, { createRng }] = await Promise.all([import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/core/rng.ts')]);
		const r = createRng(7), ring = new Float32Array(N), eps = new Float32Array(N), xt = new Float32Array(N), out = new Float32Array(N);
		for (let i = 0; i < N; i++) {
			ring[i] = Math.abs(Math.hypot((i % 28) - 13.5, Math.floor(i / 28) - 13.5) - 8) < 1.2 ? 1 : 0;
			eps[i] = r.normal();
			xt[i] = Math.SQRT1_2 * (ring[i] + eps[i]);
			out[i] = 0.8 * eps[i];
		}
		const tz = (data: Float32Array) => ({ shape: [1, 28, 28], data });
		const trace = { order: ['input', 'eps', 'xt', 'out'], t: { input: tz(ring), eps: tz(eps), xt: tz(xt), out: tz(out) } };
		const spec = { slug: 'ddpm-unet', layers: [{ id: 'xt', op: 'q_sample', in: [], outShape: [1, 28, 28], params: { abar: [0.5] } }] } as unknown as AtlasSpec;
		const rt = fakeRuntime({ slug: 'ddpm-unet', spec, trace, input: { strokes: [], t: 1 } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'mix' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
