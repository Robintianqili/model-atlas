// View 'vae-2d/pair': the learner's drawing next to the VAE's redrawing (props.mode 'redraw', the default), plus the
// loss of this drawing (mode 'loss'): the per-pixel binary cross-entropy as a map, its sum, the KL of each latent number;
// or (mode 'limits') the grey pixels (0.2 < v < 0.8, vae.ts greyShare) of both pictures marked.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { T, greyShare, lossOf } from '../../vae.ts';

export function pairPic(p: Omit<StageProps, 'ctx'>) {
	const x = T(p.rt, 'input'), y = T(p.rt, 'out');
	if (!x || !y) return null;
	const inked = (a: ArrayLike<number>) => Array.from(a).filter((v) => v > 0.5).length;
	const mode = p.props.mode === 'loss' || p.props.mode === 'limits' ? p.props.mode : 'redraw';
	return { mode, x: x.data, y: y.data, loss: lossOf(p.rt), inkIn: inked(x.data), inkOut: inked(y.data), greyIn: greyShare(x.data), greyOut: greyShare(y.data) };
}

export function describePair(p: Omit<StageProps, 'ctx'>): I18n {
	const m = pairPic(p);
	if (!m) return { en: 'Your drawing and its redrawing: nothing drawn yet.', zh: '你的画和重画：还没有画。' };
	const base = {
		en: `Your drawing (${m.inkIn} pixels more than half ink, ${fmtPct(m.greyIn)} of inked pixels grey) next to the VAE's redrawing (${m.inkOut} pixels more than half ink, ${fmtPct(m.greyOut)} grey).`,
		zh: `你的画（墨量超过一半的像素 ${m.inkIn} 个，有墨像素中 ${fmtPct(m.greyIn)} 是灰的）旁边是 VAE 的重画（墨量超过一半的像素 ${m.inkOut} 个，${fmtPct(m.greyOut)} 是灰的）。`
	};
	if (m.mode !== 'loss' || !m.loss) return base;
	const L = m.loss;
	return {
		en: `${base.en} Reconstruction loss ${fmt3(L.rec)} (summed over 784 pixels), KL ${fmt3(L.kl)}, total ${fmt3(L.total)}.`,
		zh: `${base.zh} 重画损失 ${fmt3(L.rec)}（784 个像素相加），KL ${fmt3(L.kl)}，合计 ${fmt3(L.total)}。`
	};
}

const meta = {
	key: 'vae-2d/pair',
	title: { en: 'Your drawing and its redrawing', zh: '你的画和它的重画' },
	kinds: ['trace'],
	describe: describePair,
	async demo() {
		// made-up data: a ring and a softer copy of it, with made-up scores and code numbers
		const ring = new Float32Array(784), soft = new Float32Array(784), score = new Float32Array(784);
		for (let i = 0; i < 784; i++) {
			const d = Math.abs(Math.hypot((i % 28) - 13.5, Math.floor(i / 28) - 13.5) - 8);
			ring[i] = d < 1.2 ? 1 : 0;
			soft[i] = Math.exp(-d * d / 6);
			score[i] = Math.log(soft[i] / (1 - soft[i] + 1e-6) + 1e-6);
		}
		const t = (shape: number[], data: Float32Array) => ({ shape, data });
		const trace = { order: ['input', 'mu', 'logvar', 'z', 'img', 'out'], t: { input: t([1, 28, 28], ring), mu: t([2], Float32Array.of(0.4, -0.2)), logvar: t([2], Float32Array.of(-4, -4.5)), 'z#sigma': t([2], Float32Array.of(0.135, 0.105)), img: t([1, 28, 28], score), out: t([1, 28, 28], soft) } };
		const rt = fakeRuntime({ slug: 'vae-2d', trace });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'loss' } };
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
