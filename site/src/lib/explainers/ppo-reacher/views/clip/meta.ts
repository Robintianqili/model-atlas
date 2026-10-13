// ppo-reacher view 'clip': one minibatch of a fine train tape at a 'forward' frame, every sample as a dot at its
// probability ratio r = π(a | s) / π_old(a | s) (across) and its advantage A (up). props.clip (default true) shades the
// two regions where PPO's clipped objective stops pulling (A > 0 and r > 1 + ε; A < 0 and r < 1 − ε) and rings the
// sample the lesson works out (workedOf); clipped samples are drawn hollow.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { trainingProcess } from '../../../../atlas/train/trainer.ts';
import { ppo } from '../../trainer.ts';

export interface Mb { r: number[]; A: number[]; clipped: number[]; eps: number; j: number; n: number }
export function mbOf(b: Record<string, Tensor> | undefined): Mb | null {
	if (!b?.ratio) return null;
	const r = Array.from(b.ratio.data);
	return { r, A: Array.from(b.adv.data), clipped: Array.from(b.clipped.data), eps: b.eps.data[0], j: b.j.data[0], n: r.length };
}

/** The sample the lesson works out: the clipped one farthest from r = 1, else the unclipped one farthest from it. */
export function workedOf(m: Mb) {
	let i = 0;
	for (let k = 1; k < m.n; k++) {
		const better = m.clipped[k] !== m.clipped[i] ? m.clipped[k] > m.clipped[i] : Math.abs(m.r[k] - 1) > Math.abs(m.r[i] - 1);
		if (better) i = k;
	}
	const r = m.r[i], A = m.A[i], c = Math.min(1 + m.eps, Math.max(1 - m.eps, r));
	return { i, r, A, c, s1: r * A, s2: c * A, obj: Math.min(r * A, c * A), clipped: m.clipped[i] === 1 };
}

export function describeClip(p: Omit<StageProps, 'ctx'>): I18n {
	const m = p.at.index >= 0 && p.at.tape && p.at.index < p.at.tape.length ? mbOf((p.at.tape.state(p.at.index) as { batch?: Record<string, Tensor> }).batch) : null;
	if (!m) return { en: 'The minibatch: nothing to show yet.', zh: '这个小批量：还没有可显示的内容。' };
	const lo = Math.min(...m.r), hi = Math.max(...m.r), k = m.clipped.reduce((s, x) => s + x, 0);
	return { en: `Update ${m.j}: ${m.n} samples, ratios from ${fmt3(lo)} to ${fmt3(hi)}; ${k} are clipped (ε = ${m.eps}).`, zh: `第 ${m.j} 次更新：${m.n} 个样本，比率从 ${fmt3(lo)} 到 ${fmt3(hi)}；其中 ${k} 个被截断（ε = ${m.eps}）。` };
}

const meta = {
	key: 'ppo-reacher/clip',
	title: { en: 'Ratio and advantage of every sample, and the clip', zh: '每个样本的比率和优势，以及截断' },
	kinds: ['train'],
	describe: describeClip,
	async demo() {
		const input = { target: [-0.3, 0.6], eps: 0.2 };
		const tape = record(trainingProcess(ppo), input, { steps: 16 }, { seed: 1, detail: 'fine' });
		const rt = fakeRuntime({ slug: 'ppo-reacher', input, tapes: { iter: tape } });
		return { rt, source: 'iter', at: atFor(rt, 'iter', 61), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
