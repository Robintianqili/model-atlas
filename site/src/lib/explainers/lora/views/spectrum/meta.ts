// View 'lora/spectrum': the strengths (singular values s_k) of the 16 directions of the change the new task needs, ΔW*,
// strongest first, one bar each on a linear axis from 0. The first r bars (the frame's rank, metrics.r; the learner's
// rank before a frame exists) are the directions a rank-r adapter can learn at best; the line under the plot gives their
// share of Σ s_k² and the lowest loss any rank-r change can reach (data.ts floorOf). No props.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { SPECTRUM, floorOf, pctText, rankAt, shareOf } from '../../data.ts';

export function describeSpectrum(r: number): I18n {
	const list = SPECTRUM.map((s) => fmt3(s)).join(', ');
	return {
		en: `The strengths of the 16 directions of the needed change, strongest first: ${list}. A rank-${r} adapter can learn at most the first ${r}, ${pctText(shareOf(r))} of the total of the squares, so its loss cannot go below ${fmt3(floorOf(r))}.`,
		zh: `所需改变量的 16 个方向的强度，从强到弱：${list}。秩为 ${r} 的适配器最多只能学到前 ${r} 个，占平方和的 ${pctText(shareOf(r))}，所以它的损失不可能低于 ${fmt3(floorOf(r))}。`
	};
}

const meta = {
	key: 'lora/spectrum',
	title: { en: 'The directions of the needed change', zh: '所需改变量的各个方向' },
	kinds: ['process'],
	describe: (p: StageProps) => describeSpectrum(rankAt(p)),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const rt = fakeRuntime({ slug: 'lora', input: { r: 2 }, tapes: {} });
		return { rt, source: 'fit', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
