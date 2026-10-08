// kit.probbars (W1-D2): the kit's ProbBars on a class or vocabulary output (manifest `output.view`).
import type { ViewMeta } from '../../../types/view.ts';
import { outLabels, stageModel, type Props } from '../kit.parts/stage.ts';
import { demoStage, type DemoVariant } from '../kit.parts/demo.ts';
import { fmtPct } from '../../../core/format.ts';
import { distOf } from './dist.ts';

export function describeProbBars(p: Props) {
	const d = distOf(p);
	if (!d) return { en: 'Output probabilities: nothing to show yet.', zh: '输出概率：还没有可显示的内容。' };
	const order = Array.from(d.p.keys()).sort((a, b) => d.p[b] - d.p[a]).slice(0, 3);
	const labels = outLabels(p, stageModel(p));
	const nm = (i: number, zh: boolean) => {
		const l = labels?.(i);
		return l === undefined ? `#${i}` : typeof l === 'string' ? l : zh ? (l.zh ?? l.en) : l.en;
	};
	return {
		en: `${d.label.en}: the most likely ${d.axis === 'vocab' ? (d.next ? 'next tokens' : 'tokens at this position') : 'answers'} are ${order.map((i) => `“${nm(i, false)}” ${fmtPct(d.p[i])}`).join(', ')}.`,
		zh: `${d.label.zh ?? d.label.en}：最可能的${d.axis === 'vocab' ? (d.next ? '下一个词元' : '这个位置上的词元') : '答案'}是 ${order.map((i) => `“${nm(i, true)}” ${fmtPct(d.p[i])}`).join('，')}。`
	};
}

export async function variants(): Promise<DemoVariant[]> {
	return [{ label: { en: 'next-token probabilities', zh: '下一个词元的概率' }, stage: demoStage('gpt', 'probs') }];
}

const meta = {
	key: 'kit.probbars',
	title: { en: 'Output as probability bars', zh: '用概率条显示输出' },
	kinds: ['trace'],
	accepts: () => false,
	describe: describeProbBars,
	demo: async () => demoStage('cnn', 'probs'),
	minHeight: 260
} satisfies ViewMeta;

export default meta;
