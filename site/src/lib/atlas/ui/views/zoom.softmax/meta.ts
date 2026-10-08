// zoom.softmax (W1-D2): one share of a softmax row (classes, next tokens, or the keys of an attention row).
import type { ViewMeta } from '../../../types/view.ts';
import { axesOfId, fmt3, layerLabel, outLabels, resolveUnit, tokenIds, tokenTexts, unitWhere, type Props } from '../kit.parts/stage.ts';
import { demoStage, type DemoVariant } from '../kit.parts/demo.ts';
import { fmtPct } from '../../../core/format.ts';

export function describeSoftmax(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'A softmax step: nothing to show yet.', zh: 'softmax 的一步：还没有可显示的内容。' };
	const out = r.trace?.t[e.layer];
	const axes = out ? axesOfId(r.model, e.layer, out.shape) : [];
	const T = out ? (axes[0] === 't' ? out.shape[0] : axes[1] === 'tq' ? out.shape[1] : 0) : 0;
	const where = unitWhere(e.unit, axes, T ? tokenTexts(p, T, tokenIds(r.trace, r.model)) : undefined, outLabels(p, r.model));
	const name = layerLabel(r.model, e.layer);
	const v = e.values;
	return {
		en: `${name.en}, ${where.en}: e^(${fmt3(v.x)} − ${fmt3(v.max)}) = ${fmt3(v.e)}; divided by the row sum ${fmt3(v.sum)} this is a share of ${fmtPct(e.result)} among ${v.n} entries.`,
		zh: `${name.zh ?? name.en}，${where.zh}：e^(${fmt3(v.x)} − ${fmt3(v.max)}) = ${fmt3(v.e)}；除以这一行的总和 ${fmt3(v.sum)}，在 ${v.n} 项中占 ${fmtPct(e.result)}。`
	};
}

export async function variants(): Promise<DemoVariant[]> {
	return [
		{ label: { en: 'class probabilities', zh: '类别概率' }, stage: demoStage('cnn', 'probs', [1]) },
		{ label: { en: 'attention weights of one query', zh: '一个查询的注意力权重' }, stage: demoStage('gpt', 'weights', [0, 4, 1]) },
		{ label: { en: 'next-token probabilities', zh: '下一个词元的概率' }, stage: demoStage('gpt', 'probs', [5, 6]) }
	];
}

const meta = {
	key: 'zoom.softmax',
	title: { en: 'Scores become shares', zh: '分数变成份额' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'softmax',
	describe: describeSoftmax,
	demo: async () => demoStage('cnn', 'probs', [1]),
	minHeight: 460
} satisfies ViewMeta;

export default meta;
