// zoom.weighted-sum (W1-D2): one output number as Σ x·w (+ b). Serves conv2d, linear, attention_scores,
// attention_mix (and local ops such as similarity or conv1d that explain themselves with terms).
import type { ViewMeta } from '../../../types/view.ts';
import { axesOfId, axisNames, fmt3, layerLabel, outLabels, type Props, resolveUnit, tokenIds, tokenTexts, unitWhere } from '../kit.parts/stage.ts';
import { demoStage, type DemoVariant } from '../kit.parts/demo.ts';

export function describeWeightedSum(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'A weighted sum: nothing to show yet.', zh: '加权和：还没有可显示的内容。' };
	const out = r.trace?.t[e.layer];
	const axes = out ? axesOfId(r.model, e.layer, out.shape) : [];
	const T = out && (axes[0] === 't' || axes[1] === 'tq') ? (axes[0] === 't' ? out.shape[0] : out.shape[1]) : 0;
	const where = unitWhere(e.unit, axes, T ? tokenTexts(p, T, tokenIds(r.trace, r.model)) : undefined, outLabels(p, r.model), axisNames(p));
	const name = layerLabel(r.model, e.layer);
	const n = e.terms?.length ?? 0;
	const s = e.values.s ?? (e.terms ?? []).reduce((a, x) => a + x.p, 0);
	const op = r.op?.def.op ?? r.layer?.op;
	if (op === 'attention_scores') {
		return {
			en: `${name.en}, ${where.en}: the ${n} products of query and key add up to ${fmt3(s)}; times the scale ${fmt3(e.values.scale)} this gives ${fmt3(e.result)}.`,
			zh: `${name.zh ?? name.en}，${where.zh}：查询和键的 ${n} 个乘积加起来是 ${fmt3(s)}；乘以缩放系数 ${fmt3(e.values.scale)} 得到 ${fmt3(e.result)}。`
		};
	}
	if (op === 'attention_mix') {
		return {
			en: `${name.en}, ${where.en}: ${n} values, each times its attention weight, add up to ${fmt3(e.result)}.`,
			zh: `${name.zh ?? name.en}，${where.zh}：${n} 个值各自乘以注意力权重，加起来是 ${fmt3(e.result)}。`
		};
	}
	const groups = e.groups && e.groups.length > 1 ? e.groups.length : 0;
	const g = groups ? { en: ` from ${groups} input channels`, zh: `（来自 ${groups} 个输入通道）` } : { en: '', zh: '' };
	const b = e.bias !== undefined ? { en: `; adding the bias ${fmt3(e.bias)} gives ${fmt3(e.result)}`, zh: `；再加上偏置 ${fmt3(e.bias)}，得到 ${fmt3(e.result)}` } : { en: ` = ${fmt3(e.result)}`, zh: ` = ${fmt3(e.result)}` };
	return {
		en: `${name.en}, ${where.en}: ${n} products of input and weight${g.en} add up to ${fmt3(s)}${b.en}.`,
		zh: `${name.zh ?? name.en}，${where.zh}：输入和权重的 ${n} 个乘积${g.zh}加起来是 ${fmt3(s)}${b.zh}。`
	};
}

/** More demos for the gallery (not part of ViewMeta): one per kind of weighted sum. */
export async function variants(): Promise<DemoVariant[]> {
	return [
		{ label: { en: 'conv2d on the drawing (1 channel)', zh: '画上的卷积（1 个通道）' }, stage: demoStage('cnn', 'conv1') },
		{ label: { en: 'dense on a flattened map', zh: '展平后的全连接层' }, stage: demoStage('cnn', 'dense', [1]) },
		{ label: { en: 'attention score q·k', zh: '注意力分数 q·k' }, stage: demoStage('gpt', 'scores', [1, 5, 2]) },
		{ label: { en: 'attention mix of values', zh: '值的加权混合' }, stage: demoStage('gpt', 'heads', [5, 6]) },
		{ label: { en: 'linear on one token', zh: '一个词元上的线性层' }, stage: demoStage('gpt', 'fc') }
	];
}

const meta = {
	key: 'zoom.weighted-sum',
	title: { en: 'Weighted sum, worked out', zh: '加权和的计算过程' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'weighted-sum',
	describe: describeWeightedSum,
	demo: async () => demoStage('cnn', 'conv2'),
	minHeight: 520
} satisfies ViewMeta;

export default meta;
