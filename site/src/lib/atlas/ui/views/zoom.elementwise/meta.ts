// zoom.elementwise (W1-D2): one number in, one number out (relu, tanh, sigmoid, silu, gelu, local activations), or
// a cell-by-cell sum (add).
import type { ViewMeta } from '../../../types/view.ts';
import { axesOfId, axisNames, fmt3, layerLabel, outLabels, type Props, resolveUnit, tokenIds, tokenTexts, unitWhere } from '../kit.parts/stage.ts';
import { demoStage, type DemoVariant } from '../kit.parts/demo.ts';

export function describeElementwise(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'An elementwise step: nothing to show yet.', zh: '逐元素的一步：还没有可显示的内容。' };
	const out = r.trace?.t[e.layer];
	const axes = out ? axesOfId(r.model, e.layer, out.shape) : [];
	const T = out && axes[0] === 't' ? out.shape[0] : 0;
	const where = unitWhere(e.unit, axes, T ? tokenTexts(p, T, tokenIds(r.trace, r.model)) : undefined, outLabels(p, r.model), axisNames(p));
	const name = layerLabel(r.model, e.layer);
	if (e.curve) {
		const fn = e.curve.fn.replace('_tanh', '');
		return {
			en: `${name.en}, ${where.en}: the input ${fmt3(e.curve.x)} goes through ${fn} and comes out as ${fmt3(e.curve.y)}.`,
			zh: `${name.zh ?? name.en}，${where.zh}：输入 ${fmt3(e.curve.x)} 经过 ${fn} 后变成 ${fmt3(e.curve.y)}。`
		};
	}
	const a = e.values.a ?? e.terms?.[0]?.x ?? NaN;
	const b = e.values.b ?? e.terms?.[1]?.x ?? NaN;
	return {
		en: `${name.en}, ${where.en}: ${fmt3(a)} + ${fmt3(b)} = ${fmt3(e.result)}.`,
		zh: `${name.zh ?? name.en}，${where.zh}：${fmt3(a)} + ${fmt3(b)} = ${fmt3(e.result)}。`
	};
}

export async function variants(): Promise<DemoVariant[]> {
	return [
		{ label: { en: 'ReLU on a feature map', zh: '特征图上的 ReLU' }, stage: demoStage('cnn', 'relu1', [1, 2, 3]) },
		{ label: { en: 'residual add', zh: '残差相加' }, stage: demoStage('gpt', 'resid', [4, 2]) },
		{ label: { en: 'token + position', zh: '词元 + 位置' }, stage: demoStage('gpt', 'embed', [1, 0]) }
	];
}

const meta = {
	key: 'zoom.elementwise',
	title: { en: 'One number in, one number out', zh: '一个数进，一个数出' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'elementwise',
	describe: describeElementwise,
	demo: async () => demoStage('gpt', 'gelu', [4, 9]),
	minHeight: 420
} satisfies ViewMeta;

export default meta;
