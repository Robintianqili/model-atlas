// zoom.index-map (W1-D2): an output number that is an input number moved (flatten, concat, local to_tokens /
// select_token) or replaced (causal_mask).
import type { ViewMeta } from '../../../types/view.ts';
import { axesOfId, fmt3, layerLabel, resolveUnit, tokenIds, tokenTexts, unitWhere, type Props } from '../kit.parts/stage.ts';
import { demoStage, type DemoVariant } from '../kit.parts/demo.ts';

export function describeIndexMap(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'An index map: nothing to show yet.', zh: '索引映射：还没有可显示的内容。' };
	const out = r.trace?.t[e.layer];
	const axes = out ? axesOfId(r.model, e.layer, out.shape) : [];
	const T = out ? (axes[0] === 't' ? out.shape[0] : axes[1] === 'tq' ? out.shape[1] : 0) : 0;
	const where = unitWhere(e.unit, axes, T ? tokenTexts(p, T, tokenIds(r.trace, r.model)) : undefined);
	const name = layerLabel(r.model, e.layer);
	const v = e.values;
	if (v.masked !== undefined) {
		return v.masked
			? { en: `${name.en}, ${where.en}: the key comes after the query, so the score ${fmt3(v.x)} becomes −∞.`, zh: `${name.zh ?? name.en}，${where.zh}：键在查询之后，所以分数 ${fmt3(v.x)} 变成 −∞。` }
			: { en: `${name.en}, ${where.en}: the key is not after the query, so the score ${fmt3(v.x)} passes unchanged.`, zh: `${name.zh ?? name.en}，${where.zh}：键不在查询之后，所以分数 ${fmt3(v.x)} 原样通过。` };
	}
	if (v.c !== undefined) {
		return {
			en: `${name.en}, ${where.en}: the number at channel ${v.c}, row ${v.row}, column ${v.col} of the input, ${fmt3(e.result)}, is moved to index ${v.index} of the list (counting from 0).`,
			zh: `${name.zh ?? name.en}，${where.zh}：输入中通道 ${v.c}、第 ${v.row} 行、第 ${v.col} 列的数 ${fmt3(e.result)} 被放到列表的第 ${v.index} 个位置（从 0 数起）。`
		};
	}
	return { en: `${name.en}, ${where.en}: the number ${fmt3(e.result)} is copied unchanged from its input.`, zh: `${name.zh ?? name.en}，${where.zh}：数值 ${fmt3(e.result)} 从输入原样复制。` };
}

export async function variants(): Promise<DemoVariant[]> {
	return [

		{ label: { en: 'causal mask (masked)', zh: '因果掩码（被遮住）' }, stage: demoStage('gpt', 'masked', [0, 2, 4]) },
		{ label: { en: 'causal mask (passes)', zh: '因果掩码（通过）' }, stage: demoStage('gpt', 'masked', [1, 4, 2]) }
	];
}

const meta = {
	key: 'zoom.index-map',
	title: { en: 'Numbers move, or get masked', zh: '数值换位置，或被遮住' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'index-map',
	describe: describeIndexMap,
	demo: async () => demoStage('cnn', 'flatten'),
	minHeight: 380
} satisfies ViewMeta;

export default meta;
