// zoom.normalize (W1-D2): one output number of a normalization (layernorm; local l2norm).
import type { ViewMeta } from '../../../types/view.ts';
import { axesOfId, fmt3, layerLabel, resolveUnit, tokenIds, tokenTexts, unitWhere, type Props } from '../kit.parts/stage.ts';
import { demoStage } from '../kit.parts/demo.ts';

export function describeNormalize(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'A normalization step: nothing to show yet.', zh: '归一化的一步：还没有可显示的内容。' };
	const out = r.trace?.t[e.layer];
	const axes = out ? axesOfId(r.model, e.layer, out.shape) : [];
	const T = out && axes[0] === 't' ? out.shape[0] : 0;
	const where = unitWhere(e.unit, axes, T ? tokenTexts(p, T, tokenIds(r.trace, r.model)) : undefined);
	const name = layerLabel(r.model, e.layer);
	const v = e.values;
	if (v.mean !== undefined && v.std !== undefined) {
		return {
			en: `${name.en}, ${where.en}: the row has mean ${fmt3(v.mean)} and spread ${fmt3(v.std)}. The input ${fmt3(v.x)} becomes (${fmt3(v.x)} − ${fmt3(v.mean)}) / ${fmt3(v.std)} = ${fmt3(v.xhat)}; times ${fmt3(v.gamma)} plus ${fmt3(v.beta)} gives ${fmt3(e.result)}.`,
			zh: `${name.zh ?? name.en}，${where.zh}：这一行的平均值是 ${fmt3(v.mean)}，离散程度是 ${fmt3(v.std)}。输入 ${fmt3(v.x)} 变成 (${fmt3(v.x)} − ${fmt3(v.mean)}) / ${fmt3(v.std)} = ${fmt3(v.xhat)}；乘以 ${fmt3(v.gamma)} 再加 ${fmt3(v.beta)}，得到 ${fmt3(e.result)}。`
		};
	}
	if (v.len !== undefined && v.x !== undefined)
		return {
			en: `${name.en}, ${where.en}: the row has length ${fmt3(v.len)}, so ${fmt3(v.x)} becomes ${fmt3(v.x)} / ${fmt3(v.len)} = ${fmt3(e.result)}; the whole row then has length 1.`,
			zh: `${name.zh ?? name.en}，${where.zh}：这一行的长度是 ${fmt3(v.len)}，所以 ${fmt3(v.x)} 变成 ${fmt3(v.x)} / ${fmt3(v.len)} = ${fmt3(e.result)}；整行的长度随之变成 1。`
		};
	return { en: `${name.en}, ${where.en}: normalized to ${fmt3(e.result)}.`, zh: `${name.zh ?? name.en}，${where.zh}：归一化后是 ${fmt3(e.result)}。` };
}

const meta = {
	key: 'zoom.normalize',
	title: { en: 'Normalize a row', zh: '把一行归一化' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'normalize',
	describe: describeNormalize,
	demo: async () => demoStage('gpt', 'ln'),
	minHeight: 480
} satisfies ViewMeta;

export default meta;
