// zoom.lookup (W1-D2): one output number of a table lookup (embedding, position_embedding, local cls_token).
import type { ViewMeta } from '../../../types/view.ts';
import { axesOfId, fmt3, layerLabel, resolveUnit, tokenIds, tokenTexts, unitWhere, type Props } from '../kit.parts/stage.ts';
import { demoStage, type DemoVariant } from '../kit.parts/demo.ts';

export function describeLookup(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'A table lookup: nothing to show yet.', zh: '查表的一步：还没有可显示的内容。' };
	const out = r.trace?.t[e.layer];
	const axes = out ? axesOfId(r.model, e.layer, out.shape) : [];
	const T = out && axes[0] === 't' ? out.shape[0] : 0;
	const where = unitWhere(e.unit, axes, T ? tokenTexts(p, T, tokenIds(r.trace, r.model)) : undefined);
	const name = layerLabel(r.model, e.layer);
	const v = e.values;
	if (v.id !== undefined) {
		return {
			en: `${name.en}, ${where.en}: this position holds id ${v.id}, so it copies row ${v.id} of the learned table; entry ${v.d} of that row is ${fmt3(e.result)}.`,
			zh: `${name.zh ?? name.en}，${where.zh}：这个位置的编号是 ${v.id}，所以复制学来的表的第 ${v.id} 行；这一行的第 ${v.d} 个数是 ${fmt3(e.result)}。`
		};
	}
	return {
		en: `${name.en}, ${where.en}: position ${v.t ?? 0} copies its own row of the learned table, row ${v.t ?? 0}; entry ${v.d} of that row is ${fmt3(e.result)}.`,
		zh: `${name.zh ?? name.en}，${where.zh}：位置 ${v.t ?? 0} 复制学来的表中属于自己的那一行，即第 ${v.t ?? 0} 行；这一行的第 ${v.d} 个数是 ${fmt3(e.result)}。`
	};
}

export async function variants(): Promise<DemoVariant[]> {
	return [
		{ label: { en: 'token embedding', zh: '词元嵌入' }, stage: demoStage('gpt', 'tok_emb', [2, 3]) },
		{ label: { en: 'position embedding', zh: '位置嵌入' }, stage: demoStage('gpt', 'pos_emb', [4, 1]) }
	];
}

const meta = {
	key: 'zoom.lookup',
	title: { en: 'Look up a row of a table', zh: '在表中查出一行' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'lookup',
	describe: describeLookup,
	demo: async () => demoStage('gpt', 'tok_emb', [2, 3]),
	minHeight: 420
} satisfies ViewMeta;

export default meta;
