// zoom.reduce (W1-D2): one output number from a window by max (maxpool2d) or a weighted mean (local reduces).
import type { ViewMeta } from '../../../types/view.ts';
import { axesOfId, fmt3, layerLabel, resolveUnit, unitWhere, type Props } from '../kit.parts/stage.ts';
import { demoStage } from '../kit.parts/demo.ts';

export function describeReduce(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'A pooling step: nothing to show yet.', zh: '池化的一步：还没有可显示的内容。' };
	const out = r.trace?.t[e.layer];
	const where = unitWhere(e.unit, out ? axesOfId(r.model, e.layer, out.shape) : []);
	const name = layerLabel(r.model, e.layer);
	const n = e.terms?.length ?? 0;
	const isMax = !!e.terms && e.terms.filter((x) => x.w === 1).length === 1;
	if (isMax) {
		return {
			en: `${name.en}, ${where.en}: the largest of the ${n} numbers in its window is ${fmt3(e.result)}; the others are dropped.`,
			zh: `${name.zh ?? name.en}，${where.zh}：窗口里 ${n} 个数中最大的是 ${fmt3(e.result)}，其余的被丢掉。`
		};
	}
	return {
		en: `${name.en}, ${where.en}: ${n} numbers of its window, weighted and added, give ${fmt3(e.result)}.`,
		zh: `${name.zh ?? name.en}，${where.zh}：窗口里的 ${n} 个数加权相加，得到 ${fmt3(e.result)}。`
	};
}

const meta = {
	key: 'zoom.reduce',
	title: { en: 'A window shrinks to one number', zh: '一个窗口缩成一个数' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'reduce',
	describe: describeReduce,
	demo: async () => demoStage('cnn', 'pool'),
	minHeight: 460
} satisfies ViewMeta;

export default meta;
