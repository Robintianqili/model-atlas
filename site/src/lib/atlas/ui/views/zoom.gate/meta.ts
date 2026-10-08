// zoom.gate (W1-D2): one step of a gated recurrent unit (lstm; local gru): gate bars and the cell-state line.
import type { ViewMeta } from '../../../types/view.ts';
import { fmt3, layerLabel, resolveUnit, type Props } from '../kit.parts/stage.ts';
import { demoStage, type DemoVariant } from '../kit.parts/demo.ts';

export function describeGate(p: Props) {
	const r = resolveUnit(p);
	const e = r.e;
	if (!e) return { en: 'A gated step: nothing to show yet.', zh: '带门的一步：还没有可显示的内容。' };
	const v = e.values;
	const name = layerLabel(r.model, e.layer);
	const at = { en: `time step ${v.t ?? e.unit[0]}, unit ${v.j ?? e.unit[1]} (counting from 0)`, zh: `时间步 ${v.t ?? e.unit[0]}、单元 ${v.j ?? e.unit[1]}（从 0 数起）` };
	return {
		en: `${name.en}, ${at.en}: forget gate ${fmt3(v.f)}, input gate ${fmt3(v.i)}, candidate ${fmt3(v.g)}, output gate ${fmt3(v.o)}. The cell becomes ${fmt3(v.f)} × ${fmt3(v.cPrev)} + ${fmt3(v.i)} × ${fmt3(v.g)} = ${fmt3(v.c)}, and the output is ${fmt3(v.o)} × tanh(${fmt3(v.c)}) = ${fmt3(v.h)}.`,
		zh: `${name.zh ?? name.en}，${at.zh}：遗忘门 ${fmt3(v.f)}，输入门 ${fmt3(v.i)}，候选值 ${fmt3(v.g)}，输出门 ${fmt3(v.o)}。细胞值变成 ${fmt3(v.f)} × ${fmt3(v.cPrev)} + ${fmt3(v.i)} × ${fmt3(v.g)} = ${fmt3(v.c)}，输出是 ${fmt3(v.o)} × tanh(${fmt3(v.c)}) = ${fmt3(v.h)}。`
	};
}

export async function variants(): Promise<DemoVariant[]> {
	return [
		{ label: { en: 'the hidden state h', zh: '隐藏状态 h' }, stage: demoStage('lstm', 'lstm', [3, 1]) },
		{ label: { en: 'the cell state c (sub-step)', zh: '细胞状态 c（子步骤）' }, stage: demoStage('lstm', 'lstm#c', [5, 0]) }
	];
}

const meta = {
	key: 'zoom.gate',
	title: { en: 'Gates decide what to keep', zh: '门决定保留什么' },
	kinds: ['trace'],
	accepts: (o) => o.zoom === 'gate',
	describe: describeGate,
	demo: async () => demoStage('lstm', 'lstm', [3, 1]),
	minHeight: 520
} satisfies ViewMeta;

export default meta;
