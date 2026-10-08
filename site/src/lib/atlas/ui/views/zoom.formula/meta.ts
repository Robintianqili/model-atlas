// zoom.formula (W1-D2): one formula with live numbers: an author's props.tex, a PhaseDef.formula on a process or
// training tape (kinds B and C), or the op formula of a unit (kind A).
import type { ViewMeta } from '../../../types/view.ts';
import { fmt3, type Props } from '../kit.parts/stage.ts';
import { demoStage, processStage, type DemoVariant } from '../kit.parts/demo.ts';
import { formulaOf } from './formula.ts';

export function describeFormula(p: Props) {
	const f = formulaOf(p);
	if (!f) return { en: 'A formula: nothing to show yet.', zh: '公式：还没有可显示的内容。' };
	const vals = Object.entries(f.values)
		.filter(([k]) => f.tex.includes(`{{${k}}}`))
		.map(([k, v]) => `${k} = ${typeof v === 'number' ? fmt3(v) : v}`)
		.join(', ');
	const cap = f.caption ?? { en: 'The formula', zh: '公式' };
	return { en: `${cap.en}. With the live numbers: ${vals || 'none'}.`, zh: `${cap.zh ?? cap.en}。代入的实时数值：${vals || '无'}。` };
}

export async function variants(): Promise<DemoVariant[]> {
	return [
		{ label: { en: 'a process step (update)', zh: '过程中的一步（更新）' }, stage: processStage(11) },
		{ label: { en: 'a process step (choose)', zh: '过程中的一步（选择）' }, stage: processStage(9) },
		{ label: { en: 'an op formula (kind A)', zh: '算子公式（A 类）' }, stage: demoStage('cnn', 'conv1', [1, 3, 2]) }
	];
}

const meta = {
	key: 'zoom.formula',
	title: { en: 'A formula with live numbers', zh: '代入实时数值的公式' },
	kinds: ['trace', 'train', 'process'],
	accepts: () => false,
	describe: describeFormula,
	demo: async () => processStage(11),
	minHeight: 200
} satisfies ViewMeta;

export default meta;
