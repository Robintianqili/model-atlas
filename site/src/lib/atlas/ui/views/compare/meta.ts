// compare (W1-D2): two sources side by side (StepDef.compare): two traces' outputs, or one metric of two tapes.
import type { ViewMeta } from '../../../types/view.ts';
import { fmt3, type Props } from '../kit.parts/stage.ts';
import { compareStage, type DemoVariant } from '../kit.parts/demo.ts';
import { pickComparison } from './pick.ts';
import { fmtPct } from '../../../core/format.ts';
import { outputLabel } from '../../../lesson/facts.ts';

const name = (l: string | { en: string; zh?: string }, zh: boolean) => (typeof l === 'string' ? l : zh ? (l.zh ?? l.en) : l.en);

export function describeCompare(p: Props) {
	const c = pickComparison(p);
	if (!c) return { en: 'A comparison: nothing to show yet.', zh: '对比：还没有可显示的内容。' };
	const A = c.a.label;
	const B = c.b.label;
	if (c.topA !== undefined && c.topB !== undefined && c.a.probs && c.b.probs) {
		const la = outputLabel(c.a.spec ?? null, c.topA);
		const lb = outputLabel(c.b.spec ?? null, c.topB);
		return {
			en: `${A.en}: top answer “${name(la, false)}” with ${fmtPct(c.a.probs[c.topA])}. ${B.en}: top answer “${name(lb, false)}” with ${fmtPct(c.b.probs[c.topB])}.`,
			zh: `${A.zh ?? A.en}：第一名是“${name(la, true)}”，${fmtPct(c.a.probs[c.topA])}。${B.zh ?? B.en}：第一名是“${name(lb, true)}”，${fmtPct(c.b.probs[c.topB])}。`
		};
	}
	if (c.metric && c.a.series && c.b.series) {
		const last = (s: Float64Array) => {
			for (let i = s.length - 1; i >= 0; i--) if (Number.isFinite(s[i])) return s[i];
			return NaN;
		};
		return {
			en: `${c.metric} over the frames for ${A.en} and ${B.en}; the last values are ${fmt3(last(c.a.series))} and ${fmt3(last(c.b.series))}.`,
			zh: `${A.zh ?? A.en} 和 ${B.zh ?? B.en} 在各帧上的 ${c.metric}；最后的值分别是 ${fmt3(last(c.a.series))} 和 ${fmt3(last(c.b.series))}。`
		};
	}
	return { en: `${A.en} and ${B.en}, side by side.`, zh: `${A.zh ?? A.en} 和 ${B.zh ?? B.en} 并排对比。` };
}

export async function variants(): Promise<DemoVariant[]> {
	return [{ label: { en: 'two runs of a process', zh: '一个过程的两次运行' }, stage: compareStage('tapes') }];
}

const meta = {
	key: 'compare',
	title: { en: 'Two runs side by side', zh: '两次运行并排对比' },
	kinds: ['trace', 'train', 'process'],
	accepts: () => false,
	describe: describeCompare,
	demo: async () => compareStage('traces'),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
