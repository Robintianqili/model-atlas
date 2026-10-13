// transe view 'step': one training step on one fact and its corrupted copy, read at the 'update' frame of a fine train
// source (space.ts worked()). Both distances on one axis with the margin γ marked (the loss is how far the corrupted
// copy falls short of d(fact) + γ); props.mode 'update' adds the same axis after the step and a table with one row per
// number: h, r, t, the swapped-in entity x, the gradient of h and h after the step.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ENT, REL } from '../../kg.ts';
import { worked } from '../../space.ts';
import { transe } from '../../trainer.ts';

export const GAMMA = transe.defaults.gamma;
export function describeStep(p: Omit<StageProps, 'ctx'>): I18n {
	const w = worked(p.at, GAMMA);
	if (!w) return { en: 'One training step: nothing to show yet.', zh: '一步训练：还没有可显示的内容。' };
	const n = (i: number) => ENT[i].name;
	return {
		en: `Fact (${n(w.h).en}, ${REL[w.r].en}, ${n(w.t).en}) at distance ${fmt3(w.dPos)}; corrupted copy with ${n(w.x).en} as its ${w.tail ? 'tail' : 'head'} at ${fmt3(w.dNeg)}; margin ${w.gamma}; loss ${fmt3(w.loss)}. After the step: ${fmt3(w.dPos1)} and ${fmt3(w.dNeg1)}, loss ${fmt3(w.loss1)}. Number k = ${w.k} of h (counting from 0): ${fmt3(w.old)} − ${w.lr} × (${fmt3(w.g)}) = ${fmt3(w.new)}.`,
		zh: `事实（${n(w.h).zh}，${REL[w.r].zh}，${n(w.t).zh}）的距离是 ${fmt3(w.dPos)}；把${w.tail ? '尾' : '头'}实体换成${n(w.x).zh}的损坏副本距离是 ${fmt3(w.dNeg)}；间隔 ${w.gamma}；损失 ${fmt3(w.loss)}。这一步之后：${fmt3(w.dPos1)} 和 ${fmt3(w.dNeg1)}，损失 ${fmt3(w.loss1)}。h 中编号 k = ${w.k} 的数（从 0 数起）：${fmt3(w.old)} − ${w.lr} × (${fmt3(w.g)}) = ${fmt3(w.new)}。`
	};
}

const meta = {
	key: 'transe/step',
	title: { en: 'One TransE step, number by number', zh: '一步 TransE，逐个数展开' },
	kinds: ['train'],
	describe: describeStep,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { START }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/train/trainer.ts'), import('../../kg.ts')]);
		const tape = record(trainingProcess(transe), START, { steps: 1 }, { seed: 3, detail: 'fine' });
		const rt = fakeRuntime({ slug: 'transe', input: START, tapes: { one: tape } });
		return { rt, source: 'one', at: atFor(rt, 'one', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'update' } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
