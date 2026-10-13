// ppo-reacher view 'eps': the task step's stage. Buttons set PPO's clip ε in the learner's input (setInput: the train
// sources start again with it), and two kit loss-curve charts compare, on the 'pair' tape, the run with the learner's ε
// and the same training with ε = 0.2: the average episode return and how far each iteration has moved the policy from
// the one that played its batch (KL, log scale), one point per iteration.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { DEFAULT_EPS, epsOf } from '../../sim.ts';
import { pair } from '../../process.ts';

export const epsText = (e: number): I18n => (e >= 100 ? { en: 'off', zh: '关闭' } : { en: String(e), zh: String(e) });
export const curveProps = (rt: StageProps['rt'], metric: string, log: boolean) => {
	const mine = epsText(epsOf(rt.input));
	const base = { en: `ε = ${DEFAULT_EPS}`, zh: `ε = ${DEFAULT_EPS}` };
	return { metrics: [metric, `${metric}0`], log, smooth: 0, labels: { [metric]: { en: `your ε = ${mine.en}`, zh: `你的 ε = ${mine.zh}` }, [`${metric}0`]: base } };
};

const meta = {
	key: 'ppo-reacher/eps',
	title: { en: 'Pick ε and compare the training', zh: '选一个 ε，比较训练过程' },
	kinds: ['train', 'process'],
	describe: (p: Omit<StageProps, 'ctx'>): I18n => {
		const e = epsText(epsOf(p.rt.input));
		return { en: `Buttons for the clip ε (yours is ${e.en}); the average episode return and the KL of every iteration, for your ε and for ε = ${DEFAULT_EPS}.`, zh: `截断 ε 的按钮（你的是 ${e.zh}）；你的 ε 和 ε = ${DEFAULT_EPS} 两次训练里，每次迭代的平均回合回报和 KL。` };
	},
	async demo() {
		const input = { target: [-0.3, 0.6], eps: 0.5 };
		const tape = record(pair, input, { steps: 64 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'ppo-reacher', input, tapes: { pair: tape } });
		return { rt, source: 'pair', at: atFor(rt, 'pair', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 420
} satisfies ViewMeta;

export default meta;
