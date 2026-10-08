// gridworld view 'arm-bars': the four slot machines of a 'bandit' tape (each machine's average, pulls and coins, the
// pick of this frame, UCB's bonus, ε-greedy's u against ε) or the regret curves of the 'race' tape.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ARM_NAMES } from '../../world.ts';
import { bandit } from '../../process.ts';

export const POLICY_NAMES: I18n[] = [{ en: 'greedy', zh: '贪心' }, { en: 'ε-greedy', zh: 'ε-贪心' }, { en: 'UCB', zh: 'UCB' }, { en: 'Thompson sampling', zh: '汤普森采样' }];

export function describeArms(p: Omit<StageProps, 'ctx'>): I18n {
	const tp = p.at.tape;
	if (!tp || p.at.index < 0) return { en: 'The slot machines: nothing to show yet.', zh: '老虎机：还没有可显示的内容。' };
	const m = p.at.frame?.metrics ?? {};
	if (p.props.mode === 'regret') {
		const r = [0, 1, 2, 3].map((k) => fmt3(m[`r${k}`]));
		return { en: `Regret after ${m.n} pulls: ${POLICY_NAMES.map((n, k) => `${n.en} ${r[k]}`).join(', ')}.`, zh: `拉了 ${m.n} 次后的遗憾：${POLICY_NAMES.map((n, k) => `${n.zh} ${r[k]}`).join('，')}。` };
	}
	const st = tp.state(p.at.index) as { Q?: Tensor; N?: Tensor; R?: Tensor };
	const each = ARM_NAMES.map((a, k) => `${a} ${st.R?.data[k] ?? 0}/${st.N?.data[k] ?? 0}`).join(', ');
	return { en: `Coins won / pulls per machine: ${each}.${m.arm !== undefined ? ` This frame: machine ${ARM_NAMES[m.arm]}.` : ''}`, zh: `每台机器赢的金币 / 拉的次数：${each}。${m.arm !== undefined ? `这一帧：机器 ${ARM_NAMES[m.arm]}。` : ''}` };
}

const meta = {
	key: 'gridworld/arm-bars',
	title: { en: 'Four slot machines', zh: '四台老虎机' },
	kinds: ['process'],
	describe: describeArms,
	async demo() {
		const tape = record(bandit, null, { policy: 'ucb' }, { seed: 22 });
		const rt = fakeRuntime({ slug: 'gridworld', tapes: { ucb: tape } });
		return { rt, source: 'ucb', at: atFor(rt, 'ucb', 40), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'agent', formula: true } };
	},
	minHeight: 240
} satisfies ViewMeta;

export default meta;
