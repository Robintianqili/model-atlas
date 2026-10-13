// hmm view 'memory': props.show 'limits' (the spells tape: the made-up stretch of weather as a strip, and the share of
// rainy spells of each length, as they happened and as the counted transition table implies) or 'rnn' (the HMM's memory
// next to the char-RNN's, with the char-RNN's measured numbers from facts.json and a link to its page).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { I, SPELL_MAX, START, YEAR } from '../../hmm.ts';
import { spells } from '../../process.ts';

const arr = (x: unknown): number[] => Array.from(((x as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);
/** the stretch of weather and the two spell-length distributions, from a spells tape */
export function spellsOf(p: Pick<StageProps, 'at'>): { year: number[]; model: number[]; truth: number[] } {
	const tape = p.at.tape, s = (tape && tape.length ? tape.state(Math.max(0, p.at.index)) : {}) as Record<string, unknown>;
	return { year: arr(s.year), model: arr(s.model), truth: arr(s.truth) };
}
/** a char-RNN fact (facts.json), NaN when missing */
export const factOf = (p: Pick<StageProps, 'rt'>, k: string): number => {
	const v = p.rt.fact(`fact:${k}`);
	return typeof v === 'number' ? v : NaN;
};

export function describeMemory(p: Omit<StageProps, 'ctx'>): I18n {
	if (p.props?.show === 'rnn')
		return I(
			`A hidden Markov model carries one of 2 states from day to day. The char-RNN carries ${factOf(p, 'charRnnHidden')} numbers in its hidden state h and as many in its cell state c, learned with ${factOf(p, 'charRnnParams')} parameters; its unit ${factOf(p, 'counterUnit')} (counting from 0) counts the letters of the current word (correlation ${factOf(p, 'counterR')}).`,
			`隐马尔可夫模型每天只带着 2 种状态中的一种往下传。char-RNN 在隐藏状态 h 里带着 ${factOf(p, 'charRnnHidden')} 个数，细胞状态 c 里也有同样多，由 ${factOf(p, 'charRnnParams')} 个参数学出来；它的单元 ${factOf(p, 'counterUnit')}（从 0 数起）在数当前单词已有几个字母（相关系数 ${factOf(p, 'counterR')}）。`
		);
	const s = spellsOf(p), m = p.at.frame?.metrics ?? {};
	if (!s.model.length) return I('Rainy spells: nothing yet.', '连雨天：还没有内容。');
	return I(
		`${s.year.length} made-up days in which every rainy spell lasts exactly 3 days. Counted, rain stays rain with ${fmt3(m.aRR)}. The chain then gives spells of 1 to ${s.model.length} days the shares ${s.model.map((v) => fmt3(v)).join(', ')}; as they happened: ${s.truth.map((v) => fmt3(v)).join(', ')}.`,
		`编出来的 ${s.year.length} 天里，每段连雨天都正好 3 天。数出来，雨天保持雨天的概率是 ${fmt3(m.aRR)}。这条马尔可夫链给 1 到 ${s.model.length} 天长的连雨天的比例是 ${s.model.map((v) => fmt3(v)).join('、')}；实际是 ${s.truth.map((v) => fmt3(v)).join('、')}。`
	);
}

const meta = {
	key: 'hmm/memory',
	title: I('What one hidden state can remember', '一个隐藏状态能记住什么'),
	kinds: ['process'],
	describe: describeMemory,
	async demo() {
		const tape = record(spells, { year: YEAR, maxLen: SPELL_MAX }, {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'hmm', input: START, tapes: { spells: tape } });
		return { rt, source: 'spells', at: atFor(rt, 'spells', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'limits' } };
	},
	minHeight: 280
} satisfies ViewMeta;

export default meta;
