// tinyseq2seq view 'attend': one decoder step (props.t, default 0) on the learner's date. props.mode:
//   'start'    the state the step starts from (the encoder's last state at step 0) and the symbol it reads
//   'scores'   one score per input character; the largest worked out: e = Σ v · tanh(query + key)
//   'weights'  softmax turns the scores into weights (bars); the largest worked out
//   'context'  the context vector = the encoder states mixed with those weights; its largest number worked out
//   'next'     the new state and the chances of the 13 symbols; the one written
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { chr, mixParts, scoreParts, softParts, stepAt, sym } from '../../look.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;

export function attendOf(p: P) {
	const mode = String(p.props.mode ?? 'scores');
	const t = typeof p.props.t === 'number' ? p.props.t : 0;
	const S = stepAt(p.rt, t);
	const so = Array.from({ length: t }, (_, i) => sym(S.ids[i] ?? -1)).join('');
	const j2 = S.a ? Array.from(S.a, (v, j) => (j === S.j ? -1 : v)).reduce((b, v, j, a) => (v > a[b] ? j : b), 0) : -1;
	return { mode, S, so, chars: S.chars.map(chr), score: scoreParts(p.rt, t, S.j), soft: [S.j, j2].filter((j, i) => j >= 0 && j < S.chars.length && (i === 0 || j !== S.j)).map((j) => ({ j, ...softParts(S.e, j)! })), mix: mixParts(p.rt, t) };
}

const meta = {
	key: 'tinyseq2seq/attend',
	title: { en: 'One decoder step', zh: '解码器的一步' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const A = attendOf(p);
		const S = A.S;
		if (!S.a || S.j < 0) return { en: 'One decoder step: nothing to show yet.', zh: '解码器的一步：还没有可显示的内容。' };
		const c = A.chars[S.j];
		return {
			en: `Decoder step ${S.t} (written so far: “${A.so}”). Scores ${Array.from(S.e ?? [], (v, j) => `${A.chars[j]} ${fmt3(v)}`).join(', ')}. The largest weight, ${fmtPct(S.a[S.j])}, goes to “${c}” at position ${S.j}. It writes “${sym(S.y)}” with chance ${fmtPct(S.p?.[S.y] ?? NaN)}.`,
			zh: `解码器第 ${S.t} 步（已经写下：“${A.so}”）。分数：${Array.from(S.e ?? [], (v, j) => `${A.chars[j]} ${fmt3(v)}`).join('，')}。最大的权重 ${fmtPct(S.a[S.j])} 给了位置 ${S.j} 上的“${c}”。它写下“${sym(S.y)}”，概率 ${fmtPct(S.p?.[S.y] ?? NaN)}。`
		};
	},
	demo: async () => demoProps({ mode: 'weights' }),
	minHeight: 380
} satisfies ViewMeta;

export default meta;
