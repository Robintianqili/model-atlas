// tinyseq2seq view 'align': the alignment matrix, one row per output symbol (a decoder step) and one column per input
// character, darker = more attention. On the 'decode' process it shows the rows written up to the step's frame (Play
// fills it row by row); on the trace, every row up to the end symbol. Above it: what the model wrote and the right
// answer. props.tries: buttons that type a date in a format the model never saw (the limits step).
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { argmax } from '../../seq.ts';
import { chr, look, sym, tr } from '../../look.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
/** dates in the formats of off.json (pipeline/tinyseq2seq/make_data.ts; never in training): facts offYearFirst,
 *  offDashes, offNoComma, offSpelled, offImpossible */
export const TRIES = ['2021/03/05', '05-03-2021', 'March 5 2021', 'March fifth, 2021', 'March 32, 2021'];

export function alignOf(p: P) {
	const L = look(p.rt);
	const st = p.at.state as { align?: Tensor; out?: Tensor } | undefined;
	const f = p.at.frame;
	const fromTape = !!st?.align;
	const W = fromTape ? st!.align! : tr(p.rt, 'dec#weights');
	const T = W?.shape[1] ?? 0;
	const R = fromTape ? (f?.phase === 'start' ? 0 : (f?.step ?? -1) + 1) : L.end >= 0 ? L.end + 1 : (W?.shape[0] ?? 0);
	const ids = fromTape ? Array.from(st!.out?.data ?? []) : L.ids;
	const rows = Array.from({ length: R }, (_, i) => (ids[i] >= 0 ? sym(ids[i]) : '?'));
	const weights = W ? W.data.slice(0, R * T) : new Float32Array(0);
	const tops = rows.map((_, i) => argmax(weights.subarray(i * T, (i + 1) * T)));
	const wrote = ids.slice(0, R).filter((i) => i > 1).map(sym).join('');
	const finished = !fromTape || f?.step === (L.end >= 0 ? L.end : L.ids.length - 1) && f?.phase === 'write';
	return { ...L, keys: L.chars.map(chr), rows, weights, tops, R, wrote, finished };
}

const meta = {
	key: 'tinyseq2seq/align',
	title: { en: 'The alignment matrix', zh: '对齐矩阵' },
	kinds: ['trace', 'process'],
	describe: (p: P): I18n => {
		const A = alignOf(p);
		if (!A.T) return { en: 'The alignment matrix: nothing to show yet.', zh: '对齐矩阵：还没有可显示的内容。' };
		const T = A.keys.length;
		const list = A.rows.map((r, i) => `${r} ← “${A.keys[A.tops[i]]}” ${fmtPct(A.weights[i * T + A.tops[i]])}`);
		return {
			en: `${A.R} output symbols (rows) by ${T} input characters (columns). The model wrote “${A.wrote}”${A.want ? `; the right answer is ${A.want}` : ''}. Where each row looks most: ${list.join('; ')}.`,
			zh: `${A.R} 个输出符号（行）× ${T} 个输入字符（列）。模型写下“${A.wrote}”${A.want ? `；正确答案是 ${A.want}` : ''}。每一行看得最多的地方：${list.join('；')}。`
		};
	},
	demo: async () => demoProps({}),
	minHeight: 360
} satisfies ViewMeta;

export default meta;
