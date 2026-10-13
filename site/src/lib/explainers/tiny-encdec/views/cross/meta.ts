// tiny-encdec view 'cross': the cross-attention matrix, one row per output symbol (a decoder step) and one column per
// input character, darker = more weight; both heads averaged or one head (props.head, or the buttons). On the 'decode'
// process it shows the rows written up to the step's frame (Play fills it row by row); on the trace, every row up to
// the end symbol, with props.row outlined. Above it: what the model wrote and the right answer. props.tries: buttons
// that type a date in a format the model never saw; props.gru: the measured comparison with tinyseq2seq's GRU.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { argmax } from '../../tf.ts';
import { avgHeads, chr, look, sym, tr } from '../../look.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export type Pick = number | 'avg';
/** dates in the formats of off.json (pipeline/tiny-encdec/make_data.ts; never in training): facts offYearFirst,
 *  offDashes, offNoComma, offSpelled, offImpossible */
export const TRIES = ['2021/03/05', '05-03-2021', 'March 5 2021', 'March fifth, 2021', 'March 32, 2021'];

export function crossOf(p: P, pick: Pick = 'avg') {
	const L = look(p.rt);
	const st = p.at.state as { cross?: Tensor; out?: Tensor } | undefined;
	const f = p.at.frame;
	const fromTape = !!st?.cross;
	const C = fromTape ? st!.cross! : tr(p.rt, 'dec.cattn');
	const [H, Lr, T] = C?.shape ?? [0, 0, 0];
	const R = fromTape ? (f?.phase === 'start' ? 0 : (f?.step ?? -1) + 1) : L.end >= 0 ? L.end + 1 : Lr;
	const ids = fromTape ? Array.from(st!.out?.data ?? []) : L.ids;
	const rows = Array.from({ length: R }, (_, i) => (ids[i] >= 0 ? sym(ids[i]) : '?'));
	const h = pick === 'avg' || pick >= H ? 'avg' : pick;
	const all = !C ? new Float64Array(0) : h === 'avg' ? avgHeads(C) : C.data.subarray(h * Lr * T, (h + 1) * Lr * T);
	const weights = all.slice(0, R * T);
	const tops = rows.map((_, i) => argmax(weights.subarray(i * T, (i + 1) * T)));
	const wrote = ids.slice(0, R).filter((i) => i > 1).map(sym).join('');
	const finished = !fromTape || (f?.step === (L.end >= 0 ? L.end : L.ids.length - 1) && f?.phase === 'write');
	return { ...L, H, h, keys: L.chars.map(chr), rows, weights, tops, R, T, wrote, finished };
}

const meta = {
	key: 'tiny-encdec/cross',
	title: { en: 'Cross-attention: which characters each output read', zh: '交叉注意力：每个输出读了哪些字符' },
	kinds: ['trace', 'process'],
	describe: (p: P): I18n => {
		const A = crossOf(p, typeof p.props.head === 'number' ? p.props.head : 'avg');
		if (!A.T) return { en: 'The cross-attention matrix: nothing to show yet.', zh: '交叉注意力矩阵：还没有可显示的内容。' };
		const list = A.rows.map((r, i) => `${r} ← “${A.keys[A.tops[i]]}” ${fmtPct(A.weights[i * A.T + A.tops[i]])}`);
		const who = A.h === 'avg' ? { en: 'both heads averaged', zh: '两个头取平均' } : { en: `head ${A.h}`, zh: `头 ${A.h}` };
		return {
			en: `${A.R} output symbols (rows) by ${A.T} input characters (columns), ${who.en}. The model wrote “${A.wrote}”${A.want ? `; the right answer is ${A.want}` : ''}. Where each row looks most: ${list.join('; ')}.`,
			zh: `${A.R} 个输出符号（行）× ${A.T} 个输入字符（列），${who.zh}。模型写下“${A.wrote}”${A.want ? `；正确答案是 ${A.want}` : ''}。每一行看得最多的地方：${list.join('；')}。`
		};
	},
	demo: async () => demoProps({ row: 3 }),
	minHeight: 360
} satisfies ViewMeta;

export default meta;
