// masked-lm view 'batch': a frame of the 'batch' process. Every row of the batch as tokens, each shown as the model would
// read it so far (dashed: [MASK]; coloured: a random token; underlined: chosen; outlined: the token drawing now), a line
// on the current draw, the running counts and, at the end, the counts of the same rule on a full training batch.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { vocab } from '../../adapters/masked-text/adapter.ts';
import { batchRows } from '../../process.ts';
import { say } from '../../text.ts';
import { demoProps } from '../../demo.ts';
import type { Chip } from '../fill/meta.ts';

type P = Omit<StageProps, 'ctx'>;

export function batchOf(p: P) {
	const rows = batchRows(p.rt.input as never);
	const st = (p.at.tape && p.at.index >= 0 ? p.at.tape.state(p.at.index) : null) as { fate?: Tensor; shown?: Tensor } | null;
	const m = p.at.frame?.metrics ?? {};
	const ph = p.at.frame?.phase;
	const fate = st?.fate?.data ?? [];
	const cur = ph === 'draw' || ph === 'force' ? Number(m.at) : -1;
	let at = 0;
	const lines = rows.map((ids, r) => {
		const first = at;
		at += ids.length;
		const items: Chip[] = ids.map((t, i) => {
			const s = Number(st?.shown?.data[first + i] ?? t);
			return { text: vocab(s), id: s, special: fate[first + i] === 1, tone: fate[first + i] === 2 ? 'generated' : undefined };
		});
		return { r, items, marked: ids.flatMap((_, i) => (fate[first + i] > 0 ? [i] : [])), selected: cur >= first && cur < at ? cur - first : null, label: r === 0 ? { en: 'Row 0: your text', zh: '第 0 行：你的文字' } : { en: `Row ${r}: an example sentence`, zh: `第 ${r} 行：例句` } };
	});
	const fateText = (f: number): I18n => (f === 1 ? { en: 'it becomes [MASK]', zh: '它变成 [MASK]' } : f === 2 ? { en: `it becomes a random token, “${say(m.rid)}”`, zh: `它变成一个随机词元“${say(m.rid)}”` } : { en: 'it stays as it is, but is still graded', zh: '它保持不变，但仍要被批改' });
	const now: I18n | null = cur < 0 ? null : !m.pick ? { en: `Row ${m.row}, token ${m.i}: u = ${fmt3(m.u)} is not below ${m.rate}, so it is not chosen.`, zh: `第 ${m.row} 行第 ${m.i} 个词元：u = ${fmt3(m.u)} 不小于 ${m.rate}，所以没被选中。` } : { en: `Row ${m.row}, token ${m.i}: ${ph === 'force' ? 'picked because no token of its row was chosen' : `u = ${fmt3(m.u)} is below ${m.rate}, so it is chosen`}; v = ${fmt3(m.v)}, so ${fateText(m.fate).en}.`, zh: `第 ${m.row} 行第 ${m.i} 个词元：${ph === 'force' ? '因为这一行一个都没选中而被挑出' : `u = ${fmt3(m.u)} 小于 ${m.rate}，所以被选中`}；v = ${fmt3(m.v)}，所以${fateText(m.fate).zh}。` };
	const n = rows.reduce((s, x) => s + x.length, 0);
	const tally: I18n = { en: `Chosen so far: ${m.chosen ?? 0} of ${n} tokens (${m.masked ?? 0} [MASK], ${m.random ?? 0} random, ${m.kept ?? 0} kept).`, zh: `目前选中：${n} 个词元中的 ${m.chosen ?? 0} 个（${m.masked ?? 0} 个 [MASK]，${m.random ?? 0} 个随机，${m.kept ?? 0} 个保留）。` };
	const sh = (a: number, b: number) => fmtPct(a / b);
	const big: I18n | null =
		ph === 'done'
			? { en: `The same rule on a full training batch, ${m.big} examples of ${m.bigLen} tokens (${m.bigN} tokens): ${m.bigChosen} chosen (${sh(m.bigChosen, m.bigN)}); of those ${sh(m.bigMasked, m.bigChosen)} [MASK], ${sh(m.bigRandom, m.bigChosen)} random, ${sh(m.bigKept, m.bigChosen)} kept.`, zh: `同样的规则用在一个完整的训练批次上：${m.big} 个样本、每个 ${m.bigLen} 个词元（共 ${m.bigN} 个）：选中 ${m.bigChosen} 个（${sh(m.bigChosen, m.bigN)}）；其中 ${sh(m.bigMasked, m.bigChosen)} 是 [MASK]，${sh(m.bigRandom, m.bigChosen)} 随机，${sh(m.bigKept, m.bigChosen)} 保留。` }
			: null;
	return { lines, now, tally, big };
}

const meta = {
	key: 'masked-lm/batch',
	title: { en: 'The 80/10/10 rule on a batch', zh: '一个批次上的 80/10/10 规则' },
	kinds: ['process'],
	describe: (p: P): I18n => {
		const B = batchOf(p);
		const rows = B.lines.map((l) => ({ en: `${l.label.en}: ${l.items.map((c) => c.text).join(' | ')}`, zh: `${l.label.zh}：${l.items.map((c) => c.text).join(' | ')}` }));
		const parts = [rows.map((r) => r.en).join('. '), B.now?.en, B.tally.en, B.big?.en].filter(Boolean);
		const zh = [rows.map((r) => r.zh).join('。'), B.now?.zh, B.tally.zh, B.big?.zh].filter(Boolean);
		return { en: parts.join(' '), zh: zh.join('') };
	},
	demo: async () => demoProps({ mode: 'batch' }),
	minHeight: 320
} satisfies ViewMeta;

export default meta;
