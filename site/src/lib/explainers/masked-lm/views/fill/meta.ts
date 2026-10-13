// masked-lm view 'fill': the learner's text as TinyBERT reads it ([CLS], the text with the chosen tokens replaced,
// [SEP]); a click on a text token hides it or gives it back (StageProps.setInput). Under it, for each chosen token (the
// first six), TinyBERT's three top guesses from its row at that position, the right token in the accent colour, and a
// line with the right token's share and rank.
//   props.bars   false: the token row alone (default true)
//   props.crowd  a button that hides two tokens of every three (the breakIt step)
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { CLS, contentOf, MASK, SEP, toggled, vocab, type MaskedRaw } from '../../adapters/masked-text/adapter.ts';
import { bertAt, mlmLoss, say, text } from '../../text.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export const PANELS = 6;
export type Chip = { text: string; id?: number; special?: boolean; tone?: 'generated' | 'muted' };
const FATE_TEXT: I18n[] = [{ en: '', zh: '' }, { en: 'shown as [MASK]', zh: '显示为 [MASK]' }, { en: 'shown as a random token', zh: '显示为随机词元' }, { en: 'chosen but shown as itself', zh: '被选中但原样显示' }];

/** TinyBERT's input as chips: special tokens dashed, random swaps coloured; toText maps a chip to its text token (-1: special). */
export function inputChips(p: P) {
	const x = text(p.rt);
	const items: Chip[] = x.ids.map((id) => ({ text: vocab(id), id, special: id === MASK || id === CLS || id === SEP }));
	x.fate.forEach((f, k) => {
		if (f === 2) items[x.where[k]].tone = 'generated';
	});
	return { items, toText: x.ids.map((_, q) => x.where.indexOf(q)), marked: x.graded.map((k) => x.where[k]) };
}

export function fillOf(p: P) {
	const x = text(p.rt);
	const L = mlmLoss(p.rt);
	const gaps = x.graded.slice(0, PANELS).map((k) => {
		const g = bertAt(p.rt, k);
		const ans = x.c[k];
		const bars = g.p ? g.order.slice(0, 3).map((id) => ({ label: vocab(id), p: g.p![id], id, color: id === ans ? 'var(--accent)' : undefined })) : null;
		const line: I18n = { en: `Right token “${say(ans)}”: ${fmtPct(g.pAnswer)}, rank ${g.rank}.`, zh: `正确词元“${say(ans)}”：${fmtPct(g.pAnswer)}，第 ${g.rank} 名。` };
		return { k, ans, bars, line, head: { en: `Token ${k} (counting from 0), ${FATE_TEXT[x.fate[k]].en}`, zh: `第 ${k} 个词元（从 0 数起），${FATE_TEXT[x.fate[k]].zh}` } };
	});
	const more = x.graded.length - gaps.length;
	const caption: I18n = !x.n
		? { en: 'No text yet.', zh: '还没有文字。' }
		: !x.graded.length
			? { en: 'No token is chosen: click a token to hide it.', zh: '没有选中任何词元：点一个词元把它藏起来。' }
			: { en: `${x.graded.length} of ${x.n} tokens chosen; TinyBERT’s top guess is the right token at ${L.right} of them.${more > 0 ? ` The first ${gaps.length} are shown.` : ''} Click a token to hide it or give it back.`, zh: `${x.n} 个词元中选中了 ${x.graded.length} 个；其中 ${L.right} 个，TinyBERT 的第一猜测就是正确词元。${more > 0 ? `这里显示前 ${gaps.length} 个。` : ''}点一个词元可以藏起它或把它还原。` };
	return { ...inputChips(p), gaps, caption, bars: p.props.bars !== false, crowd: p.props.crowd === true, n: x.n };
}

/** The raw input with two tokens of every three hidden (all but tokens 2, 5, 8, ...). */
export function crowded(raw: MaskedRaw): MaskedRaw {
	const n = contentOf(raw).length;
	return { ...(raw ?? {}), fate: Array.from({ length: n }, (_, k) => (k % 3 === 2 ? 0 : 1)), swap: [], picked: true, focus: 0, fromDraw: false };
}

/** The raw input after a click on chip q of TinyBERT's input (null: a special token). */
export function clicked(p: P, q: number): MaskedRaw | null {
	const k = inputChips(p).toText[q];
	return k === undefined || k < 0 ? null : toggled(p.rt.input as MaskedRaw, k);
}

const meta = {
	key: 'masked-lm/fill',
	title: { en: 'Hide tokens, read TinyBERT’s guesses', zh: '藏起词元，读 TinyBERT 的猜测' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const F = fillOf(p);
		const list = F.items.map((c) => c.text).join(' | ');
		const g = F.bars ? F.gaps.map((x) => ({ en: `${x.head.en}: top guesses ${x.bars?.map((b) => `${say(b.id)} ${fmtPct(b.p)}`).join(', ') ?? '—'}. ${x.line.en}`, zh: `${x.head.zh}：前几名猜测 ${x.bars?.map((b) => `${say(b.id)} ${fmtPct(b.p)}`).join('，') ?? '—'}。${x.line.zh}` })) : [];
		return { en: `TinyBERT reads ${F.items.length} tokens: ${list}. ${F.caption.en} ${g.map((x) => x.en).join(' ')}`, zh: `TinyBERT 读 ${F.items.length} 个词元：${list}。${F.caption.zh}${g.map((x) => x.zh).join('')}` };
	},
	demo: async () => demoProps(),
	minHeight: 320
} satisfies ViewMeta;

export default meta;
