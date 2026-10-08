// tinybert view 'chips': the learner's text as TinyBERT reads it. props.mode 'pick' (default): [CLS], the text with the
// hidden token as [MASK], [SEP]; a click on a text token hides that one instead (StageProps.setInput). 'show': the same
// row, not clickable (step 1). 'train': a frame of the 'hide' process: what each text token became for training
// ([MASK], a random token, or kept).
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { gap } from '../../bert.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export type Chip = { text: string; id?: number; special?: boolean; tone?: 'generated' | 'muted' };

export function chipsOf(p: P) {
	const g = gap(p.rt);
	if (p.props.mode === 'train') {
		const st = (p.at.tape && p.at.index >= 0 ? p.at.tape.state(p.at.index) : null) as { fate?: Tensor; shown?: Tensor } | null;
		const fate = st?.fate?.data ?? [];
		const m = p.at.frame?.metrics ?? {};
		const items: Chip[] = g.c.map((t, i) => {
			const s = Number(st?.shown?.data[i] ?? t);
			return { text: g.word(s), id: s, special: fate[i] === 1, tone: fate[i] === 2 ? 'generated' : undefined };
		});
		const n = [1, 2, 3].map((f) => Array.from(fate).filter((x) => x === f).length);
		const ph = p.at.frame?.phase;
		const cur = ph === 'draw' || ph === 'force' ? Number(m.i) : null;
		// a random token can be a lone byte 0x80-0xFF, shown as <0xC9>: a piece of a character that takes several bytes
		const byte = (id: number): I18n => (/^<0x[0-9A-F]{2}>$/.test(g.say(id)) ? { en: ' (one byte: a piece of a character such as é or 中, not a whole character)', zh: '（一个字节：是 é 或“中”这类字符的一小块，本身不是完整的字符）' } : { en: '', zh: '' });
		const fateText = (f: number): I18n => (f === 1 ? { en: 'it becomes [MASK]', zh: '它变成 [MASK]' } : f === 2 ? { en: `it becomes a random token, “${g.say(m.rid)}”${byte(m.rid).en}`, zh: `它变成一个随机词元“${g.say(m.rid)}”${byte(m.rid).zh}` } : { en: 'it stays as it is', zh: '它保持不变' });
		const now: I18n | null = cur === null ? null : !m.chosen ? { en: `Token ${cur}: u = ${fmt3(m.u)} is not below ${m.rate}, so it is not chosen.`, zh: `第 ${cur} 个词元：u = ${fmt3(m.u)} 不小于 ${m.rate}，所以没被选中。` } : { en: `Token ${cur}: ${ph === 'force' ? 'picked because no token was chosen' : `u = ${fmt3(m.u)} is below ${m.rate}, so it is chosen`}; v = ${fmt3(m.v)}, so ${fateText(m.fate).en}.`, zh: `第 ${cur} 个词元：${ph === 'force' ? '因为一个都没选中而被挑出' : `u = ${fmt3(m.u)} 小于 ${m.rate}，所以被选中`}；v = ${fmt3(m.v)}，所以${fateText(m.fate).zh}。` };
		const tally: I18n = { en: `So far: ${n[0]} [MASK], ${n[1]} random, ${n[2]} kept (underlined: chosen; dashed: [MASK]; coloured: a random token; outlined: the token drawing now).`, zh: `目前：${n[0]} 个 [MASK]，${n[1]} 个随机词元，${n[2]} 个保持不变（下划线：被选中；虚线框：[MASK]；彩色：随机词元；加框：正在抽签的词元）。` };
		const caption: I18n = now ? { en: `${now.en} ${tally.en}`, zh: `${now.zh}${tally.zh}` } : tally;
		return { mode: 'train', items, selected: cur, marked: Array.from(fate).flatMap((f, i) => (f > 0 ? [i] : [])), toText: [] as number[], caption };
	}
	const pos = g.ids.map((_, q) => g.where.indexOf(q));
	const items: Chip[] = g.ids.map((id, q) => ({ text: g.word(id), id, special: id >= 1024 }));
	const show = p.props.mode === 'show';
	const partB: I18n = g.split >= 0 ? { en: ' Part B, the rest of your text, starts after the first [SEP].', zh: '第一个 [SEP] 之后是 B 段，也就是你文字的其余部分。' } : { en: '', zh: '' };
	const caption: I18n =
		g.k < 0
			? { en: 'No text yet.', zh: '还没有文字。' }
			: show
				? { en: `${g.len} tokens; the dashed ones are special tokens.${partB.en}`, zh: `${g.len} 个词元；虚线框的是特殊词元。${partB.zh}` }
				: { en: `Hidden: “${g.say(g.hidden)}”, text token ${g.k} (counting from 0). Click another token to hide it instead.${partB.en}`, zh: `藏起来的是“${g.say(g.hidden)}”，文字里的第 ${g.k} 个词元（从 0 数起）。点另一个词元，就改为藏起它。${partB.zh}` };
	return { mode: show ? 'show' : 'pick', items, selected: show ? null : g.m >= 0 ? g.m : null, marked: [] as number[], toText: pos, caption };
}

const meta = {
	key: 'tinybert/chips',
	title: { en: 'Your text, one token hidden', zh: '你的文字，藏起一个词元' },
	kinds: ['trace', 'process'],
	describe: (p: P): I18n => {
		const c = chipsOf(p);
		const list = c.items.map((x) => x.text).join(' | ');
		return { en: `${c.items.length} tokens: ${list}. ${c.caption.en}`, zh: `${c.items.length} 个词元：${list}。${c.caption.zh}` };
	},
	demo: async () => demoProps(),
	minHeight: 120
} satisfies ViewMeta;

export default meta;
