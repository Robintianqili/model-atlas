// self-supervised-objectives view 'chips' (copied from tinybert's 'chips', credit: the tinybert package): the learner's
// text as tokens. props.mode 'plain': the text tokens alone. 'clm': what TinyGPT may read when it predicts the hidden token
// (the tokens before it; the later ones are dashed). 'pick' (default): TinyBERT's input, [CLS], the text with the hidden
// token as [MASK], [SEP]; a click on a text token hides that one instead (StageProps.setInput). 'train': a frame of the
// 'hide' process: what each text token became for training ([MASK], a random token, or kept).
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { vocab } from '../../adapters/masked-text/adapter.ts';
import { gap, say } from '../../text.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export type Chip = { text: string; id?: number; special?: boolean; tone?: 'generated' | 'muted' };

function trainChips(p: P) {
	const g = gap(p.rt);
	const st = (p.at.tape && p.at.index >= 0 ? p.at.tape.state(p.at.index) : null) as { fate?: Tensor; shown?: Tensor } | null;
	const fate = st?.fate?.data ?? [];
	const m = p.at.frame?.metrics ?? {};
	const items: Chip[] = g.c.map((t, i) => {
		const s = Number(st?.shown?.data[i] ?? t);
		return { text: vocab(s), id: s, special: fate[i] === 1, tone: fate[i] === 2 ? 'generated' : undefined };
	});
	const n = [1, 2, 3].map((f) => Array.from(fate).filter((x) => x === f).length);
	const ph = p.at.frame?.phase;
	const cur = ph === 'draw' || ph === 'force' ? Number(m.i) : null;
	const fateText = (f: number): I18n => (f === 1 ? { en: 'it becomes [MASK]', zh: '它变成 [MASK]' } : f === 2 ? { en: `it becomes a random token, “${say(m.rid)}”`, zh: `它变成一个随机词元“${say(m.rid)}”` } : { en: 'it stays as it is', zh: '它保持不变' });
	const now: I18n | null = cur === null ? null : !m.chosen ? { en: `Token ${cur}: u = ${fmt3(m.u)} is not below ${m.rate}, so it is not chosen.`, zh: `第 ${cur} 个词元：u = ${fmt3(m.u)} 不小于 ${m.rate}，所以没被选中。` } : { en: `Token ${cur}: ${ph === 'force' ? 'picked because no token was chosen' : `u = ${fmt3(m.u)} is below ${m.rate}, so it is chosen`}; v = ${fmt3(m.v)}, so ${fateText(m.fate).en}.`, zh: `第 ${cur} 个词元：${ph === 'force' ? '因为一个都没选中而被挑出' : `u = ${fmt3(m.u)} 小于 ${m.rate}，所以被选中`}；v = ${fmt3(m.v)}，所以${fateText(m.fate).zh}。` };
	const tally: I18n = { en: `So far: ${n[0]} [MASK], ${n[1]} random, ${n[2]} kept (underlined: chosen; dashed: [MASK]; coloured: a random token; outlined: the token drawing now).`, zh: `目前：${n[0]} 个 [MASK]，${n[1]} 个随机词元，${n[2]} 个保持不变（下划线：被选中；虚线框：[MASK]；彩色：随机词元；加框：正在抽签的词元）。` };
	return { mode: 'train', items, selected: cur, marked: Array.from(fate).flatMap((f, i) => (f > 0 ? [i] : [])), toText: [] as number[], caption: now ? { en: `${now.en} ${tally.en}`, zh: `${now.zh}${tally.zh}` } : tally };
}

export function chipsOf(p: P) {
	const g = gap(p.rt);
	const mode = (p.props.mode as string) ?? 'pick';
	if (mode === 'train') return trainChips(p);
	if (g.k < 0) return { mode, items: [] as Chip[], selected: null, marked: [] as number[], toText: [] as number[], caption: { en: 'No text yet.', zh: '还没有文字。' } };
	if (mode === 'plain' || mode === 'clm') {
		const clm = mode === 'clm';
		const items: Chip[] = g.c.map((id, i) => ({ text: vocab(id), id, special: clm && i > g.k }));
		const caption: I18n = !clm
			? { en: `${g.n} tokens, counted from 0.`, zh: `${g.n} 个词元，从 0 开始编号。` }
			: g.k === 0
				? { en: `To predict “${say(g.hidden)}”, the first token, TinyGPT has nothing to read: every later token (dashed) is hidden from it.`, zh: `要预测第一个词元“${say(g.hidden)}”，TinyGPT 没有任何东西可读：后面的词元（虚线框）都对它藏着。` }
				: { en: `To predict “${say(g.hidden)}” (outlined, token ${g.k}), TinyGPT reads only what comes before it, ${g.k === 1 ? '1 token' : `${g.k} tokens`}; the dashed ones come later and are hidden from it.`, zh: `要预测“${say(g.hidden)}”（加框，第 ${g.k} 个词元），TinyGPT 只读它前面的 ${g.k} 个词元；虚线框的词元在后面，对它藏着。` };
		return { mode, items, selected: clm ? g.k : null, marked: clm ? g.c.flatMap((_, i) => (i < g.k ? [i] : [])) : [], toText: [] as number[], caption };
	}
	const pos = g.ids.map((_, q) => g.where.indexOf(q));
	const items: Chip[] = g.ids.map((id) => ({ text: vocab(id), id, special: id >= 1024 }));
	const caption: I18n = { en: `Hidden: “${say(g.hidden)}”, text token ${g.k} (counting from 0). Click another token to hide it instead.`, zh: `藏起来的是“${say(g.hidden)}”，文字里的第 ${g.k} 个词元（从 0 数起）。点另一个词元，就改为藏起它。` };
	return { mode, items, selected: g.m >= 0 ? g.m : null, marked: [] as number[], toText: pos, caption };
}

const meta = {
	key: 'self-supervised-objectives/chips',
	title: { en: 'Your text as tokens', zh: '你的文字变成词元' },
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
