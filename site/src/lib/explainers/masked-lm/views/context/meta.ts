// masked-lm view 'context': masked vs causal language modelling on the same text token (the focus: the chosen token
// clicked last, else the first chosen one). Two rows of chips: what TinyBERT reads while it fills that position (every
// other position of its input, underlined) and what TinyGPT reads when it predicts that token (only the tokens before
// it; later ones dashed); under each, the model's five top guesses with the right token in the accent colour.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { vocab } from '../../adapters/masked-text/adapter.ts';
import { bertAt, gptAt, say, text } from '../../text.ts';
import { demoProps } from '../../demo.ts';
import { inputChips, type Chip } from '../fill/meta.ts';

type P = Omit<StageProps, 'ctx'>;

export function contextOf(p: P) {
	const x = text(p.rt);
	const k = x.focus;
	if (k < 0) return null;
	const ans = x.c[k];
	const bars = (g: ReturnType<typeof bertAt>) => (g.p ? g.order.slice(0, 5).map((id) => ({ label: vocab(id), p: g.p![id], id, color: id === ans ? 'var(--accent)' : undefined })) : null);
	const B = inputChips(p);
	const m = x.where[k];
	const gptItems: Chip[] = x.c.map((id, i) => ({ text: vocab(id), id, special: i > k }));
	const b = bertAt(p.rt, k);
	const g = gptAt(p.rt, k);
	const after = x.n - k - 1;
	const say2 = (w: string, q: typeof b): I18n => ({ en: `${w}: “${say(ans)}” ${fmtPct(q.pAnswer)} (rank ${q.rank}), top “${say(q.top)}” ${fmtPct(q.pTop)}.`, zh: `${w}：“${say(ans)}” ${fmtPct(q.pAnswer)}（第 ${q.rank} 名），第一名“${say(q.top)}” ${fmtPct(q.pTop)}。` });
	return {
		k,
		ans: say(ans),
		bert: { items: B.items, marked: B.items.flatMap((_, q) => (q !== m ? [q] : [])), selected: m, bars: bars(b), line: say2('TinyBERT', b), reads: { en: `TinyBERT fills token ${k} (counting from 0) reading ${k} text token${k === 1 ? '' : 's'} before it and ${after} after it, plus [CLS] and [SEP].`, zh: `TinyBERT 填第 ${k} 个词元（从 0 数起）时，读它前面的 ${k} 个文字词元和后面的 ${after} 个，再加上 [CLS] 和 [SEP]。` } as I18n },
		gpt: { items: gptItems, marked: x.c.flatMap((_, i) => (i < k ? [i] : [])), selected: k, bars: bars(g), line: g.p ? say2('TinyGPT', g) : { en: 'TinyGPT: nothing comes before the first token, so it cannot guess it.', zh: 'TinyGPT：第一个词元前面什么都没有，所以它猜不了。' }, reads: { en: `TinyGPT predicts token ${k} reading only the ${k} token${k === 1 ? '' : 's'} before it; its causal mask hides the ${after} after it (dashed).`, zh: `TinyGPT 预测第 ${k} 个词元时只读它前面的 ${k} 个词元；因果掩码把后面的 ${after} 个藏了起来（虚线框）。` } as I18n }
	};
}

const meta = {
	key: 'masked-lm/context',
	title: { en: 'Which context each model reads', zh: '每个模型读到哪些上下文' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const C = contextOf(p);
		if (!C) return { en: 'No token is chosen yet: nothing to compare.', zh: '还没有选中词元：没有可比较的内容。' };
		return { en: `The right token is “${C.ans}”. ${C.bert.reads.en} ${C.bert.line.en} ${C.gpt.reads.en} ${C.gpt.line.en}`, zh: `正确词元是“${C.ans}”。${C.bert.reads.zh}${C.bert.line.zh}${C.gpt.reads.zh}${C.gpt.line.zh}` };
	},
	demo: async () => demoProps(),
	minHeight: 360
} satisfies ViewMeta;

export default meta;
