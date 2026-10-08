// tinybert view 'both-ways': the attention weights of TinyBERT's block 1 and of TinyGPT's block 1 on the same text, one
// head at a time (props.head, else the head whose [MASK] row reads most after the gap). TinyGPT's causal mask is the
// hatched triangle; TinyBERT has none. The hidden token's row is outlined in both (in TinyGPT it is the real word).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { gap, rightHead, sides, T } from '../../bert.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;

export function bothOf(p: P, pick: number | null = null) {
	const g = gap(p.rt);
	const b = T(p.rt, 'B1.attn');
	const q = T(p.rt, 'L0.attn.weights', 'gpt');
	const H = b?.shape[0] ?? 0;
	const h = Math.max(0, Math.min(H - 1, pick ?? (typeof p.props.head === 'number' ? p.props.head : rightHead(p.rt))));
	const head = (t: typeof b) => (t ? t.data.subarray(h * t.shape[1] * t.shape[2], (h + 1) * t.shape[1] * t.shape[2]) : null);
	const s = sides(p.rt, 1, h);
	const auto = h === rightHead(p.rt);
	const after: I18n = s.nRight === 1 ? { en: 'the token after the gap', zh: '空缺后面的 1 个词元' } : { en: `the ${s.nRight} tokens after the gap`, zh: `空缺后面的 ${s.nRight} 个词元` };
	const caption: I18n = {
		en: `Head ${h} (counting from 0)${auto ? ', the head of block 1 whose [MASK] row reads most after the gap' : ''}. TinyBERT’s [MASK] row gives ${fmtPct(s.right)} of its weight to ${after.en}; in TinyGPT every weight above the diagonal is exactly 0. The two models were trained separately, so TinyGPT’s head ${h} is not a twin of TinyBERT’s.`,
		zh: `头 ${h}（从 0 数起）${auto ? '，也就是第 1 块里 [MASK] 行往空缺后面读得最多的头' : ''}。TinyBERT 的 [MASK] 行把 ${fmtPct(s.right)} 的权重给了${after.zh}；TinyGPT 对角线以上的权重全都正好是 0。两个模型是分开训练的，所以 TinyGPT 的头 ${h} 和 TinyBERT 的头 ${h} 并不对应。`
	};
	return { h, heads: H, bert: head(b), bertTokens: g.ids.map(g.word), bertRow: g.m, gpt: head(q), gptTokens: g.c.map(g.word), gptRow: g.k, caption };
}

const meta = {
	key: 'tinybert/both-ways',
	title: { en: 'Both ways vs left only', zh: '双向与只看左边' },
	kinds: ['trace'],
	describe: (p: P): I18n => (bothOf(p).bert ? bothOf(p).caption : { en: 'Attention of the two models: nothing to show yet.', zh: '两个模型的注意力：还没有可显示的内容。' }),
	demo: async () => demoProps(),
	minHeight: 360
} satisfies ViewMeta;

export default meta;
