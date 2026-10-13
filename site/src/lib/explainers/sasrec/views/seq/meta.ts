// sasrec view 'seq' (kind trace): your history and what the model does with it. props.mode:
//   pick    the shop and your clicks in order (the cards add clicks: they change the input)
//   embed   each click as its product row + position row = the row the block reads
//   attend  the last click's attention weights over your history (hidden while a predict question is open)
//   mask    every click's row of attention weights; the masked cells (later clicks) hatched
//   topk    the next-click shares of the products you have not clicked, the top 5 marked
//   order   the top 5 for your order and for the same clicks reversed; the measured cost of shuffling; links
//   limits  the new arrival (no clicks in the logs) and popularity bias (measured on the test sessions)
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { ITEMS } from '../../world.ts';
import { K, attn, clicksOf, newPlace, ranked, reversed } from '../../seq.ts';
import { demoStage } from '../../demo.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
const names = (xs: number[], ps?: number[]) => I(xs.map((i, k) => ITEMS[i].name.en + (ps ? ` ${fmtPct(ps[k])}` : '')).join(', '), xs.map((i, k) => ITEMS[i].name.zh + (ps ? ` ${fmtPct(ps[k])}` : '')).join('，'));

export function describeSeq(p: Omit<StageProps, 'ctx'>): I18n {
	const mode = String(p.props.mode ?? 'pick'), c = clicksOf(p.rt), h = names(c);
	if (mode === 'pick') return I(`The shop: ${ITEMS.length} products in six categories; press one to add a click. Your clicks: ${h.en || 'none'}.`, `商店：${ITEMS.length} 件商品，分六类；点一件就加一次点击。你的点击：${h.zh || '还没有'}。`);
	if (!c.length) return I('No clicks yet.', '还没有点击。');
	const a = attn(p.rt), r = ranked(p.rt).slice(0, K);
	if (mode === 'embed') return I(`Your ${c.length} clicks (${h.en}), each as a product row plus a position row.`, `你的 ${c.length} 次点击（${h.zh}），每次都是商品行加位置行。`);
	if ((mode === 'attend' || mode === 'mask') && a) {
		const w = names(c, a);
		return I(`Attention weights of your last click over your history: ${w.en}.${mode === 'mask' ? ' Every earlier click has its own row; later clicks are masked.' : ''}`, `你最后一次点击对历史的注意力权重：${w.zh}。${mode === 'mask' ? '每次更早的点击都有自己的一行；之后的点击被遮住。' : ''}`);
	}
	if (mode === 'order') {
		const rv = reversed(p.rt), x = names(r.map((y) => y.i)), y = names(rv.top.map((z) => z.i));
		return I(`Top ${K} for your order: ${x.en}. For the same clicks reversed: ${y.en}.`, `按你的顺序的前 ${K} 名：${x.zh}。同样的点击倒过来：${y.zh}。`);
	}
	if (mode === 'limits') {
		const np = newPlace(p.rt);
		return I(`The new drone ${np ? `ranks ${np.place} of ${np.of} for you` : 'is in your history'}; it has no clicks in the training logs. The five most clicked products take a larger share of the top-${K} slots than of the next clicks.`, `新到的无人机${np ? `对你在 ${np.of} 件里排第 ${np.place}` : '在你的历史里'}；它在训练日志里没有点击。点击最多的五件商品在前 ${K} 名推荐位里的份额，比它们在下一次点击里的份额更大。`);
	}
	const x = names(r.map((y) => y.i), r.map((y) => y.p));
	return I(`Top ${K} of the products you have not clicked: ${x.en}.`, `你没点过的商品里的前 ${K} 名：${x.zh}。`);
}

const meta = {
	key: 'sasrec/seq',
	title: { en: 'Your clicks, read in order', zh: '你的点击，按顺序读' },
	kinds: ['trace'],
	describe: describeSeq,
	demo: () => demoStage({ mode: 'attend' }),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
