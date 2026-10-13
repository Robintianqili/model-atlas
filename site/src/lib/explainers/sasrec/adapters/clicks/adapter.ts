// sasrec: local input adapter 'clicks' (catalog InputKind 'ratings'). The learner's raw input is a Clicks (world.ts): the
// products they clicked, in order, at most 12. One tensor: ids [T] (i32), the product ids in click order. No user id and
// no features: to the model a click is only its product's id and its place in the order.
import type { InputAdapter, IntTensor, Preprocessed } from '../../../../atlas/types/index.ts';
import { EDGES, ITEMS, MAX_CLICKS, PRESETS, toClicks, type Clicks } from '../../world.ts';

const adapter: InputAdapter<Clicks> = {
	id: 'clicks',
	kind: 'ratings',
	preprocess(raw): Preprocessed {
		const c = toClicks(raw).clicks;
		const ids: IntTensor = { shape: [c.length], data: Int32Array.from(c), dtype: 'i32' };
		if (!c.length) return { tensors: { ids }, steps: [], empty: true, notes: [{ en: 'No clicks yet. Click a few products in the shop and every step fills with your history.', zh: '还没有点击。在商店里点几件商品，每一步都会填上你的历史。' }] };
		return {
			tensors: { ids },
			steps: [{ id: 'ids', label: { en: `What the model sees: ${c.length} product ids, in click order`, zh: `模型看到的：按点击顺序排列的 ${c.length} 个商品编号` }, note: { en: c.map((i) => `#${i}`).join(' → '), zh: c.map((i) => `#${i}`).join(' → ') } }],
			empty: false,
			notes: [{
				en: `Your history: ${c.map((i) => ITEMS[i].name.en).join(' → ')} (${c.length} of at most ${MAX_CLICKS} clicks). The model reads only the ids and their order, never the names.`,
				zh: `你的历史：${c.map((i) => ITEMS[i].name.zh).join(' → ')}（${c.length} 次点击，最多 ${MAX_CLICKS} 次）。模型只读编号和顺序，从不读名字。`
			}]
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => EDGES,
	layer0: { view: 'sasrec/seq', narration: { what: { en: 'Your clicks become {n|int} product ids.', zh: '你的点击变成 {n|int} 个商品编号。' } } }
};

export default adapter;
