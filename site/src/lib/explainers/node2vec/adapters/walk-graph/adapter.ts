// node2vec: local input adapter 'walk-graph' (InputKind 'graph'): the kit graph adapter's input (Zachary's karate club,
// editable, and the member in focus) plus node2vec's p and q. The model's input is the walks, written out as one list of
// member ids (walk after walk). A graph without a friendship gives no move and no pair: the input is empty.
import type { I18n, InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { fmtInt } from '../../../../atlas/core/format.ts';
import { GRAPH_PRESETS } from '../../../../atlas/adapters/graph/adapter.ts';
import { ISOLATED } from '../../../../atlas/data/karate.ts';
import { inputOf, LEN, START_P, START_Q, walksOf, WALKS, type WalkRaw } from '../../walks.ts';

export const NO_WALKS: I18n = { en: 'No walk can leave its first member, so there is nothing to learn: add at least one friendship.', zh: '没有一条游走能离开它的第一个成员，所以没有可学的内容：至少加上一段朋友关系。' };
const pq = { p: START_P, q: START_Q };

const adapter: InputAdapter<WalkRaw> = {
	id: 'walk-graph',
	kind: 'graph',
	preprocess(raw): Preprocessed {
		const g = inputOf(raw), W = walksOf(g), ids = W.walks.flat();
		return {
			tensors: { input: { shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' } },
			steps: [
				{ id: 'adj', label: { en: `The club: ${g.edges.length} friendships between 34 members`, zh: `俱乐部：34 个成员之间的 ${g.edges.length} 段朋友关系` } },
				{ id: 'walks', label: { en: `Random walks: ${WALKS} from every member, at most ${LEN} members each (p = ${g.p}, q = ${g.q}), ${fmtInt(ids.length)} member ids in all`, zh: `随机游走：从每个成员出发走 ${WALKS} 次，每次最多 ${LEN} 个成员（p = ${g.p}，q = ${g.q}），一共 ${fmtInt(ids.length)} 个成员编号` } }
			],
			empty: !W.pairs.length,
			notes: W.pairs.length ? [] : [NO_WALKS]
		};
	},
	presets: () => GRAPH_PRESETS.map((p) => ({ id: p.id, label: p.label, raw: { ...p.raw, ...pq } })),
	edgeFixtures: () => [{ id: 'isolated', label: { en: 'member 11 loses its only friend (and is in focus)', zh: '成员 11 失去唯一的朋友（并被关注）' }, raw: { ...ISOLATED, ...pq } }],
	layer0: { view: 'node2vec/walk', narration: { what: { en: 'The club, and the walks that start from each member.', zh: '俱乐部，以及从每个成员出发的游走。' } } }
};

export default adapter;
