// gcn: local input adapter 'graph' (InputKind 'graph'). The raw input is a GraphRaw (graph.ts): the club's friendships
// (editable), the labelled members and the member in focus. The model reads the adjacency A and the one-hot features X.
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { adjacency, graphOf, KARATE, N, type GraphRaw } from '../../graph.ts';

export const ISOLATED: GraphRaw = { ...KARATE, edges: KARATE.edges.filter((e) => !e.includes(11)), focus: 11 };

const adapter: InputAdapter<GraphRaw> = {
	id: 'graph',
	kind: 'graph',
	preprocess(raw): Preprocessed {
		const adj = { shape: [N, N], data: Float32Array.from(adjacency(graphOf(raw))) };
		const x = { shape: [N, N], data: Float32Array.from({ length: N * N }, (_, k) => (k % (N + 1) === 0 ? 1 : 0)) };
		return {
			tensors: { adj, x },
			// not drawn again: the Capture shows A (editable); X is the same for every graph (the lesson draws it). Notes would show on every stage
			steps: [{ id: 'adj', label: { en: 'Friendships: the adjacency matrix A', zh: '朋友关系：邻接矩阵 A' } }, { id: 'x', label: { en: 'Features: the one-hot table X', zh: '特征：独热表 X' } }],
			empty: false
		};
	},
	edgeFixtures: () => [{ id: 'isolated', label: { en: 'member 11 loses his only friend', zh: '成员 11 失去唯一的朋友' }, raw: ISOLATED }],
	layer0: {
		view: 'gcn/graph',
		narration: {
			what: {
				en: 'The club becomes a table A of {n|int} numbers, shaped {shape}: row i has a 1 in column j when members i and j are friends.',
				zh: '这个俱乐部变成一张有 {n|int} 个数的表 A，形状是 {shape}：成员 i 和 j 是朋友时，第 i 行第 j 列是 1。'
			}
		}
	}
};

export default adapter;
