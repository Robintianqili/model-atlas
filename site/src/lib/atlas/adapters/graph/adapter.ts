// Input adapter 'graph' (K1; catalog InputKind 'graph'), promoted by the rule of two: the gcn and gnn-variants packages
// each wrote this adapter. The raw input is a GraphRaw (atlas/data/karate.ts): the friendships of Zachary's karate club
// (editable), the member in focus and, for pages that label members, `labelled`. Tensors: adj [34, 34] (the 0/1
// adjacency A) and x [34, 34] (one-hot features X = I). The steps are named, not drawn: the Capture draws the club
// and its table. A graph is never empty (a club without friendships still has its 34 members).
import type { InputAdapter } from '../../types/explainer.ts';
import type { Preprocessed } from '../../types/runtime.ts';
import { adjacency, graphOf, ISOLATED, KARATE_EDGES, N, oneHot, type GraphRaw } from '../../data/karate.ts';

export type { GraphRaw };

export const GRAPH_PRESETS: { id: string; label: { en: string; zh: string }; raw: GraphRaw }[] = [
	{ id: 'club', label: { en: 'Zachary’s club (1977)', zh: 'Zachary 的俱乐部（1977）' }, raw: { edges: KARATE_EDGES, focus: 8 } },
	{ id: 'alone', label: { en: 'Member 11 alone', zh: '成员 11 落单' }, raw: ISOLATED },
	{ id: 'none', label: { en: 'No friendships', zh: '没有朋友关系' }, raw: { edges: [], focus: 8 } }
];

export function preprocessGraph(raw: unknown): Preprocessed {
	const g = graphOf(raw);
	return {
		tensors: { adj: { shape: [N, N], data: Float32Array.from(adjacency(g)) }, x: { shape: [N, N], data: Float32Array.from(oneHot()) } },
		steps: [
			{ id: 'adj', label: { en: 'Friendships: the adjacency matrix A (a 1 where two members are friends)', zh: '朋友关系：邻接矩阵 A（两个成员是朋友的地方是 1）' } },
			{ id: 'x', label: { en: 'Features: the one-hot table X (every member is its own feature)', zh: '特征：独热表 X（每个成员各自是一个特征）' } }
		],
		empty: false
	};
}

const adapter: InputAdapter<GraphRaw> = {
	id: 'graph',
	kind: 'graph',
	preprocess: (raw) => preprocessGraph(raw),
	presets: () => GRAPH_PRESETS.map((p) => ({ id: p.id, label: p.label, raw: p.raw })),
	edgeFixtures: () => [{ id: 'isolated', label: { en: 'member 11 loses its only friend', zh: '成员 11 失去唯一的朋友' }, raw: ISOLATED }],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'The club becomes a table A of {n|int} numbers, shaped {shape}: row i has a 1 in column j when members i and j are friends.',
				zh: '这个俱乐部变成一张有 {n|int} 个数的表 A，形状是 {shape}：成员 i 和 j 是朋友时，第 i 行第 j 列是 1。'
			}
		}
	}
};

export default adapter;
