// gnn-variants processes (kind C). Python references: pipeline/gnn-variants/reference.py (identical frames).
//   gat   GAT layer 1 for the member in focus, one event at a time: a score per sender (LeakyReLU(a_dst·zᵢ + a_src·zⱼ)),
//         the softmax over the senders, one weighted message α × zⱼ per sender with the running sum, then + b and tanh.
//         Its input is the traced z = X Wᵀ (gat.gc1#z) and the layer's learned vectors; claims.test.ts checks that its
//         α equals the traced gat.gc1#alpha and its last row equals gat.gc1 and gat.act1.
//   drop  attention is not an explanation: remove one friendship of the member in focus at a time and re-run the whole
//         GAT (graph.ts forwardGat, f64); per friend its layer-1 and layer-2 α and how far the member's score gap (the
//         Officer's score − Mr. Hi's) moves. The last frame counts, over every member with 2 or more friends, how often
//         the friend with the largest α (layer 1: agree, layer 2: agree2) is also the one whose removal moves the gap most.
// Frames carry captions ("Score sender 2 of 6: member 0"), so the scrubber never reads "sender 0" or "message 0".
import type { ExplainerRuntime, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { adjacency, forwardGat, graphOf, leaky, N, senders, type Params } from './graph.ts';

const F = 4; // numbers per member after layer 1
const vec = (p: string, v: ArrayLike<number>) => Object.fromEntries(Array.from(v, (x, k) => [`${p}${k}`, x]));
const T = (v: ArrayLike<number>): Tensor => ({ shape: [v.length], data: Float64Array.from(v) });
const arr = (rt: ExplainerRuntime, id: string) => Array.from(rt.sources.fwd?.trace?.t[id]?.data ?? rt.model?.weights[id]?.data ?? []);

export type GatInput = { node: number; adj: number[]; z: number[]; as: number[]; ad: number[]; b: number[]; slope: number } | null;
type GatState = { node: number; w: Tensor; sum: Tensor };
/** The gat input from the runtime: the traced messages z and the layer's vectors (null before the trace exists). */
export function gatInput(rt: ExplainerRuntime): GatInput {
	const z = arr(rt, 'gat.gc1#z');
	if (!z.length || !rt.model?.weights['gat.gc1.bias']) return null;
	const slope = Number(rt.spec?.layers.find((l) => l.id === 'gat.gc1')?.params?.slope ?? 0.2);
	const g = graphOf(rt.input);
	return { node: g.focus, adj: Array.from(adjacency(g)), z, as: arr(rt, 'gat.gc1.att_src'), ad: arr(rt, 'gat.gc1.att_dst'), b: arr(rt, 'gat.gc1.bias'), slope };
}

export const gat: ProcessDef<GatInput, Record<string, never>, GatState> = {
	id: 'gat',
	kind: 'algorithm',
	title: { en: 'GAT: attention into one member (layer 1)', zh: 'GAT：流入一个成员的注意力（第 1 层）' },
	phases: { score: { label: { en: 'Score one sender', zh: '给一个发送者打分' }, kind: 'other' }, softmax: { label: { en: 'Softmax over the senders', zh: '对发送者做 softmax' }, kind: 'other' }, message: { label: { en: 'A weighted message', zh: '一条加权消息' }, kind: 'message' }, activate: { label: { en: 'Add the bias, then tanh', zh: '加上偏置，再用 tanh' }, kind: 'forward' } },
	schema: { node: { kind: 'int', label: { en: 'Member in focus', zh: '关注的成员' }, tol: 'exact' }, w: { kind: 'tensor', label: { en: 'Score, then weight α, per member', zh: '每个成员的分数，之后是权重 α' }, shape: [N], tol: 1e-12 }, sum: { kind: 'tensor', label: { en: 'Messages added so far', zh: '目前加起来的消息' }, shape: [F], tol: 1e-12 } },
	defaults: {},
	*run(inp): ProcessRun<GatState, Record<string, never>> {
		if (!inp) return;
		const { node: i, z, slope } = inp;
		const S = senders(inp.adj, N, i);
		const zj = (j: number) => z.slice(j * F, j * F + F);
		const dot = (v: number[], j: number) => { let s = 0; for (let o = 0; o < F; o++) s += z[j * F + o] * v[o]; return s; }; // prettier-ignore
		const friends = (j: number) => senders(inp.adj, N, j).length - 1;
		const w = new Array<number>(N).fill(0);
		const st = (sum: number[]) => ({ node: i, w: T(w), sum: T(sum) });
		const sd = dot(inp.ad, i);
		const cap = (en: string, zh: string, k: number, j: number) => ({ en: `${en} ${k + 1} of ${S.length}: member ${j}${j === i ? ' itself' : ''}`, zh: `${zh}（共 ${S.length} 个）第 ${k + 1} 个：成员 ${j}${j === i ? ' 自己' : ''}` });
		const e: number[] = [];
		for (const [k, j] of S.entries()) {
			const ss = dot(inp.as, j);
			e.push(leaky(sd + ss, slope));
			w[j] = e[k];
			yield { phase: 'score', step: k + 1, state: st([0, 0, 0, 0]), caption: cap('Score sender', '给发送者打分', k, j), metrics: { node: i, j, k: k + 1, of: S.length, dj: friends(j), sd, ss, raw: sd + ss, e: e[k], slope, ...vec('x', zj(j)) } };
		}
		const m = Math.max(...e);
		const ex = e.map((v) => Math.exp(v - m));
		let tot = 0;
		for (const v of ex) tot += v;
		S.forEach((j, k) => (w[j] = ex[k] / tot));
		let top = S.length > 1 ? S[1] : i; // the friend with the largest α (ties: the smaller member)
		for (const j of S.slice(1)) if (w[j] > w[top]) top = j;
		yield { phase: 'softmax', step: 0, state: st([0, 0, 0, 0]), metrics: { node: i, of: S.length, friends: S.length - 1, max: m, tot, self: w[i], top, aTop: w[top], dTop: friends(top), eTop: e[S.indexOf(top)], even: 1 / S.length } };
		const sum = [0, 0, 0, 0];
		for (const [k, j] of S.entries()) {
			const msg = zj(j).map((v) => w[j] * v);
			msg.forEach((v, f) => (sum[f] += v));
			yield { phase: 'message', step: k + 1, state: st(sum), caption: cap('Message', '消息', k, j), metrics: { node: i, j, k: k + 1, of: S.length, a: w[j], ...vec('x', zj(j)), ...vec('m', msg), ...vec('s', sum) } };
		}
		const y = sum.map((v, f) => v + inp.b[f]);
		yield { phase: 'activate', step: 0, state: st(sum), caption: { en: 'Add the bias, then tanh', zh: '加上偏置，再用 tanh' }, metrics: { node: i, ...vec('s', sum), ...vec('b', inp.b), ...vec('y', y), ...vec('h', y.map(Math.tanh)) } };
	}
};

export type DropInput = { edges: [number, number][]; node: number; slope: number; p: Record<string, number[]> } | null;
type DropState = { node: number; alpha: Tensor; delta: Tensor };
/** The drop input: the learner's graph, the member in focus and every GAT weight (null before the model is loaded). */
export function dropInput(rt: ExplainerRuntime): DropInput {
	const W = rt.model?.weights;
	if (!W?.['gat.head.bias']) return null;
	const g = graphOf(rt.input);
	const p = Object.fromEntries(Object.entries(W).filter(([k]) => k.startsWith('gat.')).map(([k, v]) => [k, Array.from(v.data)]));
	return { edges: g.edges, node: g.focus, slope: Number(rt.spec?.layers.find((l) => l.id === 'gat.gc1')?.params?.slope ?? 0.2), p };
}
/** Per friend j of member i: α (layer 1), α2 (layer 2) and the move of i's score gap when the friendship i–j is removed. */
function ablate(edges: [number, number][], i: number, p: Params, slope: number) {
	const adj = adjacency({ edges, focus: i });
	const base = forwardGat(p, adj, slope);
	const nb = senders(adj, N, i).slice(1);
	const rows = nb.map((j) => {
		const cut = edges.filter(([a, b]) => !((a === i && b === j) || (a === j && b === i)));
		return { j, a: base.alpha[i * N + j], a2: base.alpha2[i * N + j], d: forwardGat(p, adjacency({ edges: cut, focus: i }), slope).gap[i] - base.gap[i] };
	});
	const best = (f: (r: (typeof rows)[number]) => number) => rows.reduce((b, r) => (f(r) > f(b) ? r : b), rows[0]);
	return { gap: base.gap[i], rows, topA: best((r) => r.a), topA2: best((r) => r.a2), topE: best((r) => Math.abs(r.d)) };
}

export const drop: ProcessDef<DropInput, Record<string, never>, DropState> = {
	id: 'drop',
	kind: 'algorithm',
	title: { en: 'Remove one friendship at a time', zh: '一次去掉一段朋友关系' },
	phases: { base: { label: { en: 'The answer with every friendship', zh: '所有朋友关系都在时的答案' }, kind: 'forward' }, remove: { label: { en: 'Remove one friendship and re-run', zh: '去掉一段朋友关系，重新计算' }, kind: 'other' }, verdict: { label: { en: 'Most attention vs largest effect', zh: '注意力最大与影响最大' }, kind: 'output' } },
	schema: { node: { kind: 'int', label: { en: 'Member in focus', zh: '关注的成员' }, tol: 'exact' }, alpha: { kind: 'tensor', label: { en: 'Attention α in layer 1', zh: '第 1 层的注意力 α' }, shape: [N], tol: 1e-12 }, delta: { kind: 'tensor', label: { en: 'Move of the score gap without that friendship', zh: '去掉这段朋友关系后分数差的变化' }, shape: [N], tol: 1e-12 } },
	defaults: {},
	*run(inp): ProcessRun<DropState, Record<string, never>> {
		if (!inp) return;
		const { node: i, p, slope } = inp;
		const r = ablate(inp.edges, i, p, slope);
		const alpha = new Array<number>(N).fill(0);
		const delta = new Array<number>(N).fill(0);
		r.rows.forEach((x) => (alpha[x.j] = x.a));
		const st = () => ({ node: i, alpha: T(alpha), delta: T(delta) });
		yield { phase: 'base', step: 0, state: st(), metrics: { node: i, gap: r.gap, friends: r.rows.length } };
		for (const [k, x] of r.rows.entries()) {
			delta[x.j] = x.d;
			yield { phase: 'remove', step: k, state: st(), metrics: { node: i, j: x.j, k: k + 1, of: r.rows.length, a: x.a, a2: x.a2, d: x.d, gap: r.gap, after: r.gap + x.d, flip: Math.sign(r.gap + x.d) !== Math.sign(r.gap) ? 1 : 0 } };
		}
		let agree = 0, agree2 = 0, members = 0; // prettier-ignore
		for (let m = 0; m < N; m++) {
			const q = ablate(inp.edges, m, p, slope);
			if (q.rows.length < 2) continue;
			members++;
			if (q.topA.j === q.topE.j) agree++;
			if (q.topA2.j === q.topE.j) agree2++;
		}
		const { topA: A, topA2: A2, topE: E } = r;
		const pair: Record<string, number> = A ? { topA: A.j, aTopA: A.a, dTopA: A.d, topE: E.j, aTopE: E.a, a2TopE: E.a2, dTopE: E.d, same: A.j === E.j ? 1 : 0, topA2: A2.j, aTopA2: A2.a2, same2: A2.j === E.j ? 1 : 0 } : {};
		yield { phase: 'verdict', step: 0, state: st(), metrics: { node: i, friends: r.rows.length, gap: r.gap, ...pair, agree, agree2, members } };
	}
};
