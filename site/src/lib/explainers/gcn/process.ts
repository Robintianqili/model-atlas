// gcn processes (kind C).
//   mp      message passing into the member in focus, layer 1 of the exported model, one event at a time: self-loop ->
//           normalize (Â) -> transform (X Wᵀ) -> one message per incoming edge -> aggregate (+ b) -> activate (tanh).
//           It replays the traced numbers of the fwd source (norm#deg, norm, gc1#xw, gc1, act1): every number on its
//           frames is a traced value or a product and running sum of traced values (claims.test.ts checks the sums).
//   smooth  over-smoothing: the one-hot rows averaged with Â again and again, no learned weights (k = 0 … rounds). Per
//           round: how alike Mr. Hi's and the Officer's rows are (cosine similarity), and each member's lean toward one.
import type { ExplainerRuntime, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { adjacency, graphOf, N, neighbours, normalize } from './graph.ts';

const F = 4; // numbers per member after layer 1
export type MpInput = { node: number; nb: number[]; deg: number[]; ahat: number[]; xw: number[]; b: number[]; z: number[]; h: number[] } | null;
type MpState = { node: number; sum: Tensor };
const vec = (p: string, v: ArrayLike<number>) => Object.fromEntries(Array.from(v, (x, k) => [`${p}${k}`, x]));

/** The mp input from the runtime: the traced numbers for the member in focus (null before the trace exists). */
export function mpInput(rt: ExplainerRuntime): MpInput {
	const t = rt.sources.fwd?.trace?.t;
	const b = rt.model?.weights['gc1.bias'];
	if (!t?.norm || !t['gc1#xw'] || !t['norm#deg'] || !b) return null;
	const g = graphOf(rt.input);
	const i = g.focus;
	const row = (x: Tensor, r: number, w: number) => Array.from(x.data.subarray(r * w, (r + 1) * w));
	return { node: i, nb: neighbours(g, i), deg: Array.from(t['norm#deg'].data), ahat: row(t.norm, i, N), xw: Array.from(t['gc1#xw'].data), b: Array.from(b.data), z: row(t.gc1, i, F), h: row(t.act1, i, F) };
}

export const mp: ProcessDef<MpInput, Record<string, never>, MpState> = {
	id: 'mp',
	kind: 'algorithm',
	title: { en: 'Messages into one member (graph layer 1)', zh: '消息流入一个成员（图卷积层 1）' },
	phases: {
		loop: { label: { en: 'Add a self-loop', zh: '加上自环' }, kind: 'input' },
		norm: { label: { en: 'Weigh each edge (Â)', zh: '给每条边定权重（Â）' }, kind: 'other' },
		transform: { label: { en: 'Transform every member (X Wᵀ)', zh: '变换每个成员（X Wᵀ）' }, kind: 'forward' },
		message: { label: { en: 'A message along one edge', zh: '沿一条边的消息' }, kind: 'message' },
		aggregate: { label: { en: 'Add up the messages and the bias', zh: '把消息和偏置加起来' }, kind: 'aggregate' },
		activate: { label: { en: 'Tanh', zh: 'Tanh' }, kind: 'forward' }
	},
	schema: {
		node: { kind: 'int', label: { en: 'Member in focus', zh: '关注的成员' }, tol: 'exact' },
		sum: { kind: 'tensor', label: { en: 'Messages added so far', zh: '目前加起来的消息' }, shape: [F], tol: 1e-12 }
	},
	defaults: {},
	*run(inp): ProcessRun<MpState, Record<string, never>> {
		if (!inp) return;
		const { node: i, nb, deg, ahat, xw } = inp;
		const st = (sum: ArrayLike<number>) => ({ node: i, sum: { shape: [F], data: Float64Array.from(sum) } });
		const zero = new Array<number>(F).fill(0);
		yield { phase: 'loop', step: 0, state: st(zero), metrics: { node: i, friends: nb.length, d: deg[i] } };
		const by = (f: (a: number, b: number) => boolean) => nb.reduce((best, j) => (f(deg[j], deg[best]) ? j : best), nb[0] ?? i);
		const hub = by((a, b) => a > b);
		const low = by((a, b) => a < b);
		yield { phase: 'norm', step: 0, state: st(zero), metrics: { node: i, d: deg[i], self: ahat[i], hub, dHub: deg[hub], aHub: ahat[hub], low, dLow: deg[low], aLow: ahat[low], friends: nb.length } };
		yield { phase: 'transform', step: 0, state: st(zero), metrics: { node: i, ...vec('x', xw.slice(i * F, i * F + F)) } };
		const sum = zero.slice();
		// messages count from 1 (step = k + 1): the scrubber reads "message 1" for the first, the member's own
		const senders = [i, ...nb];
		for (const [k, j] of senders.entries()) {
			const x = xw.slice(j * F, j * F + F);
			const m = x.map((v) => ahat[j] * v);
			m.forEach((v, f) => (sum[f] += v));
			yield { phase: 'message', step: k + 1, state: st(sum), metrics: { node: i, j, k: k + 1, of: senders.length, a: ahat[j], dj: deg[j], ...vec('x', x), ...vec('m', m), ...vec('s', sum) } };
		}
		yield { phase: 'aggregate', step: 0, state: st(sum), metrics: { node: i, ...vec('s', sum), ...vec('b', inp.b), ...vec('z', inp.z) } };
		yield { phase: 'activate', step: 0, state: st(sum), metrics: { node: i, ...vec('s', sum), ...vec('b', inp.b), ...vec('z', inp.z), ...vec('h', inp.h) } };
	}
};

export type SmoothOpt = { rounds: number };
type SmoothState = { lean: Tensor };
const cosRows = (z: Float64Array, a: number, b: number) => {
	let ab = 0, aa = 0, bb = 0;
	for (let k = 0; k < N; k++) (ab += z[a * N + k] * z[b * N + k]), (aa += z[a * N + k] * z[a * N + k]), (bb += z[b * N + k] * z[b * N + k]);
	return ab / Math.sqrt(aa * bb);
};

export const smooth: ProcessDef<unknown, SmoothOpt, SmoothState> = {
	id: 'smooth',
	kind: 'algorithm',
	title: { en: 'Average with the neighbours, again and again', zh: '和邻居平均，一遍又一遍' },
	phases: { round: { label: { en: 'Average with the neighbours', zh: '和邻居平均' }, kind: 'aggregate' } },
	schema: { lean: { kind: 'tensor', label: { en: 'Lean toward the Officer (+) or Mr. Hi (−)', zh: '偏向干事（+）还是 Mr. Hi（−）' }, shape: [N], tol: 1e-12 } },
	defaults: { rounds: 8 },
	*run(raw, ctx): ProcessRun<SmoothState, SmoothOpt> {
		const { ahat } = normalize(adjacency(graphOf(raw)), N);
		let z = Float64Array.from({ length: N * N }, (_, k) => (k % (N + 1) === 0 ? 1 : 0));
		for (let k = 0; k <= ctx.opt.rounds; k++) {
			if (k > 0) {
				const next = new Float64Array(N * N);
				for (let r = 0; r < N; r++) for (let j = 0; j < N; j++) if (ahat[r * N + j]) for (let c = 0; c < N; c++) next[r * N + c] += ahat[r * N + j] * z[j * N + c];
				z = next;
			}
			const lean = Float64Array.from({ length: N }, (_, r) => cosRows(z, r, 33) - cosRows(z, r, 0));
			yield { phase: 'round', step: k, state: { lean: { shape: [N], data: lean } }, metrics: { k, leaders: cosRows(z, 0, 33), spread: lean.reduce((s, v) => s + Math.abs(v), 0) / N } };
		}
	}
};
