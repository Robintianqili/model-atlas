// tinymamba local op 'ssm_scan': Mamba's selective scan (Gu and Dao 2023, selective_scan with delta_softplus), unbatched.
//   inputs  u [T, D] (the channels), dt [T, D] (raw step sizes), B [T, N], C [T, N];  weights A_log [D, N], D [D]
//   delta_t = softplus(dt_t)  (log(1 + e^x); x itself above 20, as PyTorch)          sub-step '#delta' [T, D]
//   h_t[d, n] = exp(delta_t[d] · A[d, n]) · h_{t−1}[d, n] + delta_t[d] · B_t[n] · u_t[d],   A = −exp(A_log), h_0 = 0
//                                                                                    sub-step '#h' [T, D, N]
//   y_t[d] = Σ_n C_t[n] · h_t[d, n] + D[d] · u_t[d]                                   output [T, D]
// Accumulates in f64 and rounds each stored number to f32 (Δ, h between tokens, y), as PyTorch keeps them; explain([t, d])
// is the read y = Σ C·h + D·u from the traced previous state (also for the sub-steps #delta [t, d] and #h [t, d, n]).
// Python mirror: pipeline/tinymamba/ops_ssm_scan.py.
import type { AnyTensor, OpDef, Tensor, UnitExplanation } from '../../../atlas/types/index.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, needRank, outArray, t, unitBase, weightName } from '../../../atlas/ops/common.ts';

export type ScanP = { state: number };
export const softplus = (x: number) => (x > 20 ? x : Math.log1p(Math.exp(x)));
const rounder = (like: ArrayLike<number>) => (like instanceof Float64Array ? (v: number) => v : Math.fround);

/** One token s, one channel d: Δ, the N new state numbers (from the previous ones, hPrev(n)) and y, each rounded to the
 *  output precision as it is stored (PyTorch keeps h in float32 between tokens too). Shared by forward and explain. */
function cell(io: { xs: AnyTensor[]; Al: ArrayLike<number>; Dk: ArrayLike<number>; rd: (v: number) => number }, s: number, d: number, N: number, hPrev: (n: number) => number) {
	const [u, dt, B, C] = io.xs;
	const D = u.shape[1];
	const dl = io.rd(softplus(dt.data[s * D + d]));
	const x = u.data[s * D + d];
	const h: number[] = [];
	let sum = 0;
	for (let n = 0; n < N; n++) {
		h.push(io.rd(Math.exp(-dl * Math.exp(io.Al[d * N + n])) * hPrev(n) + dl * B.data[s * N + n] * x));
		sum += C.data[s * N + n] * h[n];
	}
	const skip = io.Dk[d] * x;
	return { dl, h, sum, skip, y: io.rd(sum + skip) };
}

export const ssm_scan: OpDef<ScanP> = {
	op: 'ssm_scan',
	version: 1,
	attrs: { state: { type: 'int', required: true, doc: 'N: state slots per channel' } },
	category: 'recurrent',
	label: t('Selective scan', '选择性扫描'),
	weights: [{ slot: 'A_log' }, { slot: 'D' }],
	subs: [
		{ name: 'delta', label: t('step size Δ', '步长 Δ'), axes: ['t', 'd'] },
		{ name: 'h', label: t('state h', '状态 h'), axes: ['t', 'd', 'n'] }
	],
	inferShape(ins, p, w) {
		const [u, dt, B, C] = ins;
		if (ins.length !== 4 || u.length !== 2) throw new Error('ssm_scan: inputs are u [T, D], dt [T, D], B [T, N], C [T, N]');
		if (dt[1] !== u[1] || B[1] !== p.state || C[1] !== p.state) throw new Error(`ssm_scan: dt must be [T, ${u[1]}], B and C [T, ${p.state}]`);
		if (w.A_log && (w.A_log[0] !== u[1] || w.A_log[1] !== p.state)) throw new Error(`ssm_scan: A_log must be [${u[1]}, ${p.state}]`);
		return [u[0], u[1]];
	},
	axes: () => ['t', 'd'],
	forward(xs, p, ctx) {
		const [u, , , C] = xs;
		needRank(u, 2, 'ssm_scan u');
		const [T, D] = u.shape;
		const N = p.state;
		const Y = outArray(T * D, u);
		const Dl = outArray(T * D, u);
		const H = outArray(T * D * N, u);
		const io = { xs, Al: ctx.w('A_log').data, Dk: ctx.w('D').data, rd: rounder(Y) };
		for (let s = 0; s < T; s++)
			for (let d = 0; d < D; d++) {
				const r = cell(io, s, d, N, (n) => (s ? H[((s - 1) * D + d) * N + n] : 0));
				Dl[s * D + d] = r.dl;
				r.h.forEach((v, n) => (H[(s * D + d) * N + n] = v));
				Y[s * D + d] = r.y;
			}
		ctx.emit('delta', { shape: [T, D], data: Dl });
		ctx.emit('h', { shape: [T, D, N], data: H });
		return { shape: [T, D], data: Y };
	},
	unit: { axes: (out) => out.slice(), scan: 'time', pick: 'last' },
	explain(q): UnitExplanation {
		const [s, d, k = 0] = q.unit;
		const [u, , , C] = q.ins;
		const D = u.shape[1];
		const N = q.p.state;
		const hId = `${q.layer.id}#h`;
		const hT = q.trace.t[hId] as Tensor | undefined;
		if (!hT) throw new Error(`ssm_scan explain: the trace has no ${hId}`);
		const io = { xs: q.ins, Al: q.w('A_log').data, Dk: q.w('D').data, rd: rounder(q.out.data) };
		const r = cell(io, s, d, N, (n) => (s ? hT.data[((s - 1) * D + d) * N + n] : 0));
		const at = (n: number) => (s * D + d) * N + n;
		const terms = r.h.map((x, n) => ({ x, w: C.data[s * N + n], p: x * C.data[s * N + n], xAt: at(n), wAt: s * N + n }));
		const e = unitBase('weighted-sum', q.layer, [s, d], r.y, q.out.data[s * D + d]);
		e.reads = [{ tensor: hId, cells: terms.map((x) => x.xAt) }, { tensor: q.layer.in[3], cells: terms.map((x) => x.wAt) }, { tensor: weightName(q.layer, 'D'), cells: [d] }];
		e.terms = terms;
		e.bias = r.skip;
		e.values = { n: N, s: r.sum, b: r.skip, y: r.y, delta: r.dl, h: r.h[k], t: s, d, tNo: s + 1, dNo: d + 1 };
		return e;
	},
	formula: `${sym('out', 'y')}_t = \\textstyle\\sum_n ${sym('param', 'C')}_{t,n}\\, ${sym('act', 'h')}_{t,n} + D\\,u_t = {{s}} + {{b}} = {{y}}`,
	narration: {
		what: t(
			'The selective scan reads the tokens one at a time and keeps a state of {state|int} numbers per channel. Each token decides, through its step size Δ, how much of the old state to keep and how much of itself to write in.',
			'选择性扫描一次读一个词元，为每个通道保留一份 {state|int} 个数的状态。每个词元通过自己的步长 Δ 决定保留多少旧状态、写入多少自己。'
		),
		how: t(
			'For each channel: h = exp(Δ·A)·h_prev + Δ·B·u, then the output is y = Σ C·h + D·u. A is negative, so a big Δ forgets the old state and writes the new token; a Δ near 0 keeps the state as it is.',
			'对每个通道：h = exp(Δ·A)·h_prev + Δ·B·u，输出是 y = Σ C·h + D·u。A 是负数，所以 Δ 大时忘掉旧状态、写入新词元；Δ 接近 0 时状态保持不变。'
		),
		look: t('Follow one channel: its state changes at some tokens and stays put at others.', '跟踪一个通道：它的状态在某些词元处改变，在另一些词元处保持不动。'),
		unit: t(
			'At token {t|int}, channel {d|int} (counting from 0) reads its {n|int} state numbers with C: the products add up to {s}, plus D·u = {b}, gives {y}.',
			'在第 {t|int} 个词元、通道 {d|int}（从 0 数起），用 C 读出它的 {n|int} 个状态数：乘积之和是 {s}，加上 D·u = {b}，得到 {y}。'
		)
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), state: Number((layer.params as Partial<ScanP>)?.state ?? NaN) }),
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'h = torch.exp(delta[:, None] * A) * h + delta[:, None] * B[None] * u[:, None];  y = (h * C).sum(-1) + D * u',
	terms: ['hidden-state']
};

export default ssm_scan;
