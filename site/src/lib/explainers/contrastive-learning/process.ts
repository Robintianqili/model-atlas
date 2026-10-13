// contrastive-learning processes (kind C):
//   nce       the learner's batch (batch.ts batchIn: N image and N caption vectors from the shipped TinyCLIP, and τ) ->
//             the N×N cosines and scores cos / τ, the row and column softmax, the symmetric InfoNCE loss and its
//             gradient on the scores (kit autograd). Python twin: pipeline/contrastive-learning/reference.py (float64).
//   progress  the same batch through each checkpoint of the training re-run (batch.ts progressIn; models by reference,
//             run on the main thread): one frame per checkpoint with the cosines and the loss at that checkpoint's own
//             learned temperature.
//   simclr    SimCLR's version: two randomly distorted views of every drawing (augment()), the shipped TinyCLIP's image
//             tower as the encoder, the 2N×2N cosines, the loss over the 2N views (NT-Xent) at the learner's τ, and one
//             drawing moved to the other side (the invariance step).
import type { Frame, GraphModel, ProcessDef, ProcessRun, Rng, Tensor } from '../../atlas/types/index.ts';
import { backward, input as inputVar } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { infoNCE } from '../../atlas/autograd/loss.ts';
import { render, type Pt } from './scenes.ts';
import { imageVec, pairVecs, scaleIn, type BatchIn } from './batch.ts';
import { cosines, diagOf, lossAt, mean, offMax, offOf, wins } from './data.ts';
import { fmtInt } from '../../atlas/core/format.ts';

const I18 = (en: string, zh: string) => ({ en, zh });

type NceState = { C: Tensor; L: Tensor; I: Tensor; T: Tensor; P: Tensor; Q: Tensor; G: Tensor };
const mat = (R: number, K: number, d: ArrayLike<number>): Tensor => ({ shape: [R, K], data: Float64Array.from(d) });

export const nce: ProcessDef<BatchIn | null, Record<string, never>, NceState> = {
	id: 'nce',
	kind: 'training',
	title: I18('Score a batch and its loss', '给一批图文对打分并算损失'),
	phases: {
		scores: { label: I18('Cosines and scores of every pair', '每一对的余弦和分数'), kind: 'forward', formula: { tex: '\\cos = \\textstyle\\sum_d a_d\\, b_d = {{cos}} \\qquad \\cos / \\tau = {{cos}} / {{tau}} = {{l}}', values: (f) => ({ cos: f.metrics!.cos00, tau: f.metrics!.tau, l: f.metrics!.l00 }) } },
		softmax: { label: I18('Softmax along rows and columns', '沿行和列做 softmax'), kind: 'loss' },
		grad: { label: I18('Gradient on the scores', '分数上的梯度'), kind: 'backward' }
	},
	schema: {
		C: { kind: 'tensor', label: I18('Cosines (drawing i with caption j)', '余弦（画 i 与描述 j）'), role: 'activation', tol: 1e-12 },
		L: { kind: 'tensor', label: I18('Scores: cosine ÷ τ', '分数：余弦 ÷ τ'), role: 'activation', tol: 1e-12 },
		I: { kind: 'tensor', label: I18('Image vectors (length 1)', '图像向量（长度 1）'), role: 'data', tol: 1e-12 },
		T: { kind: 'tensor', label: I18('Caption vectors (length 1)', '描述向量（长度 1）'), role: 'data', tol: 1e-12 },
		P: { kind: 'tensor', label: I18('Row softmax: which caption fits each drawing', '行 softmax：每幅画配哪句描述'), role: 'output', tol: 1e-12 },
		Q: { kind: 'tensor', label: I18('Column softmax: which drawing fits each caption', '列 softmax：每句描述配哪幅画'), role: 'output', tol: 1e-12 },
		G: { kind: 'tensor', label: I18('Gradient of the loss on each score', '损失对每个分数的梯度'), role: 'grad', tol: 1e-12 }
	},
	defaults: {},
	*run(b): ProcessRun<NceState, Record<string, never>> {
		if (!b || b.N < 2) return;
		const { N, D, tau } = b;
		const C = cosines(b.I, b.T, N, N, D);
		const L = inputVar({ shape: [N, N], data: Float64Array.from(C, (c) => c / tau) }, true);
		const h = offMax(C, N);
		yield {
			phase: 'scores',
			state: { C: mat(N, N, C), L: mat(N, N, L.data), I: mat(N, D, b.I), T: mat(N, D, b.T) },
			metrics: { N, D, tau, cos00: C[0], l00: L.data[0], diagMean: mean(diagOf(C, N)), offMean: mean(offOf(C, N)), hard: h.v, hardI: h.i, hardJ: h.j }
		};
		const P = op.softmax(L, 1).data, Q = op.softmax(L, 0).data;
		const loss = infoNCE(L);
		const w = wins(C, N);
		yield {
			phase: 'softmax',
			state: { P: mat(N, N, P), Q: mat(N, N, Q) },
			metrics: { p00: P[0], q00: Q[0], l0: -Math.log(P[0]), rowLoss: -mean(diagOf(P, N).map(Math.log)), colLoss: -mean(diagOf(Q, N).map(Math.log)), loss: loss.data[0], chance: Math.log(N), rowRight: w.rows, colRight: w.cols }
		};
		backward(loss);
		const G = L.grad!.data;
		const g = offMax(G, N);
		yield { phase: 'grad', state: { G: mat(N, N, G) }, metrics: { g00: G[0], gOffMax: g.v, gOffI: g.i, gOffJ: g.j } } as Frame<NceState>;
	}
};

// ------------------------------------------------------------------ progress: the batch through the training checkpoints
export type ProgressIn = { pairs: { strokes: Pt[][]; caption: string }[]; cks: { update: number; model: GraphModel }[] };
type ProgressState = { C: Tensor };
export const progress: ProcessDef<ProgressIn | null, Record<string, never>, ProgressState> = {
	id: 'progress',
	kind: 'training',
	title: I18('Your batch through a training run', '你的这一批在训练过程中的样子'),
	phases: { ck: { label: I18('A checkpoint of the run', '训练中的一个检查点'), kind: 'update' } },
	schema: { C: { kind: 'tensor', label: I18('Cosines (drawing i with caption j)', '余弦（画 i 与描述 j）'), role: 'activation', tol: 1e-9 } },
	defaults: {},
	*run(b): ProcessRun<ProgressState, Record<string, never>> {
		if (!b || b.pairs.length < 2) return;
		const N = b.pairs.length;
		const last = b.cks[b.cks.length - 1].update;
		for (let k = 0; k < b.cks.length; k++) {
			const { update, model } = b.cks[k];
			const v = b.pairs.map((p) => pairVecs(model, p.strokes, p.caption));
			const C = cosines(v.flatMap((x) => x[0]), v.flatMap((x) => x[1]), N, N, v[0][0].length);
			const s = scaleIn(model);
			const w = wins(C, N);
			yield {
				phase: 'ck',
				step: k,
				// the scrubber's label of this frame: the training update it was saved after
				caption: update === 0 ? I18('Before training (update 0)', '训练前（第 0 次更新）') : I18(`After ${fmtInt(update)} of ${fmtInt(last)} updates`, `${fmtInt(last)} 次更新中的第 ${fmtInt(update)} 次之后`),
				state: { C: mat(N, N, C) },
				metrics: { update, last, N, scale: s, tau: 1 / s, loss: lossAt(C, N, 1 / s), right: w.rows, diag: mean(diagOf(C, N)), off: mean(offOf(C, N)), chance: Math.log(N) }
			};
		}
	}
};

// ------------------------------------------------------------------ simclr: two views of each drawing
/** A random distortion about the drawing's own centre: a turn of up to 0.3 rad, a stretch of 0.85 to 1.15, a shift of up
 *  to 0.05 and a little jitter on every point; points stay inside the frame. */
export function augment(strokes: Pt[][], r: Rng): Pt[][] {
	const all = strokes.flat();
	if (!all.length) return strokes;
	const u = (s: number) => (r.next() * 2 - 1) * s;
	const cx = (Math.min(...all.map((p) => p.x)) + Math.max(...all.map((p) => p.x))) / 2;
	const cy = (Math.min(...all.map((p) => p.y)) + Math.max(...all.map((p) => p.y))) / 2;
	const th = u(0.3), sc = 1 + u(0.15), dx = u(0.05), dy = u(0.05);
	const c = Math.cos(th), s = Math.sin(th);
	const clip = (v: number) => Math.min(0.98, Math.max(0.02, v));
	return strokes.map((line) => line.map((p) => {
		const x = p.x - cx, y = p.y - cy;
		return { x: clip(cx + dx + sc * (c * x - s * y) + u(0.006)), y: clip(cy + dy + sc * (s * x + c * y) + u(0.006)) };
	}));
}
/** The drawing moved sideways into the other half of the frame (to the left when it sits in the middle). */
export function moveAcross(strokes: Pt[][]): Pt[][] {
	const all = strokes.flat();
	if (!all.length) return strokes;
	const x0 = Math.min(...all.map((p) => p.x)), x1 = Math.max(...all.map((p) => p.x));
	const cx = (x0 + x1) / 2;
	const want = cx > 0.6 ? 0.3 : cx < 0.4 ? 0.7 : 0.25;
	const d = Math.min(0.98 - x1, Math.max(0.02 - x0, want - cx));
	return strokes.map((line) => line.map((p) => ({ x: p.x + d, y: p.y })));
}
export type SimclrIn = { strokes: Pt[][][]; model: GraphModel; tau: number };
type SimclrState = { V: Tensor; S: Tensor; P: Tensor; A: Tensor; B: Tensor };
/** view k's partner: views 2i and 2i + 1 are the two views of drawing i */
export const partner = (k: number) => k ^ 1;
export const simclr: ProcessDef<SimclrIn | null, Record<string, never>, SimclrState> = {
	id: 'simclr',
	kind: 'training',
	title: I18('Two views of every drawing (SimCLR)', '每幅画的两个视图（SimCLR）'),
	phases: {
		views: { label: I18('Two distorted views per drawing, and their cosines', '每幅画两个变形视图，以及它们的余弦'), kind: 'forward' },
		loss: { label: I18('Softmax over the other views and the loss', '在其他视图上做 softmax，并算损失'), kind: 'loss' },
		shift: { label: I18('The same drawing, moved across', '同一幅画，移到另一边'), kind: 'forward' }
	},
	schema: {
		V: { kind: 'tensor', label: I18('The views (28×28 each)', '视图（每个 28×28）'), role: 'data', tol: 'exact' },
		S: { kind: 'tensor', label: I18('Cosines of every two views', '每两个视图的余弦'), role: 'activation', tol: 1e-9 },
		P: { kind: 'tensor', label: I18('Softmax of each row over the other views', '每行在其他视图上的 softmax'), role: 'output', tol: 1e-9 },
		A: { kind: 'tensor', label: I18('Drawing 0 as drawn', '画 0 的原样'), role: 'data', tol: 'exact' },
		B: { kind: 'tensor', label: I18('Drawing 0 moved across', '画 0 移到另一边'), role: 'data', tol: 'exact' }
	},
	defaults: {},
	*run(b, ctx): ProcessRun<SimclrState, Record<string, never>> {
		if (!b || b.strokes.length < 2) return;
		const N = b.strokes.length, M = 2 * N;
		const V = new Float32Array(M * 784);
		b.strokes.forEach((st, i) => {
			V.set(render(augment(st, ctx.rng.fork(`view${i}a`))), 2 * i * 784);
			V.set(render(augment(st, ctx.rng.fork(`view${i}b`))), (2 * i + 1) * 784);
		});
		const Z = Array.from({ length: M }, (_, k) => imageVec(b.model, V.subarray(k * 784, (k + 1) * 784)));
		const S = cosines(Z.flat(), Z.flat(), M, M, Z[0].length);
		const pos = Array.from({ length: M }, (_, k) => S[k * M + partner(k)]);
		let neg = { v: -Infinity, i: 0, j: 0 };
		for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) if (j !== i && j !== partner(i) && S[i * M + j] > neg.v) neg = { v: S[i * M + j], i, j };
		yield { phase: 'views', state: { V: { shape: [M, 28, 28], data: V }, S: mat(M, M, S) }, metrics: { N, M, posMean: mean(pos), pos0: S[1], negMax: neg.v, negI: neg.i, negJ: neg.j } };
		// NT-Xent: each view's row of cos / τ without the view itself; its partner is the right answer
		const P = new Float64Array(M * M);
		let loss = 0, right = 0;
		for (let k = 0; k < M; k++) {
			let mx = -Infinity;
			for (let j = 0; j < M; j++) if (j !== k) mx = Math.max(mx, S[k * M + j] / b.tau);
			let z = 0;
			for (let j = 0; j < M; j++) if (j !== k) z += Math.exp(S[k * M + j] / b.tau - mx);
			for (let j = 0; j < M; j++) P[k * M + j] = j === k ? 0 : Math.exp(S[k * M + j] / b.tau - mx) / z;
			loss -= Math.log(P[k * M + partner(k)]) / M;
			let ok = true;
			for (let j = 0; j < M; j++) if (j !== k && j !== partner(k) && S[k * M + j] >= S[k * M + partner(k)]) ok = false;
			right += +ok;
		}
		yield { phase: 'loss', state: { P: mat(M, M, P) }, metrics: { tau: b.tau, loss, chance: Math.log(M - 1), right, p01: P[1] } };
		const A = render(b.strokes[0]), B = render(moveAcross(b.strokes[0]));
		const za = imageVec(b.model, A), zb = imageVec(b.model, B);
		yield { phase: 'shift', state: { A: { shape: [28, 28], data: A }, B: { shape: [28, 28], data: B } }, metrics: { shift0: za.reduce((s, x, d) => s + x * zb[d], 0), jitter0: S[1] } } as Frame<SimclrState>;
	}
};
