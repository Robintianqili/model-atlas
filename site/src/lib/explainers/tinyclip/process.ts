// tinyclip processes (kind C):
//   synth  frame 0: the fixed drawings (presets, partner pairs, edge cases); then the training set: random scenes as strokes, rendered by render() (drawing28), template captions as word ids.
//          One frame per chunk of `chunk` pairs; pair i draws only from ctx.rng.fork(`s${i}`), so the first 12 pairs are
//          the same whatever n is. `npm run atlas -- run tinyclip synth --opt '{"n":24000}' --save` writes the set that
//          pipeline/tinyclip/train.py reads (it checks every chunk's `hash` on the bytes it read).
//   nce    a batch of N image vectors and N caption vectors (batch.ts, from the exported model) -> the N×N scores, the
//          row and column softmax, the InfoNCE loss and its gradient on the scores (kit autograd). Python twin:
//          pipeline/tinyclip/reference.py (torch float64).
import type { Frame, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { backward, input as inputVar } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { infoNCE } from '../../atlas/autograd/loss.ts';
import { FIXED, MAX_WORDS, captionOf, crcBytes, randomScene, render, strokesFor, strokesOf, tokenize } from './scenes.ts';

const I18 = (en: string, zh: string) => ({ en, zh });

export type SynthOpt = { n: number; chunk: number; twoShare: number };
type I32 = { shape: number[]; data: Int32Array; dtype: 'i32' };
type SynthState = { images: Tensor; ids: I32; scenes: I32; captions: string[] | string[][] };

export const synth: ProcessDef<unknown, SynthOpt, SynthState> = {
	id: 'synth',
	kind: 'algorithm',
	title: I18('Make the training pairs', '生成训练用的图文对'),
	phases: { fixed: { label: I18('The fixed drawings', '固定的画'), kind: 'input' }, chunk: { label: I18('A chunk of pairs', '一批图文对'), kind: 'input' } },
	schema: {
		images: { kind: 'tensor', label: I18('Drawings (28×28 each)', '画（每幅 28×28）'), role: 'data', tol: 'exact' },
		ids: { kind: 'tensor', label: I18('Caption word ids (0 pads)', '描述的词编号（0 是填充）'), role: 'data', tol: 'exact' },
		scenes: { kind: 'tensor', label: I18('Scene codes', '场景编码'), role: 'target', tol: 'exact' },
		captions: { kind: 'json', label: I18('Captions', '描述'), role: 'data' }
	},
	defaults: { n: 12, chunk: 1000, twoShare: 0.25 },
	*run(_input, ctx): ProcessRun<SynthState, SynthOpt> {
		const { n, chunk, twoShare } = ctx.opt;
		// frame 0: the presets, the page's partner pairs and the edge cases (export.py's reference cases), up to 3 captions each
		const F = FIXED.length;
		const fi = new Float32Array(F * 784), fids = new Int32Array(F * 3 * MAX_WORDS);
		FIXED.forEach((f, k) => {
			fi.set(render(strokesFor(f)), k * 784);
			f.captions.forEach((c, j) => fids.set(tokenize(c).ids, (k * 3 + j) * MAX_WORDS));
		});
		yield { phase: 'fixed', state: { images: { shape: [F, 784], data: fi }, ids: { shape: [F, 3, MAX_WORDS], data: fids, dtype: 'i32' }, captions: FIXED.map((f) => f.captions) }, metrics: { count: F, hash: crcBytes(fi) } };
		for (let from = 0, k = 0; from < n && !ctx.signal.cancelled; from += chunk, k++) {
			const c = Math.min(chunk, n - from);
			const images = new Float32Array(c * 784);
			const ids = new Int32Array(c * MAX_WORDS);
			const scenes = new Int32Array(c * 5).fill(-1);
			const captions: string[] = [];
			for (let i = 0; i < c; i++) {
				const r = ctx.rng.fork(`s${from + i}`);
				const sc = randomScene(r, twoShare);
				images.set(render(strokesOf(sc, r)), i * 784);
				const cap = captionOf(sc, r);
				captions.push(cap);
				ids.set(tokenize(cap).ids, i * MAX_WORDS);
				scenes.set(sc.kind === 'one' ? [0, sc.shape, sc.size, sc.place] : [1, sc.a, sc.b, sc.rel], i * 5);
			}
			yield {
				phase: 'chunk',
				step: k,
				state: { images: { shape: [c, 784], data: images }, ids: { shape: [c, MAX_WORDS], data: ids, dtype: 'i32' }, scenes: { shape: [c, 5], data: scenes, dtype: 'i32' }, captions },
				metrics: { from, count: c, hash: crcBytes(images) }
			};
		}
	}
};

/** nce input: N image vectors and N caption vectors (row i of each is pair i), D numbers each, and the learned scale. */
export type BatchIn = { I: number[]; T: number[]; N: number; D: number; scale: number };
type NceState = { L: Tensor; P: Tensor; Q: Tensor; G: Tensor };
const mat = (N: number, d: ArrayLike<number>): Tensor => ({ shape: [N, N], data: Float64Array.from(d) });

export const nce: ProcessDef<BatchIn | null, Record<string, never>, NceState> = {
	id: 'nce',
	kind: 'training',
	title: I18('Score a batch and its loss', '给一批图文对打分并算损失'),
	phases: {
		scores: { label: I18('Scores of every pair', '每一对的分数'), kind: 'forward', formula: { tex: 's \\cdot \\cos = {{scale}} \\times {{cos}} = {{l}}', values: (f) => ({ scale: f.metrics!.scale, cos: f.metrics!.cos00, l: f.metrics!.l00 }) } },
		softmax: { label: I18('Softmax along rows and columns', '沿行和列做 softmax'), kind: 'loss' },
		grad: { label: I18('Gradient on the scores', '分数上的梯度'), kind: 'backward' }
	},
	schema: {
		L: { kind: 'tensor', label: I18('Scores (scale × cosine)', '分数（缩放 × 余弦）'), role: 'activation', tol: 1e-12 },
		P: { kind: 'tensor', label: I18('Row softmax: which caption fits each drawing', '行 softmax：每幅画配哪句描述'), role: 'output', tol: 1e-12 },
		Q: { kind: 'tensor', label: I18('Column softmax: which drawing fits each caption', '列 softmax：每句描述配哪幅画'), role: 'output', tol: 1e-12 },
		G: { kind: 'tensor', label: I18('Gradient of the loss on each score', '损失对每个分数的梯度'), role: 'grad', tol: 1e-12 }
	},
	defaults: {},
	*run(b): ProcessRun<NceState, Record<string, never>> {
		if (!b || b.N < 1) return;
		const { N, D, scale } = b;
		const cos = new Float64Array(N * N);
		for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
			let s = 0;
			for (let d = 0; d < D; d++) s += b.I[i * D + d] * b.T[j * D + d];
			cos[i * N + j] = s;
		}
		const L = inputVar({ shape: [N, N], data: Float64Array.from(cos, (c) => scale * c) }, true);
		const diag = (a: ArrayLike<number>) => Array.from({ length: N }, (_, i) => a[i * N + i]);
		const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
		const off = Array.from(L.data).filter((_, k) => k % (N + 1) !== 0);
		yield { phase: 'scores', state: { L: mat(N, L.data) }, metrics: { N, D, scale, cos00: cos[0], l00: L.data[0], diagMean: mean(diag(L.data)), offMean: off.length ? mean(off) : 0 } };
		const P = op.softmax(L, 1).data, Q = op.softmax(L, 0).data;
		const rowLoss = -mean(diag(P).map(Math.log)), colLoss = -mean(diag(Q).map(Math.log));
		const loss = infoNCE(L);
		yield { phase: 'softmax', state: { P: mat(N, P), Q: mat(N, Q) }, metrics: { p00: P[0], q00: Q[0], rowLoss, colLoss, loss: loss.data[0], chance: Math.log(N) } };
		backward(loss);
		const G = L.grad!.data;
		const offG = Array.from(G).filter((_, k) => k % (N + 1) !== 0);
		yield { phase: 'grad', state: { G: mat(N, G) }, metrics: { g00: G[0], gMin: Math.min(...diag(G)), gOffMax: offG.length ? Math.max(...offG) : 0, gSum: Array.from(G).reduce((x, y) => x + y, 0) } } as Frame<NceState>;
	}
};
