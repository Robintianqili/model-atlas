// positional-encodings: the process 'pe' (kind C, inference): TinyGPT (static/models/tinygpt, read only; the kit's own
// forward pass) and the four encodings of pe.ts on the learner's text and positions. One frame per phase, in this order:
//   order     TinyGPT on the text as typed and shuffled (pe.ts shuffleOf: all tokens but the last), once with its
//             position table set to zero and once as trained: the last token's layer-0 attention weights (mean of the
//             4 heads), how much its layer-0 attention output and its last hidden state change, and the top next token
//   learned   TinyGPT's own position table (64 rows of 76) and how alike its rows are (cosine similarity)
//   sin       the sinusoidal table for 128 positions of width 76
//   rope      head 0 of TinyGPT's layer-0 query (the last token) and key (the first token), without positions, turned by
//             RoPE to the query position p + k and the key position p
//   relative  their head-0 dot product as the key moves along (query k positions later): with RoPE, and with TinyGPT's
//             learned rows added to the tokens instead
//   alibi     ALiBi's bias −m·distance for TinyGPT's 4 heads, distance 0 to 127
//   extrap    the precomputed sweep of extrap.ts (five TinyGPT-shaped models, trained at 64 tokens, scored at 256)
// Its input is TinyGPT's spec and weights (copied from the trace source 'gpt', explainer.ts) with the learner's settings,
// so it runs in the package worker.   npm run atlas -- run positional-encodings pe --frames 0..6
import type { AtlasSpec, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { BASE, cosRows, D, dot, freq, H, HD, project, rope, rowNorm, rowOf, ROT, shuffleOf, SIN_ROWS, sinTable, slope } from './pe.ts';
import { BIN, EVAL_CTX, LOSS, SCHEMES } from './extrap.ts';

export type PeInput = { spec: AtlasSpec; weights: Record<string, Tensor>; ids: number[]; cut: number; shuffle: number; p: number; k: number } | null;
const V = 1024;
export const ROWS = 64; // TinyGPT's position table
export const ALIBI_D = 128;
const vec = (a: ArrayLike<number>, shape: number[] = [a.length]): Tensor => ({ shape, data: Float64Array.from(a) });
const argmax = (a: ArrayLike<number>, off: number, n: number) => {
	let k = 0;
	for (let i = 1; i < n; i++) if (a[off + i] > a[off + k]) k = i;
	return k;
};
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>, off: number, n: number) => {
	let d = 0;
	for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(a[off + i] - b[off + i]));
	return d;
};

/** TinyGPT on ids: the last token's layer-0 weights (mean over heads), its layer-0 attention output (all heads), its
 *  last hidden state and the next-token distribution. */
function lastRow(model: ReturnType<typeof fromParts>, ids: number[]) {
	const T = ids.length;
	const t = model.forward({ shape: [T], data: Int32Array.from(ids), dtype: 'i32' }, { record: 'layers' }).t;
	const w = t['L0.attn.weights'].data;
	const mean = Float64Array.from({ length: T }, (_, j) => {
		let s = 0;
		for (let h = 0; h < H; h++) s += w[h * T * T + (T - 1) * T + j] / H;
		return s;
	});
	return { mean, heads: t['L0.attn.heads'].data, hidden: t['L3.resid2'].data, probs: t['probs'].data };
}

export const pe: ProcessDef<PeInput, Record<string, never>> = {
	id: 'pe',
	kind: 'inference',
	title: { en: 'Positions in TinyGPT and four encodings', zh: 'TinyGPT 中的位置与四种编码' },
	phases: {
		order: { label: { en: 'Shuffle the tokens', zh: '打乱词元' }, kind: 'forward' },
		learned: { label: { en: 'The learned table', zh: '可学习的位置表' }, kind: 'observe' },
		sin: { label: { en: 'The sinusoidal table', zh: '正弦位置表' }, kind: 'observe' },
		rope: { label: { en: 'RoPE turns a query and a key', zh: 'RoPE 旋转查询和键' }, kind: 'forward' },
		relative: { label: { en: 'Move both positions', zh: '同时移动两个位置' }, kind: 'forward' },
		alibi: { label: { en: 'ALiBi’s bias ramp', zh: 'ALiBi 的偏置斜坡' }, kind: 'observe' },
		extrap: { label: { en: 'Beyond the trained length', zh: '超出训练长度' }, kind: 'output' }
	},
	schema: {
		typed: { kind: 'tensor', label: { en: 'Token ids as typed', zh: '原顺序的词元 id' }, shape: ['T'], role: 'data', tol: 'exact' },
		shuffled: { kind: 'tensor', label: { en: 'Token ids shuffled', zh: '打乱后的词元 id' }, shape: ['T'], role: 'data', tol: 'exact' },
		w0: { kind: 'tensor', label: { en: 'Last token’s weights, no positions, as typed', zh: '最后一个词元的权重（无位置，原顺序）' }, shape: ['T'], scale: 'sequential', role: 'activation', tol: 1e-6 },
		w0s: { kind: 'tensor', label: { en: 'Last token’s weights, no positions, shuffled', zh: '最后一个词元的权重（无位置，打乱）' }, shape: ['T'], scale: 'sequential', role: 'activation', tol: 1e-6 },
		w1: { kind: 'tensor', label: { en: 'Last token’s weights, TinyGPT’s positions, as typed', zh: '最后一个词元的权重（TinyGPT 的位置，原顺序）' }, shape: ['T'], scale: 'sequential', role: 'activation', tol: 1e-6 },
		w1s: { kind: 'tensor', label: { en: 'Last token’s weights, TinyGPT’s positions, shuffled', zh: '最后一个词元的权重（TinyGPT 的位置，打乱）' }, shape: ['T'], scale: 'sequential', role: 'activation', tol: 1e-6 },
		table: { kind: 'tensor', label: { en: 'Position table (rows: positions)', zh: '位置表（每行一个位置）' }, shape: ['P', D], scale: 'diverging', role: 'param', tol: 1e-6 },
		sim: { kind: 'tensor', label: { en: 'Cosine similarity of row p with every row', zh: '第 p 行与每一行的余弦相似度' }, shape: [ROWS], role: 'output', tol: 1e-9 },
		q: { kind: 'tensor', label: { en: 'Query, head 0, no position', zh: '查询（头 0，无位置）' }, shape: [HD], role: 'activation', tol: 1e-6 },
		kv: { kind: 'tensor', label: { en: 'Key, head 0, no position', zh: '键（头 0，无位置）' }, shape: [HD], role: 'activation', tol: 1e-6 },
		qr: { kind: 'tensor', label: { en: 'Query turned to position p + k', zh: '转到位置 p + k 的查询' }, shape: [HD], role: 'activation', tol: 1e-6 },
		kr: { kind: 'tensor', label: { en: 'Key turned to position p', zh: '转到位置 p 的键' }, shape: [HD], role: 'activation', tol: 1e-6 },
		ropeBy: { kind: 'tensor', label: { en: 'RoPE dot product by key position', zh: '按键位置的 RoPE 点积' }, shape: [ROWS], role: 'output', tol: 1e-6 },
		absBy: { kind: 'tensor', label: { en: 'Learned-table dot product by key position', zh: '按键位置的可学习位置表点积' }, shape: [ROWS], role: 'output', tol: 1e-6 },
		bias: { kind: 'tensor', label: { en: 'ALiBi bias by head and distance', zh: '按头和距离的 ALiBi 偏置' }, shape: [H, ALIBI_D], scale: 'diverging', role: 'param', tol: 1e-12 },
		loss: { kind: 'tensor', label: { en: 'Loss by position bin, per encoding', zh: '各编码按位置区间的损失' }, shape: [SCHEMES.length, EVAL_CTX / BIN], role: 'output', tol: 1e-9 }
	},
	defaults: {},
	*run(input): ProcessRun<Record<string, Tensor>, Record<string, never>> {
		if (!input || !input.ids.length) return;
		const { spec, weights: W, ids, p, k } = input;
		const T = ids.length;
		// ---- order: as typed and shuffled, positions off (a zero table) and on
		const perm = shuffleOf(T, input.shuffle);
		const sh = perm.map((i) => ids[i]);
		const zero: Record<string, Tensor> = { ...W, 'pos_emb.weight': { shape: W['pos_emb.weight'].shape, data: new Float32Array(W['pos_emb.weight'].data.length) } };
		const m0 = fromParts(spec, zero, CORE_OPS);
		const m1 = fromParts(spec, W, CORE_OPS);
		const [a0, b0, a1, b1] = [lastRow(m0, ids), lastRow(m0, sh), lastRow(m1, ids), lastRow(m1, sh)];
		const L = (T - 1) * D;
		const top = (r: typeof a0) => argmax(r.probs, (T - 1) * V, V);
		const pr = (r: typeof a0) => r.probs[(T - 1) * V + top(r)];
		yield {
			phase: 'order',
			step: 0,
			state: { typed: vec(ids), shuffled: vec(sh), w0: vec(a0.mean), w0s: vec(b0.mean), w1: vec(a1.mean), w1s: vec(b1.mean) },
			metrics: {
				n: T,
				cut: input.cut,
				same: sh.every((v, i) => v === ids[i]) ? 1 : 0,
				d0: maxDiff(a0.heads, b0.heads, L, D),
				d1: maxDiff(a1.heads, b1.heads, L, D),
				dF0: maxDiff(a0.hidden, b0.hidden, L, D),
				rows: ROWS,
				top1: top(a1),
				p1: pr(a1),
				top1s: top(b1),
				p1s: pr(b1)
			}
		};
		// ---- learned: TinyGPT's table
		const P = W['pos_emb.weight'].data;
		const E = W['tok_emb.weight'].data;
		const meanCos = (dist: number) => {
			let s = 0;
			for (let i = 0; i + dist < ROWS; i++) s += cosRows(P, D, i, i + dist);
			return s / (ROWS - dist);
		};
		const inTable = p < ROWS;
		yield {
			phase: 'learned',
			step: 0,
			state: { table: vec(P, [ROWS, D]), sim: vec(Array.from({ length: ROWS }, (_, j) => (inTable ? cosRows(P, D, p, j) : NaN))) },
			metrics: {
				p,
				k,
				d: D,
				rows: ROWS,
				tokNorm: ids.reduce((s, id) => s + rowNorm(E, D, id), 0) / T,
				normP: inTable ? rowNorm(P, D, p) : NaN,
				c1: p + 1 < ROWS ? cosRows(P, D, p, p + 1) : NaN,
				ck: p + k < ROWS ? cosRows(P, D, p, p + k) : NaN,
				mean1: meanCos(1),
				meank: meanCos(k),
				far: meanCos(ROWS - 1)
			}
		};
		// ---- sin: the formula at p
		const last = D / 2 - 1;
		yield {
			phase: 'sin',
			step: 0,
			state: { table: vec(sinTable(), [SIN_ROWS, D]) },
			metrics: { p, d: D, base: BASE, s0: Math.sin(p), c0: Math.cos(p), last, wLast: freq(last, D), aLast: p * freq(last, D), turnLast: (2 * Math.PI) / freq(last, D) }
		};
		// ---- rope: head 0 of the last token's query and the first token's key, without positions
		const q = project(W, rowOf(E, ids[T - 1]), 'q').slice(0, HD);
		const kv = project(W, rowOf(E, ids[0]), 'k').slice(0, HD);
		const [mq, nk] = [p + k, p];
		const [qr, kr] = [rope(q, mq), rope(kv, nk)];
		yield {
			phase: 'rope',
			step: 0,
			state: { q: vec(q), kv: vec(kv), qr: vec(qr), kr: vec(kr) },
			metrics: { m: mq, n: nk, k, pairs: ROT / 2, thetaLast: freq(ROT / 2 - 1, ROT), pairRaw: dot(q, kv, 2), pair: dot(qr, kr, 2), raw: dot(q, kv), score: dot(qr, kr), score0: dot(rope(q, k), rope(kv, 0)) }
		};
		// ---- relative: the key at s, the query at s + k; RoPE against TinyGPT's learned rows
		const ropeBy = Array.from({ length: ROWS }, (_, s) => dot(rope(q, s + k), rope(kv, s)));
		const absAt = (s: number) => (s + k < ROWS ? dot(project(W, rowOf(E, ids[T - 1], P, s + k), 'q').slice(0, HD), project(W, rowOf(E, ids[0], P, s), 'k').slice(0, HD)) : NaN);
		const absBy = Array.from({ length: ROWS }, (_, s) => absAt(s));
		const ok = absBy.filter(Number.isFinite);
		// the shift the narration compares p with: key position 0, or (when p is 0) the last one that still fits
		const ref = p === 0 ? ROWS - 1 : 0;
		const refA = p === 0 ? ROWS - 1 - k : 0;
		yield {
			phase: 'relative',
			step: 0,
			state: { ropeBy: vec(ropeBy), absBy: vec(absBy) },
			metrics: { p, k, last: ROWS - 1, ropeP: dot(qr, kr), rope0: ropeBy[0], ref, ropeRef: ropeBy[ref], absP: absAt(p), abs0: absBy[0], refA, absRef: absAt(refA), absMin: Math.min(...ok), absMax: Math.max(...ok) }
		};
		// ---- alibi
		const bias = new Float64Array(H * ALIBI_D);
		const am: Record<string, number> = { k };
		for (let h = 0; h < H; h++) {
			const m = slope(h);
			for (let d = 0; d < ALIBI_D; d++) bias[h * ALIBI_D + d] = -m * d;
			Object.assign(am, { [`m${h}`]: m, [`b${h}`]: -m * k, [`r${h}`]: Math.exp(-m * k) });
		}
		yield { phase: 'alibi', step: 0, state: { bias: vec(bias, [H, ALIBI_D]) }, metrics: am };
		// ---- extrap: the precomputed sweep
		const nb = EVAL_CTX / BIN;
		const loss = Float64Array.from({ length: SCHEMES.length * nb }, (_, i) => LOSS[SCHEMES[Math.floor(i / nb)]][i % nb] ?? NaN);
		yield { phase: 'extrap', step: 0, state: { loss: { shape: [SCHEMES.length, nb], data: loss } }, metrics: { bins: nb, bin: BIN } };
	}
};
