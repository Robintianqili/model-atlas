// attention-variants: the processes (kind C).
//   cache   the KV cache of the learner's configuration (data.ts, analytic): one frame 'cache' with the numbers per token and
//           block, per token and for the whole sequence, and the same for MHA (K = H) and MQA (K = 1).
//   share   TinyGPT (static/models/tinygpt, the kit's own forward pass) on the probe prompt with its keys and values shared
//           across query heads. Frame 'mha' (as trained), two 'shared' frames (only block opt.layer converted, to 2 and
//           to 1 key/value heads) and three 'eval' frames (every block converted to 4, 2 and 1 key/value heads, scored on
//           the four evaluation windows of evalset.ts, 256 next tokens). Converting = mean pooling the key and value
//           projections of each group of query heads (Ainslie et al. 2023); the browser copies the pooled rows to every
//           head of the group, which gives exactly grouped-query attention. State: the attention weights w and scores s
//           [4, T] of block opt.layer at the probe's last token (the query that predicts the next token).
//   npm run atlas -- run attention-variants share --frames 0..5
import type { AtlasSpec, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { EVAL_IDS, PROBE_IDS } from './evalset.ts';
import { configOf, kindOf, perBlock, perToken, scoreMacs, total, withKv } from './data.ts';

// ---------------------------------------------------------------- cache

export type CacheState = { per: Tensor };
/** Bytes per token for K = H (MHA), the learner's configuration and K = 1 (MQA), in that order. */
export const cache: ProcessDef<unknown, Record<string, never>, CacheState> = {
	id: 'cache',
	kind: 'algorithm',
	title: { en: 'The KV cache of a configuration', zh: '一种配置的 KV 缓存' },
	phases: {
		cache: {
			label: { en: 'Count the cache', zh: '计算缓存' },
			kind: 'output',
			narration: { what: { en: 'The cache holds {frame.total|int} bytes.', zh: '缓存一共占 {frame.total|int} 字节。' } }
		}
	},
	schema: { per: { kind: 'tensor', label: { en: 'Bytes per token: MHA, yours, MQA', zh: '每个词元的字节数：MHA、你的配置、MQA' }, shape: [3], role: 'output', tol: 'exact' } },
	defaults: {},
	*run(input): ProcessRun<CacheState, Record<string, never>> {
		const c = configOf(input);
		const mha = withKv(c, c.heads);
		const mqa = withKv(c, 1);
		yield {
			phase: 'cache',
			step: 0,
			state: { per: { shape: [3], data: Float64Array.of(perToken(mha), perToken(c), perToken(mqa)) } },
			metrics: {
				layers: c.layers,
				heads: c.heads,
				kvHeads: c.kvHeads,
				dHead: c.dHead,
				tokens: c.tokens,
				bytes: c.bytes,
				latent: c.latent,
				kind: kindOf(c),
				group: c.heads / c.kvHeads,
				perBlock: perBlock(c),
				perTokenNums: c.layers * perBlock(c),
				perToken: perToken(c),
				total: total(c),
				mhaPerToken: perToken(mha),
				mhaTotal: total(mha),
				mqaPerToken: perToken(mqa),
				mqaTotal: total(mqa),
				saving: perToken(mha) / perToken(c),
				macs: scoreMacs(c)
			}
		};
	}
};

// ---------------------------------------------------------------- share

export type GptInput = { spec: AtlasSpec; weights: Record<string, Tensor> } | null;
export type ShareOpt = { layer: number };
export type ShareState = { w: Tensor; s: Tensor };

export const H = 4;
export const KVS = [4, 2, 1];
export const FOCUS_LAYER = 2;

/** TinyGPT's weights with the key and value projections of the given blocks shared by groups of H / kv query heads: each
 *  group's rows (weight and bias) replaced by their mean, copied to every head of the group. */
export function pooled(weights: Record<string, Tensor>, kv: number, blocks: number[], dHead: number): Record<string, Tensor> {
	const out = { ...weights };
	const g = H / kv;
	for (const l of blocks)
		for (const nm of ['k', 'v'])
			for (const part of ['weight', 'bias']) {
				const key = `L${l}.attn.${nm}.${part}`;
				const t = weights[key];
				const n = dHead * (part === 'weight' ? t.shape[1] : 1);
				const src = t.data;
				const dst = new Float32Array(src.length);
				for (let G = 0; G < kv; G++)
					for (let i = 0; i < n; i++) {
						let s = 0;
						for (let j = 0; j < g; j++) s += src[(G * g + j) * n + i];
						for (let j = 0; j < g; j++) dst[(G * g + j) * n + i] = s / g;
					}
				out[key] = { shape: t.shape.slice(), data: dst };
			}
	return out;
}

function logSoftmax(row: ArrayLike<number>): Float64Array {
	let m = -Infinity;
	for (let i = 0; i < row.length; i++) m = Math.max(m, row[i]);
	let s = 0;
	for (let i = 0; i < row.length; i++) s += Math.exp(row[i] - m);
	const lse = m + Math.log(s);
	const out = new Float64Array(row.length);
	for (let i = 0; i < row.length; i++) out[i] = row[i] - lse;
	return out;
}
export const argmax = (a: ArrayLike<number>) => {
	let k = 0;
	for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
	return k;
};

type Model = ReturnType<typeof fromParts>;
/** One forward pass on the probe: the last row of block `layer`'s attention weights and scores ([H, T]), the block's
 *  mixed values (attn.heads) and the last position's log-probabilities. */
function probe(m: Model, layer: number) {
	const T = PROBE_IDS.length;
	const t = m.forward({ shape: [T], data: Int32Array.from(PROBE_IDS), dtype: 'i32' }, { record: 'layers' }).t;
	const rows = (x: Tensor) => {
		const out = new Float64Array(H * T);
		for (let h = 0; h < H; h++) for (let j = 0; j < T; j++) out[h * T + j] = x.data[h * T * T + (T - 1) * T + j];
		return out;
	};
	const V = t.logits.shape[1];
	return { w: rows(t[`L${layer}.attn.weights`]), s: rows(t[`L${layer}.attn.scores`]), heads: t[`L${layer}.attn.heads`], last: logSoftmax((t.logits.data as Float32Array).subarray((T - 1) * V, T * V)) };
}

export const share: ProcessDef<GptInput, ShareOpt, ShareState> = {
	id: 'share',
	kind: 'inference',
	title: { en: 'TinyGPT with shared keys and values', zh: '共享键和值的 TinyGPT' },
	phases: {
		mha: { label: { en: 'Multi-head attention, as trained', zh: '训练好的多头注意力' }, kind: 'forward', narration: { what: { en: 'Head 0 puts its largest weight, {frame.top0|pct}, on position {frame.arg0}.', zh: '头 0 把最大的权重 {frame.top0|pct} 放在位置 {frame.arg0} 上。' } } },
		shared: { label: { en: 'One block with shared keys and values', zh: '一个块共享键和值' }, kind: 'forward', narration: { what: { en: 'With {frame.kv} key/value heads, head 0 puts {frame.top0|pct} on position {frame.arg0}.', zh: '只有 {frame.kv} 个键值头时，头 0 把 {frame.top0|pct} 放在位置 {frame.arg0} 上。' } } },
		eval: { label: { en: 'Every block converted, scored', zh: '所有块都转换后评测' }, kind: 'output', narration: { what: { en: 'With {frame.kv} key/value heads in every block, the perplexity is {frame.ppl}.', zh: '每个块都只有 {frame.kv} 个键值头时，困惑度是 {frame.ppl}。' } } }
	},
	schema: {
		w: { kind: 'tensor', label: { en: 'Attention weights of the last token, per head', zh: '最后一个词元的注意力权重（每个头一行）' }, shape: [H, 'T'], scale: 'sequential', role: 'output', tol: 1e-4 },
		s: { kind: 'tensor', label: { en: 'Attention scores of the last token, per head', zh: '最后一个词元的注意力分数（每个头一行）' }, shape: [H, 'T'], role: 'output', tol: 1e-4 }
	},
	defaults: { layer: FOCUS_LAYER },
	metricTol: 1e-4,
	*run(input, ctx): ProcessRun<ShareState, ShareOpt> {
		if (!input || !input.weights) return;
		const { spec, weights } = input;
		const layer = ctx.opt.layer ?? FOCUS_LAYER;
		const dHead = weights['L0.attn.q.weight'].shape[0] / H;
		const nLayer = Object.keys(weights).filter((k) => /^L\d+\.ln1\.weight$/.test(k)).length;
		const T = PROBE_IDS.length;
		const model = (w: Record<string, Tensor>) => fromParts(spec, w, CORE_OPS);
		const base = probe(model(weights), layer);
		const top0 = argmax(base.last);
		const headsOf = (w: Float64Array) => {
			const m: Record<string, number> = {};
			for (let h = 0; h < H; h++) {
				const row = w.subarray(h * T, (h + 1) * T);
				m[`arg${h}`] = argmax(row);
				m[`top${h}`] = row[argmax(row)];
			}
			return m;
		};
		const state = (p: { w: Float64Array; s: Float64Array }) => ({ w: { shape: [H, T], data: p.w }, s: { shape: [H, T], data: p.s } });
		const common = { T, layer, dHead, nLayer };
		yield { phase: 'mha', step: 0, state: state(base), metrics: { ...common, kv: H, kvNums: 2 * H * dHead * T, ...headsOf(base.w), next: top0, nextP: Math.exp(base.last[top0]) } };

		const D = H * dHead;
		for (let k = 1; k < KVS.length; k++) {
			const kv = KVS[k];
			const p = probe(model(pooled(weights, kv, [layer], dHead)), layer);
			let dw = 0;
			for (let i = 0; i < p.w.length; i++) dw = Math.max(dw, Math.abs(p.w[i] - base.w[i]));
			let dn = 0;
			let n0 = 0;
			for (let i = 0; i < D; i++) {
				const a = p.heads.data[(T - 1) * D + i];
				const b = base.heads.data[(T - 1) * D + i];
				dn += (a - b) ** 2;
				n0 += b * b;
			}
			const next = argmax(p.last);
			yield {
				phase: 'shared',
				step: k - 1,
				state: state(p),
				metrics: { ...common, kv, kvNums: 2 * kv * dHead * T, ...headsOf(p.w), dwMax: dw, outChange: Math.sqrt(dn / n0), next, nextP: Math.exp(p.last[next]), next0P: Math.exp(p.last[top0]) }
			};
		}

		const tokens = EVAL_IDS.reduce((a, w) => a + w.length - 1, 0);
		const blocks = Array.from({ length: nLayer }, (_, i) => i);
		let lp0: Float64Array[][] = [];
		for (let k = 0; k < KVS.length; k++) {
			const kv = KVS[k];
			const m = model(kv === H ? weights : pooled(weights, kv, blocks, dHead));
			const lp = EVAL_IDS.map((win) => {
				const ids = win.slice(0, -1);
				const t = m.forward({ shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' }, { record: 'layers' }).t.logits;
				const V = t.shape[1];
				return ids.map((_, i) => logSoftmax((t.data as Float32Array).subarray(i * V, (i + 1) * V)));
			});
			if (k === 0) lp0 = lp;
			let nll = 0;
			let kl = 0;
			let agree = 0;
			EVAL_IDS.forEach((win, w) =>
				lp[w].forEach((row, t) => {
					nll -= row[win[t + 1]];
					const r0 = lp0[w][t];
					for (let j = 0; j < row.length; j++) kl += Math.exp(r0[j]) * (r0[j] - row[j]);
					if (argmax(row) === argmax(r0)) agree++;
				})
			);
			const p = probe(m, layer);
			const next = argmax(p.last);
			yield {
				phase: 'eval',
				step: k,
				state: state(p),
				metrics: { ...common, kv, kvNums: 2 * kv * dHead * T, ...headsOf(p.w), ppl: Math.exp(nll / tokens), kl: kl / tokens, agree: agree / tokens, tokens, next, nextP: Math.exp(p.last[next]), next0P: Math.exp(p.last[top0]), cacheBytes: 4 * 2 * kv * dHead * nLayer }
			};
		}
	}
};
