// kv-cache: the processes (kind C), on TinyGPT's spec and weights (static/models/tinygpt, read only; gpt.ts runs them).
//   gen     prompt A continued greedily for NEW tokens, twice in lockstep: without a cache (every pass computes every token
//           again) and with a KV cache (a pass computes only the newest token). One frame per pass k (counting from 0):
//           'prefill' (k = 0, the prompt) and 'decode' (k ≥ 1). Metrics: the multiply-adds of both modes (this pass and so
//           far), the cache's size, the largest change of the earlier keys and values between two uncached passes (dKV),
//           the largest logit difference of the two modes (dLogit) and whether they chose the same token; the last frame
//           also compares every pass's logits with the kit's own forward pass (fromParts, parity-checked against PyTorch).
//           State: the token ids (prompt, then the new tokens; −1 after them) and every layer's attention weights averaged
//           over the heads ([L, C, C], filled by the cached run), shared by all frames: a frame shows the rows it has read.
//   prefix  prompts A and B: B's prefill from scratch and with A's cached prefix reused; one frame 'share'. Past the first
//           difference, the keys and values of the tokens both prompts have in the same place, layer by layer, and B's
//           next-token logits when such rows are (wrongly) taken from A's cache too.
//   npm run atlas -- run kv-cache gen --frames 0..30
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { argmax, cacheBytes, dimsOf, KVCache, maxDiff, run, softmaxAt, type Count, type Gpt } from './gpt.ts';
import { NEW, PROMPT_MAX, promptOf, samePlaces, sharedPrefix } from './data.ts';

export type GenInput = (Gpt & { ids: number[] }) | null;
export type PrefixInput = (Gpt & { a: number[]; b: number[] }) | null;
export type GenState = { ids: Tensor; att: Tensor };
export type PrefixState = { a: Tensor; b: Tensor; dK: Tensor; dV: Tensor };

const vec = (a: ArrayLike<number>, n = a.length, fill = -1): Tensor => {
	const d = new Float64Array(n).fill(fill);
	for (let i = 0; i < Math.min(n, a.length); i++) d[i] = a[i];
	return { shape: [n], data: d };
};

/** The kit's own forward pass (static/models/tinygpt is parity-checked against PyTorch) over a sequence: logits [T, V]. */
function kitLogits(g: Gpt, ids: number[]): { data: Float32Array; V: number } {
	const t = fromParts(g.spec, g.weights, CORE_OPS).forward({ shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' }, { record: 'layers' }).t.logits;
	return { data: t.data as Float32Array, V: t.shape[1] };
}

export const gen: ProcessDef<GenInput, { maxNew: number }, GenState> = {
	id: 'gen',
	kind: 'inference',
	title: { en: 'TinyGPT generating, with and without a KV cache', zh: 'TinyGPT 生成文本：有无 KV 缓存' },
	phases: {
		prefill: {
			label: { en: 'Prefill: read the prompt', zh: '预填充：读入提示词' },
			kind: 'forward',
			narration: { what: { en: 'The first pass reads all {frame.n} prompt tokens and stores their keys and values.', zh: '第一遍读入全部 {frame.n} 个提示词词元，并存下它们的键和值。' } }
		},
		decode: {
			label: { en: 'Decode: one new token', zh: '解码：一个新词元' },
			kind: 'forward',
			narration: { what: { en: 'Pass {frame.step} reads {frame.n} tokens: {frame.macsFull|int} multiply-adds without the cache, {frame.macsCache|int} with it.', zh: '第 {frame.step} 遍读入 {frame.n} 个词元：不用缓存要做 {frame.macsFull|int} 次乘加，用缓存只要 {frame.macsCache|int} 次。' } }
		}
	},
	schema: {
		ids: { kind: 'tensor', label: { en: 'Token ids: the prompt, then the new tokens', zh: '词元 id：先是提示词，然后是新词元' }, shape: [64], role: 'data', tol: 'exact' },
		att: { kind: 'tensor', label: { en: 'Attention weights per layer, averaged over the heads', zh: '每层的注意力权重（各头平均）' }, shape: [4, 64, 64], scale: 'sequential', role: 'output', tol: 1e-5 }
	},
	defaults: { maxNew: NEW },
	metricTol: { dLogit: 1e-5, dKit: 1e-4, dKV: 1e-6, p: 1e-5 },
	*run(input, ctx): ProcessRun<GenState, { maxNew: number }> {
		if (!input || !input.weights) return;
		const g: Gpt = { spec: input.spec, weights: input.weights };
		const d = dimsOf(g.spec);
		const seq = promptOf(input.ids);
		const P = seq.length;
		const maxNew = Math.max(1, Math.min(ctx.opt.maxNew ?? NEW, d.C - P));
		const att = new Float32Array(d.L * d.C * d.C);
		const cache = new KVCache(d);
		let prev: KVCache | null = null;
		let cumFull = 0;
		let cumCache = 0;
		let dLogitMax = 0;
		let same = 0;
		const rows: Float32Array[] = [];
		const frames: { phase: string; step: number; metrics: Record<string, number> }[] = [];
		for (let k = 0; k < maxNew; k++) {
			const n = seq.length;
			// without a cache: a fresh, empty cache every pass, so every token is computed again
			const full = new KVCache(d);
			const cf: Count = { macs: 0 };
			const lf = run(g, seq, full, cf);
			// with the cache: only the tokens it does not hold yet (all of the prompt at k = 0, then one)
			const cc: Count = { macs: 0 };
			const lc = run(g, seq, cache, cc, att);
			// the earlier tokens' keys and values: did the new token change them? (uncached pass k against pass k − 1)
			let dKV = 0;
			if (prev) for (let l = 0; l < d.L; l++) dKV = Math.max(dKV, maxDiff(prev.k[l], full.k[l], prev.n * d.D), maxDiff(prev.v[l], full.v[l], prev.n * d.D));
			prev = full;
			const token = argmax(lc);
			const tokFull = argmax(lf);
			const dLogit = maxDiff(lf, lc);
			dLogitMax = Math.max(dLogitMax, dLogit);
			if (token === tokFull) same++;
			cumFull += cf.macs;
			cumCache += cc.macs;
			rows.push(lc);
			frames.push({
				phase: k === 0 ? 'prefill' : 'decode',
				step: k,
				metrics: {
					k, P, n, token, tokFull, p: softmaxAt(lc, token), macsFull: cf.macs, macsCache: cc.macs, cumFull, cumCache,
					bytes: cacheBytes(d, cache.n), bytesFull: cacheBytes(d, d.C), dKV, dLogit, same: token === tokFull ? 1 : 0, L: d.L, H: d.H, dh: d.dh, C: d.C, maxNew
				}
			});
			seq.push(token);
		}
		// the kit's forward pass over every token read: row P − 1 + k holds pass k's next-token logits
		const kit = kitLogits(g, seq.slice(0, seq.length - 1));
		let dKit = 0;
		rows.forEach((r, k) => (dKit = Math.max(dKit, maxDiff(r, kit.data.subarray((P - 1 + k) * kit.V, (P + k) * kit.V)))));
		const state: GenState = { ids: vec(seq, d.C), att: { shape: [d.L, d.C, d.C], data: att } };
		for (let i = 0; i < frames.length; i++) {
			const f = frames[i];
			const last = i === frames.length - 1;
			yield { phase: f.phase, step: f.step, state, metrics: last ? { ...f.metrics, dKit, dLogitMax, sameAll: same, last: 1 } : f.metrics };
		}
	}
};

export const prefix: ProcessDef<PrefixInput, Record<string, never>, PrefixState> = {
	id: 'prefix',
	kind: 'inference',
	title: { en: 'Two prompts, one shared prefix', zh: '两个提示词，一个共享前缀' },
	phases: {
		share: {
			label: { en: 'Prompt B reuses prompt A’s cache', zh: '提示词 B 复用提示词 A 的缓存' },
			kind: 'forward',
			narration: { what: { en: 'Prompt B shares its first {frame.s} tokens with prompt A.', zh: '提示词 B 的前 {frame.s} 个词元与提示词 A 相同。' } }
		}
	},
	schema: {
		a: { kind: 'tensor', label: { en: 'Prompt A (token ids)', zh: '提示词 A（词元 id）' }, shape: [PROMPT_MAX], role: 'data', tol: 'exact' },
		b: { kind: 'tensor', label: { en: 'Prompt B (token ids)', zh: '提示词 B（词元 id）' }, shape: [PROMPT_MAX], role: 'data', tol: 'exact' },
		dK: { kind: 'tensor', label: { en: 'Largest key difference per layer, same token in the same place', zh: '同一位置同一词元，每层键的最大差异' }, shape: [4], role: 'output', tol: 1e-6 },
		dV: { kind: 'tensor', label: { en: 'Largest value difference per layer, same token in the same place', zh: '同一位置同一词元，每层值的最大差异' }, shape: [4], role: 'output', tol: 1e-6 }
	},
	defaults: {},
	metricTol: { dReuse: 1e-6, dWrong: 1e-5, pB: 1e-5, pWrong: 1e-5 },
	*run(input): ProcessRun<PrefixState, Record<string, never>> {
		if (!input || !input.weights) return;
		const g: Gpt = { spec: input.spec, weights: input.weights };
		const d = dimsOf(g.spec);
		const A = promptOf(input.a);
		const B = promptOf(input.b);
		const s = sharedPrefix(A, B);
		const reused = Math.min(s, B.length - 1); // B's last token is always computed: its logits give B's next token
		const ca = new KVCache(d);
		run(g, A, ca, { macs: 0 });
		const cb = new KVCache(d);
		const mb: Count = { macs: 0 };
		const lb = run(g, B, cb, mb);
		const mr: Count = { macs: 0 };
		const lr = run(g, B, KVCache.prefixOf(d, ca, reused), mr);
		// past the first difference: same token, same place, but a different past
		const same = samePlaces(A, B);
		const dK = new Float64Array(d.L);
		const dV = new Float64Array(d.L);
		for (let l = 0; l < d.L; l++)
			for (const j of same) {
				dK[l] = Math.max(dK[l], maxDiff(ca.k[l], cb.k[l], (j + 1) * d.D, j * d.D));
				dV[l] = Math.max(dV[l], maxDiff(ca.v[l], cb.v[l], (j + 1) * d.D, j * d.D));
			}
		// wrong reuse: B's cache with A's rows wherever the token matches (also after the difference), then B's last token
		const cw = KVCache.prefixOf(d, cb, B.length - 1);
		for (let l = 0; l < d.L; l++)
			for (let j = 0; j < B.length - 1; j++)
				if (j < A.length && A[j] === B[j]) {
					cw.k[l].set(ca.k[l].subarray(j * d.D, (j + 1) * d.D), j * d.D);
					cw.v[l].set(ca.v[l].subarray(j * d.D, (j + 1) * d.D), j * d.D);
				}
		const lw = run(g, B, cw, { macs: 0 });
		const tokB = argmax(lb);
		yield {
			phase: 'share',
			step: 0,
			state: { a: vec(A, PROMPT_MAX), b: vec(B, PROMPT_MAX), dK: { shape: [d.L], data: dK }, dV: { shape: [d.L], data: dV } },
			metrics: {
				PA: A.length, PB: B.length, s, reused, computed: B.length - reused, macsB: mb.macs, macsReuse: mr.macs, saved: mb.macs - mr.macs, bytesShared: cacheBytes(d, reused),
				dReuse: maxDiff(lb, lr), nSame: same.length, dK0: dK[0], dKdeep: Math.max(...Array.from(dK).slice(1)), dVdeep: Math.max(...Array.from(dV).slice(1)),
				dWrong: maxDiff(lb, lw), tokB, pB: softmaxAt(lb, tokB), pWrong: softmaxAt(lw, tokB), L: d.L
			}
		};
	}
};
