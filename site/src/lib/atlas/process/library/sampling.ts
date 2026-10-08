// Sampling the next token, with TinyGPT semantics (models/tinygpt.ts generate/samplingDistribution, copied, not imported):
//   softmax(logits / temperature) in float64 -> top-k -> top-p -> renormalize; ties broken by the smaller id;
//   u = rng.next() (only next(), so Python reproduces it bit for bit); the token is the first i with cumulative p > u.
//   temperature <= 0 is greedy: argmax, ties to the smaller id, no random number drawn.
import type { I18n, Tensor } from '../../types/core.ts';
import type { Control, Frame, ProcessDef, RunCtx } from '../../types/process.ts';

export function samplingDistribution(logits: ArrayLike<number>, temperature: number, topK = 0, topP = 1): Float64Array {
	const V = logits.length;
	const p = new Float64Array(V);
	let m = -Infinity;
	for (let i = 0; i < V; i++) if (logits[i] / temperature > m) m = logits[i] / temperature;
	let s = 0;
	for (let i = 0; i < V; i++) {
		p[i] = Math.exp(logits[i] / temperature - m);
		s += p[i];
	}
	for (let i = 0; i < V; i++) p[i] /= s;
	if ((topK > 0 && topK < V) || topP < 1) {
		const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
		let keep = topK > 0 && topK < V ? topK : V;
		if (topP < 1) {
			let cum = 0;
			let n = 0;
			while (n < keep) {
				cum += p[idx[n]];
				n++;
				if (cum >= topP) break;
			}
			keep = Math.max(1, n);
		}
		for (let r = keep; r < V; r++) p[idx[r]] = 0;
		let z = 0;
		for (let i = 0; i < V; i++) z += p[i];
		for (let i = 0; i < V; i++) p[i] /= z;
	}
	return p;
}

/** The first token whose cumulative probability passes u (tokens with p = 0 are skipped). */
export function pickToken(dist: ArrayLike<number>, u: number): number {
	let token = -1;
	let cum = 0;
	for (let i = 0; i < dist.length; i++) {
		if (dist[i] === 0) continue;
		cum += dist[i];
		token = i;
		if (cum > u) break;
	}
	return token;
}

export function argmax(x: ArrayLike<number>): number {
	let best = 0;
	for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
	return best;
}

export function topEntries(p: ArrayLike<number>, n: number): { id: number; p: number }[] {
	const idx = Array.from({ length: p.length }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
	return idx.slice(0, n).map((id) => ({ id, p: p[id] }));
}

export type SamplingOpt = { maxNew: number; temperature: number; topK: number; topP: number; stopId: number; topN: number };
export type SamplingInput = { ids: number[] };
export type SamplingState = {
	ids: number[];
	logits?: Tensor;
	dist?: Tensor;
	token?: number;
	p?: number;
	pModel?: number;
	u?: number;
	top?: { id: number; p: number }[];
	contextStart?: number;
};

const L = (en: string, zh: string): I18n => ({ en, zh });

/** A sampling process over any next-token model. Coarse: one 'sample' frame per token. Fine: 'logits' -> 'distribution'
 *  -> 'sample' per token. A 'set' control (e.g. temperature) applies from the next token on. */
export function samplingProcess(def: {
	id: string;
	title?: I18n;
	/** logits for the next token after `ids` (the last `context` ids are passed) */
	logits: (ids: number[]) => ArrayLike<number>;
	context?: number;
	defaults?: Partial<SamplingOpt>;
}): ProcessDef<SamplingInput, SamplingOpt, SamplingState> {
	return {
		id: def.id,
		kind: 'inference',
		title: def.title ?? L('Sampling one token at a time', '一次采样一个词元'),
		phases: {
			prompt: { label: L('The prompt', '提示'), kind: 'input' },
			logits: { label: L('Scores for every token', '每个词元的分数'), kind: 'forward' },
			distribution: { label: L('Turn scores into chances', '把分数变成概率'), kind: 'other' },
			sample: { label: L('Pick one token', '选出一个词元'), kind: 'sample' }
		},
		schema: {
			ids: { kind: 'json', label: L('Token ids so far', '目前的词元 id'), tol: 'exact', role: 'data' },
			logits: { kind: 'tensor', label: L('Scores (logits)', '分数（logits）'), scale: 'diverging', role: 'activation', tol: 1e-5 },
			dist: { kind: 'tensor', label: L('Sampling distribution', '采样分布'), scale: 'sequential', role: 'output', tol: 1e-9 },
			token: { kind: 'int', label: L('Chosen token', '选中的词元'), tol: 'exact', role: 'output' },
			p: { kind: 'scalar', label: L('Its chance', '它的概率'), tol: 1e-9 },
			pModel: { kind: 'scalar', label: L('Its chance at temperature 1', '温度为 1 时的概率'), tol: 1e-9 },
			u: { kind: 'scalar', label: L('Random number (NaN when greedy)', '随机数（贪心时为 NaN）'), tol: 'exact' },
			top: { kind: 'json', label: L('Most likely tokens', '最可能的词元'), tol: 1e-9 },
			contextStart: { kind: 'int', label: L('First token the model saw', '模型看到的第一个词元'), tol: 'exact' }
		},
		defaults: { maxNew: 24, temperature: 1, topK: 0, topP: 1, stopId: -1, topN: 10, ...(def.defaults ?? {}) },
		*run(input: SamplingInput, ctx: RunCtx<SamplingOpt>): Generator<Frame<SamplingState>, void, Control<SamplingOpt> | undefined> {
			const ids = input.ids.slice();
			yield { phase: 'prompt', state: { ids: ids.slice() }, metrics: { length: ids.length } };
			const C = def.context ?? Infinity;
			for (let s = 0; s < ctx.opt.maxNew && !ctx.signal.cancelled; s++) {
				const start = Math.max(0, ids.length - C);
				const logits = def.logits(ids.slice(start));
				const V = logits.length;
				const lt: Tensor = { shape: [V], data: Float32Array.from(logits) };
				const { temperature, topK, topP, topN } = ctx.opt;
				const pModel = samplingDistribution(logits, 1);
				if (ctx.detail === 'fine') {
					const c = yield { phase: 'logits', step: s, state: { logits: lt }, metrics: { V } };
					if (c?.type === 'set') Object.assign(ctx.opt, c.opt);
				}
				let token: number;
				let dist: Float64Array;
				let u = NaN;
				if (temperature <= 0) {
					token = argmax(logits);
					dist = new Float64Array(V);
					dist[token] = 1;
				} else {
					dist = samplingDistribution(logits, temperature, topK, topP);
					if (ctx.detail === 'fine') {
						const c = yield { phase: 'distribution', step: s, state: { dist: { shape: [V], data: dist } }, metrics: { temperature } };
						if (c?.type === 'set') Object.assign(ctx.opt, c.opt);
					}
					u = ctx.rng.next();
					token = pickToken(dist, u);
				}
				ids.push(token);
				const shown = temperature <= 0 ? pModel : dist;
				const c = yield {
					phase: 'sample',
					step: s,
					state: { ids: ids.slice(), token, p: dist[token], pModel: pModel[token], u, top: topEntries(shown, topN), contextStart: start },
					metrics: { p: dist[token], pModel: pModel[token], u, bits: -Math.log2(pModel[token]) }
				};
				if (c?.type === 'set') Object.assign(ctx.opt, c.opt);
				if (token === ctx.opt.stopId) break;
			}
		}
	};
}
