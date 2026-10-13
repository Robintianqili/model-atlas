// decoding: the generating process (kind C), the kit's sampling process (atlas/process/library/sampling.ts: softmax in
// float64 at temperature T, top-k, then top-p, renormalize; u = rng.next(); the first token whose running total passes u;
// temperature 0 = greedy, no random number) wrapped around TinyGPT (static/models/tinygpt, read only) run by the kit's
// graph executor. One 'prompt' frame, then one 'sample' frame per new token: the ids so far, the token, its chance under
// the settings (p) and under the model (pModel, temperature 1), u, and the ten most likely tokens of the distribution it
// was drawn from (top). The input carries TinyGPT's spec and weights (copied from the trace source, explainer.ts gptInput),
// the prompt's ids and, for the learner's own run, the settings and the seed; without them the source's opt applies.
// It runs in the package worker.   npm run atlas -- run decoding greedy --frames 0..12
import type { AtlasSpec, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { samplingProcess, type SamplingOpt, type SamplingState } from '../../atlas/process/library/sampling.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { CONTEXT } from './decode.ts';

export type DecodeInput = {
	spec: AtlasSpec;
	weights: Record<string, Tensor>;
	ids: number[];
	temperature?: number;
	topK?: number;
	topP?: number;
	seed?: number;
} | null;

const shape = samplingProcess({ id: 'decode', logits: () => [] });
const L = (en: string, zh: string) => ({ en, zh });

export const decode: ProcessDef<DecodeInput, SamplingOpt, SamplingState> = {
	...shape,
	id: 'decode',
	title: L('TinyGPT writes one token at a time', 'TinyGPT 一次写一个词元'),
	defaults: { ...shape.defaults, maxNew: 48, topN: 10 },
	phases: {
		...shape.phases,
		prompt: { ...shape.phases.prompt, narration: { what: L('TinyGPT starts from your prompt, {frame.length|int} tokens.', 'TinyGPT 从你的提示出发，共 {frame.length|int} 个词元。') } },
		sample: { ...shape.phases.sample, narration: { what: L('It picks a token whose chance was {frame.p|pct}.', '它选出一个概率为 {frame.p|pct} 的词元。') } }
	},
	*run(input, ctx): ProcessRun<SamplingState, SamplingOpt> {
		if (!input || !input.weights || !input.ids?.length) return;
		const m = fromParts(input.spec, input.weights, CORE_OPS);
		const logits = (ids: number[]) => {
			const t = m.forward({ shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' }, { record: 'layers' }).t.logits;
			const n = t.shape[1];
			return (t.data as Float32Array).subarray((t.shape[0] - 1) * n, t.shape[0] * n);
		};
		for (const k of ['temperature', 'topK', 'topP'] as const) if (typeof input[k] === 'number') ctx.opt[k] = input[k]!;
		const rng = typeof input.seed === 'number' ? createRng(input.seed) : ctx.rng;
		yield* samplingProcess({ id: 'decode', context: CONTEXT, logits }).run({ ids: input.ids.slice(-CONTEXT) }, { ...ctx, rng });
	}
};
