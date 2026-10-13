// View 'quantization/gpt': TinyGPT with quantized weights (process 'sweep'), at the step's frame: which weights are
// quantized and how, the perplexity on the evaluation text against fp32, the mean KL divergence, the top-1 agreement and
// the model's size; then the probe prompt's five most likely next tokens in fp32, each with its probability (and logit)
// in fp32 and quantized. Props:
//   table?: 'layer' | 'all'   a table of every frame of that phase (the frame's row outlined); the step's cursor should be
//                             the phase's last frame so that every row exists
//   hist?: boolean            the histogram of the quantized layer's weights (the trace source 'gpt' holds them), with the
//                             levels of a per-tensor scale at the frame's bit width
//   controls?: boolean        buttons for every frame of the run (the concept page's real-model panel)
//   probe?: boolean           the probe's bars (default true)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { Frame } from '../../../../atlas/types/process.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { PROBE_TEXT, VOCAB } from '../../evalset.ts';

const GRAN: I18n[] = [
	{ en: 'per tensor', zh: '按张量' },
	{ en: 'per channel', zh: '按通道' },
	{ en: 'per block of 32', zh: '每 32 个一块' }
];

/** What a sweep frame quantized: "fp32", "int4 per channel, one layer", ... */
export function configText(f: Frame | undefined): I18n {
	const m = f?.metrics;
	if (!f || !m) return { en: '', zh: '' };
	if (f.phase === 'fp32') return { en: 'fp32 (as trained)', zh: 'fp32（原模型）' };
	const g = GRAN[m.gran] ?? GRAN[0];
	const scope = f.phase === 'layer' ? { en: 'one layer (L0.mlp.fc)', zh: '只量化一层（L0.mlp.fc）' } : { en: 'every weight matrix', zh: '所有权重矩阵' };
	return { en: `int${m.bits} ${g.en}, ${scope.en}`, zh: `int${m.bits} ${g.zh}，${scope.zh}` };
}

/** A token's text for display (spaces and line breaks made visible). */
export const tokText = (id: number) => (VOCAB[id] ?? `#${id}`).replace(/\n/g, '↵').replace(/^ /, '␣');

export function describeGpt(f: Frame | undefined, top: number[], p0: number[], p: number[]): I18n {
	const m = f?.metrics;
	if (!m) return { en: 'TinyGPT is not ready yet.', zh: 'TinyGPT 还没有准备好。' };
	const c = configText(f);
	const bars = top.map((id, i) => `${tokText(id)} ${fmt3(p0[i])} → ${fmt3(p[i])}`).join('; ');
	return {
		en: `TinyGPT, ${c.en}: perplexity ${fmt3(m.ppl)} (fp32 ${fmt3(m.ppl0)}), KL ${fmt3(m.kl)}, top-1 agreement ${fmt3(m.agree * 100)}%, size ${fmt3(m.bytes / 1e6)} MB. After "${PROBE_TEXT}": ${bars}.`,
		zh: `TinyGPT，${c.zh}：困惑度 ${fmt3(m.ppl)}（fp32 为 ${fmt3(m.ppl0)}），KL ${fmt3(m.kl)}，top-1 一致率 ${fmt3(m.agree * 100)}%，大小 ${fmt3(m.bytes / 1e6)} MB。在“${PROBE_TEXT}”之后：${bars}。`
	};
}

const arr = (t: unknown) => Array.from(((t as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);
export function probeOf(at: StageProps['at']) {
	const s = (at?.state ?? {}) as Record<string, unknown>;
	return { top: arr(s.top), p0: arr(s.p0), p: arr(s.p), l0: arr(s.l0), l: arr(s.l) };
}

const meta = {
	key: 'quantization/gpt',
	title: { en: 'TinyGPT with quantized weights', zh: '权重量化后的 TinyGPT' },
	kinds: ['process'],
	describe: (p: StageProps) => {
		const q = probeOf(p.at);
		return describeGpt(p.at?.frame, q.top, q.p0, q.p);
	},
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const { TapeStore } = await import('../../../../atlas/process/tape.ts');
		const { sweep } = await import('../../process.ts');
		const tape = new TapeStore({ process: 'sweep', schema: sweep.schema, phases: sweep.phases, provenance: 'illustrative' });
		const v = (a: number[]) => ({ shape: [a.length], data: Float64Array.from(a) });
		tape.push({ phase: 'layer', step: 1, state: { top: v([1, 2, 3, 4, 5]), p0: v([0.5, 0.2, 0.1, 0.05, 0.02]), p: v([0.3, 0.3, 0.1, 0.05, 0.05]), l0: v([9, 8, 7, 6, 5]), l: v([8, 8, 7, 6, 6]) }, metrics: { bits: 4, gran: 0, ppl0: 6, ppl: 8, kl: 0.3, agree: 0.7, dmax: 5, bytes: 1.4e6, bytes0: 1.46e6 } } as never);
		const rt = fakeRuntime({ slug: 'quantization', input: {}, tapes: { sweep: tape } });
		return { rt, source: 'sweep', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
