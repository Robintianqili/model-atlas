// View 'number-formats/gpt': TinyGPT with its weight matrices cast to a format (process 'sweep'), at the step's frame:
// the setting, the perplexity on the evaluation text against fp32, the mean KL divergence, the top-1 agreement and the
// size of the weights; then the probe prompt's five most likely next tokens in fp32, each with its probability in fp32
// and after the cast. Props:
//   table?: boolean     a table of every frame (the frame's row outlined); the step's cursor should be the last frame
//   controls?: boolean  buttons for every frame of the run (the concept page's real-model panel)
//   probe?: boolean     the probe's bars (default true)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { Frame } from '../../../../atlas/types/process.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { PROBE_TEXT, VOCAB } from '../../evalset.ts';
import { CASTS } from '../../process.ts';
import { NAMES } from '../bits/meta.ts';

/** What a sweep frame did: "fp32 (as trained)", "FP8 e4m3, one scale per tensor", ... */
export function configText(f: Frame | undefined): I18n {
	const m = f?.metrics;
	if (!f || !m) return { en: '', zh: '' };
	const c = CASTS[m.k];
	if (f.phase === 'fp32' || !c) return { en: 'fp32 (as trained)', zh: 'fp32（原模型）' };
	const name = NAMES[c.fmt];
	const fmt = { en: name.en.replace(/ \(.*\)$/, ''), zh: (name.zh ?? name.en).replace(/（.*）$/, '') };
	if (c.scale === 'tensor') return { en: `${fmt.en}, one scale per matrix`, zh: `${fmt.zh}，每个矩阵一个缩放比例` };
	if (c.scale === 'block') return { en: `${fmt.en}, a scale per block of ${c.block}`, zh: `${fmt.zh}，每 ${c.block} 个一块、各配缩放比例` };
	return { en: `${fmt.en}, no scale`, zh: `${fmt.zh}，不缩放` };
}

/** A token's text for display (spaces and line breaks made visible). */
export const tokText = (id: number) => (VOCAB[id] ?? `#${id}`).replace(/\n/g, '↵').replace(/^ /, '␣');

export function describeGpt(f: Frame | undefined, top: number[], p0: number[], p: number[]): I18n {
	const m = f?.metrics;
	if (!m) return { en: 'TinyGPT is not ready yet.', zh: 'TinyGPT 还没有准备好。' };
	const c = configText(f);
	const bars = top.map((id, i) => `${tokText(id)} ${fmt3(p0[i])} → ${fmt3(p[i])}`).join('; ');
	return {
		en: `TinyGPT, ${c.en}: perplexity ${fmt3(m.ppl)} (fp32 ${fmt3(m.ppl0)}), KL ${fmt3(m.kl)}, top-1 agreement ${fmt3(m.agree * 100)}%, ${fmt3(m.bpw)} bits per weight. After "${PROBE_TEXT}": ${bars}.`,
		zh: `TinyGPT，${c.zh}：困惑度 ${fmt3(m.ppl)}（fp32 为 ${fmt3(m.ppl0)}），KL ${fmt3(m.kl)}，top-1 一致率 ${fmt3(m.agree * 100)}%，每个权重 ${fmt3(m.bpw)} 位。在“${PROBE_TEXT}”之后：${bars}。`
	};
}

const arr = (t: unknown) => Array.from(((t as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);
export function probeOf(at: StageProps['at']) {
	const s = (at?.state ?? {}) as Record<string, unknown>;
	return { top: arr(s.top), p0: arr(s.p0), p: arr(s.p) };
}

const meta = {
	key: 'number-formats/gpt',
	title: { en: 'TinyGPT with its weights in other formats', zh: '权重换成其他格式的 TinyGPT' },
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
		tape.push({ phase: 'cast', step: 3, state: { top: v([1, 2, 3, 4, 5]), p0: v([0.5, 0.2, 0.1, 0.05, 0.02]), p: v([0.3, 0.3, 0.1, 0.05, 0.05]), hist: v([1, 2, 3]) }, metrics: { k: 3, bits: 8, bpw: 8, ppl0: 6, ppl: 8, kl: 0.3, agree: 0.7, dmax: 5, bytes: 1.4e6, bytes0: 1.46e6 } } as never);
		const rt = fakeRuntime({ slug: 'number-formats', input: {}, tapes: { sweep: tape } });
		return { rt, source: 'sweep', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
