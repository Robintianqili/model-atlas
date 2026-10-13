// View 'kv-cache/cost': the work of every pass of TinyGPT's generation (process 'gen'), without and with the KV cache: the
// multiply-adds of each pass (or, with the toggle, all passes so far), one line per run, the frame's pass marked. Props:
//   memory?: boolean        also the cache's size: layers × heads × tokens × head size × 2 × bytes, for TinyGPT now and at
//                           its full 64-token window, and for GPT-2 small and Llama 2 7B at their windows (facts.json)
//   cumulative?: boolean    start with the running totals (default: per pass)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { Frame } from '../../../../atlas/types/process.ts';
import type { ExplainerRuntime } from '../../../../atlas/types/runtime.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtBytes, fmtInt } from '../../../../atlas/core/format.ts';

export type MemRow = { label: I18n; L: number; H: number; n: number; dh: number; bytes: number; total: number; live: boolean };

/** The memory table: TinyGPT now and at its full window (live), GPT-2 small and Llama 2 7B (facts.json, fp16). */
export function memRows(rt: ExplainerRuntime, m: Record<string, number> | undefined): MemRow[] {
	if (!m) return [];
	const f = (k: string) => rt.num(`fact:${k}`);
	const row = (label: I18n, L: number, H: number, n: number, dh: number, bytes: number, live: boolean): MemRow => ({ label, L, H, n, dh, bytes, total: L * H * n * dh * 2 * bytes, live });
	return [
		row({ en: 'TinyGPT, this pass (fp32)', zh: 'TinyGPT，这一遍（fp32）' }, m.L, m.H, m.n, m.dh, 4, true),
		row({ en: 'TinyGPT, full window (fp32)', zh: 'TinyGPT，满窗口（fp32）' }, m.L, m.H, m.C, m.dh, 4, true),
		row({ en: 'GPT-2 small, full window (fp16)', zh: 'GPT-2 small，满窗口（fp16）' }, f('gpt2Layers'), f('gpt2Heads'), f('gpt2Ctx'), f('gpt2DHead'), f('halfBytes'), false),
		row({ en: 'Llama 2 7B, full window (fp16)', zh: 'Llama 2 7B，满窗口（fp16）' }, f('llamaLayers'), f('llamaHeads'), f('llamaCtx'), f('llamaDHead'), f('halfBytes'), false)
	];
}

export function describeCost(f: Frame | undefined, last: Frame | undefined): I18n {
	const m = f?.metrics;
	const z = last?.metrics;
	if (!m || !z) return { en: 'TinyGPT has not generated anything yet.', zh: 'TinyGPT 还没有生成任何内容。' };
	return {
		en: `Multiply-adds per pass: at pass ${m.k}, ${fmtInt(m.macsFull)} without the cache and ${fmtInt(m.macsCache)} with it. All ${z.maxNew} passes: ${fmtInt(z.cumFull)} against ${fmtInt(z.cumCache)}. The cache holds ${fmtBytes(m.bytes)}.`,
		zh: `每一遍的乘加次数：在第 ${m.k} 遍，不用缓存 ${fmtInt(m.macsFull)} 次，用缓存 ${fmtInt(m.macsCache)} 次。全部 ${z.maxNew} 遍合计：${fmtInt(z.cumFull)} 次对 ${fmtInt(z.cumCache)} 次。缓存占 ${fmtBytes(m.bytes)}。`
	};
}

const meta = {
	key: 'kv-cache/cost',
	title: { en: 'Work per new token, and the cache’s memory', zh: '每个新词元的计算量，以及缓存的内存' },
	kinds: ['process'],
	describe: (p: StageProps) => describeCost(p.at?.frame, p.at?.tape?.length ? p.at.tape.frame(p.at.tape.length - 1) : undefined),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const { TapeStore } = await import('../../../../atlas/process/tape.ts');
		const { gen } = await import('../../process.ts');
		const tape = new TapeStore({ process: 'gen', schema: gen.schema, phases: gen.phases, provenance: 'illustrative' });
		const st = { ids: { shape: [64], data: new Float64Array(64).fill(-1) }, att: { shape: [4, 64, 64], data: new Float64Array(4 * 64 * 64) } };
		let cf = 0;
		let cc = 0;
		for (let k = 0; k < 6; k++) {
			const mf = 4e6 + k * 3e5;
			const mc = k ? 3.6e5 : mf;
			cf += mf;
			cc += mc;
			tape.push({ phase: k ? 'decode' : 'prefill', step: k, state: st, metrics: { k, P: 10, n: 10 + k, maxNew: 6, macsFull: mf, macsCache: mc, cumFull: cf, cumCache: cc, bytes: 608 * (10 + k), L: 4, H: 4, dh: 19, C: 64 } } as never);
		}
		const rt = fakeRuntime({ slug: 'kv-cache', input: {}, tapes: { gen: tape } });
		return { rt, source: 'gen', at: { index: 3, frame: tape.frame(3), state: tape.state(3), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
