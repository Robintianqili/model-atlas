// tinymamba: process 'scan' (kind C): one block's selective scan worked out token by token, in float64 from the exported
// weights (mamba.ts), one frame per token. The state h [D, N] is stored whole; the metrics hold the arithmetic of one
// channel `ch` and one slot `slot` (chosen by the source's input function, scanInput in lesson.ts), so the narration
// and the formula can quote h_new = exp(Δ·A)·h_old + Δ·B·u with real numbers.
// pipeline/tinymamba/reference.py is the independent Python run (atlas parity: the same frames, to 1e-12).
//   npm run atlas -- run tinymamba scan --frames 0..8
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { sym } from '../../atlas/core/tex.ts';
import { blocks, sizes, step, type Weights } from './mamba.ts';
import { ASK, isLetter } from './task.ts';

export type ScanInput = { ids: number[]; w: Weights | null; block: number; ch: number; slot: number };
export type ScanState = { h: Tensor };
type Run = ProcessRun<ScanState, Record<string, never>>;

const L = (en: string, zh: string) => ({ en, zh });

export const scan: ProcessDef<ScanInput, Record<string, never>, ScanState> = {
	id: 'scan',
	kind: 'inference',
	title: L('The scan, one token at a time', '扫描，一次一个词元'),
	phases: {
		start: { label: L('Empty state', '空状态'), kind: 'input', narration: { what: L('Before the first token the state is all zeros.', '读第一个词元之前，状态全是 0。') } },
		token: {
			label: L('Read one token', '读一个词元'),
			kind: 'forward',
			formula: {
				tex: `\\begin{aligned} ${sym('act', 'h')}_{\\text{new}} &= e^{${sym('accent', '\\Delta')} A}\\, ${sym('act', 'h')}_{\\text{old}} + ${sym('accent', '\\Delta')}\\, ${sym('param', 'B')}\\, ${sym('in', 'u')} \\\\ &= e^{ {{delta}} \\times ({{A}}) } \\times {{hOld}} + {{delta}} \\times {{B}} \\times {{u}} \\\\ &= {{decay}} \\times {{hOld}} + {{write}} = {{hNew}} \\end{aligned}`,
				values: (f) => f.metrics ?? {}
			},
			narration: { what: L('Token {frame.t|int} (counting from 0): Δ = {frame.delta}, so the old state is kept × {frame.decay} and {frame.write} is written.', '第 {frame.t|int} 个词元（从 0 数起）：Δ = {frame.delta}，所以旧状态乘以 {frame.decay} 保留下来，再写入 {frame.write}。') }
		}
	},
	schema: { h: { kind: 'tensor', label: L('state h (channels × slots)', '状态 h（通道 × 槽）'), shape: ['D', 'N'], scale: 'diverging', tol: 1e-12 } },
	defaults: {},
	*run(input): Run {
		if (!input.w) throw new Error('tinymamba scan: the model is not loaded (npm run atlas -- export tinymamba)');
		const { D, N } = sizes(input.w);
		const ids = input.ids;
		if (!ids.length) return;
		const r = blocks(input.w, ids)[input.block - 1];
		const h = new Float64Array(D * N);
		const c = input.ch;
		const k = c * N + input.slot;
		yield { phase: 'start', state: { h: { shape: [D, N], data: h.slice() } }, metrics: { T: ids.length, D, N, block: input.block, ch: c, slot: input.slot } };
		for (let t = 0; t < ids.length; t++) {
			const hOld = h[k];
			const y = step(r, t, h, N);
			const dl = r.delta[t][c];
			let mean = 0;
			for (let j = 0; j < D; j++) mean += r.delta[t][j] / D;
			const metrics = { t, id: ids[t], letter: isLetter(ids[t]) ? 1 : 0, ask: ids[t] === ASK ? 1 : 0, delta: dl, A: r.A[k], decay: Math.exp(dl * r.A[k]), B: r.B[t][input.slot], u: r.u[t][c],
				write: dl * r.B[t][input.slot] * r.u[t][c], hOld, hNew: h[k], C: r.C[t][input.slot], y: y[c], meanDelta: mean };

			yield { phase: 'token', step: t, state: { h: { shape: [D, N], data: h.slice() } }, metrics };
		}
	}
};
