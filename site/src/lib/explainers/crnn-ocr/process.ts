// crnn-ocr: CTC greedy decoding as a process (kind C), on the per-column chances of the traced model (float32 values,
// worked in float64). One frame per column: its top symbol and what CTC does with it (a blank is dropped, a repeat of
// the column before is merged, anything else adds a character); a last frame with the text, the chance of the path and
// the total chance of the text over every path (ctc.ts forward). pipeline/crnn-ocr/reference.py writes the same frames
// with numpy and PyTorch's own CTC loss, and atlas parity compares them.
//   npm run atlas -- run crnn-ocr decode --frames 0..33
import type { I18n, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { bestPath, forward, idsOf, pathLog, type Probs } from './ctc.ts';
import { SYMBOLS } from './glyphs.ts';
import { textOf } from './adapters/word-strokes/adapter.ts';

export type DecodeInput = { T: number; V: number; probs: number[]; target: string };
export type DecodeState = { path: Tensor; text: string };
/** the chance and the CTC loss of the text the learner meant when it has a path, else of the text read */
export function lossOf(mt: Record<string, number>) {
	const lp = Number.isFinite(mt.logTarget) ? mt.logTarget : mt.logText;
	const p = Math.exp(lp);
	// near 1, enough decimals that 1 − p keeps 3 significant digits, so −log of the printed chance gives the printed loss
	const pY = p > 0.99 && p < 1 ? p.toFixed(Math.min(15, Math.ceil(-Math.log10(1 - p)) + 2)) : p < 1e-3 ? p.toExponential(3).replace(/e\+?(-?\d+)/, ' \\times 10^{$1}') : p.toPrecision(5);
	return { pY, loss: -lp, paths: Number.isFinite(mt.logTarget) ? mt.pathsTarget : mt.paths };
}
/** what CTC does with a column's top symbol */
export const ACT = { blank: 0, repeat: 1, emit: 2 } as const;

const L = (en: string, zh: string): I18n => ({ en, zh });

/** the process input: the traced chances (none while the pad is empty) and the text the learner says they wrote */
export function decodeInput(rt: { input: unknown; pre: { empty: boolean } | null; sources: Record<string, { trace?: { t: Record<string, Tensor> } | null } | undefined> }): DecodeInput {
	const t = rt.pre && !rt.pre.empty ? rt.sources.fwd?.trace?.t.probs : undefined;
	return t ? { T: t.shape[0], V: t.shape[1], probs: Array.from(t.data), target: textOf(rt.input) } : { T: 0, V: 0, probs: [], target: '' };
}

export const decode: ProcessDef<DecodeInput, Record<string, never>, DecodeState> = {
	id: 'decode',
	kind: 'algorithm',
	title: L('CTC greedy decoding, one column at a time', 'CTC 贪心解码，一次一列'),
	phases: {
		read: { label: L('Read a column', '读一列'), kind: 'output', narration: { what: L('Column {frame.t|int}: its top symbol has a chance of {frame.p|pct}.', '第 {frame.t|int} 列：排第一的符号概率是 {frame.p|pct}。') } },
		done: {
			label: L('The text', '读出的文字'),
			kind: 'output',
			// the CTC loss of the text the learner meant (else of the text read): −log of the sum over its paths π of Π p_t(π_t)
			formula: { tex: 'L = -\\log \\sum_{\\pi \\to y} \\prod_t p_t(\\pi_t) = -\\log {{pY}} = {{loss}}', values: (f) => lossOf(f.metrics!) },
			narration: { what: L('The {frame.T|int} columns collapse to {frame.n|int} characters.', '这 {frame.T|int} 列合并成 {frame.n|int} 个字符。') }
		}
	},
	schema: {
		path: { kind: 'tensor', label: L('top symbol of each column so far (−1: not read yet)', '到目前为止每列排第一的符号（−1：还没读）'), tol: 'exact' },
		text: { kind: 'text', label: L('text so far', '到目前为止的文字'), tol: 'exact' }
	},
	metricTol: { logText: 1e-9, logTarget: 1e-9 },
	defaults: {},
	*run(input): ProcessRun<DecodeState, Record<string, never>> {
		if (!input?.probs?.length) return;
		const q: Probs = { T: input.T, V: input.V, p: input.probs };
		const top = bestPath(q);
		const path = new Float64Array(q.T).fill(-1);
		let text = '';
		for (let t = 0; t < q.T; t++) {
			const k = top[t];
			const act = k === 0 ? ACT.blank : t > 0 && top[t - 1] === k ? ACT.repeat : ACT.emit;
			path[t] = k;
			if (act === ACT.emit) text += SYMBOLS[k];
			yield { phase: 'read', step: t, state: { path: { shape: [q.T], data: path.slice() }, text }, metrics: { t, k, p: q.p[t * q.V + k], act, n: text.length } };
		}
		const fw = forward(q, idsOf(text));
		const tg = input.target ? forward(q, idsOf(input.target)) : null;
		const target: Record<string, number> = tg && tg.paths > 0 ? { logTarget: tg.logP, pathsTarget: tg.paths } : {};
		yield { phase: 'done', step: q.T, state: { path: { shape: [q.T], data: path.slice() }, text }, metrics: { T: q.T, n: text.length, logPath: pathLog(q, top), logText: fw.logP, paths: fw.paths, ...target } };
	}
};
