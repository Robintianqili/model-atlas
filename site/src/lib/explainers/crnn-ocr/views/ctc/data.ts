// crnn-ocr/ctc view data (pure TS: the view, its describe() and the lesson read it). One strip of writing, its 32
// columns' chances and what CTC makes of them:
//   decode  the top symbol of each column up to the step's column (the decode process frame), merged and dropped
//   paths   a path the learner edits column by column (starts as the top symbols), its chance and its text
//   chances the chances of every symbol in every column, as a grid under the strip
import type { ExplainerRuntime, GraphModel, StageProps, Tensor } from '../../../../atlas/types/index.ts';
import { bestPath, collapse, forward, idsOf, pathLog, probsOf, textOf, type Probs } from '../../ctc.ts';
import adapter, { presetRaw, textOf as wroteOf } from '../../adapters/word-strokes/adapter.ts';

export type Mode = 'decode' | 'paths' | 'chances';
export type Strip = { ink: Tensor; q: Probs; best: number[]; upTo: number; text: string; wrote: string; mode: Mode };

const memo = new WeakMap<GraphModel, Map<string, { ink: Tensor; probs: Tensor }>>();
/** the model on a preset instead of the learner's input (the limits step), memoized per model */
export function variant(rt: Pick<ExplainerRuntime, 'model'>, id: string): { ink: Tensor; probs: Tensor } | null {
	const m = rt.model;
	if (!m) return null;
	let byId = memo.get(m);
	if (!byId) memo.set(m, (byId = new Map()));
	let v = byId.get(id);
	if (!v) {
		const ink = adapter.preprocess(presetRaw(id), m.spec, {}).tensors.input as Tensor;
		v = { ink, probs: m.forward({ input: ink }, { record: 'layers' }).t.probs };
		byId.set(id, v);
	}
	return v;
}

/** the strip a step shows: props.variant (a preset id) or the learner's input; null before the model has run */
export function stripOf(P: Pick<StageProps, 'rt' | 'at' | 'props'>): Strip | null {
	const mode = (P.props?.mode as Mode) ?? 'decode';
	const v = typeof P.props?.variant === 'string' ? variant(P.rt, P.props.variant) : null;
	const ink = v?.ink ?? (P.rt.pre?.empty ? undefined : (P.rt.pre?.tensors.input as Tensor | undefined));
	const q = probsOf(v?.probs ?? P.rt.sources.fwd?.trace?.t.probs);
	if (!ink || !q) return null;
	const best = bestPath(q);
	const f = P.at?.frame;
	const upTo = f?.phase === 'read' && typeof f.metrics?.t === 'number' ? f.metrics.t : q.T - 1;
	return { ink, q, best, upTo, text: textOf(collapse(best)), wrote: v ? '' : wroteOf(P.rt.input), mode };
}

/** a path's text, its log chance, and the total log chance and path count of that text */
export function pathFacts(s: Strip, path: number[]) {
	const text = textOf(collapse(path));
	const fw = forward(s.q, idsOf(text));
	return { text, logP: pathLog(s.q, path), logText: fw.logP, paths: fw.paths };
}

export const chance = (s: Strip, t: number, k: number) => s.q.p[t * s.q.V + k];
