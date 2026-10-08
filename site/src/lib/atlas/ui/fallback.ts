// Pure helpers behind FallbackView: the generic stage StageHost shows when a step's view key is not registered in this
// build (a lane without W1-D2's zoom views, or a typo that lint reports as AK405). It reads only the At and the runtime:
// the frame's tensor, the unit explanation, or the frame's scalars and metrics.
import type { I18n, Tensor, Trace } from '../types/core.ts';
import type { UnitExplanation } from '../types/op.ts';
import type { StageProps } from '../types/view.ts';
import type { ExplainerRuntime } from '../types/runtime.ts';
import { fmt3, shapeText } from '../core/format.ts';
import { outputId, outputLabel, outputProbs, outputRow, vocabLabel } from '../lesson/facts.ts';
import type { Preprocessed } from '../types/runtime.ts';
import { tokenizerOf } from '../adapters/text-bpe/adapter.ts';

const isTensor = (v: unknown): v is Tensor => !!v && typeof v === 'object' && Array.isArray((v as Tensor).shape) && ArrayBuffer.isView((v as Tensor).data);

export interface FallbackModel {
	kind: 'unit' | 'tensor' | 'frame' | 'empty';
	layer?: string;
	unit?: UnitExplanation;
	/** the layer's whole output (unit and tensor kinds) */
	out?: Tensor;
	/** the tensor the unit reads from, with its box */
	read?: { id: string; t: Tensor; box?: { y0: number; x0: number; y1: number; x1: number }; channel: number };
	/** scalars of a process / train frame */
	scalars?: { name: string; value: number | string }[];
	/** tensors of a process / train frame */
	tensors?: { name: string; t: Tensor }[];
	phase?: string;
	step?: number;
}

function traceOf(p: Pick<StageProps, 'rt' | 'source'>): Trace | null {
	const st = p.rt.sources[p.source];
	if (st?.trace) return st.trace;
	for (const s of Object.values(p.rt.sources)) if (s.kind === 'trace' && s.trace) return s.trace;
	return null;
}

/** What the generic stage shows for these props. */
export function fallbackModel(p: Pick<StageProps, 'rt' | 'source' | 'at'>): FallbackModel {
	const at = p.at;
	const state = (at.state ?? {}) as Record<string, unknown>;
	const trace = traceOf(p);
	const layer = (at.layer ?? (typeof state.layer === 'string' ? state.layer : undefined)) as string | undefined;
	const unit = at.unit;
	const out = isTensor(state.out) ? state.out : layer && trace?.t[layer] ? trace.t[layer] : undefined;
	if (unit) {
		const m: FallbackModel = { kind: 'unit', layer: unit.layer, unit, out: trace?.t[unit.layer] ?? out };
		const r = unit.reads[0];
		const rt = r && trace?.t[r.tensor];
		if (r && rt) {
			let channel = 0;
			if (unit.groups?.length && rt.shape.length === 3) {
				let best = 0;
				unit.groups.forEach((g, i) => {
					if (Math.abs(g.sum) > Math.abs(unit.groups![best].sum)) best = i;
				});
				channel = Math.min(rt.shape[0] - 1, best);
			}
			m.read = { id: r.tensor, t: rt, box: r.box, channel };
		}
		return m;
	}
	if (out && at.index >= 0) return { kind: 'tensor', layer, out };
	if (at.index < 0 || !at.frame) return { kind: 'empty' };
	const scalars: { name: string; value: number | string }[] = [];
	const tensors: { name: string; t: Tensor }[] = [];
	for (const [k, v] of Object.entries(state)) {
		if (typeof v === 'number' || typeof v === 'string') scalars.push({ name: k, value: v });
		else if (isTensor(v)) tensors.push({ name: k, t: v });
		else if (v && typeof v === 'object' && !Array.isArray(v)) for (const [k2, v2] of Object.entries(v)) if (isTensor(v2)) tensors.push({ name: `${k}.${k2}`, t: v2 });
	}
	for (const [k, v] of Object.entries(at.frame.metrics ?? {})) if (!scalars.some((s) => s.name === k)) scalars.push({ name: k, value: v });
	return { kind: 'frame', scalars, tensors: tensors.slice(0, 4), phase: at.frame.phase, step: at.frame.step };
}

/** The terms worth listing: in order when they fit, else the largest |product| first. */
export function shownTerms(u: UnitExplanation, max = 9): { terms: NonNullable<UnitExplanation['terms']>; more: number; ordered: boolean } {
	const all = u.terms ?? [];
	if (all.length <= max) return { terms: all, more: 0, ordered: true };
	const top = all
		.map((t, i) => ({ t, i }))
		.sort((a, b) => Math.abs(b.t.p) - Math.abs(a.t.p) || a.i - b.i)
		.slice(0, max)
		.map((e) => e.t);
	return { terms: top, more: all.length - max, ordered: false };
}

export function isProbabilities(t: Tensor): boolean {
	if (t.shape.length !== 1 && !(t.shape.length === 2 && t.shape[0] === 1)) return false;
	let s = 0;
	for (let i = 0; i < t.data.length; i++) {
		const v = t.data[i];
		if (!(v >= 0 && v <= 1)) return false;
		s += v;
	}
	return Math.abs(s - 1) < 1e-3;
}

function minMax(t: Tensor): [number, number] {
	let lo = Infinity;
	let hi = -Infinity;
	for (let i = 0; i < t.data.length; i++) {
		const v = t.data[i];
		if (v < lo) lo = v;
		if (v > hi) hi = v;
	}
	return [lo, hi];
}

/** The text summary of the generic stage (live region / "Show as text"). */
export function describeFallback(p: Pick<StageProps, 'rt' | 'source' | 'at'>): I18n {
	const m = fallbackModel(p);
	if (m.kind === 'unit' && m.unit) {
		const u = m.unit;
		const at = `[${u.unit.join(', ')}]`;
		const n = u.terms?.length ?? 0;
		const s = u.values.s;
		if (u.kind === 'weighted-sum' && n)
			return {
				en: `${u.layer}, output ${at}: ${n} products of input and weight add up to ${fmt3(s ?? u.result - (u.bias ?? 0))}${u.bias !== undefined ? `, plus the bias ${fmt3(u.bias)}` : ''}, giving ${fmt3(u.result)}.`,
				zh: `${u.layer}，输出 ${at}：${n} 个“输入 × 权重”的乘积加起来是 ${fmt3(s ?? u.result - (u.bias ?? 0))}${u.bias !== undefined ? `，再加偏置 ${fmt3(u.bias)}` : ''}，得到 ${fmt3(u.result)}。`
			};
		if (u.kind === 'reduce' && n) return { en: `${u.layer}, output ${at}: the largest of ${n} numbers is ${fmt3(u.result)}.`, zh: `${u.layer}，输出 ${at}：${n} 个数里最大的是 ${fmt3(u.result)}。` };
		return { en: `${u.layer}, output ${at}: ${fmt3(u.result)} (${u.kind}).`, zh: `${u.layer}，输出 ${at}：${fmt3(u.result)}（${u.kind}）。` };
	}
	if ((m.kind === 'tensor' || m.kind === 'unit') && m.out) {
		const [lo, hi] = minMax(m.out);
		const sh = shapeText(m.out.shape);
		return {
			en: `${m.layer ?? 'Output'}: ${m.out.data.length} numbers in a ${sh} grid, from ${fmt3(lo)} to ${fmt3(hi)}.`,
			zh: `${m.layer ?? '输出'}：${sh} 网格里的 ${m.out.data.length} 个数，从 ${fmt3(lo)} 到 ${fmt3(hi)}。`
		};
	}
	if (m.kind === 'frame') {
		const list = (m.scalars ?? []).slice(0, 6).map((s) => `${s.name} ${typeof s.value === 'number' ? fmt3(s.value) : s.value}`).join(', ');
		return { en: `Frame ${p.at.index + 1}, phase ${m.phase}${list ? `: ${list}` : ''}.`, zh: `第 ${p.at.index + 1} 帧，阶段 ${m.phase}${list ? `：${list}` : ''}。` };
	}
	return { en: 'Nothing to show yet.', zh: '还没有可显示的内容。' };
}

/** A readable label for output index i: the spec's class name, else the text of vocabulary id i when the adapter gives
 *  it (Preprocessed.vocab, K1: a character model showed '#1'), else the token text of a BPE tokenizer (tokenizer.json
 *  among its assets or adapter options), else '#i'. */
export function outputItemLabel(
	rt: Pick<ExplainerRuntime, 'spec'> & { pre?: Preprocessed | null; assets?: Record<string, unknown>; pkg?: { manifest?: { input?: { options?: Record<string, unknown> } } } },
	i: number
): string | I18n {
	const l = outputLabel(rt.spec, i);
	if (typeof l !== 'string' || !l.startsWith('#')) return l;
	if (rt.pre?.vocab) {
		const v = vocabLabel(rt.pre, i);
		if (typeof v === 'string') return v;
	}
	const tok = tokenizerOf(rt.assets?.['tokenizer.json'] ?? rt.pkg?.manifest?.input?.options?.tokenizer);
	if (!tok || i >= tok.vocabSize) return l;
	return `“${tok.tokenToDisplay(i, { visibleWhitespace: true })}”`;
}

/** Is this frame the model's one answer (the output layer, with class or vocab scores)? K1: an output whose last axis is
 *  not class or vocab (an image, a 2-D point) is not, even when the spec lists classes; nor is a per-item output such as
 *  a GCN's [n, class] (one answer per node): only a [class] vector or a sequence's [t, vocab | class] rows (read at the
 *  answer row: K2, manifest output.row, default the last) give one answer. */
export function isAnswerLayer(rt: Pick<ExplainerRuntime, 'spec'>, layer: string | undefined, trace: Trace | null): boolean {
	if (!layer || !rt.spec) return false;
	if (layer !== outputId(rt.spec, trace)) return false;
	const l = rt.spec.layers.find((x) => x.id === layer);
	const axes = l?.axes ?? [];
	const out = trace?.t[layer];
	const rank = out?.shape.length ?? l?.outShape.length ?? 1;
	const last = axes[axes.length - 1];
	if (axes.length) {
		if (last !== 'class' && last !== 'vocab') return false;
		return rank === 1 || ((axes[0] === 't' || axes[0] === 'tq') && rank === 2);
	}
	// no axes recorded: a classes list that matches the output's last dim
	const classes = rt.spec.classes;
	const V = out?.shape[out.shape.length - 1];
	return Array.isArray(classes) && rank === 1 && V === classes.length;
}

/** The top answers of an output tensor (a [T, V] sequence output is read at its answer row: K2, manifest output.row,
 *  default the last row). */
export function topAnswers(rt: Parameters<typeof outputItemLabel>[0], out: Tensor, k = 10): { label: string | I18n; p: number; i: number }[] {
	const p = outputProbs(out, outputRow(rt, out));
	return Array.from(p, (v, i) => ({ v, i }))
		.sort((a, b) => b.v - a.v)
		.slice(0, k)
		.map(({ v, i }) => ({ label: outputItemLabel(rt, i), p: v, i }));
}
