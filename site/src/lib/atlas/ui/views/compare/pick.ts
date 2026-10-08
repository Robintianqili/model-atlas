// What the compare view puts side by side (pure TS; meta.ts uses it for describe()).
//   sources: props.compare {a, b, label} (the step's compare, passed through props), else props.a / props.b, else the
//            first two sources of the runtime
//   traces:  each side's layer (props.layer, else its output layer) from its own trace
//   tapes:   a metric (props.metric, else the first metric both tapes record) over the frames
import type { AtlasSpec, I18n, Tape, Tensor } from '../../../types/index.ts';
import { outputId, outputProbs, outputRow } from '../../../lesson/facts.ts';
import type { Props } from '../kit.parts/stage.ts';
import { layerLabel } from '../kit.parts/stage.ts';

export interface Side {
	source: string;
	label: I18n;
	kind: 'trace' | 'tape';
	layer?: string;
	t?: Tensor;
	probs?: Float64Array;
	spec?: AtlasSpec | null;
	tape?: Tape;
	series?: Float64Array;
}

export interface Comparison {
	a: Side;
	b: Side;
	metric?: string;
	/** traces: same shape → largest |a − b|; distributions → top-1 of each */
	maxDiff?: number;
	topA?: number;
	topB?: number;
}

function labelOf(p: Props, i: 0 | 1, source: string): I18n {
	const c = p.props.compare as { label?: [I18n, I18n] } | undefined;
	const l = (c?.label ?? (p.props.labels as [I18n, I18n] | undefined))?.[i];
	return l ?? { en: source, zh: source };
}

export function pickComparison(p: Props): Comparison | null {
	const c = p.props.compare as { a?: string; b?: string } | undefined;
	const names = Object.keys(p.rt.sources);
	const a = c?.a ?? (typeof p.props.a === 'string' ? p.props.a : names[0]);
	const b = c?.b ?? (typeof p.props.b === 'string' ? p.props.b : names.find((n) => n !== a));
	if (!a || !b) return null;
	const sa = p.rt.sources[a];
	const sb = p.rt.sources[b];
	if (!sa || !sb) return null;
	const side = (name: string, i: 0 | 1): Side | null => {
		const s = p.rt.sources[name];
		if (s.trace) {
			const spec = s.model?.spec ?? p.rt.spec;
			const layer = (typeof p.props.layer === 'string' && s.trace.t[p.props.layer] ? p.props.layer : undefined) ?? outputId(spec, s.trace);
			const t = layer ? s.trace.t[layer] : undefined;
			if (!layer || !t) return null;
			const last = (s.model?.spec.layers.find((l) => l.id === layer)?.axes ?? []).slice(-1)[0];
			// K2: a [T, V] output at the package's answer row (manifest output.row) when it is the output layer
			const probs = last === 'class' || last === 'vocab' ? outputProbs(t, layer === outputId(spec, s.trace) ? outputRow(p.rt, t) : undefined) : undefined;
			return { source: name, label: labelOf(p, i, name), kind: 'trace', layer, t, probs, spec };
		}
		if (s.tape) return { source: name, label: labelOf(p, i, name), kind: 'tape', tape: s.tape };
		return null;
	};
	const A = side(a, 0);
	const B = side(b, 1);
	if (!A || !B) return null;
	const out: Comparison = { a: A, b: B };
	if (A.kind === 'trace' && B.kind === 'trace') {
		if (A.probs && B.probs) {
			const arg = (x: Float64Array) => x.reduce((bi, v, i) => (v > x[bi] ? i : bi), 0);
			out.topA = arg(A.probs);
			out.topB = arg(B.probs);
		}
		if (A.t && B.t && A.t.data.length === B.t.data.length) {
			let m = 0;
			for (let i = 0; i < A.t.data.length; i++) {
				const d = Math.abs(A.t.data[i] - B.t.data[i]);
				if (Number.isFinite(d) && d > m) m = d;
			}
			out.maxDiff = m;
		}
	} else if (A.kind === 'tape' && B.kind === 'tape' && A.tape && B.tape) {
		const want = typeof p.props.metric === 'string' ? p.props.metric : undefined;
		const first = (tp: Tape) => {
			for (let i = 0; i < Math.min(tp.length, 50); i++) {
				const m = tp.frame(i).metrics;
				if (m) return Object.keys(m);
			}
			return [];
		};
		const common = first(A.tape).filter((k) => first(B.tape!).includes(k));
		const metric = want ?? common.find((k) => /loss/.test(k)) ?? common[0];
		if (metric) {
			out.metric = metric;
			A.series = A.tape.metric(metric);
			B.series = B.tape.metric(metric);
		}
	}
	return out;
}

export function sideName(p: Props, s: Side): I18n {
	if (s.kind === 'trace' && s.layer) {
		const model = p.rt.sources[s.source]?.model ?? null;
		const l = layerLabel(model, s.layer);
		return { en: `${s.label.en} · ${l.en}`, zh: `${s.label.zh ?? s.label.en} · ${l.zh ?? l.en}` };
	}
	return s.label;
}
