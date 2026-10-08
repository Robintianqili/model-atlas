// AtlasSpec -> the kit's ModelSpec (design §5.1 "Adapting to the kit"). The kit readers (specToLayers, ProvenanceBadge,
// FactStrip, ModelPageShell) get a ModelSpec built here, never an AtlasSpec: symbolic dims become numbers, and a parity
// record that has not been measured yet (max_abs_diff: null) is left out, so no badge prints a tolerance that was never
// measured. The kit itself is not changed.
import type { ModelSpec, SpecLayer } from '$lib/kit/spec';
import type { LayerNode, PredictQuestion as KitPredictQuestion } from '$lib/kit/types';
import type { PredictQuestion as AtlasPredictQuestion } from '../types/core.ts';
import type { AtlasSpec, Dim } from '../types/spec.ts';
import type { I18n } from '../types/core.ts';
import type { SourceParity } from '../types/runtime.ts';

// Atlas PredictQuestion is a structural copy of the kit's (atlas never imports the kit): both directions must assign.
// svelte-check fails here if either type changes.
const toKitQ = (q: AtlasPredictQuestion): KitPredictQuestion => q;
const toAtlasQ = (q: KitPredictQuestion): AtlasPredictQuestion => q;
export const predictQuestionCompat = { toKitQ, toAtlasQ };

/** A numeric shape: symbolic dims take `dims[name]`, else the dim's declared max, else 0 (unknown). */
export function numericShape(shape: readonly Dim[], spec: Pick<AtlasSpec, 'dims'> | null, dims: Record<string, number> = {}): number[] {
	return shape.map((d) => (typeof d === 'number' ? d : (dims[d] ?? spec?.dims?.[d]?.max ?? 0)));
}

/** The shape as text for labels: symbolic dims keep their names unless `dims` gives a value ('T×76' or '9×76'). */
export function shapeLabel(shape: readonly Dim[], dims: Record<string, number> = {}): (number | string)[] {
	return shape.map((d) => (typeof d === 'number' ? d : (dims[d] ?? d)));
}

/** The kit ModelSpec of an AtlasSpec. `dims`: the current values of symbolic dims (e.g. the trace's T). */
export function toKitSpec(s: AtlasSpec, dims: Record<string, number> = {}): ModelSpec {
	const layers: SpecLayer[] = s.layers.map((l) => {
		const out: SpecLayer = { id: l.id, op: l.op, label: l.label, in: l.in, outShape: numericShape(l.outShape, s, dims) };
		if (l.params) out.params = l.params;
		return out;
	});
	const inp = s.inputs?.[0] ?? (s.input ? { kind: s.input.kind, shape: s.input.shape } : undefined);
	const spec: ModelSpec = {
		slug: s.slug,
		name: s.name,
		version: s.version,
		params: s.params,
		precision: s.precision,
		tensors: s.tensors,
		layers
	};
	if (inp) spec.input = { kind: inp.kind, shape: numericShape(inp.shape, s, dims) };
	// K1: every input of a multi-input model (the facts named only the first: TinyCLIP's drawing, not its caption)
	if ((s.inputs?.length ?? 0) > 1) spec.inputs = s.inputs!.map((i) => ({ kind: i.kind, shape: numericShape(i.shape, s, dims) }));
	if (s.training) spec.training = { dataset: s.training.dataset, steps: s.training.steps, metrics: s.training.metrics };
	const p = s.parity;
	if (p && typeof p.max_abs_diff === 'number' && Number.isFinite(p.max_abs_diff)) spec.parity = { against: p.against, max_abs_diff: p.max_abs_diff, checked: p.checked ?? '' };
	return spec;
}

/** The badge numbers of a spec: tolerance only when parity was measured. */
export function parityBadge(s: AtlasSpec | null): { tolerance?: number; against?: string; precision?: string } {
	if (!s) return {};
	const p = s.parity;
	const out: { tolerance?: number; against?: string; precision?: string } = { precision: s.precision };
	if (p && typeof p.max_abs_diff === 'number' && Number.isFinite(p.max_abs_diff)) {
		out.tolerance = p.max_abs_diff;
		out.against = p.against;
	}
	return out;
}

/** The badge of a train or process step from the package's parity record (explainers/<slug>/parity.json, written by
 *  atlas parity --write; K0): design §6.2's texts. Nothing for a source without a passing twin or reference. */
export function sourceBadge(rec: SourceParity | undefined): { tolerance?: number; against?: string; precision?: string; note?: I18n; short?: I18n } {
	if (!rec || !rec.pass) return {};
	if (rec.scope === 'train') {
		const digits = rec.bar && rec.bar > 0 ? Math.round(-Math.log10(rec.bar)) : 7;
		const steps = rec.steps ?? 50;
		return {
			precision: 'f64',
			tolerance: rec.max,
			against: 'PyTorch fp64 (relative)',
			note: { en: `f64 in your browser · matches PyTorch fp64 to ${digits} significant digits over ${steps} steps.`, zh: `在你的浏览器中用 f64 计算 · ${steps} 步内与 PyTorch fp64 吻合到 ${digits} 位有效数字。` },
			// K1: shown next to the compact badge, so the check is visible at a glance (the note is in its tooltip)
			short: { en: '= PyTorch fp64', zh: '= PyTorch fp64' }
		};
	}
	const more = rec.fixtures ? { en: ` (also on ${rec.fixtures} other inputs)`, zh: `（另外 ${rec.fixtures} 个输入上也一样）` } : { en: '', zh: '' };
	return {
		tolerance: rec.max,
		against: 'an independent Python run',
		note: { en: `Same steps as an independent Python run${more.en}.`, zh: `与独立的 Python 实现逐步相同${more.zh}。` },
		short: { en: '= Python run', zh: '= Python 运行' }
	};
}

/** LayerNode[] for ModelOverview and Inspector, with symbolic dims kept as names unless `dims` gives values. */
export function layerNodes(s: AtlasSpec, dims: Record<string, number> = {}): LayerNode[] {
	return s.layers.map((l) => ({ id: l.id, label: l.label, op: l.op, outShape: shapeLabel(l.outShape, dims) }));
}
