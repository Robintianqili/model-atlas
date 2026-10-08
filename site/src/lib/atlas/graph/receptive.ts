// Receptive fields (design §3.2: tinycnn.ts receptiveField generalized). Walks from a layer back to the input image
// along each layer's first input:
//   - an op with receptive() maps the box (conv2d, maxpool2d; elementwise ops return it unchanged);
//   - any other op (flatten, linear, softmax, …) sees the whole image, as in tinycnn.ts;
//   - the box is clamped to each input grid (padding cells are not real pixels).
// Returns an inclusive box in the [H, W] grid of the declared input, or null when the input is not a 2-D image.
import type { AtlasLayer, AtlasSpec, Box, Dim, OpRegistry } from '../types/index.ts';
import { mergeParams } from '../ops/registry.ts';

function hw(shape: readonly Dim[] | undefined): [number, number] | null {
	if (!shape || shape.length < 2) return null;
	const H = shape[shape.length - 2];
	const W = shape[shape.length - 1];
	return typeof H === 'number' && typeof W === 'number' ? [H, W] : null;
}

function clamp(b: Box, size: [number, number] | null): Box {
	if (!size) return b;
	const [H, W] = size;
	return { y0: Math.max(0, b.y0), x0: Math.max(0, b.x0), y1: Math.min(H - 1, b.y1), x1: Math.min(W - 1, b.x1) };
}

/** The declared input a layer chain starts from (following first inputs). */
function rootInput(spec: AtlasSpec, layerId: string): { name: string; shape: Dim[] } | null {
	const byId = new Map(spec.layers.map((l) => [l.id, l]));
	let id: string | undefined = layerId;
	const seen = new Set<string>();
	while (id !== undefined && !seen.has(id)) {
		seen.add(id);
		const l = byId.get(id);
		if (!l) {
			const d = spec.inputs?.find((i) => i.name === id);
			return d ? { name: d.name, shape: d.shape } : null;
		}
		if (l.op === 'input') {
			const d = spec.inputs?.find((i) => i.name === (l.in[0] ?? 'input'));
			return { name: l.id, shape: d?.shape ?? l.outShape };
		}
		id = l.in[0];
	}
	return null;
}

export function receptiveField(spec: AtlasSpec, ops: OpRegistry, layerId: string, box: Box): Box | null {
	const byId = new Map(spec.layers.map((l) => [l.id, l]));
	const root = rootInput(spec, layerId);
	const img = hw(root?.shape);
	if (!root || !img) return null;
	const whole: Box = { y0: 0, x0: 0, y1: img[0] - 1, x1: img[1] - 1 };
	let L = byId.get(layerId);
	if (!L) return null;
	let b: Box = { ...box };
	const seen = new Set<string>();
	while (L && L.op !== 'input' && !seen.has(L.id)) {
		seen.add(L.id);
		const r = ops.get(L.op);
		if (!r.def.receptive) return whole;
		b = r.def.receptive(b, mergeParams(r.def, r.params, L.params));
		const src: string | undefined = L.in[0];
		const next: AtlasLayer | undefined = src !== undefined ? byId.get(src) : undefined;
		const srcShape = next ? next.outShape : spec.inputs?.find((i) => i.name === src)?.shape;
		b = clamp(b, hw(srcShape));
		if (!next) break;
		L = next;
	}
	return clamp(b, img);
}
