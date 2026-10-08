// tinymamba: illustrative props for the package views' gallery demos (ViewMeta.demo): a made-up trace of the right
// shapes on a short row (seeded numbers that mean nothing), served by the kit's fakeRuntime.
import type { GraphModel, StageProps, Tensor } from '../../atlas/types/index.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';
import { createRng } from '../../atlas/core/rng.ts';

export function demoProps(props: Record<string, unknown> = {}): Omit<StageProps, 'ctx'> {
	const r = createRng(5);
	const ids = [0, 3, 0, 0, 1, 0, 9, 9];
	const [n, D, N, V] = [ids.length, 4, 2, 10];
	const ten = (shape: number[], f: () => number): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, f) });
	const t: Record<string, Tensor> = {
		'B1.scan#delta': ten([n, D], () => r.next()),
		'B1.B': ten([n, N], () => r.next() - 0.5),
		'B1.C': ten([n, N], () => r.next() - 0.5),
		'B1.scan#h': ten([n, D, N], () => r.next() - 0.5),
		probs: ten([n, V], () => r.next() / 5)
	};
	const model = { weights: { 'B1.scan.A_log': ten([D, N], () => r.next()) } } as unknown as GraphModel;
	const rt = fakeRuntime({
		spec: { slug: 'tinymamba', version: 'demo', layers: [], config: { d: 4, D, N, layers: 1, vocab: V } } as never,
		model,
		pre: { tensors: { input: { shape: [n], data: Int32Array.from(ids), dtype: 'i32' } }, steps: [], empty: false },
		trace: { order: Object.keys(t), t }
	});
	return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
