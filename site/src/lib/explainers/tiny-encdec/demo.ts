// tiny-encdec: illustrative stage props for the package view's demo() (the gallery and the view contract test): the
// text "Mar 5, 2021" with a made-up trace (two heads, smooth made-up numbers). Nothing here comes from the model.
import type { StageProps, Tensor, Trace } from '../../atlas/types/index.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';

const TEXT = 'Mar 5, 2021';
const OUT = [4, 2, 4, 3, 12, 2, 5, 12, 2, 7, 1]; // 2021-03-05, then the end symbol
const ten = (shape: number[], f: (i: number) => number): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)) });

export function demoProps(props: Record<string, unknown> = {}): Omit<StageProps, 'ctx'> {
	const T = TEXT.length;
	const look = [7, 8, 9, 10, 3, 2, 2, 3, 4, 4, 5]; // the input position each made-up step looks at
	const t: Record<string, Tensor> = {
		'dec.cattn': ten([2, 11, T], (i) => (i % T === look[Math.floor(i / T) % 11] ? 0.7 : 0.3 / (T - 1))),
		greedy: ten([11], (i) => (i ? OUT[i - 1] : 0)),
		probs: ten([11, 13], (i) => (i % 13 === OUT[Math.floor(i / 13)] ? 0.9 : 0.1 / 12))
	};
	const trace: Trace = { order: Object.keys(t), t };
	const pre = { tensors: {}, steps: [], empty: false, tokens: [...TEXT] };
	const rt = fakeRuntime({ slug: 'tiny-encdec', input: TEXT, pre, trace });
	return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
