// tinyseq2seq: illustrative stage props for the package views' demo() (the gallery and the view contract test): the
// text "Mar 5, 2021" with a made-up trace (small sizes, smooth made-up numbers). Nothing here comes from the model.
import type { StageProps, Tensor, Trace } from '../../atlas/types/index.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';

const TEXT = 'Mar 5, 2021';
const OUT = [4, 2, 4, 3, 12, 2, 5, 12, 2, 7, 1]; // 2021-03-05, then the end symbol
const ten = (shape: number[], f: (i: number) => number): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)) });

export function demoProps(props: Record<string, unknown> = {}): Omit<StageProps, 'ctx'> {
	const T = TEXT.length;
	const look = [8, 8, 9, 10, 8, 8, 2, 10, 8, 4, 8]; // the input position each made-up step looks at
	const w = ten([11, T], (i) => (i % T === look[Math.floor(i / T)] ? 0.7 : 0.3 / (T - 1)));
	const t: Record<string, Tensor> = {
		enc: ten([T, 6], (i) => Math.tanh(Math.sin(i * 0.7))),
		'dec#scores': ten([11, T], (i) => Math.log(w.data[i] * 10)),
		'dec#weights': w,
		'dec#context': ten([11, 6], (i) => Math.sin(i * 0.4)),
		'dec#next': ten([11], (i) => OUT[i]),
		probs: ten([11, 13], (i) => (i % 13 === OUT[Math.floor(i / 13)] ? 0.9 : 0.1 / 12))
	};
	const trace: Trace = { order: Object.keys(t), t };
	const pre = { tensors: {}, steps: [], empty: false, tokens: [...TEXT] };
	const rt = fakeRuntime({ slug: 'tinyseq2seq', input: TEXT, pre, trace });
	return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
