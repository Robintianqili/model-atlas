// self-supervised-objectives: illustrative stage props for the package views' demo() (the gallery and the view contract
// test): a short example sentence with made-up traces (adapted from tinybert's demo.ts). Nothing here comes from a model.
import type { StageProps, Tensor, Trace } from '../../atlas/types/index.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';
import { record } from '../../atlas/process/record.ts';
import { atFor } from '../../atlas/runtime/sources.ts';
import { hide } from './process.ts';

const ten = (shape: number[], f: (i: number) => number): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)) });
/** rows of a softmax-like table: 0.6 on one id, the rest spread evenly */
const softRows = (rows: number, V: number, peak: (r: number) => number) => ten([rows, V], (i) => (i % V === peak(Math.floor(i / V)) ? 0.6 : 0.4 / (V - 1)));

export function demoProps(props: Record<string, unknown> = {}): Omit<StageProps, 'ctx'> {
	// "Tom was sad because he lost his ball." with "sad" hidden
	const ids = [735, 282, 493, 711, 285, 435, 350, 341, 477, 46];
	const raw = { text: 'Tom was sad because he lost his ball.', ids, mask: 2 };
	const bert: Trace = { order: ['probs'], t: { probs: softRows(ids.length + 2, 1027, () => 493) } };
	const gpt: Trace = { order: ['probs'], t: { probs: softRows(ids.length, 1024, (r) => ids[r + 1] ?? 46) } };
	const rt = fakeRuntime({ slug: 'self-supervised-objectives', input: raw, trace: bert, traceSource: 'bert', sources: { gpt: { kind: 'trace', status: 'ready', provenance: 'illustrative', trace: gpt } }, tapes: { hide: record(hide, raw, {}, { seed: 7 }) } });
	const train = props.mode === 'train';
	return { rt, source: train ? 'hide' : 'bert', at: train ? atFor(rt, 'hide', 2) : { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
