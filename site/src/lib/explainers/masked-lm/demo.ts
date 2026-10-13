// masked-lm: illustrative stage props for the package views' demo() (the gallery and the view contract test): a short
// example sentence with made-up traces (adapted from tinybert's demo.ts). Nothing here comes from a model.
import type { StageProps, Tensor, Trace } from '../../atlas/types/index.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';
import { record } from '../../atlas/process/record.ts';
import { atFor } from '../../atlas/runtime/sources.ts';
import { batch } from './process.ts';

const ten = (shape: number[], f: (i: number) => number): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)) });
/** rows of a softmax-like table: 0.6 on one id, the rest spread evenly */
const softRows = (rows: number, V: number, peak: (r: number) => number) => ten([rows, V], (i) => (i % V === peak(Math.floor(i / V)) ? 0.6 : 0.4 / (V - 1)));

export function demoProps(props: Record<string, unknown> = {}): Omit<StageProps, 'ctx'> {
	// "Tom was sad because he lost his ball." with "sad" and "ball" hidden
	const ids = [735, 282, 493, 711, 285, 435, 350, 341, 477, 46];
	const raw = { text: 'Tom was sad because he lost his ball.', ids, fate: ids.map((_, i) => (i === 2 || i === 8 ? 1 : 0)), picked: true };
	const bert: Trace = { order: ['probs'], t: { probs: softRows(ids.length + 2, 1027, (r) => ids[r - 1] ?? 46) } };
	const gpt: Trace = { order: ['probs'], t: { probs: softRows(ids.length, 1024, (r) => ids[r + 1] ?? 46) } };
	const rt = fakeRuntime({ slug: 'masked-lm', input: raw, trace: bert, traceSource: 'bert', sources: { gpt: { kind: 'trace', status: 'ready', provenance: 'illustrative', trace: gpt } }, tapes: { batch: record(batch, raw, { big: 4 }, { seed: 7 }) } });
	const train = props.mode === 'batch' || props.draw === true;
	return { rt, source: train ? 'batch' : 'bert', at: train ? atFor(rt, 'batch', (rt.sources.batch?.tape?.length ?? 0) - 1) : { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
