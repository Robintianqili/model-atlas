// tinybert: illustrative stage props for the package views' demo() (the gallery and the view contract test): a made-up
// five-token text with made-up traces. Nothing here comes from the model.
import type { StageProps, Tensor, Trace } from '../../atlas/types/index.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';
import { record } from '../../atlas/process/record.ts';
import { atFor } from '../../atlas/runtime/sources.ts';
import { hide } from './process.ts';

const NAMES: Record<number, string> = { 1: 'The', 2: ' cat', 3: ' sat', 4: '.', 1024: '[MASK]', 1025: '[CLS]', 1026: '[SEP]' };
const vocab = Array.from({ length: 1027 }, (_, i) => NAMES[i] ?? `#${i}`);
const ten = (shape: number[], f: (i: number) => number): Tensor => ({ shape, data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => f(i)) });
const softRows = (rows: number, V: number, peak: (r: number) => number) => ten([rows, V], (i) => (i % V === peak(Math.floor(i / V)) ? 0.6 : 0.4 / (V - 1)));
const attn = (H: number, n: number, causal: boolean) => ten([H, n, n], (i) => {
	const q = Math.floor(i / n) % n;
	const k = i % n;
	return causal ? (k <= q ? 1 / (q + 1) : 0) : 1 / n;
});

export function demoProps(props: Record<string, unknown> = {}): Omit<StageProps, 'ctx'> {
	const raw = { text: 'The cat sat.', ids: [1, 2, 3, 4], mask: 1 };
	const fwd: Trace = { order: ['tok', 'pos', 'seg', 'embed', 'B1.attn', 'probs'], t: { tok: ten([6, 4], (i) => Math.sin(i)), pos: ten([6, 4], (i) => Math.cos(i) / 2), seg: ten([6, 4], (i) => (i % 4) / 10), embed: ten([6, 4], (i) => Math.sin(i) + Math.cos(i) / 2 + (i % 4) / 10), 'B1.attn': attn(2, 6, false), probs: softRows(6, 1027, () => 2) } };
	const gpt: Trace = { order: ['L0.attn.weights', 'probs'], t: { 'L0.attn.weights': attn(2, 4, true), probs: softRows(4, 1024, (r) => r + 2) } };
	const spec = { slug: 'tinybert', version: 'demo', layers: [{ id: 'B1.scores', op: 'attention_scores', in: [], outShape: [2, 'T', 'T'], params: { nHead: 2, dHead: 2 } }], vocab, tokenFlags: '' } as unknown as StageProps['rt']['spec'];
	const rt = fakeRuntime({ slug: 'tinybert', spec, input: raw, trace: fwd, sources: { gpt: { kind: 'trace', status: 'ready', provenance: 'illustrative', trace: gpt } }, tapes: { hide: record(hide, raw, {}, { seed: 7 }) } });
	const train = props.mode === 'train';
	return { rt, source: train ? 'hide' : 'fwd', at: train ? atFor(rt, 'hide', 2) : { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}
