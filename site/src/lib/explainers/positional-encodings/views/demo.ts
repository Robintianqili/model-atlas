// positional-encodings: the gallery demo of the package's views (ViewMeta.demo): a tape of process 'pe' with one made-up
// frame per phase (the sinusoidal, RoPE and ALiBi numbers are computed; the token weights and the "learned" table are
// stand-ins, so the provenance is illustrative). Pure TS.
import type { StageProps, Tensor } from '../../../atlas/types/index.ts';
import { rope, sinTable, slope } from '../pe.ts';

const v = (a: ArrayLike<number>, shape = [a.length]): Tensor => ({ shape, data: Float64Array.from(a) });

export async function demoStage(phase: string, props: Record<string, unknown> = {}): Promise<Omit<StageProps, 'ctx'>> {
	const [{ fakeRuntime }, { TapeStore }, { pe, ROWS, ALIBI_D }, { LOSS, SCHEMES }] = await Promise.all([
		import('../../../atlas/runtime/fake.ts'),
		import('../../../atlas/process/tape.ts'),
		import('../process.ts'),
		import('../extrap.ts')
	]);
	const tape = new TapeStore({ process: 'pe', schema: pe.schema, phases: pe.phases, provenance: 'illustrative' });
	const w = [0.1, 0.2, 0.25, 0.1, 0.35];
	const ws = [0.25, 0.1, 0.2, 0.1, 0.35];
	tape.push({ phase: 'order', step: 0, state: { typed: v([428, 437, 258, 397, 46]), shuffled: v([258, 428, 437, 397, 46]), w0: v(w), w0s: v(ws), w1: v(w), w1s: v([0.3, 0.1, 0.1, 0.15, 0.35]) }, metrics: { n: 5, cut: 0, same: 0, d0: 0, d1: 0.2, dF0: 0.9, top1: 512, p1: 0.9, top1s: 46, p1s: 0.3 } } as never);
	const sin = sinTable(ROWS, 76).map((x) => x * 0.2);
	tape.push({ phase: 'learned', step: 0, state: { table: v(sin, [ROWS, 76]), sim: v(Array.from({ length: ROWS }, (_, j) => Math.cos((j - 5) / 12))) }, metrics: { p: 5, k: 3, rows: ROWS } } as never);
	tape.push({ phase: 'sin', step: 0, state: { table: v(sinTable(), [128, 76]) }, metrics: { p: 5 } } as never);
	const q = Array.from({ length: 19 }, (_, i) => Math.cos(i + 1));
	const kv = Array.from({ length: 19 }, (_, i) => Math.sin(i + 2));
	tape.push({ phase: 'rope', step: 0, state: { q: v(q), kv: v(kv), qr: v(rope(q, 8)), kr: v(rope(kv, 5)) }, metrics: { m: 8, n: 5, k: 3, pairRaw: 0.2, pair: -0.1, raw: 1.5, score: 1.2 } } as never);
	const by = (f: (s: number) => number) => v(Array.from({ length: ROWS }, (_, s) => f(s)));
	tape.push({ phase: 'relative', step: 0, state: { ropeBy: by(() => -2), absBy: by((s) => (s < 61 ? -3 + Math.sin(s / 6) : NaN)) }, metrics: { p: 5, k: 3 } } as never);
	tape.push({ phase: 'alibi', step: 0, state: { bias: v(Array.from({ length: 4 * ALIBI_D }, (_, i) => -slope(Math.floor(i / ALIBI_D)) * (i % ALIBI_D)), [4, ALIBI_D]) }, metrics: { k: 3 } } as never);
	const nb = LOSS[SCHEMES[0]].length;
	tape.push({ phase: 'extrap', step: 0, state: { loss: v(SCHEMES.flatMap((s) => LOSS[s].map((x) => x ?? NaN)), [SCHEMES.length, nb]) }, metrics: { bins: nb, bin: 8 } } as never);
	const rt = fakeRuntime({ slug: 'positional-encodings', input: {}, tapes: { pe: tape } });
	const i = Math.max(0, tape.find({ phase }));
	return { rt, source: 'pe', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props };
}

/** The state of the step's frame as tensors by name (views read it). */
export const stateOf = (p: Pick<StageProps, 'at'>) => (p.at?.state ?? {}) as Record<string, Tensor | undefined>;
export const arr = (t: Tensor | undefined) => (t ? Array.from(t.data as ArrayLike<number>) : []);
