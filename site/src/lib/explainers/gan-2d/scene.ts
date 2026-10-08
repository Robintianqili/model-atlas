// gan-2d: what the arena view draws and the lesson narrates, from one frame of a train source (pure TS, so the view,
// the narration and the claim tests read the same numbers).
//   coarse frames (init, step, snapshot): the weights of the latest snapshot; the shown noise Z_SHOW (240 points)
//   fine 'update' frames (the slowed-down steps): the weights before and after the update and the step's own batch
import type { At, ExplainerRuntime, Tape } from '../../atlas/types/index.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';
import { realOf } from './data.ts';
import { gen, hasNets, logits, match, mean, modesHit, sigmoid, SPAN, turnOf, uphill, Z_SHOW, type Params } from './net.ts';

type Tensors = Record<string, { shape: number[]; data: Float64Array }>;
type St = { params?: Params; prev?: Params; batch?: Tensors; step?: number };

export interface Scene {
	source: string;
	step: number;
	/** fine update frames: 0 = a D step, 1 = a G step; null on coarse frames */
	turn: 0 | 1 | null;
	real: Points2D;
	/** the real points of the step's batch (fine frames), else all of them */
	realShown: Float64Array;
	params: Params;
	/** fine update frames: the weights before the update */
	before: Params | null;
	z: Float64Array;
	fake: Float64Array;
	fakeBefore: Float64Array | null;
	/** mean D(x) on realShown and on fake, with the weights drawn (and before the update, on fine frames) */
	dReal: number;
	dFake: number;
	dRealBefore: number;
	dFakeBefore: number;
	recall: number;
	precision: number;
	centres: [number, number][] | null;
	modes: number;
	/** generated points outside the drawn plane (a fast generator can throw them off it) */
	off: number;
}

const meanD = (p: Params, x: Float64Array) => (x.length ? mean(logits(p, x).map(sigmoid)) : NaN);

/** The scene of `source` at the step's cursor (another source: its frame at the same training step, else its last). */
export function sceneOf(rt: ExplainerRuntime, at: At | undefined, source: string, own = source): Scene | null {
	const tape = own === source ? at?.tape : rt.sources[source]?.tape;
	let s: St | undefined;
	try {
		if (own === source) s = at?.state as St | undefined;
		else if (tape?.length) {
			const i = typeof at?.frame?.step === 'number' ? tape.find({ step: at.frame.step }) : -1;
			s = tape.state(i >= 0 ? i : tape.length - 1) as St;
		}
	} catch {
		s = undefined;
	}
	if (!s || !hasNets(s.params)) return null;
	const opt = (tape?.opt ?? {}) as { data?: unknown; centres?: [number, number][]; dSteps?: number };
	const real = realOf(rt.input, opt);
	const fine = !!s.prev && !!s.batch?.z;
	const params = s.params;
	const before = fine ? s.prev! : null;
	const z = fine ? s.batch!.z.data : Z_SHOW;
	const realShown = fine ? s.batch!.x.data : real.xy;
	const fake = gen(params, z);
	const fakeBefore = before ? gen(before, z) : null;
	const m = fine ? { recall: NaN, precision: NaN } : match(real, fake);
	const step = s.step ?? 0;
	return {
		source,
		step,
		turn: fine ? (turnOf(step, opt.dSteps ?? 1) as 0 | 1) : null,
		real,
		realShown,
		params,
		before,
		z,
		fake,
		fakeBefore,
		dReal: meanD(params, realShown),
		dFake: meanD(params, fake),
		dRealBefore: before ? meanD(before, realShown) : NaN,
		dFakeBefore: before ? meanD(before, fakeBefore!) : NaN,
		recall: m.recall,
		precision: m.precision,
		centres: opt.centres ?? null,
		modes: opt.centres ? modesHit(opt.centres, fake) : NaN,
		off: Array.from({ length: fake.length / 2 }, (_, j) => Math.max(Math.abs(fake[2 * j]), Math.abs(fake[2 * j + 1])) > SPAN).filter(Boolean).length
	};
}

/** a metric over the snapshots in the second half of a tape's steps, up to frame `upto` (from: the first such step) */
export function lateRange(tape: Tape | null | undefined, k: string, upto = (tape?.length ?? 0) - 1): { from: number; lo: number; hi: number } {
	const half = Math.ceil((tape && upto >= 0 ? (tape.frame(upto).step ?? 0) : 0) / 2);
	const fs = Array.from({ length: upto + 1 }, (_, i) => tape!.frame(i)).filter((f) => f.phase === 'snapshot' && (f.step ?? 0) >= half && typeof f.metrics?.[k] === 'number');
	const vs = fs.map((f) => f.metrics![k]);
	return { from: fs[0]?.step ?? NaN, lo: vs.length ? Math.min(...vs) : NaN, hi: vs.length ? Math.max(...vs) : NaN };
}

/** −ln σ(l) = ln(1 + e^{−l}), written stably */
const softplus = (v: number) => Math.max(v, 0) + Math.log1p(Math.exp(-Math.abs(v)));
/** G's loss −mean ln D(G(z)) for weights p, given the generated points */
export const gLossOf = (p: Params, fake: Float64Array) => mean(logits(p, fake).map((l) => softplus(-l)));
/** how many generated points of a G step moved along their arrow (less than 90° off the direction that raises D) */
export function alongArrows(s: Scene): number {
	if (!s.fakeBefore) return NaN;
	const u = uphill(s.params, s.fakeBefore);
	let k = 0;
	for (let j = 0; j < s.fake.length / 2; j++) if ((s.fake[2 * j] - s.fakeBefore[2 * j]) * u[2 * j] + (s.fake[2 * j + 1] - s.fakeBefore[2 * j + 1]) * u[2 * j + 1] > 0) k++;
	return k;
}

/** −mean ln D(real) and −mean ln(1 − D(fake)): the two parts of D's loss, for weights p on the scene's batch */
export function dLossParts(p: Params, real: Float64Array, fake: Float64Array): [number, number] {
	const lr = logits(p, real);
	const lf = logits(p, fake);
	// −ln(1 − σ(l)) = ln(1 + e^{l})
	return [mean(lr.map((l) => softplus(-l))), mean(lf.map((l) => softplus(l)))];
}
