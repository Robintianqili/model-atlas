// egnn process (kind C).
//   spin   the exported EGNN on the learner's molecule turned by 0, 30, …, 330 degrees about its centre, one frame per
//          turn: the energy, its difference from the unturned energy, the direction of the force on the followed atom
//          (look.ts focus) and how far it turned, and the largest difference between the forces on the turned molecule
//          and the unturned forces turned by the same angle. It runs on the main thread (its input holds the model, the
//          parity-checked executor), so every number comes from the same forward pass as the trace.
import type { ExplainerRuntime, GraphModel, ProcessDef, ProcessRun } from '../../atlas/types/index.ts';
import { angleOf, molOf, turn, turnVec } from './mol.ts';
import { focus, run, wrap } from './look.ts';

export const STEP_DEG = 30;
export type SpinInput = { model: GraphModel; pos: Float32Array; kind: Float32Array; at: number } | null;
type SpinState = { pos: number[]; f: number[]; deg: number };

/** the spin input from the runtime: the learner's molecule as the adapter gives it to the model (float32 positions,
 *  one-hot kinds; it needs no trace, so a deep link to the step works) and the model; null without atoms */
export function spinInput(rt: ExplainerRuntime): SpinInput {
	const m = molOf(rt.input);
	if (!rt.model || !m.n) return null;
	const kind = new Float32Array(3 * m.n);
	m.kind.forEach((k, i) => (kind[3 * i + k] = 1));
	return { model: rt.model, pos: Float32Array.from(m.pos), kind, at: focus(rt) };
}

export const spin: ProcessDef<SpinInput, Record<string, never>, SpinState> = {
	id: 'spin',
	kind: 'inference',
	title: { en: 'One full turn of your molecule', zh: '把你的分子转一整圈' },
	phases: { turn: { label: { en: 'Turn and run the network again', zh: '转动，再跑一遍网络' }, kind: 'forward' } },
	schema: {
		pos: { kind: 'json', label: { en: 'Turned positions', zh: '转动后的位置' } },
		f: { kind: 'json', label: { en: 'Forces on the turned molecule', zh: '转动后分子上的力' } },
		deg: { kind: 'int', label: { en: 'Turn (degrees)', zh: '转角（度）' }, tol: 'exact' }
	},
	defaults: {},
	*run(inp): ProcessRun<SpinState, Record<string, never>> {
		if (!inp) return;
		const { model, pos, kind, at } = inp;
		const o0 = run(model, pos, kind);
		const n = pos.length / 2;
		const a0 = angleOf(o0.f[2 * at], o0.f[2 * at + 1]);
		for (let k = 0; k * STEP_DEG < 360; k++) {
			const deg = k * STEP_DEG;
			const o = deg ? run(model, turn(pos, deg), kind) : o0;
			let df = 0;
			for (let i = 0; i < n; i++) {
				const [x, y] = turnVec(o0.f[2 * i], o0.f[2 * i + 1], deg);
				df = Math.max(df, Math.abs(o.f[2 * i] - x), Math.abs(o.f[2 * i + 1] - y));
			}
			const ang = angleOf(o.f[2 * at], o.f[2 * at + 1]);
			yield {
				phase: 'turn',
				step: k + 1,
				state: { pos: Array.from(o.pos), f: Array.from(o.f), deg },
				metrics: { deg, E: o.E, dE: o.E - o0.E, at, ang, turned: wrap(ang - a0), df, n }
			};
		}
	}
};
