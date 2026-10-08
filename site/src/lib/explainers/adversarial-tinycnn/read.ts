// adversarial-tinycnn: small readers shared by the lesson, the view and the claim tests.
import type { ExplainerRuntime, I18n, Tape } from '../../atlas/types/index.ts';
import type { AttackState } from './process.ts';
import { fmtSetting } from '../../atlas/core/format.ts';

export type RT = ExplainerRuntime;
export type St = Partial<AttackState>;
/** the ε of the measured facts (pipeline/adversarial-tinycnn/stats.py): the table rows (fgsm05 ... pgd20) and the
 *  transfer numbers (transferVit, vitPgd10), which are at TABLE_EPS */
export const TABLE = [{ eps: 0.05, key: '05' }, { eps: 0.1, key: '10' }, { eps: 0.2, key: '20' }];
export const TABLE_EPS = 0.1;
/** the task's tolerance: within this much of the smallest flipping ε */
export const NEAR = 0.01;

/** an ε (a multiple of 0.005, or a measured median) as the setting it is: '0.1', '0.065', never '0.100' (the kit's
 *  fmtSetting, K2); NaN when missing, so lint sees it */
export const epsText = (v: number | undefined): string | number => (typeof v === 'number' && Number.isFinite(v) ? fmtSetting(v) : NaN);

/** a class name from TinyCNN's spec (the trace source's model) */
export function cls(rt: RT, i: number): I18n {
	const c = (rt.sources.fwd?.model?.spec.classes as { en: string; zh: string }[] | undefined)?.[i];
	return c ? { en: c.en, zh: c.zh } : { en: `#${i}`, zh: `#${i}` };
}

/** the state at the first frame of a phase (the FGSM step, the clean drawing) */
export function stateOf(tape: Tape | null | undefined, phase: string): St | null {
	if (!tape) return null;
	for (let i = 0; i < tape.length; i++) if (tape.frame(i).phase === phase) return tape.state(i) as St;
	return null;
}

/** the PGD curve up to frame `upto`: p(y) and the best other class's share at steps 0 (your drawing) .. k */
export function pgdCurve(tape: Tape | null | undefined, upto: number) {
	const py: number[] = [], other: number[] = [];
	if (!tape) return { py, other };
	for (let i = 0; i <= Math.min(upto, tape.length - 1); i++) {
		const f = tape.frame(i);
		if (f.phase !== 'clean' && f.phase !== 'pgd') continue;
		const p = (tape.state(i) as St).p?.data;
		const y = f.metrics?.y ?? 0;
		if (!p) continue;
		py.push(p[y]);
		let o = 0;
		for (let j = 0; j < p.length; j++) if (j !== y) o = Math.max(o, p[j]);
		other.push(o);
	}
	return { py, other };
}

/** the task: one FGSM step flips the answer at the learner's ε, and that ε is within NEAR of the smallest one */
export function nearSmallest(m: Record<string, number> | undefined): boolean {
	return !!m && m.found === 1 && m.flipped === 1 && m.eps <= m.minEps + NEAR + 1e-9;
}
