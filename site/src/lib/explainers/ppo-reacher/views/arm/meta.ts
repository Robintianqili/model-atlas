// ppo-reacher view 'arm': the two-link arm seen from above, the target and the fingertip's path. On the 'episode' tape it
// shows the episode up to the step's frame; an 'act' frame adds the two torques' bell curves (μ, σ and the sampled a).
// On a train tape it plays the policy at that frame's weights without its randomness (a = μ) to the learner's target;
// props.curve adds the average episode return of every iteration up to the frame, props.eval the limits step's test:
// the trained arm sent to 8 targets above the shoulder line and their 8 mirror images below it.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { EVAL_DOWN, EVAL_UP, meanEpisode, targetOf, tip, type ArmState, type Params, type Pt } from '../../sim.ts';
import { episode } from '../../process.ts';
import { ppo } from '../../trainer.ts';

export interface Act { mu: Pt; sigma: Pt; z: Pt; a: Pt; logp: number }
export interface Scene { states: ArmState[]; target: Pt; d: number[]; act: Act | null; policy: boolean; curve?: number[]; iters?: number; eval?: { g: Pt; p: Pt; d: number; up: boolean }[] }
const S = (m: Record<string, number>): ArmState => [m.q1, m.q2, m.w1, m.w2];

/** The limits step's test, also read by the lesson and the claims: the trained arm's final fingertip for every target. */
export function evalTargets(p: Params) {
	return [...EVAL_UP.map((g) => [g, true] as const), ...EVAL_DOWN.map((g) => [g, false] as const)].map(([g, up]) => {
		const e = meanEpisode(p, g, ppo.defaults.T);
		return { g, p: tip(e.states[e.states.length - 1]), d: e.final, up };
	});
}

export function sceneOf(p: Omit<StageProps, 'ctx'>): Scene | null {
	const tape = p.at.tape;
	// a new input restarts the sources: until the lesson re-resolves the step, its frame index may not be recorded yet
	if (!tape || p.at.index < 0 || p.at.index >= tape.length) return null;
	const target = targetOf(p.rt.input), st = tape.state(p.at.index) as { params?: Params };
	if (st.params) {
		const e = meanEpisode(st.params, target, ppo.defaults.T), U = ppo.defaults.snapshotEvery;
		// one point per iteration: the snapshot that starts it (its batch was played by that iteration's policy)
		const curve = p.props.curve ? Array.from({ length: p.at.index + 1 }, (_, i) => tape.frame(i)).filter((f) => f.phase === 'snapshot' && (f.step ?? 0) % U === 0).map((f) => f.metrics?.epRet ?? NaN) : undefined;
		const iters = Math.floor(Number((tape.opt as { steps?: number } | undefined)?.steps ?? ppo.defaults.steps) / U);
		return { states: e.states, target, d: e.d, act: null, policy: true, curve, iters, eval: p.props.eval ? evalTargets(st.params) : undefined };
	}
	const states: ArmState[] = [], d: number[] = [];
	let act: Act | null = null, g = target;
	for (let i = 0; i <= p.at.index; i++) {
		const f = tape.frame(i), m = f.metrics ?? {};
		if (f.phase === 'start') states.push(S(m)), (g = [m.tx, m.ty]);
		if (f.phase === 'move') states.push(S(m)), d.push(m.d);
		if (i === p.at.index && f.phase === 'act') act = { mu: [m.mu1, m.mu2], sigma: [m.s1, m.s2], z: [m.z1, m.z2], a: [m.a1, m.a2], logp: m.logp };
	}
	return states.length ? { states, target: g, d, act, policy: false } : null;
}

export function describeArm(p: Omit<StageProps, 'ctx'>): I18n {
	const s = sceneOf(p);
	if (!s) return { en: 'The arm: nothing to show yet.', zh: '机械臂：还没有可显示的内容。' };
	const last = s.states[s.states.length - 1], f = tip(last), dd = Math.hypot(s.target[0] - f[0], s.target[1] - f[1]);
	const head = { en: `Target at (${fmt3(s.target[0])}, ${fmt3(s.target[1])}) m; fingertip at (${fmt3(f[0])}, ${fmt3(f[1])}) m, ${fmt3(dd)} m away, after ${s.states.length - 1} steps.`, zh: `目标在 (${fmt3(s.target[0])}, ${fmt3(s.target[1])}) m；${s.states.length - 1} 步之后，指尖在 (${fmt3(f[0])}, ${fmt3(f[1])}) m，相距 ${fmt3(dd)} m。` };
	const a = s.act ? { en: ` Torque means μ = (${fmt3(s.act.mu[0])}, ${fmt3(s.act.mu[1])}), σ = (${fmt3(s.act.sigma[0])}, ${fmt3(s.act.sigma[1])}), sampled a = (${fmt3(s.act.a[0])}, ${fmt3(s.act.a[1])}).`, zh: `力矩均值 μ = (${fmt3(s.act.mu[0])}, ${fmt3(s.act.mu[1])})，σ = (${fmt3(s.act.sigma[0])}, ${fmt3(s.act.sigma[1])})，采样得到 a = (${fmt3(s.act.a[0])}, ${fmt3(s.act.a[1])})。` } : { en: '', zh: '' };
	const c = s.curve?.length ? { en: ` Iteration ${s.curve.length - 1}: average episode return ${fmt3(s.curve[s.curve.length - 1])}.`, zh: `第 ${s.curve.length - 1} 次迭代：平均回合回报 ${fmt3(s.curve[s.curve.length - 1])}。` } : { en: '', zh: '' };
	const ev = s.eval ? (() => { const m = (up: boolean) => fmt3(s.eval!.filter((e) => e.up === up).reduce((x, e) => x + e.d, 0) / 8); return { en: ` Test targets: average miss ${m(true)} m above the shoulder line, ${m(false)} m below it.`, zh: `测试目标：肩膀那条线以上平均差 ${m(true)} m，以下平均差 ${m(false)} m。` }; })() : { en: '', zh: '' };
	return { en: `${s.policy ? 'The policy without randomness. ' : ''}${head.en}${a.en}${c.en}${ev.en}`, zh: `${s.policy ? '去掉随机性的策略。' : ''}${head.zh}${a.zh}${c.zh}${ev.zh}` };
}

const meta = {
	key: 'ppo-reacher/arm',
	title: { en: 'The arm, the target and the fingertip’s path', zh: '机械臂、目标和指尖的轨迹' },
	kinds: ['process', 'train'],
	describe: describeArm,
	async demo() {
		const input = { target: [-0.3, 0.6], eps: 0.2 };
		const tape = record(episode, input, {}, { seed: 1, recordDraws: true });
		const rt = fakeRuntime({ slug: 'ppo-reacher', input, tapes: { episode: tape } });
		return { rt, source: 'episode', at: atFor(rt, 'episode', 21), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
