// cart-pole view 'scene': the cart on its track and the pole, the four numbers of the state and the pole's angle over
// the episode. On an 'episode' tape it shows the episode up to the step's frame (an 'act' frame adds the p(left) gauge
// with the draw u). On a train tape it plays one episode of the policy at that frame's weights from the learner's start
// state, with the same draws as the episode process (rng.fork('batch') of seed 1), so only the weights differ, and shows its
// first push (p(left) and u at the start);
// props.curve draws the average episode length of every round up to the frame instead of the first push.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { createRng } from '../../../../atlas/core/rng.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { deg, lenText, rollout, startOf, type Params, type State } from '../../sim.ts';
import { episode } from '../../process.ts';
import { reinforce } from '../../trainer.ts';

export interface Scene {
	states: State[];
	t: number;
	len: number;
	end: number;
	act: { p: number; u: number; left: number } | null;
	/** a policy's episode (without the curve): its first push, from p(left) and u at the start */
	first?: { p: number; u: number; left: number };
	force: number;
	policy: boolean;
	curve?: number[];
	rounds?: number;
	maxLen?: number;
}
const S = (m: Record<string, number>, k = ''): State => [m[`x${k}`], m[`xd${k}`], m[`th${k}`], m[`thd${k}`]];

/** The trained (or fresh) policy's episode from the learner's start: the lesson's numbers use the same function. */
export function policyEpisode(params: Params, raw: unknown) {
	return rollout(params, startOf(raw), createRng(1).fork('batch'), reinforce.defaults.maxLen);
}

export function sceneOf(p: Omit<StageProps, 'ctx'>): Scene | null {
	const tape = p.at.tape;
	// a new input restarts the sources: until the lesson re-resolves the step, its frame index may not be recorded yet
	if (!tape || p.at.index < 0 || p.at.index >= tape.length) return null;
	const st = tape.state(p.at.index) as { params?: Params };
	if (st.params) {
		const e = policyEpisode(st.params, p.rt.input);
		const sc: Scene = { states: [...e.states, e.last], t: e.len, len: e.len, end: e.end, act: null, force: 0, policy: true };
		if (p.props.curve) {
			const curve: number[] = [];
			for (let i = 0; i <= p.at.index; i++) {
				const v = tape.frame(i).metrics?.meanLen;
				if (typeof v === 'number') curve.push(v);
			}
			Object.assign(sc, { curve, rounds: Number((tape.opt as { steps?: number } | undefined)?.steps ?? reinforce.defaults.steps), maxLen: reinforce.defaults.maxLen });
		} else sc.first = { p: e.p[0], u: e.u[0], left: e.left[0] };
		return sc;
	}
	const states: State[] = [];
	let act: Scene['act'] = null, force = 0, end = 0;
	for (let i = 0; i <= p.at.index; i++) {
		const f = tape.frame(i), m = f.metrics ?? {};
		if (f.phase === 'start') states.push(S(m));
		if (f.phase === 'move') states.push(S(m, '2'));
		if (i === p.at.index && f.phase === 'act') act = { p: m.p, u: m.u, left: m.left };
		if (i === p.at.index && f.phase === 'move') force = m.force;
		if (f.phase === 'end') end = m.end;
	}
	return states.length ? { states, t: states.length - 1, len: states.length - 1, end, act, force, policy: false } : null;
}

export function describeScene(p: Omit<StageProps, 'ctx'>): I18n {
	const s = sceneOf(p);
	if (!s) return { en: 'The cart-pole: nothing to show yet.', zh: '倒立摆：还没有可显示的内容。' };
	const [x, xd, th, thd] = s.states[s.policy ? 0 : s.t];
	const head = { en: `Cart at x = ${fmt3(x)} m moving at ${fmt3(xd)} m/s; pole at ${fmt3(deg(th))}° from upright, turning at ${fmt3(thd)} rad/s.`, zh: `小车在 x = ${fmt3(x)} m，速度 ${fmt3(xd)} m/s；杆偏离竖直 ${fmt3(deg(th))}°，角速度 ${fmt3(thd)} rad/s。` };
	const c = s.curve?.length ? { en: ` Round ${s.curve.length - 1}: episodes last ${lenText(s.curve[s.curve.length - 1])} steps on average.`, zh: `第 ${s.curve.length - 1} 轮：回合平均持续 ${lenText(s.curve[s.curve.length - 1])} 步。` } : { en: '', zh: '' };
	const f = s.first ? { en: ` First push: p(left) = ${fmt3(s.first.p)}, u = ${fmt3(s.first.u)}, so ${s.first.left ? 'left' : 'right'}.`, zh: `第一推：p(左) = ${fmt3(s.first.p)}，u = ${fmt3(s.first.u)}，所以向${s.first.left ? '左' : '右'}。` } : { en: '', zh: '' };
	if (s.policy) return { en: `The policy plays from your start. ${head.en}${f.en} The episode lasts ${s.len} steps.${c.en}`, zh: `策略从你的起点开始玩。${head.zh}${f.zh}这个回合持续 ${s.len} 步。${c.zh}` };
	if (s.act) return { en: `${head.en} p(left) = ${fmt3(s.act.p)}, the draw u = ${fmt3(s.act.u)}: push ${s.act.left ? 'left' : 'right'}.`, zh: `${head.zh}p(左) = ${fmt3(s.act.p)}，随机数 u = ${fmt3(s.act.u)}：向${s.act.left ? '左' : '右'}推。` };
	return { en: `Step ${s.t} of the episode. ${head.en}`, zh: `回合的第 ${s.t} 步。${head.zh}` };
}

const meta = {
	key: 'cart-pole/scene',
	title: { en: 'The cart, the pole and the four numbers', zh: '小车、杆和四个数' },
	kinds: ['process', 'train'],
	describe: describeScene,
	async demo() {
		const input = { start: [0, 0, 0.05, 0] };
		const tape = record(episode, input, {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'cart-pole', input, tapes: { episode: tape } });
		return { rt, source: 'episode', at: atFor(rt, 'episode', 21), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
