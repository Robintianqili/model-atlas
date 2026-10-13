// View 'fno/field': the learner's curve and what the network and the solver do to it. props.mode:
//   input    the starting curve            solver   a frame of the solver (source solve) over the starting curve
//   lift     the 4 channels after the lift fft      the wave heights of the followed channel (layer f0.fft)
//   keep     the kept and dropped modes, and the channel rebuilt from the kept ones
//   back     inverse FFT, pointwise path, their sum and GELU, for the followed channel
//   compare  the prediction against the solver   gain     the learned filter next to the solver's, mode by mode
//   modes    the slider for the kept modes and the largest error for every choice
//   limits   the curve made TALL times taller: solver and network
// Pure helpers here, the drawing is View.svelte.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { I, N, lowPass, presetY, solve } from '../../heat.ts';
import { K } from '../../adapters/series/adapter.ts';
import { CH, T, col, errByKeep, fewest, heights, learnedGain, solverGain, stateOf, taller } from '../../fno.ts';

export type Mode = 'input' | 'solver' | 'lift' | 'fft' | 'keep' | 'back' | 'compare' | 'gain' | 'modes' | 'limits';
export const TALL = 4;

export function fieldData(p: Omit<StageProps, 'ctx'>) {
	const mode = (p.props.mode ?? 'compare') as Mode, rt = p.rt, st = stateOf(rt), ch = CH(rt);
	const frameU = (p.at.state as { u?: Tensor } | undefined)?.u;
	const lifted = col(T(rt, 'lift'), ch);
	return {
		mode, st, ch, frameU: frameU ? Array.from(frameU.data) : null, step: Number(p.at.frame?.step ?? NaN),
		chans: [0, 1, 2, 3].map((c) => col(T(rt, 'lift'), c)),
		spec: heights(T(rt, 'f0.fft'), ch),
		lifted, rebuilt: st && mode === 'keep' ? Array.from(lowPass(lifted, st.keep)) : [],
		back: ['f0.ifft', 'w0', 'f0.sum', 'f0.gelu'].map((id) => col(T(rt, id), ch)),
		gain: mode === 'gain' && rt.model ? { learned: learnedGain(rt.model), solver: solverGain() } : null,
		errs: mode === 'modes' && rt.model && st ? errByKeep(rt.model, st.u0) : [],
		tall: mode === 'limits' ? taller(rt, TALL) : null
	};
}

export function describeField(p: Omit<StageProps, 'ctx'>): I18n {
	const d = fieldData(p), s = d.st;
	if (!s) return I('No curve yet: pick one or draw one.', '还没有曲线：选一条或画一条。');
	const f = fmt3;
	switch (d.mode) {
		case 'solver': return I(`The solver after step ${d.step}: the curve spans ${f(Math.max(...(d.frameU ?? [])) - Math.min(...(d.frameU ?? [])))}.`, `求解器第 ${d.step} 步之后：曲线高低相差 ${f(Math.max(...(d.frameU ?? [])) - Math.min(...(d.frameU ?? [])))}。`);
		case 'lift': return I(`Four channels, each a scaled and shifted copy of the curve; channel ${d.ch} is followed.`, `四个通道，每个都是曲线缩放并平移后的副本；跟踪的是通道 ${d.ch}。`);
		case 'fft': case 'keep': return I(`Wave heights of channel ${d.ch} for modes 0 to ${d.spec.length - 1}; ${s.keep} modes kept.`, `通道 ${d.ch} 在模式 0 到 ${d.spec.length - 1} 上的波高；保留 ${s.keep} 个模式。`);
		case 'back': return I(`Channel ${d.ch}: inverse FFT, pointwise path, their sum and GELU over the 64 points.`, `通道 ${d.ch}：逆 FFT、逐点通路、两者之和以及 GELU，覆盖 64 个点。`);
		case 'gain': return I('The share of each wave the network keeps, next to the solver’s share, modes 0 to 32.', '网络对每个波保留的比例，与求解器的比例对照，模式 0 到 32。');
		case 'modes': return I(`${s.keep} modes kept: largest error ${f(s.err)}; the fewest modes with an error below 0.02: ${fewest(d.errs)}.`, `保留 ${s.keep} 个模式：最大误差 ${f(s.err)}；误差低于 0.02 的最少模式数：${fewest(d.errs)}。`);
		case 'limits': return I(`The curve ${TALL} times taller: largest error ${f(d.tall?.err ?? NaN)}, against ${f(s.err)} for your curve.`, `曲线加高到 ${TALL} 倍：最大误差 ${f(d.tall?.err ?? NaN)}，而你的曲线是 ${f(s.err)}。`);
		case 'compare': return I(`Prediction and solver: largest difference ${f(s.err)} at point ${s.at}.`, `预测和求解器：最大差距 ${f(s.err)}，在点 ${s.at}。`);
		default: return I(`A curve of ${N} heights from ${f(Math.min(...s.u0))} to ${f(Math.max(...s.u0))}.`, `一条由 ${N} 个高度组成的曲线，从 ${f(Math.min(...s.u0))} 到 ${f(Math.max(...s.u0))}。`);
	}
}

const meta = {
	key: 'fno/field',
	title: I('The curve, its waves and the prediction', '曲线、它的波和预测'),
	kinds: ['trace', 'process'],
	describe: describeField,
	async demo() {
		// made-up prediction: the solver's answer itself (the demo has no model)
		const u = Float32Array.from(presetY('square')), t = (shape: number[], data: Float32Array) => ({ shape, data });
		const rt = fakeRuntime({ slug: 'fno', input: { y: Array.from(u) }, trace: { order: ['input', 'proj'], t: { input: t([N, 1], u), proj: t([N, 1], Float32Array.from(solve(u))) } } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'compare' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
export { K };
