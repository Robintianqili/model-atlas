// View 'flow-2d/plane' (kind B on train:flow sources): the flow-matching plane, from −2.5 to 2.5 in x and y. Props:
//   mode    'data' | 'noise' | 'pairs' | 'slider' | 'pair' | 'field' | 'euler' | 'few'
//   pair    the index of the pair the lesson follows (ringed, with its ends x₀, x₁ and its point x_t labelled)
//   arrows  mode 'pair': which arrows of that pair to draw at its point x_t: 'target' (x1 − x0), 'pred' (the network)
//   t       the start value of the time slider (modes 'slider' and 'field')
// Numbers come from the step's frame: the batch of a fine frame (state.batch: inp = (x_t, y_t, t) and target = x1 − x0, so
// x0 = x_t − t·target and x1 = x_t + (1 − t)·target) and the network's outputs (state.acts['net.6']); the weights
// (state.params: the arrows over the plane); the sampled points (state.x) and their paths (the tape's earlier frames).
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { dataOf, type Params } from '../../flow.ts';
import type { Points2D } from '../../../../atlas/data/points2d.ts';

export type Mode = 'data' | 'noise' | 'pairs' | 'slider' | 'pair' | 'field' | 'euler' | 'few';
export interface Pairs {
	n: number;
	x0: Float64Array;
	x1: Float64Array;
	t: Float64Array;
	target: Float64Array;
}

const st = (p: Pick<StageProps, 'at'>) => (p.at?.state ?? {}) as Record<string, unknown>;

/** the batch of the step's frame as pairs (null without a batch) */
export function pairsOf(p: Pick<StageProps, 'at'>): Pairs | null {
	const b = st(p).batch as Record<string, Tensor> | undefined;
	if (!b?.inp || !b.target) return null;
	const n = b.inp.shape[0];
	const out: Pairs = { n, x0: new Float64Array(2 * n), x1: new Float64Array(2 * n), t: new Float64Array(n), target: Float64Array.from(b.target.data) };
	for (let i = 0; i < n; i++) {
		const t = b.inp.data[3 * i + 2];
		out.t[i] = t;
		for (let c = 0; c < 2; c++) {
			out.x0[2 * i + c] = b.inp.data[3 * i + c] - t * b.target.data[2 * i + c];
			out.x1[2 * i + c] = b.inp.data[3 * i + c] + (1 - t) * b.target.data[2 * i + c];
		}
	}
	return out;
}

/** the network's arrow for pair i of the frame's batch */
export function predOf(p: Pick<StageProps, 'at'>, i = 0): [number, number] | null {
	const y = (st(p).acts as Record<string, Tensor> | undefined)?.['net.6'];
	return y && y.data.length >= 2 * i + 2 ? [y.data[2 * i], y.data[2 * i + 1]] : null;
}

/** sampled points of the step's frame and their paths: every frame of the same run from its start to this frame */
export function samplesOf(p: Pick<StageProps, 'at'>): { x: Tensor; path: Float64Array[]; run: 'move' | 'moveFew' } | null {
	const x = st(p).x as Tensor | undefined;
	const tape = p.at?.tape;
	const ph = p.at?.frame?.phase;
	if (!x || !tape || !ph) return null;
	const run = ph === 'noiseFew' || ph === 'moveFew' ? 'moveFew' : 'move';
	const path: Float64Array[] = [];
	for (let i = tape.find({ phase: run === 'move' ? 'noise' : 'noiseFew' }); i >= 0 && i <= p.at!.index; i++) path.push(Float64Array.from((tape.state(i) as { x: Tensor }).x.data));
	return { x, path, run };
}

/** the last frame of a run of the step's tape: its points and metrics */
export function runEnd(p: Pick<StageProps, 'at'>, phase: 'move' | 'moveFew') {
	const tape = p.at?.tape;
	const i = tape ? tape.find({ phase, last: true }) : -1;
	if (!tape || i < 0) return null;
	const s = tape.state(i) as { x: Tensor };
	const start = tape.frame(tape.find({ phase: phase === 'move' ? 'noise' : 'noiseFew' }));
	return { x: s.x, K: start?.metrics?.K ?? NaN, dist: tape.frame(i).metrics?.dist ?? NaN, dist0: start?.metrics?.dist ?? NaN };
}

/** the middle of the data: the mean of its points */
export function middleOf(data: Points2D): [number, number] {
	let [mx, my] = [0, 0];
	for (let j = 0; j < data.n; j++) (mx += data.xy[2 * j] / data.n), (my += data.xy[2 * j + 1] / data.n);
	return [mx, my];
}
/** the mean distance of n points (xy [n·2]) from a point m: how far out from the middle they sit */
export function spreadAround(xy: ArrayLike<number>, n: number, m: [number, number]): number {
	let s = 0;
	for (let i = 0; i < n; i++) s += Math.hypot(xy[2 * i] - m[0], xy[2 * i + 1] - m[1]);
	return n ? s / n : NaN;
}

export const paramsOf = (p: Pick<StageProps, 'at'>) => (st(p).params as Params | undefined) ?? null;
export const modeOf = (p: Pick<StageProps, 'props'>) => ((p.props?.mode as Mode | undefined) ?? 'data');
export const pairIndex = (p: Pick<StageProps, 'props'>) => Math.max(0, Math.floor(Number(p.props?.pair ?? 0)) || 0);

export function describePlane(p: Pick<StageProps, 'rt' | 'at' | 'props'>): I18n {
	const data = dataOf(p.rt.input);
	const mode = modeOf(p);
	if (!data.n) return { en: 'The plane has no points yet.', zh: '平面上还没有点。' };
	const parts: I18n[] = [{ en: `${fmtInt(data.n)} data points (green).`, zh: `${fmtInt(data.n)} 个数据点（绿色）。` }];
	const pr = pairsOf(p);
	if (pr && ['noise', 'pairs', 'slider', 'pair'].includes(mode)) {
		parts.push({ en: ` ${fmtInt(pr.n)} noise points (grey)`, zh: `${fmtInt(pr.n)} 个噪声点（灰色）` });
		const k = Math.min(pairIndex(p), pr.n - 1);
		const at2 = (a: Float64Array) => `(${fmt3(a[2 * k])}, ${fmt3(a[2 * k + 1])})`;
		const how = mode === 'pairs' ? { en: 'the thick line', zh: '粗线' } : { en: 'ringed', zh: '圈出的' };
		if (mode !== 'noise') parts.push({ en: `, each joined to a data point by a straight line; pair ${k} (${how.en}) goes from ${at2(pr.x0)} to ${at2(pr.x1)}.`, zh: `，每个都用一条直线连到一个数据点；第 ${k} 对（${how.zh}）从 ${at2(pr.x0)} 连到 ${at2(pr.x1)}。` });
		else parts.push({ en: '.', zh: '。' });
		const q = predOf(p, k);
		if (mode === 'pair' && q) parts.push({ en: ` Target arrow ${at2(pr.target)}, the network's arrow (${fmt3(q[0])}, ${fmt3(q[1])}).`, zh: `目标箭头 ${at2(pr.target)}，网络的箭头 (${fmt3(q[0])}, ${fmt3(q[1])})。` });
	}
	if (mode === 'field' && paramsOf(p)) parts.push({ en: ` Arrows: the network's velocity on a 20 × 20 grid, after ${fmtInt(Number(st(p).step ?? 0))} training steps.`, zh: `箭头：网络在 20 × 20 网格上的速度，训练了 ${fmtInt(Number(st(p).step ?? 0))} 步。` });
	const s = samplesOf(p);
	if (mode === 'euler' && s) parts.push({ en: ` ${fmtInt(s.x.shape[0])} sampled points (pink) after ${fmtInt(s.path.length - 1)} Euler steps, with their paths.`, zh: `${fmtInt(s.x.shape[0])} 个采样点（粉色），走了 ${fmtInt(s.path.length - 1)} 步欧拉法，并画出路径。` });
	const a = runEnd(p, 'move'), b = runEnd(p, 'moveFew');
	if (mode === 'few' && a && b) {
		const m = middleOf(data);
		const [r4, r50, rd] = [spreadAround(b.x.data, b.x.shape[0], m), spreadAround(a.x.data, a.x.shape[0], m), spreadAround(data.xy, data.n, m)].map((v) => fmt3(v));
		parts.push({ en: ` Left: ${fmtInt(b.K)} steps, mean distance to the data ${fmt3(b.dist)}, to the middle ${r4}. Right: ${fmtInt(a.K)} steps, ${fmt3(a.dist)} and ${r50}. Your points lie ${rd} from the middle (the cross).`, zh: `左：${fmtInt(b.K)} 步，到数据的平均距离 ${fmt3(b.dist)}，到中间 ${r4}。右：${fmtInt(a.K)} 步，${fmt3(a.dist)} 和 ${r50}。你的点离中间 ${rd}（十字处）。` });
	}
	return { en: parts.map((x) => x.en).join(''), zh: parts.map((x) => x.zh).join('') };
}

const meta = {
	key: 'flow-2d/plane',
	title: { en: 'Noise, pairs, arrows and samples on the plane', zh: '平面上的噪声、配对、箭头和样本' },
	kinds: ['train', 'process'],
	describe: (p: StageProps): I18n => describePlane(p),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { presetOf }, { default: flow }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../adapters/points2d/adapter.ts'), import('../../process.ts')]);
		const raw = presetOf('ring');
		const tape = record(flow, raw, { steps: 60, snapshotEvery: 20, particles: 100, K: 10 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'flow-2d', input: raw, tapes: { flow: tape } });
		return { rt, source: 'flow', at: atFor(rt, 'flow', tape.find({ phase: 'move', last: true })), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'euler' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
