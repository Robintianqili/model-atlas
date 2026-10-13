// View 'optimizers/surface': a loss surface of two weights seen from above (the kit's sequential scale on the log of the
// loss), with each run's path so far, where it is now and the lowest points. The surface is the frame's metrics.surf
// (data.ts SURFACES). A race frame (process 'race') has four runs (SGD, momentum, Adam, AdamW); a cliff frame (process
// 'cliff') has two (without and with clipping; metrics.mine is the learner's, the other is drawn dashed). Props:
//   show?: RacerKey[]              which racers to draw (default all four)
//   glyph?: 'momentum' | 'adam'    the optimizer's state: momentum's last step (−η·v) next to the step the gradient alone
//                                  would take (−η·g); Adam's squared-gradient squares (side √v̂ per weight, so the
//                                  area is v̂) and the step they give
//   formula?: true                 the frame's update formula with its numbers (zoom.formula), under the plot
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { BETAS, EPS, RACERS, SURFACES, type RacerKey, type Surface } from '../../data.ts';

export interface Run {
	key: string;
	name: I18n;
	color: number;
	dashed: boolean;
	path: number[][];
	now: number[];
	loss: number;
	inside: boolean;
}
export interface Glyph {
	kind: 'momentum' | 'adam';
	/** the last step taken, from → to (weights), and for momentum the gradient-only step from the same point */
	from: number[];
	to: number[];
	plain: number[] | null;
	/** momentum: v and g; adam: m̂, √v̂ and the step, per weight (x, y) */
	a: [number, number];
	b: [number, number];
	step: [number, number];
	t: number;
}
export interface SurfacePic {
	surface: Surface;
	runs: Run[];
	glyph: Glyph | null;
	steps: number;
}

const CLIFF_RUNS: I18n[] = [
	{ en: 'without clipping', zh: '不裁剪' },
	{ en: 'with clipping', zh: '裁剪' }
];

const rowsOf = (t: Tensor | undefined, r: number): number[][] => {
	if (!t || t.shape.length !== 3) return [];
	const n = t.shape[1];
	return Array.from({ length: n }, (_, i) => [t.data[(r * n + i) * 2], t.data[(r * n + i) * 2 + 1]]);
};
export const inWindow = (s: Surface, q: number[]) => q[0] >= s.xr[0] && q[0] <= s.xr[1] && q[1] >= s.yr[0] && q[1] <= s.yr[1];

/** What the view draws (pure; also the text form). Null when the frame has no runs. */
export function surfacePic(p: Pick<StageProps, 'at' | 'props'>): SurfacePic | null {
	const st = (p.at?.state ?? {}) as { path?: Tensor; mom?: Tensor; am?: Tensor; av?: Tensor };
	const m = p.at?.frame?.metrics ?? {};
	const surface = SURFACES[m.surf ?? -1];
	if (!surface || !st.path) return null;
	const cliff = surface.id === 'cliff';
	const show = (p.props?.show as RacerKey[] | undefined) ?? RACERS.map((r) => r.key);
	const runs: Run[] = [];
	const n = st.path.shape[0];
	for (let i = 0; i < n; i++) {
		const path = rowsOf(st.path, i);
		const now = path[path.length - 1];
		if (!now) continue;
		const key = cliff ? (i ? 'clip' : 'plain') : RACERS[i].key;
		if (!cliff && !show.includes(key as RacerKey)) continue;
		runs.push({ key, name: cliff ? CLIFF_RUNS[i] : RACERS[i].name, color: i, dashed: cliff && i !== m.mine, path, now, loss: surface.f(now[0], now[1]), inside: inWindow(surface, now) });
	}
	let glyph: Glyph | null = null;
	const g = p.props?.glyph;
	if (!cliff && (g === 'momentum' || g === 'adam')) {
		const path = rowsOf(st.path, g === 'momentum' ? 1 : 2);
		const t = path.length - 1;
		if (t >= 1) {
			const from = path[t - 1];
			const to = path[t];
			const step: [number, number] = [to[0] - from[0], to[1] - from[1]];
			if (g === 'momentum' && st.mom) {
				const gr = surface.grad(from[0], from[1]);
				const lr = RACERS[1].lr;
				glyph = { kind: g, from, to, plain: [from[0] - lr * gr[0], from[1] - lr * gr[1]], a: [st.mom.data[0], st.mom.data[1]], b: gr, step, t };
			} else if (g === 'adam' && st.am && st.av) {
				const c1 = 1 - Math.pow(BETAS[0], t);
				const c2 = 1 - Math.pow(BETAS[1], t);
				const mh: [number, number] = [st.am.data[0] / c1, st.am.data[1] / c1];
				const sv: [number, number] = [Math.sqrt(st.av.data[0] / c2), Math.sqrt(st.av.data[1] / c2)];
				const lr = RACERS[2].lr;
				glyph = { kind: g, from, to, plain: null, a: mh, b: sv, step: [(-lr * mh[0]) / (sv[0] + EPS), (-lr * mh[1]) / (sv[1] + EPS)], t };
			}
		}
	}
	return { surface, runs, glyph, steps: Math.max(0, ...runs.map((r) => r.path.length - 1)) };
}

const pair = (v: number[]) => `(${fmt3(v[0])}, ${fmt3(v[1])})`;

export function describeSurface(pic: SurfacePic | null): I18n {
	if (!pic) return { en: 'No runs to draw yet.', zh: '还没有可画的运行。' };
	const s = pic.surface;
	const [ax, ay] = s.axes;
	const runs = pic.runs.map((r) => ({
		en: ` ${r.name.en}: at ${pair(r.now)}, loss ${fmt3(r.loss)}${r.inside ? '' : ', outside the window'}.`,
		zh: `${r.name.zh ?? r.name.en}：位于 ${pair(r.now)}，损失 ${fmt3(r.loss)}${r.inside ? '' : '，在窗口之外'}。`
	}));
	const low = s.min.length ? { en: ` The lowest loss, 0, is at ${s.min.map(pair).join(' and ')}.`, zh: `最低损失 0 位于 ${s.min.map(pair).join(' 和 ')}。` } : { en: '', zh: '' };
	let gl = { en: '', zh: '' };
	const g = pic.glyph;
	if (g?.kind === 'momentum')
		gl = {
			en: ` Momentum's last step moved the weights by ${pair(g.step)}, using its velocity v = ${pair(g.a)}; the gradient alone, ${pair(g.b)}, would have moved them to ${pair(g.plain!)}.`,
			zh: `动量法的上一步让权重移动了 ${pair(g.step)}，用的是速度 v = ${pair(g.a)}；只用梯度 ${pair(g.b)} 的话，会移到 ${pair(g.plain!)}。`
		};
	if (g?.kind === 'adam')
		gl = {
			en: ` Adam after ${fmtInt(g.t)} updates: m̂ = ${pair(g.a)}, √v̂ = ${pair(g.b)} (the squares' sides), so the step it just took was ${pair(g.step)}.`,
			zh: `Adam 更新 ${fmtInt(g.t)} 次之后：m̂ = ${pair(g.a)}，√v̂ = ${pair(g.b)}（方块的边长），所以它刚走的一步是 ${pair(g.step)}。`
		};
	return {
		en: `The loss over every setting of ${ax.en} and ${ay.en} on the ${s.name.en}, ${ax.en} from ${s.xr[0]} to ${s.xr[1]} and ${ay.en} from ${s.yr[0]} to ${s.yr[1]}, after ${fmtInt(pic.steps)} updates.${low.en}${runs.map((r) => r.en).join('')}${gl.en}`,
		zh: `${s.name.zh}上 ${ax.zh} 和 ${ay.zh} 每种取值的损失（${ax.zh} 从 ${s.xr[0]} 到 ${s.xr[1]}，${ay.zh} 从 ${s.yr[0]} 到 ${s.yr[1]}），更新 ${fmtInt(pic.steps)} 次之后。${low.zh}${runs.map((r) => r.zh).join('')}${gl.zh}`
	};
}

const meta = {
	key: 'optimizers/surface',
	title: { en: 'Optimizers on a loss surface', zh: '损失曲面上的优化器' },
	kinds: ['process'],
	describe: (p: StageProps) => describeSurface(surfacePic(p)),
	demo: async () => {
		const [{ record }, { race }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { surface: 'valley', x: -4, y: 1 };
		const tape = record(race, input, { steps: 20 });
		const rt = fakeRuntime({ slug: 'optimizers', input, tapes: { race: tape } });
		const i = tape.length - 1;
		return { rt, source: 'race', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { glyph: 'momentum' } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
