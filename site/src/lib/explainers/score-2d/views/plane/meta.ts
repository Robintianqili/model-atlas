// View 'score-2d/plane': the plane from −2.5 to 2.5 in x and y. Props:
//   mode     'data' | 'density' | 'score' | 'pairs' | 'field' | 'chains' | 'limits'
//   sigma    the noise level of modes 'density', 'score' and 'limits' (default 0.16; 'limits' 0.04)
//   level    mode 'field': the start position of the σ slider (an index into SIGMAS)
//   sliders  mode 'chains': show the step-size and noise sliders (they re-run the chains in the view)
// Numbers come from the step's frame: the batch of a fine frame (state.batch: inp = (x̃, ỹ, log σ), target = −ε, so
// x = x̃ + σ·target), the weights (state.params: the learned arrows), the chains (state.x, and their paths: the tape's
// earlier frames); the exact density and score come from the learner's points (score.ts exact()).
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
import { dataOf, exact, probeOf, SHOW, SMIN, type Params } from '../../score.ts';

export type Mode = 'data' | 'density' | 'score' | 'pairs' | 'field' | 'chains' | 'limits';
const st = (p: Pick<StageProps, 'at'>) => (p.at?.state ?? {}) as Record<string, unknown>;
export const modeOf = (p: Pick<StageProps, 'props'>) => (p.props?.mode as Mode | undefined) ?? 'data';
export const sigmaOf = (p: Pick<StageProps, 'props'>) => Number(p.props?.sigma ?? (modeOf(p) === 'limits' ? SMIN : SHOW));
export const paramsOf = (p: Pick<StageProps, 'at'>) => (st(p).params as Params | undefined) ?? null;


export interface Pairs { n: number; x: Float64Array; xt: Float64Array; s: Float64Array; e: Float64Array }
/** the batch of the step's frame as denoising examples (null without a batch) */
export function pairsOf(p: Pick<StageProps, 'at'>): Pairs | null {
	const b = st(p).batch as Record<string, Tensor> | undefined;
	if (!b?.inp || !b.target) return null;
	const n = b.inp.shape[0];
	const out: Pairs = { n, x: new Float64Array(2 * n), xt: new Float64Array(2 * n), s: new Float64Array(n), e: new Float64Array(2 * n) };
	for (let i = 0; i < n; i++) {
		out.s[i] = Math.exp(b.inp.data[3 * i + 2]);
		for (let c = 0; c < 2; c++) {
			out.xt[2 * i + c] = b.inp.data[3 * i + c];
			out.e[2 * i + c] = -b.target.data[2 * i + c];
			out.x[2 * i + c] = out.xt[2 * i + c] - out.s[i] * out.e[2 * i + c];
		}
	}
	return out;
}
/** the example the lesson follows: the first of the batch with σ between 0.2 and 0.5 and |ε| between 0.8 and 2 (else 0) */
export function followed(q: Pairs): number {
	for (let i = 0; i < q.n; i++) if (q.s[i] >= 0.2 && q.s[i] <= 0.5 && Math.hypot(q.e[2 * i], q.e[2 * i + 1]) >= 0.8 && Math.hypot(q.e[2 * i], q.e[2 * i + 1]) <= 2) return i;
	return 0;
}
/** the network's output d for example i of the frame's batch */
export function predOf(p: Pick<StageProps, 'at'>, i: number): [number, number] | null {
	const y = (st(p).acts as Record<string, Tensor> | undefined)?.['net.4'];
	return y && y.data.length >= 2 * i + 2 ? [y.data[2 * i], y.data[2 * i + 1]] : null;
}

/** the chains of the step's frame and their paths: every frame of the run from its start to this frame */
export function chainsOf(p: Pick<StageProps, 'at'>): { x: Tensor; path: Float64Array[] } | null {
	const x = st(p).x as Tensor | undefined;
	const tape = p.at?.tape;
	if (!x || !tape) return null;
	const path: Float64Array[] = [];
	for (let i = Math.max(0, tape.find({ phase: 'start' })); i <= p.at!.index; i++) path.push((tape.state(i) as { x: Tensor }).x.data as Float64Array);
	return { x, path };
}


export function describePlane(p: Pick<StageProps, 'rt' | 'at' | 'props'>): I18n {
	const data = dataOf(p.rt.input);
	const mode = modeOf(p);
	const s = sigmaOf(p);
	if (!data.n) return { en: 'The plane has no points yet.', zh: '平面上还没有点。' };
	const parts: I18n[] = [{ en: `${fmtInt(data.n)} of your points (green).`, zh: `你的 ${fmtInt(data.n)} 个点（绿色）。` }];
	const [px, py] = probeOf(data);
	const e = exact(data, px, py, s);
	if (mode === 'density') parts.push({ en: ` Shading: their density blurred by σ = ${fmt3(s)}. At the ringed point (${fmt3(px)}, ${fmt3(py)}) it is ${fmt3(e.density)}.`, zh: `底色：用 σ = ${fmt3(s)} 模糊后的密度。圈出的点 (${fmt3(px)}, ${fmt3(py)}) 处是 ${fmt3(e.density)}。` });
	if (mode === 'score') parts.push({ en: ` Arrows: the score at σ = ${fmt3(s)} on a grid; at the ringed point it is (${fmt3(e.sx)}, ${fmt3(e.sy)}).`, zh: `箭头：σ = ${fmt3(s)} 时网格上的分数；圈出的点处是 (${fmt3(e.sx)}, ${fmt3(e.sy)})。` });
	const q = mode === 'pairs' ? pairsOf(p) : null;
	if (q) {
		const i = followed(q);
		const at2 = (a: Float64Array) => `(${fmt3(a[2 * i])}, ${fmt3(a[2 * i + 1])})`;
		parts.push({ en: ` ${fmtInt(q.n)} noisy copies (grey). Example ${i} (ringed): ${at2(q.x)} moved by σ = ${fmt3(q.s[i])} times ε ${at2(q.e)} to ${at2(q.xt)}; the arrow points back.`, zh: `${fmtInt(q.n)} 个加噪副本（灰色）。第 ${i} 个（圈出）：${at2(q.x)} 加上 σ = ${fmt3(q.s[i])} 乘 ε ${at2(q.e)}，到了 ${at2(q.xt)}；箭头指回去。` });
	}
	if (mode === 'field' && paramsOf(p)) parts.push({ en: ` Arrows: the network's score on a grid, after ${fmtInt(Number(st(p).step ?? 0))} training steps; a slider picks σ.`, zh: `箭头：网络在网格上给出的分数，训练了 ${fmtInt(Number(st(p).step ?? 0))} 步；滑块选 σ。` });
	const c = mode === 'chains' ? chainsOf(p) : null;
	if (c) {
		const m = p.at?.frame?.metrics ?? {};
		parts.push({ en: ` ${fmtInt(c.x.shape[0])} Langevin chains (pink) after ${fmtInt(c.path.length - 1)} steps${m.sigma ? ` at σ = ${fmt3(m.sigma)}` : ''}, with their paths.`, zh: `${fmtInt(c.x.shape[0])} 条朗之万链（粉色），走了 ${fmtInt(c.path.length - 1)} 步${m.sigma ? `（σ = ${fmt3(m.sigma)}）` : ''}，并画出路径。` });
		if (typeof m.dist === 'number') parts.push({ en: ` Mean distance to your points ${fmt3(m.dist)}; ${fmtPct(m.far ?? NaN)} of the chains are farther than 0.2.`, zh: `到你的点的平均距离 ${fmt3(m.dist)}；${fmtPct(m.far ?? NaN)} 的链离得比 0.2 还远。` });
	}
	if (mode === 'limits') parts.push({ en: ` Arrows at σ = ${fmt3(s)}: the true score (green) and the network's (pink), lengths on the same log scale.`, zh: `σ = ${fmt3(s)} 时的箭头：真实的分数（绿色）和网络的分数（粉色），长度按同一个对数比例画出。` });
	return { en: parts.map((x) => x.en).join(''), zh: parts.map((x) => x.zh).join('') };
}

const meta = {
	key: 'score-2d/plane',
	title: { en: 'Density, score arrows and Langevin chains on the plane', zh: '平面上的密度、分数箭头和朗之万链' },
	kinds: ['train', 'process'],
	describe: (p: StageProps): I18n => describePlane(p),
	async demo() {
		const [{ fakeRuntime }, { presetRaw }] = await Promise.all([import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/adapters/points2d/adapter.ts')]);
		const rt = fakeRuntime({ slug: 'score-2d', input: presetRaw('ring') });
		return { rt, source: 'train', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'score' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
