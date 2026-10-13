// View 'realnvp-2d/plane': the flow on a plane. Props:
//   mode   'data'     your points on the data plane (±1.5)
//          'couple'   one coupling layer of process 'flow': the grid and points before it (faint) and after it, point 0
//                     ringed with its move; back: true draws the move undone (the inverse) and its formula
//          'warp'     the frame's points (flow: pts, green; sample: x, pink) and grid on the noise plane (±3), with the
//                     circles |z| = 1 and 2 of the noise N(0, I); cell: true adds the small square around point 0 pushed
//                     through the same layers
//          'density'  p(x) = e^(log p) on the data plane from the 'density' frame of process 'flow'
//          'learn'    the same density from a train frame's weights (a coarser grid, computed here)
//          'samples'  the samples of a 'sample' frame (pink) and the points they were trained on (green) on the data
//                     plane; gap: true draws the rings' gap (0.6 < |x| < 0.75)
// Numbers come from the step's frame (state.pts / state.x / state.grid / state.logp, metrics) and the trained weights.
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
import { dataOf, DSPAN, forward, LAYERS, ZSPAN, type Params } from '../../nvp.ts';
import { RINGS, trained, trainedOf } from '../../explainer.ts';
import { GAP } from '../../process.ts';

export type Mode = 'data' | 'couple' | 'warp' | 'density' | 'learn' | 'samples';
type P = Pick<StageProps, 'rt' | 'at' | 'props'> & { source?: string };
/** the step's source name */
const srcOf = (p: P) => p.source ?? p.at?.source ?? '';
const st = (p: Pick<StageProps, 'at'>) => (p.at?.state ?? {}) as Record<string, Tensor | number | Params | undefined>;
export const modeOf = (p: Pick<StageProps, 'props'>) => (p.props?.mode as Mode | undefined) ?? 'data';
export const spanOf = (p: Pick<StageProps, 'props'>) => (modeOf(p) === 'warp' ? ZSPAN : DSPAN);
/** the points of the step's frame: the flow's (pts) or the samples (x) */
export const pointsOf = (p: Pick<StageProps, 'at'>) => (st(p).pts ?? st(p).x) as Tensor | undefined;
export const isSample = (p: Pick<StageProps, 'at'>) => !!st(p).x;
/** the points the frame's source trained on: the two rings for the rings sources, else yours */
export const dataFor = (p: P) => dataOf(srcOf(p).startsWith('ring') ? RINGS : p.rt.input);
export const metricsOf = (p: Pick<StageProps, 'at'>) => p.at?.frame?.metrics ?? {};
/** how many layers the frame's points went through (data -> noise): the start 0, layer k k + 1 */
export const layersDone = (p: Pick<StageProps, 'at'>) => (p.at?.frame?.phase === 'couple' ? (p.at.frame.step ?? 0) + 1 : p.at?.frame?.phase === 'density' ? LAYERS : 0);

/** the small square (side 0.02) around point 0, 8 points per side, pushed through the first k layers; its area over
 *  the square's (shoelace) */
export const SIDE = 0.02;
export function cellOf(params: Params, x: number, y: number, k: number) {
	const sq: number[] = [];
	const h = SIDE / 2;
	for (let side = 0; side < 4; side++)
		for (let q = 0; q < 8; q++) {
			const a = -h + (q * SIDE) / 8;
			const [dx, dy] = [[a, -h], [h, a], [-a, h], [-h, -a]][side];
			sq.push(x + dx, y + dy);
		}
	const { z } = forward(params, sq, 32, k);
	let area = 0;
	for (let i = 0; i < 32; i++) area += z[2 * i] * z[2 * ((i + 1) % 32) + 1] - z[2 * ((i + 1) % 32)] * z[2 * i + 1];
	return { poly: z, ratio: Math.abs(area) / 2 / (SIDE * SIDE) };
}
/** the trained weights of the frame's train source */
export const paramsFor = (p: P) => (srcOf(p).startsWith('ring') ? trainedOf(p.rt, 'rings', RINGS) : trained(p.rt))?.params ?? null;

/** the coordinate names of a layer: [kept, moved] */
export const namesOf = (keep: number) => (keep === 0 ? ['x', 'y'] : ['y', 'x']);
export function describePlane(p: P): I18n {
	const data = dataFor(p);
	const mode = modeOf(p);
	if (!data.n) return { en: 'The plane has no points yet.', zh: '平面上还没有点。' };
	const m = metricsOf(p);
	const parts: I18n[] = [{ en: `${fmtInt(data.n)} points (green).`, zh: `${fmtInt(data.n)} 个点（绿色）。` }];
	if (mode === 'couple' && typeof m.s === 'number') {
		const [u, v] = namesOf(m.keep);
		parts.push({
			en: ` Coupling layer ${m.k} (counting from 0) keeps ${u} and moves ${v}: the faint grid and points are before it, the solid ones after. Point 0 moves from ${v} = ${fmt3(m.v)} to ${fmt3(m.v2)} (s = ${fmt3(m.s)}, t = ${fmt3(m.t)}).`,
			zh: ` 耦合层 ${m.k}（从 0 数起）保持 ${u} 不变、移动 ${v}：淡的网格和点是它之前，实的是它之后。0 号点的 ${v} 从 ${fmt3(m.v)} 变成 ${fmt3(m.v2)}（s = ${fmt3(m.s)}，t = ${fmt3(m.t)}）。`
		});
	}
	if (mode === 'warp') {
		const k = layersDone(p);
		parts.push(
			isSample(p)
				? { en: ` ${fmtInt(pointsOf(p)?.shape[0] ?? 0)} samples (pink) and the grid of the noise plane, pulled back through ${fmtInt(p.at?.frame?.phase === 'uncouple' ? (p.at.frame.step ?? 0) + 1 : 0)} of the ${LAYERS} layers.`, zh: ` ${fmtInt(pointsOf(p)?.shape[0] ?? 0)} 个样本（粉色）和噪声平面上的网格，已倒着穿过 ${LAYERS} 层中的 ${fmtInt(p.at?.frame?.phase === 'uncouple' ? (p.at.frame.step ?? 0) + 1 : 0)} 层。` }
				: { en: ` Your points and a grid over them after ${fmtInt(k)} of the ${LAYERS} coupling layers; dashed circles: radius 1 and 2 of the noise.`, zh: ` 你的点和覆盖它们的网格，已穿过 ${LAYERS} 个耦合层中的 ${fmtInt(k)} 个；虚线圆：噪声的半径 1 和 2。` }
		);
	}
	if (mode === 'density' && typeof m.logp === 'number') parts.push({ en: ` Shading: the flow's density p(x). At point 0, log p = ${fmt3(m.logp)}; the highest log p on the grid is ${fmt3(m.lpMax)}.`, zh: ` 底色：流模型的密度 p(x)。0 号点处 log p = ${fmt3(m.logp)}；网格上最高的 log p 是 ${fmt3(m.lpMax)}。` });
	if (mode === 'learn') parts.push({ en: ` Shading: the density of the flow after ${fmtInt(Number(st(p).step ?? 0))} training steps.`, zh: ` 底色：训练 ${fmtInt(Number(st(p).step ?? 0))} 步后流模型的密度。` });
	if (mode === 'samples' && pointsOf(p)) {
		const g = m.gap;
		parts.push({ en: ` ${fmtInt(pointsOf(p)!.shape[0])} samples of the flow (pink)${typeof g === 'number' && p.props?.gap ? `; ${fmtPct(g)} of them lie in the gap between the rings (dashed)` : ''}.`, zh: ` 流模型的 ${fmtInt(pointsOf(p)!.shape[0])} 个样本（粉色）${typeof g === 'number' && p.props?.gap ? `；其中 ${fmtPct(g)} 落在两个圆环之间的空隙里（虚线）` : ''}。` });
	}
	return { en: parts.map((x) => x.en).join(''), zh: parts.map((x) => x.zh).join('') };
}

const meta = {
	key: 'realnvp-2d/plane',
	title: { en: 'Coupling layers warping the plane', zh: '耦合层扭曲平面' },
	kinds: ['train', 'process'],
	describe: (p: StageProps): I18n => describePlane(p),
	async demo() {
		const [{ fakeRuntime }, { presetRaw }] = await Promise.all([import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/adapters/points2d/adapter.ts')]);
		const rt = fakeRuntime({ slug: 'realnvp-2d', input: presetRaw('ring') });
		return { rt, source: 'train', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'data' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
