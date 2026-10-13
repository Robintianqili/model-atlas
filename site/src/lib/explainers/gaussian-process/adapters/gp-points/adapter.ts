// gaussian-process: local input adapter 'gp-points' (catalog InputKind 'points2d'). The learner's raw input is a GpRaw
// (gp.ts): a preset or the points the learner clicked (in the order added), the kernel's settings ℓ and σf², and the
// seed of the sample functions. It becomes the tensors 'x' [n] and 'y' [n]; the process reads dataOf(raw).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { I, MAXP, PRESETS, pointsOf, XHI, YHI, type GpRaw } from '../../gp.ts';

const adapter: InputAdapter<GpRaw> = {
	id: 'gp-points',
	kind: 'points2d',
	preprocess(raw): Preprocessed {
		const p = pointsOf(raw), n = p.length;
		return {
			tensors: { x: { shape: [n], data: Float64Array.from(p.map((q) => q.x)) }, y: { shape: [n], data: Float64Array.from(p.map((q) => q.y)) } },
			steps: [],
			empty: n === 0,
			notes: n ? [] : [I('There are no points yet: click the plot to add some, or pick a preset.', '还没有点：在图上点击添加，或者选一个预设。')]
		};
	},
	presets: () => PRESETS.map((p) => ({ id: p.id, label: p.label, raw: { preset: p.id } })),
	edgeFixtures: () => [
		{ id: 'empty', label: I('No points', '没有点'), raw: { points: [] } },
		{ id: 'one', label: I('One point', '一个点'), raw: { points: [{ x: 0.5, y: 1.2 }] } },
		{ id: 'twin', label: I('Two points at the same x, far apart in y', '同一个 x 上的两个点，y 相差很远'), raw: { points: [{ x: -1, y: -2 }, { x: -1, y: 2 }], ls: 0.25, sf2: 0.25 } },
		{ id: 'full', label: I('The most points, at the corners, extreme settings', '点数最多，贴着角落，极端的设置'), raw: { points: Array.from({ length: MAXP + 3 }, (_, i) => ({ x: (i % 2 ? 1 : -1) * XHI, y: (i % 3 ? 1 : -1) * YHI + i / 100 })), ls: 3, sf2: 2, seed: 9 } }
	],
	layer0: { view: 'gaussian-process/plot', narration: { what: { en: 'The data are {n|int} numbers: an x and a y for every point.', zh: '数据是 {n|int} 个数：每个点一个 x、一个 y。' } } }
};

export default adapter;
