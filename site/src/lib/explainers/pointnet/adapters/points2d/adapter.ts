// pointnet: the kit's points2d adapter (its Capture.svelte is used as it is, unlabelled: options.classes 1) with the
// page's own presets (shapes.ts) and the model's input: the painted points, centred and scaled (normalize), as
// `input` [n, 2] float32. The presets start with the star (the page opens on it). The raw value may carry `cut` (the stage's "remove a critical point" button: how many it
// removed and the class before the first removal), which the Capture keeps when it paints or erases.
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import kit, { type PointsRaw } from '../../../../atlas/adapters/points2d/adapter.ts';
import { LABELS, MAX_N, cloudOf, normalize, presetPoints } from '../../shapes.ts';

export interface CloudRaw extends PointsRaw {
	cut?: { n: number; cls: number; back?: { x: number; y: number }[] };
}

const adapter: InputAdapter<CloudRaw> = {
	...kit,
	preprocess(raw): Preprocessed {
		const pts = cloudOf(raw);
		const xy = Float64Array.from(pts.flatMap((q) => [q.x, q.y]));
		const many = ((raw as PointsRaw | null)?.points?.length ?? 0) > MAX_N;
		const notes = pts.length ? [] : [{ en: 'There are no points yet. Draw a shape, or pick one.', zh: '还没有点。画一个形状，或者选一个。' }];
		if (many) notes.push({ en: `Only the first ${MAX_N} points are used.`, zh: `只用前 ${MAX_N} 个点。` });
		if (!pts.length) return { tensors: { xy: { shape: [0, 2], data: xy } }, steps: [{ id: 'points', label: { en: 'Your points: x and y', zh: '你的点：x 和 y' } }], empty: true, notes };
		const m = normalize(xy);
		return {
			tensors: { input: { shape: [m.n, 2], data: m.xy } },
			steps: [
				{ id: 'points', label: { en: 'Your points: x and y', zh: '你的点：x 和 y' }, tensor: { shape: [m.n, 2], data: xy } },
				{ id: 'input', label: { en: 'Centred on the middle of their box, farthest point at distance 1', zh: '以方框中心为原点，最远的点距离为 1' }, tensor: { shape: [m.n, 2], data: m.xy } }
			],
			empty: false,
			notes
		};
	},
	presets: () => (['star', 'square', 'triangle', 'circle', 'ring'] as const).map((k) => ({ id: k, label: LABELS[k], raw: { points: presetPoints(k) } })),
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'No points', zh: '没有点' }, raw: { points: [] } },
		{ id: 'one-point', label: { en: 'One point', zh: '一个点' }, raw: { points: [{ x: 0.2, y: -0.3 }] } },
		{ id: 'line', label: { en: 'A straight line', zh: '一条直线' }, raw: { points: Array.from({ length: 20 }, (_, i) => ({ x: -0.8 + (1.6 * i) / 19, y: 0.1 })) } }
	],
	layer0: {
		view: 'pointnet/cloud',
		narration: {
			what: { en: 'Each point is two numbers, x and y, so your points form a table of {shape} numbers.', zh: '每个点是两个数 x 和 y，所以你的点组成一张 {shape} 的数表。' },
			how: { en: 'The points are moved so the middle of the box around them sits at 0, then scaled so the farthest one is at distance 1 from 0.', zh: '这些点先被平移，让包住它们的方框的中心落在 0；再缩放，让最远的点到 0 的距离为 1。' }
		}
	}
};

export default adapter;
