// tiny-unet-seg: local input adapter 'shapes32'. The learner's raw input is a scene (scenes.ts SegRaw): stamped shapes,
// painted pixels (each with the class the learner painted) and a noise seed. One tensor comes out, the picture
// input [1, 32, 32]; the true class of every pixel comes from the same render() (truthOf), for the IoU and the task.
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { render, S, type SegRaw } from '../../scenes.ts';

export const rawOf = (raw: unknown): SegRaw => (raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as SegRaw) : {});
export const truthOf = (raw: unknown): Uint8Array => render(rawOf(raw)).label;

/** Paint entries for a ring of class k around (x, y): every pixel whose centre is within w/2 of radius r. */
export function ringPaint(x: number, y: number, r: number, k: number, w = 2): number[] {
	const out: number[] = [];
	for (let i = 0; i < S * S; i++) if (Math.abs(Math.hypot((i % S) + 0.5 - x, Math.floor(i / S) + 0.5 - y) - r) <= w / 2) out.push(i * 4 + k);
	return out;
}

/** Inputs every step must survive (also reference cases for parity): only noise, a circle far bigger than any in
 *  training, a hand-painted ring (the kind of drawing the model gets wrong), a square cut by the border. */
export const EDGES: { id: string; label: { en: string; zh: string }; raw: SegRaw }[] = [
	{ id: 'blank', label: { en: 'no shapes, only noise', zh: '没有形状，只有噪声' }, raw: { shapes: [], seed: 21 } },
	{ id: 'giant', label: { en: 'one giant circle', zh: '一个巨大的圆' }, raw: { shapes: [{ k: 1, x: 16, y: 16, r: 14 }], seed: 22 } },
	{ id: 'ring', label: { en: 'a painted ring, labelled circle', zh: '画出的一个圈，标为圆' }, raw: { paint: ringPaint(16, 16, 8, 1), seed: 23 } },
	{ id: 'cut', label: { en: 'a square cut by the border', zh: '被边框切掉一半的正方形' }, raw: { shapes: [{ k: 2, x: 1, y: 16, r: 7 }], seed: 24 } }
];

const adapter: InputAdapter<SegRaw> = {
	id: 'shapes32',
	kind: 'drawing',
	preprocess(raw): Preprocessed {
		const input = { shape: [1, S, S], data: render(rawOf(raw)).image };
		const r = rawOf(raw);
		const notes = (r.shapes?.length ?? 0) + (r.paint?.length ?? 0) ? [] : [{ en: 'No shapes yet: the picture is only background noise.', zh: '还没有形状：这幅图只有背景噪声。' }];
		return { tensors: { input }, steps: [{ id: 'input', label: { en: 'What the model sees: 32×32 grey numbers', zh: '模型看到的：32×32 个灰度数' }, tensor: input }], empty: false, notes };
	},
	edgeFixtures: () => EDGES.map((e) => ({ ...e, raw: structuredClone(e.raw) })),
	layer0: {
		view: 'tiny-unet-seg/mask',
		narration: {
			what: { en: 'The model sees {n|int} grey numbers in a {shape} grid: dark background with noise, brighter shapes.', zh: '模型看到 {shape} 网格里的 {n|int} 个灰度数：带噪声的暗背景，和更亮的形状。' },
			look: { en: 'Each cell is one number; the shapes are brighter than the background.', zh: '每个格子是一个数；形状比背景更亮。' }
		}
	}
};

export default adapter;
