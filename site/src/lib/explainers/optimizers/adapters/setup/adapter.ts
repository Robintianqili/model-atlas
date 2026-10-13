// optimizers: local input adapter 'setup'. The learner's input is the page's set of controls: the landscape of the race
// and its start (x, y), the learning-rate schedule with or without warm-up, and the gradient-clipping switch
// (data.ts Setup). The page reads the start as a [2] tensor named 'input'. Pure TS (Capture.svelte next to it is the
// widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { START, setupOf, surfaceOf, type Setup } from '../../data.ts';

/** The starts the widget offers (one per landscape), the lint fixtures with the schedule and clipping choices. */
export const STARTS: { id: string; label: { en: string; zh: string }; raw: Setup }[] = [
	{ id: 'valley', label: { en: 'Long valley, constant η', zh: '狭长山谷，恒定 η' }, raw: START },
	{ id: 'saddle', label: { en: 'Saddle, cosine decay with warm-up', zh: '鞍点，带预热的余弦衰减' }, raw: { ...START, surface: 'saddle', x: surfaceOf('saddle').start[0], y: surfaceOf('saddle').start[1], schedule: 'cosine', warmup: true } },
	{ id: 'bumpy', label: { en: 'Bumpy bowl, step decay, clipping on', zh: '坑洼的碗，阶梯衰减，开启裁剪' }, raw: { ...START, surface: 'bumpy', x: surfaceOf('bumpy').start[0], y: surfaceOf('bumpy').start[1], schedule: 'step', clip: true } }
];

const adapter: InputAdapter<Setup> = {
	id: 'setup',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const s = setupOf(raw);
		const t = { shape: [2], data: Float64Array.of(s.x, s.y) };
		return {
			tensors: { input: t },
			steps: [{ id: 'start', label: { en: 'The start (x, y)', zh: '起点 (x, y)' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'ridge', label: { en: 'saddle, start exactly on the ridge (y = 0)', zh: '鞍点，起点正好在脊线上（y = 0）' }, raw: { ...START, surface: 'saddle', x: -2.5, y: 0 } },
		{ id: 'corner', label: { en: 'valley, start in the far corner', zh: '狭长山谷，起点在远角' }, raw: { ...START, x: -4.5, y: -1.5, warmup: true } },
		{ id: 'bottom', label: { en: 'bumpy bowl, start at its bottom', zh: '坑洼的碗，起点就在碗底' }, raw: { ...START, surface: 'bumpy', x: 1, y: 0.5, schedule: 'cosine', clip: true } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'The start is {n|int} numbers, the weights x and y. Every optimizer starts from them.',
				zh: '起点是 {n|int} 个数：权重 x 和 y。每个优化器都从这里出发。'
			}
		}
	}
};

export default adapter;
