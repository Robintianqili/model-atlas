// regularization: local input adapter 'knobs'. The learner's input is five knobs ({ noise, ratio, width, l2, dropout }):
// two for the data (how noisy the points are, what share of them is used for training), one for the network's size
// and two regularizers (λ of the L2 penalty, p of dropout). The page reads them as a [5] tensor named 'input'. Pure TS
// (the Capture component next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { START, knobsOf, type Knobs } from '../../data.ts';

/** Starting knobs the widget offers as buttons (and lint runs every step on). */
export const STARTS: { id: string; label: { en: string; zh: string }; short: { en: string; zh: string }; raw: Knobs }[] = [
	{ id: 'overfit', label: { en: 'Big network, few points (the start)', zh: '大网络、少量点（起点）' }, short: { en: 'Start', zh: '起点' }, raw: { ...START } },
	{ id: 'weight-decay', label: { en: 'Add an L2 penalty (λ = 0.003)', zh: '加上 L2 惩罚（λ = 0.003）' }, short: { en: 'L2', zh: 'L2' }, raw: { ...START, l2: 0.003 } },
	{ id: 'dropout', label: { en: 'Add dropout (p = 0.5)', zh: '加上 dropout（p = 0.5）' }, short: { en: 'Dropout', zh: 'Dropout' }, raw: { ...START, dropout: 0.5 } },
	{ id: 'more-data', label: { en: 'Train on 70% of the points', zh: '用 70% 的点训练' }, short: { en: 'More data', zh: '更多数据' }, raw: { ...START, ratio: 0.7 } },
	{ id: 'small', label: { en: 'A small network (2 units per layer)', zh: '小网络（每层 2 个单元）' }, short: { en: 'Small', zh: '小网络' }, raw: { ...START, width: 2 } }
];

const adapter: InputAdapter<Knobs> = {
	id: 'knobs',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const k = knobsOf(raw);
		const t = { shape: [5], data: Float64Array.of(k.noise, k.ratio, k.width, k.l2, k.dropout) };
		return {
			tensors: { input: t },
			steps: [{ id: 'knobs', label: { en: 'Your knobs (noise, train share, width, λ, p)', zh: '你的旋钮（噪声、训练比例、宽度、λ、p）' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'clean-few', label: { en: 'no noise, 10 % training points, 2 units', zh: '无噪声、10% 训练点、2 个单元' }, raw: { noise: 0, ratio: 0.1, width: 2, l2: 0, dropout: 0 } },
		{ id: 'everything', label: { en: 'every knob at its highest', zh: '每个旋钮都调到最高' }, raw: { noise: 0.5, ratio: 0.9, width: 32, l2: 0.1, dropout: 0.5 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'Your settings are {n|int} numbers: the noise, the share of points used for training, the units per layer, λ and p.',
				zh: '你的设置是 {n|int} 个数：噪声、用来训练的点的比例、每层的单元数、λ 和 p。'
			}
		}
	}
};

export default adapter;
