// residual-connections: local input adapter 'setup'. The learner's input is the deep MLP's setup ({ depth, skip }): how
// many linear layers (3 to 20) and whether the hidden layers have skip connections (1) or not (0). The page reads it as
// a [2] tensor named 'input'. Pure TS (the Capture component next to it is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { MAX_DEPTH, MIN_DEPTH, START, setupOf, type Setup } from '../../data.ts';

/** Starting setups the widget offers as buttons (and lint runs every step on). */
export const STARTS: { id: string; label: { en: string; zh: string }; short: { en: string; zh: string }; raw: Setup }[] = [
	{ id: 'plain-20', label: { en: '20 layers, no skips (the start)', zh: '20 层，无跳跃连接（起点）' }, short: { en: '20 plain', zh: '20 层无跳跃' }, raw: { ...START } },
	{ id: 'skips-20', label: { en: '20 layers with skips', zh: '20 层，有跳跃连接' }, short: { en: '20 skips', zh: '20 层有跳跃' }, raw: { depth: MAX_DEPTH, skip: 1 } },
	{ id: 'plain-6', label: { en: '6 layers, no skips', zh: '6 层，无跳跃连接' }, short: { en: '6 plain', zh: '6 层无跳跃' }, raw: { depth: 6, skip: 0 } }
];

const adapter: InputAdapter<Setup> = {
	id: 'setup',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const s = setupOf(raw);
		const t = { shape: [2], data: Float64Array.of(s.depth, s.skip) };
		return {
			tensors: { input: t },
			steps: [{ id: 'setup', label: { en: 'Your setup (layers, skips on = 1)', zh: '你的设置（层数，跳跃连接打开 = 1）' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'shallow-plain', label: { en: 'the fewest layers, no skips', zh: '层数最少，无跳跃连接' }, raw: { depth: MIN_DEPTH, skip: 0 } },
		{ id: 'shallow-skips', label: { en: 'the fewest layers, with skips', zh: '层数最少，有跳跃连接' }, raw: { depth: MIN_DEPTH, skip: 1 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'Your setup is {n|int} numbers: how many layers the network has, and 1 when its hidden layers have skip connections.',
				zh: '你的设置是 {n|int} 个数：网络有多少层；隐藏层带跳跃连接时第二个数为 1。'
			}
		}
	}
};

export default adapter;
