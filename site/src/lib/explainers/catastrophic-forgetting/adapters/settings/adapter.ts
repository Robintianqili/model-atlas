// catastrophic-forgetting: local input adapter 'settings'. The learner's input is how the network trains on task B:
// the replay share, the pull λ, a frozen first layer and a ten times smaller learning rate ({ replay, pull, freeze,
// slow }; data.ts settingsOf). The page reads them as a [4] tensor named 'input'. Pure TS (Capture.svelte is the widget).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { MAX_PULL, MAX_REPLAY, PLAIN, settingsOf, type Settings } from '../../data.ts';

/** Ready-made settings the widget offers as buttons (and lint runs every step on). */
export const PRESETS: { id: string; label: { en: string; zh: string }; short: { en: string; zh: string }; raw: Settings }[] = [
	{ id: 'plain', label: { en: 'Plain fine-tuning (no protection)', zh: '普通微调（不加保护）' }, short: { en: 'Plain', zh: '普通' }, raw: { ...PLAIN } },
	{ id: 'replay', label: { en: 'Replay 25% of task A', zh: '回放 25% 的任务 A' }, short: { en: 'Replay', zh: '回放' }, raw: { ...PLAIN, replay: 0.25 } },
	{ id: 'pull', label: { en: 'Pull toward the old weights (λ = 0.8)', zh: '拉向旧权重（λ = 0.8）' }, short: { en: 'Pull', zh: '拉力' }, raw: { ...PLAIN, pull: 0.8 } },
	{ id: 'freeze', label: { en: 'Freeze the first layer', zh: '冻结第一层' }, short: { en: 'Freeze', zh: '冻结' }, raw: { ...PLAIN, freeze: true } },
	{ id: 'slow', label: { en: 'Learning rate ten times smaller', zh: '学习率缩小到十分之一' }, short: { en: 'Slow', zh: '慢速' }, raw: { ...PLAIN, slow: true } }
];

const adapter: InputAdapter<Settings> = {
	id: 'settings',
	kind: 'tabular',
	preprocess(raw): Preprocessed {
		const s = settingsOf(raw);
		const t = { shape: [4], data: Float64Array.of(s.replay, s.pull, s.freeze ? 1 : 0, s.slow ? 1 : 0) };
		return {
			tensors: { input: t },
			steps: [{ id: 'settings', label: { en: 'Your settings (replay, pull, freeze, slow)', zh: '你的设置（回放、拉力、冻结、慢速）' }, tensor: t }],
			empty: false
		};
	},
	edgeFixtures: () => [
		{ id: 'all-on', label: { en: 'every setting at its strongest', zh: '所有设置都开到最强' }, raw: { replay: MAX_REPLAY, pull: MAX_PULL, freeze: true, slow: true } },
		{ id: 'weak-pull', label: { en: 'a weak pull (λ = 0.3)', zh: '较弱的拉力（λ = 0.3）' }, raw: { ...PLAIN, pull: 0.3 } },
		{ id: 'tiny-replay', label: { en: 'replay 5%', zh: '回放 5%' }, raw: { ...PLAIN, replay: 0.05 } }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: {
				en: 'Your {n|int} settings for training on task B: the replay share, the pull λ, the frozen first layer (1 = on) and the slower learning rate (1 = on).',
				zh: '你在任务 B 上训练时的 {n|int} 个设置：回放比例、拉力 λ、是否冻结第一层（1 = 开）和是否用更小的学习率（1 = 开）。'
			}
		}
	}
};

export default adapter;
