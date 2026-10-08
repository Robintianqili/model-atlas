// View 'train-controls' (kind B): play / pause / step / reset and the learning rate for a live train source, with the
// current step, loss, learning rate, speed and a small loss sparkline. All numbers are read from the recorded frames.
// Props (StageProps.props):
//   lrs?: number[]       learning rates to offer (default 0.001 … 3)
//   target?: number      a step to reach: Play pauses there and the stage reports done (ctx.done / onend)
//   showReset?: boolean  (default true)
// On a recorded (not live) tape the buttons are disabled and the view only reports the numbers.
import type { I18n } from '../../../types/core.ts';
import type { StageProps, ViewMeta } from '../../../types/view.ts';
import type { Tape } from '../../../types/process.ts';
import { fmt3, fmtInt } from '../../../core/format.ts';
import { isLive, tapeOf } from '../../train/refs.ts';

export const DEFAULT_LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];

export interface ControlsInfo {
	live: boolean;
	length: number;
	step: number;
	loss: number;
	lr: number;
	paused: boolean;
	where: string;
}

function lastMetric(t: Tape, name: string, upTo: number): number {
	for (let i = Math.min(upTo, t.length - 1); i >= 0; i--) {
		const v = t.frame(i).metrics?.[name];
		if (typeof v === 'number') return v;
	}
	return NaN;
}

/** What the controls show for a tape (latest frame of a live tape, the cursor's frame of a recorded one). */
export function controlsInfo(p: Pick<StageProps, 'rt' | 'source' | 'at'>): ControlsInfo | null {
	const t = tapeOf(p.rt, p.source);
	if (!t || t.length === 0) return null;
	const live = isLive(t);
	const i = live ? t.length - 1 : Math.max(0, Math.min(t.length - 1, p.at.index));
	const f = t.frame(i);
	return {
		live,
		length: t.length,
		step: typeof f.step === 'number' ? f.step : lastMetric(t, 'step', i),
		loss: lastMetric(t, 'loss', i),
		lr: lastMetric(t, 'lr', i),
		paused: live ? ((t as { paused?: boolean }).paused ?? true) : true,
		where: (t as { where?: string }).where ?? 'main'
	};
}

const meta = {
	key: 'train-controls',
	title: { en: 'Training controls', zh: '训练控制' },
	kinds: ['train'],
	accepts: (o) => o.field === 'lr',
	describe(p: StageProps): I18n {
		const c = controlsInfo(p);
		if (!c) return { en: 'Training has not started yet.', zh: '训练还没有开始。' };
		const nums = { en: `step ${fmtInt(c.step)}, loss ${fmt3(c.loss)}, learning rate ${fmt3(c.lr)}`, zh: `第 ${fmtInt(c.step)} 步，损失 ${fmt3(c.loss)}，学习率 ${fmt3(c.lr)}` };
		if (!c.live) return { en: `A recorded run (the controls need a live source): ${nums.en}.`, zh: `一次记录好的运行（控制按钮需要实时的数据源）：${nums.zh}。` };
		const state = c.paused ? { en: 'paused', zh: '已暂停' } : { en: 'running', zh: '正在运行' };
		const where = c.where === 'worker' ? { en: 'in a background thread', zh: '在后台线程里' } : { en: 'on this page', zh: '在本页面上' };
		return { en: `Training controls: ${nums.en}. Training is ${state.en}, ${where.en}.`, zh: `训练控制：${nums.zh}。训练${state.zh}，${where.zh}。` };
	},
	// the demo helpers (training code) load only when a demo is asked for, so this meta stays light
	demo: async () => {
		const { classifierRun, demoProps } = await import('../../train/demo.ts');
		return demoProps(classifierRun('coarse', 300), 'last');
	},
	minHeight: 150
} satisfies ViewMeta;

export default meta;
