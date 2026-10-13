// View 'train-step' (kind B): one training step slowed down, on a fine-detail train source. The frame's phase picks
// what is shown: 'forward' (the batch and the network's outputs), 'loss' (one number for the batch), 'backward'
// (the gradient of every weight tensor) or 'update' (one weight before and after, with the rule that ran).
// The weight in focus: a Focus { field: 'params' | 'grads', key: 'net.0.weight', index: [o, i] }, else props.key/index,
// else the largest gradient of the first weight tensor. The SGD rule old − lr × g = new is shown only when the source
// trained with plain SGD (train/helpers.ts updateOf checks it); Adam and momentum show what actually changed.
// Props: key?: string; index?: number[]; rows?: number (examples listed in 'forward', default 4).
import type { I18n, Tensor } from '../../../types/core.ts';
import type { StageProps, ViewMeta } from '../../../types/view.ts';
import { fmt3, fmtInt, fmtSetting, paren } from '../../../core/format.ts';
import { updateOf } from '../../../train/helpers.ts';

export const PHASES = ['forward', 'loss', 'backward', 'update'] as const;
export type StepPhase = (typeof PHASES)[number];

export interface StepData {
	phase: StepPhase | null;
	step: number;
	loss: number;
	lr: number;
	optimizer: string;
	gradNorm: number;
	key: string | null;
	index: number[] | null;
	/** the focused weight's numbers (NaN where the phase does not have them) */
	old: number;
	g: number;
	now: number;
	m: number;
	v: number;
	/** old − lr × g = new holds for this update (plain SGD) */
	sgd: boolean;
	batch: Record<string, Tensor>;
	output: Tensor | null;
	grads: Record<string, Tensor>;
}

function flat(shape: number[], idx: number[]): number {
	return idx.reduce((a, k, d) => a * shape[d] + k, 0);
}

function argmaxAbs(d: ArrayLike<number>): number {
	let b = 0;
	for (let i = 1; i < d.length; i++) if (Math.abs(d[i]) > Math.abs(d[b])) b = i;
	return b;
}

/** K4: the significant digits of an update's old and new value: props.digits (2-6), else the fewest from 3 to 6 that
 *  print old and new differently (with 3, a small step printed "0.512 − 0.05 × 0.0123 = 0.512"). */
export function updateDigits(d: { old: number; now: number }, digits?: unknown): number {
	if (Number.isInteger(digits) && (digits as number) >= 2 && (digits as number) <= 6) return digits as number;
	for (let k = 3; k <= 6; k++) if (fmt3(d.old, k) !== fmt3(d.now, k)) return k;
	return 3;
}

/** Everything the view shows (pure). */
export function stepData(p: Pick<StageProps, 'at' | 'props' | 'focus'>): StepData {
	const f = p.at?.frame;
	const s = (p.at?.state ?? {}) as Record<string, unknown>;
	const phase = f && (PHASES as readonly string[]).includes(f.phase) ? (f.phase as StepPhase) : null;
	const params = (s.params ?? {}) as Record<string, Tensor>;
	const grads = (s.grads ?? {}) as Record<string, Tensor>;
	const prev = (s.prev ?? {}) as Record<string, Tensor>;
	const moments = (s.opt ?? {}) as Record<string, { m?: Tensor; v?: Tensor }>;
	let key: string | null = null;
	let index: number[] | null = null;
	for (const fc of p.focus ?? []) {
		if ((fc.field === 'params' || fc.field === 'grads') && fc.key && Array.isArray(fc.index) && typeof fc.index[0] === 'number') {
			key = fc.key;
			index = fc.index as number[];
		}
	}
	if (!key && typeof p.props?.key === 'string') {
		key = p.props.key as string;
		index = Array.isArray(p.props.index) ? (p.props.index as number[]) : null;
	}
	if (!key) key = Object.keys(params).find((k) => k.endsWith('.weight') && params[k].shape.length === 2) ?? Object.keys(params)[0] ?? null;
	const w = key ? (params[key] ?? grads[key]) : undefined;
	if (key && w && !index) {
		const g = grads[key];
		const k = g ? argmaxAbs(g.data) : 0;
		index = [];
		let r = k;
		for (let d = w.shape.length - 1; d >= 0; d--) {
			index.unshift(r % w.shape[d]);
			r = Math.floor(r / w.shape[d]);
		}
	}
	const at = key && w && index && index.length === w.shape.length && index.every((v, d) => v >= 0 && v < w.shape[d]) ? flat(w.shape, index) : -1;
	const num = (t: Tensor | undefined) => (t && at >= 0 && at < t.data.length ? t.data[at] : NaN);
	const metric = (n: string) => (typeof f?.metrics?.[n] === 'number' ? (f.metrics[n] as number) : typeof s[n] === 'number' ? (s[n] as number) : NaN);
	let sgd = false;
	if (phase === 'update' && key && index && at >= 0) {
		try {
			updateOf(p.at, key, index);
			sgd = true;
		} catch {
			sgd = false;
		}
	}
	const acts = (s.acts ?? {}) as Record<string, Tensor>;
	const actKeys = Object.keys(acts);
	return {
		phase,
		step: typeof f?.step === 'number' ? f.step : metric('step'),
		loss: metric('loss'),
		lr: metric('lr'),
		optimizer: typeof s.optimizer === 'string' ? s.optimizer : '',
		gradNorm: metric('gradNorm'),
		key,
		index: at >= 0 ? index : null,
		old: phase === 'update' ? num(prev[key ?? '']) : num(params[key ?? '']),
		g: num(grads[key ?? '']),
		now: phase === 'update' ? num(params[key ?? '']) : NaN,
		m: num(moments[key ?? '']?.m),
		v: num(moments[key ?? '']?.v),
		sgd,
		batch: (s.batch ?? {}) as Record<string, Tensor>,
		output: actKeys.length ? acts[actKeys[actKeys.length - 1]] : null,
		grads
	};
}

const NAMES: Record<StepPhase, I18n> = {
	forward: { en: 'Forward pass', zh: '前向计算' },
	loss: { en: 'Loss', zh: '损失' },
	backward: { en: 'Backward pass', zh: '反向传播' },
	update: { en: 'Update', zh: '更新' }
};
export const phaseName = (ph: StepPhase): I18n => NAMES[ph];

const meta = {
	key: 'train-step',
	title: { en: 'One training step', zh: '一步训练' },
	kinds: ['train'],
	accepts: (o) => o.field === 'grads',
	describe(p: StageProps): I18n {
		const d = stepData(p);
		if (!d.phase) return { en: 'This frame is not part of a slowed-down training step (use a fine-detail train source).', zh: '这一帧不属于放慢的训练步（需要细节模式的训练数据源）。' };
		const head = { en: `Training step ${fmtInt(d.step)}, ${NAMES[d.phase].en.toLowerCase()}: `, zh: `第 ${fmtInt(d.step)} 步训练，${NAMES[d.phase].zh}：` };
		const where = d.key && d.index ? `${d.key}[${d.index.join(', ')}]` : '';
		let body: I18n;
		if (d.phase === 'forward') {
			const B = Object.values(d.batch)[0]?.shape[0] ?? 0;
			body = { en: `a batch of ${fmtInt(B)} examples goes through the network with the current weights.`, zh: `一批 ${fmtInt(B)} 个样例用当前的权重通过网络。` };
		} else if (d.phase === 'loss') body = { en: `the loss on this batch is ${fmt3(d.loss)}.`, zh: `这一批的损失是 ${fmt3(d.loss)}。` };
		else if (d.phase === 'backward')
			body = { en: `every weight gets a gradient; for ${where} it is ${fmt3(d.g)} (size of all gradients together: ${fmt3(d.gradNorm)}).`, zh: `每个权重都得到一个梯度；${where} 的梯度是 ${fmt3(d.g)}（全部梯度合起来的大小：${fmt3(d.gradNorm)}）。` };
		else if (d.sgd) {
			const k = updateDigits(d, p.props?.digits);
			const v = (x: number) => fmt3(x, k);
			body = { en: `${where} was ${v(d.old)}; with gradient ${fmt3(d.g)} and learning rate ${fmtSetting(d.lr)} it becomes ${v(d.old)} − ${fmtSetting(d.lr)} × ${paren(d.g)} = ${v(d.now)}.`, zh: `${where} 原来是 ${v(d.old)}；梯度是 ${fmt3(d.g)}，学习率是 ${fmtSetting(d.lr)}，所以它变成 ${v(d.old)} − ${fmtSetting(d.lr)} × ${paren(d.g)} = ${v(d.now)}。` };
		} else
			body = { en: `${where} went from ${fmt3(d.old)} to ${fmt3(d.now)} (optimizer ${d.optimizer || 'unknown'}, learning rate ${fmt3(d.lr)}, gradient ${fmt3(d.g)}).`, zh: `${where} 从 ${fmt3(d.old)} 变成 ${fmt3(d.now)}（优化器 ${d.optimizer || '未知'}，学习率 ${fmt3(d.lr)}，梯度 ${fmt3(d.g)}）。` };
		return { en: head.en + body.en, zh: head.zh + body.zh };
	},
	// the demo helpers (training code) load only when a demo is asked for, so this meta stays light
	demo: async () => {
		const { classifierRun, demoProps } = await import('../../train/demo.ts');
		return demoProps(classifierRun('fine', 4), { phase: 'update', step: 1 }, {}, [{ field: 'params', key: 'net.0.weight', index: [3, 1], role: 'param' }]);
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
