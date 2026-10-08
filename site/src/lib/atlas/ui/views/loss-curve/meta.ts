// View 'loss-curve' (kind B, also kind C metrics): a metric over the recorded steps of a source, with the cursor marked.
// Props (StageProps.props):
//   metric?: string            the metric to draw (default 'loss')
//   metrics?: string[]         several metrics of the same source (loss parts), one line each
//   compare?: string[]         other sources drawn with the same metric (e.g. lr = 3 next to lr = 0.03)
//   labels?: Record<string, I18n>  legend names by metric or source name
//   log?: boolean | 'auto'     log scale for y (auto: when the values span more than 1.5 decades and are all > 0)
//   smooth?: number            also draw the trailing mean over this many steps (a noisy batch loss shows its trend)
//   refs?: { y: number; label?: I18n }[]   K2: dashed reference lines (an equilibrium value, a baseline), inside the y range
// A live source redraws as frames arrive. Non-finite values (divergence) are marked, not drawn.
import type { I18n } from '../../../types/core.ts';
import type { StageProps, ViewMeta } from '../../../types/view.ts';
import type { Tape } from '../../../types/process.ts';
import { fmt3, fmtInt } from '../../../core/format.ts';
import { tapeOf } from '../../train/refs.ts';

export interface Series {
	key: string;
	label: I18n;
	source: string;
	metric: string;
	x: Float64Array;
	y: Float64Array;
	/** first step whose value is not finite (divergence), or -1 */
	diverged: number;
}

const DEFAULT_LABELS: Record<string, I18n> = { loss: { en: 'loss', zh: '损失' }, lr: { en: 'learning rate', zh: '学习率' }, gradNorm: { en: 'gradient size', zh: '梯度大小' }, accuracy: { en: 'accuracy', zh: '准确率' } };

function seriesOf(tape: Tape, source: string, metric: string, label: I18n | undefined): Series {
	const y = tape.metric(metric);
	const st = tape.metric('step');
	const x = new Float64Array(y.length);
	let diverged = -1;
	for (let i = 0; i < y.length; i++) {
		x[i] = Number.isFinite(st[i]) ? st[i] : i;
		const f = tape.frame(i).metrics;
		if (diverged < 0 && f && metric in f && !Number.isFinite(y[i])) diverged = x[i];
		if (diverged < 0 && f?.diverged === 1) diverged = x[i];
	}
	return { key: `${source}:${metric}`, label: label ?? DEFAULT_LABELS[metric] ?? { en: metric, zh: metric }, source, metric, x, y, diverged };
}

/** Every line the view draws (pure; the view and describe() share it). */
export function lossSeries(p: Pick<StageProps, 'rt' | 'source' | 'props'>): Series[] {
	const props = p.props ?? {};
	const labels = (props.labels ?? {}) as Record<string, I18n>;
	const metrics = Array.isArray(props.metrics) && props.metrics.length ? (props.metrics as string[]) : [typeof props.metric === 'string' ? props.metric : 'loss'];
	const out: Series[] = [];
	const own = tapeOf(p.rt, p.source);
	if (own && own.length) for (const m of metrics) out.push(seriesOf(own, p.source, m, labels[m]));
	for (const s of (props.compare as string[] | undefined) ?? []) {
		const t = tapeOf(p.rt, s);
		if (t && t.length) out.push(seriesOf(t, s, metrics[0], labels[s] ?? { en: s, zh: s }));
	}
	return out;
}

export interface LossSummary {
	n: number;
	first: number;
	last: number;
	lastStep: number;
	min: number;
	minStep: number;
	atStep: number;
	atValue: number;
	diverged: number;
}

export function summarize(s: Series | undefined, atIndex: number): LossSummary | null {
	if (!s || s.y.length === 0) return null;
	let first = NaN;
	let last = NaN;
	let lastStep = NaN;
	let min = Infinity;
	let minStep = NaN;
	let n = 0;
	for (let i = 0; i < s.y.length; i++) {
		const v = s.y[i];
		if (!Number.isFinite(v)) continue;
		n++;
		if (Number.isNaN(first)) first = v;
		last = v;
		lastStep = s.x[i];
		if (v < min) {
			min = v;
			minStep = s.x[i];
		}
	}
	let atStep = NaN;
	let atValue = NaN;
	if (atIndex >= 0 && atIndex < s.y.length) {
		atStep = s.x[atIndex];
		for (let i = atIndex; i >= 0; i--)
			if (Number.isFinite(s.y[i])) {
				atValue = s.y[i];
				break;
			}
	}
	return { n, first, last, lastStep, min: n ? min : NaN, minStep, atStep, atValue, diverged: s.diverged };
}

const meta = {
	key: 'loss-curve',
	title: { en: 'Loss curve', zh: '损失曲线' },
	kinds: ['train', 'process'],
	accepts: (o) => o.field === 'loss' || o.field === 'metric',
	describe(p: StageProps): I18n {
		const all = lossSeries(p);
		const s = summarize(all[0], p.at.index);
		if (!s || s.n === 0) return { en: 'No training steps have been recorded yet.', zh: '还没有记录任何训练步。' };
		const name = all[0].label;
		const div = s.diverged >= 0 ? { en: ` The values stop being finite numbers at step ${fmtInt(s.diverged)}: training diverged.`, zh: `从第 ${fmtInt(s.diverged)} 步起不再是有限的数：训练发散了。` } : { en: '', zh: '' };
		const cur = Number.isFinite(s.atValue) ? { en: ` The marker is at step ${fmtInt(s.atStep)}, where it is ${fmt3(s.atValue)}.`, zh: `标记在第 ${fmtInt(s.atStep)} 步，那里是 ${fmt3(s.atValue)}。` } : { en: '', zh: '' };
		const refs = (Array.isArray(p.props?.refs) ? (p.props.refs as { y: number; label?: I18n }[]) : []).filter((r) => r && Number.isFinite(r.y));
		const rl = refs.length ? { en: ` Dashed lines mark ${refs.map((r) => `${r.label ? `${r.label.en} ` : ''}${fmt3(r.y)}`).join(', ')}.`, zh: `虚线标出 ${refs.map((r) => `${r.label ? `${r.label.zh ?? r.label.en} ` : ''}${fmt3(r.y)}`).join('、')}。` } : { en: '', zh: '' };
		const more = all.length > 1 ? { en: ` ${all.length - 1} more line(s) are drawn for comparison.`, zh: `另外画了 ${all.length - 1} 条线作对比。` } : { en: '', zh: '' };
		return {
			en: `The ${name.en} over ${fmtInt(s.n)} recorded steps: it starts at ${fmt3(s.first)} and is ${fmt3(s.last)} at step ${fmtInt(s.lastStep)} (lowest ${fmt3(s.min)} at step ${fmtInt(s.minStep)}).${cur.en}${div.en}${more.en}${rl.en}`,
			zh: `${name.zh ?? name.en}在 ${fmtInt(s.n)} 个记录的步上的变化：开始是 ${fmt3(s.first)}，第 ${fmtInt(s.lastStep)} 步是 ${fmt3(s.last)}（最低 ${fmt3(s.min)}，在第 ${fmtInt(s.minStep)} 步）。${cur.zh}${div.zh}${more.zh}${rl.zh}`
		};
	},
	// the demo helpers (training code) load only when a demo is asked for, so this meta stays light
	demo: async () => {
		const { classifierRun, demoProps } = await import('../../train/demo.ts');
		return demoProps(classifierRun('coarse', 300), { phase: 'snapshot', step: 150 });
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
