// View 'positional-encodings/lines': a few lines on shared axes (process 'pe'), picked by the frame's phase:
//   relative  head 0's query–key dot product as the key moves (the query k positions later): RoPE against TinyGPT's
//             learned rows; the learner's key position p marked
//   alibi     ALiBi's bias −m·distance for TinyGPT's 4 heads; the learner's distance k marked
//   extrap    the precomputed sweep: loss by position (bins of 8) for five encodings; the trained length marked
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { arr, demoStage, stateOf } from '../demo.ts';
import { slope } from '../../pe.ts';

export type Line = { label: I18n; x: number[]; y: number[] };
export type Chart = { lines: Line[]; xLabel: I18n; yLabel: I18n; mark: { x: number; label: I18n } | null };

export const SCHEME_NAMES: Record<string, I18n> = {
	none: { en: 'no positions (NoPE)', zh: '无位置（NoPE）' },
	learned: { en: 'learned table', zh: '可学习位置表' },
	sinusoidal: { en: 'sinusoidal', zh: '正弦' },
	rope: { en: 'RoPE', zh: 'RoPE' },
	alibi: { en: 'ALiBi', zh: 'ALiBi' }
};
const ix = <T = number>(n: number, f: (i: number) => T = (i) => i as T) => Array.from({ length: n }, (_, i) => f(i));

export function chartOf(p: Pick<StageProps, 'at'>): Chart | null {
	const s = stateOf(p);
	const f = p.at?.frame;
	const m = f?.metrics ?? {};
	if (f?.phase === 'relative') {
		const [a, b] = [arr(s.ropeBy), arr(s.absBy)];
		return {
			lines: [
				{ label: { en: 'RoPE', zh: 'RoPE' }, x: ix(a.length), y: a },
				{ label: { en: 'TinyGPT’s learned rows', zh: 'TinyGPT 学到的位置行' }, x: ix(b.length), y: b }
			],
			xLabel: { en: 'key position (the query sits k later)', zh: '键的位置（查询在其后 k 个位置）' },
			yLabel: { en: 'head-0 dot product q·k', zh: '头 0 的点积 q·k' },
			mark: { x: m.p, label: { en: `p = ${m.p}`, zh: `p = ${m.p}` } }
		};
	}
	if (f?.phase === 'alibi') {
		const t = s.bias;
		const n = t?.shape[1] ?? 0;
		const d = arr(t);
		return {
			lines: ix(t?.shape[0] ?? 0, (h) => ({ label: { en: `head ${h}: m = 1/${1 / slope(h)}`, zh: `头 ${h}：m = 1/${1 / slope(h)}` }, x: ix(n), y: d.slice(h * n, (h + 1) * n) })),
			xLabel: { en: 'distance from the query (tokens back)', zh: '与查询的距离（往前数的词元数）' },
			yLabel: { en: 'bias added to the score', zh: '加到分数上的偏置' },
			mark: { x: m.k, label: { en: `k = ${m.k}`, zh: `k = ${m.k}` } }
		};
	}
	if (f?.phase === 'extrap') {
		const t = s.loss;
		const n = t?.shape[1] ?? 0;
		const d = arr(t);
		const bin = m.bin || 8;
		const keys = Object.keys(SCHEME_NAMES);
		return {
			lines: ix(t?.shape[0] ?? 0, (r) => ({ label: SCHEME_NAMES[keys[r]], x: ix(n, (b) => b * bin + (bin - 1) / 2), y: d.slice(r * n, (r + 1) * n) })),
			xLabel: { en: 'position in the text (mean of 8 positions)', zh: '在文本中的位置（每 8 个位置取平均）' },
			yLabel: { en: 'next-token loss (nats)', zh: '下一个词元的损失（nats）' },
			mark: { x: 64, label: { en: 'trained length 64', zh: '训练长度 64' } }
		};
	}
	return null;
}

export function describeChart(c: Chart | null): I18n {
	if (!c) return { en: 'Nothing to draw yet.', zh: '还没有可画的内容。' };
	const one = (l: Line, zh: boolean) => {
		const fin = l.y.map((y, i) => [l.x[i], y]).filter(([, y]) => Number.isFinite(y));
		if (!fin.length) return '';
		const [x0, y0] = fin[0];
		const [x1, y1] = fin[fin.length - 1];
		return zh ? `${l.label.zh}：x = ${fmt3(x0)} 时为 ${fmt3(y0)}，x = ${fmt3(x1)} 时为 ${fmt3(y1)}` : `${l.label.en}: ${fmt3(y0)} at x = ${fmt3(x0)}, ${fmt3(y1)} at x = ${fmt3(x1)}`;
	};
	return {
		en: `${c.yLabel.en} against ${c.xLabel.en}. ${c.lines.map((l) => one(l, false)).join('; ')}.`,
		zh: `${c.yLabel.zh}随${c.xLabel.zh}的变化。${c.lines.map((l) => one(l, true)).join('；')}。`
	};
}

const meta = {
	key: 'positional-encodings/lines',
	title: { en: 'Lines on shared axes', zh: '同一坐标轴上的几条线' },
	kinds: ['process'],
	describe: (p: StageProps) => describeChart(chartOf(p)),
	demo: () => demoStage('extrap'),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
