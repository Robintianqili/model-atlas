// View 'residual-connections/bars': one row per layer (or per depth) with horizontal bars, the network without skips in
// the plain color and with skips in the accent color. Props:
//   mode?: 'grads' (default)  the size of each layer's weight gradient, log axis 1e−7 .. 10 (frame state gW, other)
//          'flow'             the size of the gradient at each hidden state's output, same log axis (state gH)
//          'acts'             the size (RMS) of each hidden state, linear axis from 0 (state rms, otherRms)
//          'depths'           one row per depth: the loss after training, linear axis from 0, a dashed line at ln 2
//                             (the frames of process 'depths')
//   both?: boolean            also draw the other setting of the skip switch (grads, acts: the same starting weights)
//   plainOnly?: boolean       depths: only the networks without skips
//   mark?: 'ends' | 'mid'     outline layer 0 and the last layer (grads), or the middle layer's output and input (flow)
// At a frame before the backward pass (phase 'forward') the gradient modes show the rows without bars and a note.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps } from '../../../../atlas/types/view.ts';
import type { Tensor } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { midLayer } from '../../data.ts';

export type Mode = 'grads' | 'flow' | 'acts' | 'depths';
/** One bar: its series (plain or with skips) and value. */
export type Bar = { skip: number; value: number };
export type Row = { label: I18n; bars: Bar[]; mark: boolean; note?: string };
export interface BarsPic {
	mode: Mode;
	log: boolean;
	lo: number;
	hi: number;
	rows: Row[];
	/** the series shown, in drawing order (0 = plain, 1 = with skips) */
	series: number[];
	/** a dashed reference line (ln 2 in depths mode) */
	ref?: { value: number; label: I18n };
	/** shown instead of bars (a frame before the backward pass) */
	waiting?: I18n;
}

export const LOG_LO = 1e-7;
export const LOG_HI = 10;
export const NAMES: Record<number, I18n> = { 0: { en: 'no skips', zh: '无跳跃连接' }, 1: { en: 'with skips', zh: '有跳跃连接' } };
export const AXES: Record<Mode, I18n> = {
	grads: { en: 'size of the layer\'s weight gradient (log scale)', zh: '该层权重梯度的大小（对数刻度）' },
	flow: { en: 'size of the gradient at the layer\'s output (log scale)', zh: '该层输出处梯度的大小（对数刻度）' },
	acts: { en: 'size (RMS) of the layer\'s output', zh: '该层输出的大小（均方根）' },
	depths: { en: 'loss after training', zh: '训练后的损失' }
};
const layer = (l: number): I18n => ({ en: `layer ${l}`, zh: `第 ${l} 层` });
const data = (t: unknown): number[] | null => (t && typeof t === 'object' && 'data' in (t as Tensor) ? Array.from((t as Tensor).data) : null);

export function barsOf(p: Pick<StageProps, 'at' | 'props'>): BarsPic | null {
	const props = p.props ?? {};
	const mode = (['grads', 'flow', 'acts', 'depths'] as Mode[]).includes(props.mode as Mode) ? (props.mode as Mode) : 'grads';
	const at = p.at;
	if (mode === 'depths') {
		const tape = at?.tape;
		if (!tape || !tape.length) return null;
		const plainOnly = props.plainOnly === true;
		const rows: Row[] = [];
		const last = Math.min(at.index ?? tape.length - 1, tape.length - 1);
		for (let i = 0; i <= last; i++) {
			const m = tape.frame(i).metrics ?? {};
			const bars: Bar[] = [{ skip: 0, value: m.lossPlain }];
			if (!plainOnly) bars.push({ skip: 1, value: m.loss });
			rows.push({ label: { en: `${m.depth} layers`, zh: `${m.depth} 层` }, bars, mark: false, note: plainOnly ? fmtPct(m.accPlain) : undefined });
		}
		const top = Math.max(0.8, ...rows.flatMap((r) => r.bars.map((b) => (Number.isFinite(b.value) ? b.value : 0))));
		return { mode, log: false, lo: 0, hi: top, rows, series: plainOnly ? [0] : [0, 1], ref: { value: Math.LN2, label: { en: 'coin flip, ln 2', zh: '抛硬币，ln 2' } } };
	}
	const m = at?.frame?.metrics ?? {};
	const st = (at?.state ?? {}) as Record<string, unknown>;
	const depth = m.depth;
	if (!Number.isFinite(depth)) return null;
	const skip = m.skip ? 1 : 0;
	const both = props.both === true;
	const series = both ? [0, 1] : [skip];
	const mine = data(mode === 'grads' ? st.gW : mode === 'flow' ? st.gH : st.rms);
	const other = data(mode === 'grads' ? st.other : mode === 'acts' ? st.otherRms : null);
	const n = mode === 'grads' ? depth : depth - 1;
	const mid = midLayer(depth);
	const rows: Row[] = [];
	for (let l = 0; l < n; l++) {
		const bars: Bar[] = [];
		for (const s of series) {
			const v = s === skip ? mine?.[l] : other?.[l];
			if (v !== undefined) bars.push({ skip: s, value: v });
		}
		const mark = props.mark === 'ends' ? l === 0 || l === n - 1 : props.mark === 'mid' ? l === mid || l === mid - 1 : false;
		rows.push({ label: layer(l), bars, mark });
	}
	const waiting: I18n | undefined = mine ? undefined : { en: 'The backward pass has not run yet: no layer has a gradient so far.', zh: '反向传播还没有运行：到目前为止，没有哪一层有梯度。' };
	if (mode === 'acts') {
		const top = Math.max(1, ...rows.flatMap((r) => r.bars.map((b) => b.value)));
		return { mode, log: false, lo: 0, hi: Math.ceil(top), rows, series };
	}
	return { mode, log: true, lo: LOG_LO, hi: LOG_HI, rows, series: both ? [0, 1] : [skip], waiting };
}

/** A bar's length as a share of the track (0..1), on the picture's axis. */
export function lengthOf(pic: BarsPic, v: number): number {
	if (!Number.isFinite(v) || v <= 0) return 0;
	const f = pic.log ? (Math.log10(v) - Math.log10(pic.lo)) / (Math.log10(pic.hi) - Math.log10(pic.lo)) : (v - pic.lo) / (pic.hi - pic.lo);
	return Math.max(0, Math.min(1, f));
}

export function describeBars(pic: BarsPic | null): I18n {
	if (!pic) return { en: 'Nothing has been computed yet.', zh: '还没有算出任何数值。' };
	if (pic.waiting) return pic.waiting;
	const list = (lang: 'en' | 'zh') =>
		pic.rows.map((r) => `${r.label[lang]}: ${r.bars.map((b) => `${NAMES[b.skip][lang]} ${fmt3(b.value)}`).join(lang === 'en' ? ', ' : '，')}`).join(lang === 'en' ? '; ' : '；');
	return { en: `${AXES[pic.mode].en}. ${list('en')}.`, zh: `${AXES[pic.mode].zh}。${list('zh')}。` };
}

const meta = {
	key: 'residual-connections/bars',
	title: { en: 'Bars per layer, with and without skips', zh: '逐层条形图：有无跳跃连接' },
	kinds: ['process'],
	describe: (p: StageProps): I18n => describeBars(barsOf(p)),
	async demo() {
		const [{ record }, { grads }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { depth: 8, skip: 0 };
		const tape = record(grads, input, {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'residual-connections', input, tapes: { bars: tape } });
		return { rt, source: 'bars', at: { index: 1, frame: tape.frame(1), state: tape.state(1), tape }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { both: true, mark: 'ends' } };
	},
	minHeight: 300
};

export default meta;
