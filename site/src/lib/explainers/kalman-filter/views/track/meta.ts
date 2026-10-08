// kalman-filter view 'track': position over time. The dashed line is the true track, the orange dots the sensor
// readings, the teal line the filter's estimate with its band of two standard deviations. props.show picks what else is
// drawn: noisy (the readings only), predict / grow / measure / gain / update / shrink (one second of the cycle, zoomed:
// the guess before, the prediction, the reading, the new estimate, each with its band; the formula of the step; for
// grow and shrink the covariance ellipse), run (the whole run and the three errors), tune (+ the q and r sliders), turn
// (a window of props.from..props.to seconds). Pure helpers here, the drawing is View.svelte.
import type { I18n, StageProps, Tape, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { dataOf, I, KDEMO, START } from '../../track.ts';
import { kf } from '../../process.ts';

export type Show = 'noisy' | 'predict' | 'grow' | 'measure' | 'gain' | 'update' | 'shrink' | 'run' | 'tune' | 'turn';
export const CYCLE: Show[] = ['predict', 'grow', 'measure', 'gain', 'update', 'shrink'];

/** the filter's estimate and standard deviation at every second, as far as frame i of a 'kf' tape */
export function history(tape: Tape | undefined, i: number): { est: number[]; sd: number[] } {
	const est: number[] = [], sd: number[] = [];
	for (let j = 0; tape && j <= i && j < tape.length; j++) {
		const f = tape.frame(j), m = f.metrics ?? {};
		if (f.phase === 'start' || f.phase === 'update') (est[m.k] = m.p), (sd[m.k] = Math.sqrt(m.P00));
	}
	return { est, sd };
}
/** the turn track around second tk: the root-mean-square error before (seconds 1 to tk − 1) and after (tk + 1 to the
 *  end) the turn, how many seconds after it the truth lies outside the band, and the largest error from tk on */
export function turnStats(tape: Tape | undefined, i: number, tk: number): { pre: number; post: number; out: number; after: number; worst: number } {
	const h = history(tape, i), truth = tape && tape.length ? ((tape.state(0) as { truth?: { data: ArrayLike<number> } }).truth?.data ?? []) : [];
	const e = (s: number) => h.est[s] - truth[s], rms = (a: number, b: number) => Math.sqrt(Array.from({ length: b - a + 1 }, (_, j) => e(a + j) ** 2).reduce((x, y) => x + y, 0) / (b - a + 1));
	const n = h.est.length, after = Array.from({ length: Math.max(0, n - tk - 1) }, (_, j) => tk + 1 + j);
	return { pre: rms(1, tk - 1), post: after.length ? rms(tk + 1, n - 1) : NaN, out: after.filter((s) => Math.abs(e(s)) > 2 * h.sd[s]).length, after: after.length, worst: Math.max(...after.concat(tk).map((s) => Math.abs(e(s)))) };
}
/** the done frame's metrics of a finished tape (empty before) */
export const doneOf = (tape: Tape | undefined): Record<string, number> => (tape && tape.length ? (tape.frame(tape.length - 1).phase === 'done' ? tape.frame(tape.length - 1).metrics ?? {} : {}) : {});
/** how the whole run went (the run step's last sentence): the filter beats the raw readings and the moving average
 *  ('wins'); it does not, but the best pair on the sliders does ('tune'); not even the best pair does ('model') */
export type RunCase = 'wins' | 'tune' | 'model';
export const runCase = (d: Record<string, number>): RunCase =>
	d.rmseKf < d.rmseZ && d.rmseKf < d.rmseMa ? 'wins' : d.bestRmse < Math.min(d.rmseZ, d.rmseMa) ? 'tune' : 'model';
/** a setting as set (0.01, not 0.0100) */
export const setting = (v: number) => String(v);
/** about n round tick values from lo to hi (steps of 1, 2 or 5 times a power of ten) */
export const tick = (lo: number, hi: number, n = 5): number[] => {
	const span = Math.max(1e-9, hi - lo), raw = span / n, p = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map((s) => s * p).find((s) => s >= raw) ?? 10 * p;
	const out: number[] = [];
	for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step);
	return out;
};

export function describeTrack(p: Omit<StageProps, 'ctx'>): I18n {
	const f = p.at.frame, m = f?.metrics ?? {};
	if (!f) return I('The track: nothing to show yet.', '轨迹：还没有可显示的内容。');
	const d = doneOf(p.at.tape);
	if (p.props?.show === 'turn' && f.phase === 'done') {
		const tk = Number(p.props.turnAt ?? 0), s = turnStats(p.at.tape, p.at.index, tk);
		return I(`The cart turns back at second ${tk}. With Q = ${setting(d.q)}: largest error from then on ${fmt3(s.worst)} m; after the turn the truth lies outside the band for ${s.out} of ${s.after} seconds; typical error ${fmt3(s.pre)} m before the turn and ${fmt3(s.post)} m after it.`, `小车在第 ${tk} 秒掉头。Q = ${setting(d.q)} 时：此后最大误差 ${fmt3(s.worst)} 米；掉头后的 ${s.after} 秒中有 ${s.out} 秒真实位置落在误差带之外；典型误差掉头前 ${fmt3(s.pre)} 米，掉头后 ${fmt3(s.post)} 米。`);
	}
	const tail: I18n =
		f.phase === 'start' ? I(`${m.n} sensor readings of the cart's position, one per second, with the true track dashed. Their typical error is ${fmt3(m.rmseZ)} m.`, `小车位置的 ${m.n} 个传感器读数，每秒一个；虚线是真实轨迹。读数的典型误差是 ${fmt3(m.rmseZ)} 米。`)
		: f.phase === 'predict' ? I(`Second ${m.k}: the guess ${fmt3(m.p)} m moves on to the prediction ${fmt3(m.pp)} m; its band widens from ±${fmt3(2 * m.sd)} to ±${fmt3(2 * m.sdp)} m.`, `第 ${m.k} 秒：猜测 ${fmt3(m.p)} 米往前推到预测 ${fmt3(m.pp)} 米；误差带从 ±${fmt3(2 * m.sd)} 变宽到 ±${fmt3(2 * m.sdp)} 米。`)
		: f.phase === 'measure' ? I(`Second ${m.k}: the reading ${fmt3(m.z)} m differs from the prediction ${fmt3(m.pp)} m by ${fmt3(m.y)} m.`, `第 ${m.k} 秒：读数 ${fmt3(m.z)} 米和预测 ${fmt3(m.pp)} 米相差 ${fmt3(m.y)} 米。`)
		: f.phase === 'gain' ? I(`Second ${m.k}: the Kalman gain is ${fmt3(m.K)}.`, `第 ${m.k} 秒：卡尔曼增益是 ${fmt3(m.K)}。`)
		: f.phase === 'update' ? I(`Second ${m.k}: the new estimate is ${fmt3(m.p)} m, band ±${fmt3(2 * m.sd)} m.`, `第 ${m.k} 秒：新的估计是 ${fmt3(m.p)} 米，误差带 ±${fmt3(2 * m.sd)} 米。`)
		: I(`The whole run: typical error ${fmt3(d.rmseKf)} m for the filter, ${fmt3(d.rmseZ)} m for the readings, ${fmt3(d.rmseMa)} m for the moving average.`, `整段运行：滤波器的典型误差 ${fmt3(d.rmseKf)} 米，读数 ${fmt3(d.rmseZ)} 米，移动平均 ${fmt3(d.rmseMa)} 米。`);
	return tail;
}

const meta = {
	key: 'kalman-filter/track',
	title: I('The track, the readings and the filter', '轨迹、读数和滤波器'),
	kinds: ['process'],
	describe: describeTrack,
	async demo() {
		const tape = record(kf, dataOf(START, { fixed: true }), {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'kalman-filter', input: START, tapes: { cycle: tape } });
		return { rt, source: 'cycle', at: atFor(rt, 'cycle', tape.find({ phase: 'update', step: KDEMO })), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'shrink' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
