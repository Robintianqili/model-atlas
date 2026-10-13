// gaussian-process view 'plot': x across, y up. The orange dots are the points, the purple line the posterior mean and
// the shaded band mean ± 2 standard deviations. props.show picks what else is drawn: prior (the prior and three
// functions drawn from it, sliders), kernel (the closest and the farthest pair, and the kernel as a curve of distance),
// first (one point, worked out), gap (the widest gap between neighbouring points), points (the posterior point by
// point), matrix (the kernel matrix with its numbers and the weights α), samples (three functions drawn from the
// posterior), evidence and tune (the log marginal likelihood for every length-scale; tune adds the sliders), cost (what
// the solve costs), choice (the step data, for a compare). Pure helpers here, the drawing is View.svelte.
import type { I18n, StageProps, Tape, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3, fmtNum } from '../../../../atlas/core/format.ts';
import { dataOf, I, START } from '../../gp.ts';
import { gp } from '../../process.ts';

export type Show = 'prior' | 'kernel' | 'first' | 'gap' | 'points' | 'matrix' | 'samples' | 'evidence' | 'tune' | 'cost' | 'choice';
/** the metrics of the first frame of a phase in a finished (or running) tape; empty before it is recorded */
export function phaseOf(tape: Tape | undefined, phase: string): Record<string, number> {
	for (let i = 0; tape && i < tape.length; i++) if (tape.frame(i).phase === phase) return tape.frame(i).metrics ?? {};
	return {};
}
/** a setting as set (0.25, not 0.250) */
export const setting = (v: number) => String(v);

export function describePlot(p: Omit<StageProps, 'ctx'>): I18n {
	const f = p.at.frame, m = f?.metrics ?? {}, show = String(p.props?.show ?? 'prior');
	if (!f) return I('The plot: nothing to show yet.', '图：还没有可显示的内容。');
	const d = phaseOf(p.at.tape, 'done'), e = phaseOf(p.at.tape, 'evidence'), pr = phaseOf(p.at.tape, 'prior');
	const head = I(`${pr.n} points with an RBF kernel, length-scale ${setting(pr.ls)} and signal variance ${setting(pr.sf2)}. `, `${pr.n} 个点，RBF 核，长度尺度 ${setting(pr.ls)}，信号方差 ${setting(pr.sf2)}。`);
	const tail: I18n =
		show === 'prior' ? I(`Before any data the mean is 0 and the band is ±${fmt3(m.band0)} everywhere; three functions are drawn from the prior.`, `还没有数据时，均值处处为 0，误差带处处为 ±${fmt3(m.band0)}；图中画了三条从先验中抽取的函数。`)
		: show === 'kernel' || show === 'matrix' ? I(`The kernel matrix has ${m.n} × ${m.n} entries; the diagonal is ${fmt3(m.kd)}. The mean at x = 0 is ${fmt3(m.mq)}.`, `核矩阵有 ${m.n} × ${m.n} 个元素，对角线是 ${fmt3(m.kd)}。x = 0 处的均值是 ${fmt3(m.mq)}。`)
		: f.phase === 'add' ? I(`After point ${m.i} (counting from 0) at x = ${fmt3(m.xi)}: the mean there is ${fmt3(m.mAt)} and the band ±${fmt3(2 * m.sdAt)}; the average band is ±${fmt3(m.avgBand)}.`, `加入编号 ${m.i}（从 0 数起）的点（x = ${fmt3(m.xi)}）后：那里的均值是 ${fmt3(m.mAt)}，误差带 ±${fmt3(2 * m.sdAt)}；误差带的平均宽度是 ±${fmt3(m.avgBand)}。`)
		: show === 'samples' ? I(`Three functions drawn from the posterior: they spread ${fmt3(m.spreadNear)} apart at x = ${fmt3(m.xNear)} and ${fmt3(m.spreadFar)} at x = ${fmt3(m.xFar)}.`, `三条从后验中抽取的函数：在 x = ${fmt3(m.xNear)} 处相差 ${fmt3(m.spreadNear)}，在 x = ${fmt3(m.xFar)} 处相差 ${fmt3(m.spreadFar)}。`)
		: show === 'evidence' || show === 'tune' ? I(`Log marginal likelihood ${fmt3(e.lml)} at this length-scale; the best on the slider is ℓ = ${setting(e.bestLs)} with ${fmt3(e.bestLml)}.`, `这个长度尺度的对数边缘似然是 ${fmt3(e.lml)}；滑块上最好的是 ℓ = ${setting(e.bestLs)}，为 ${fmt3(e.bestLml)}。`)
		: show === 'cost' ? I(`Factoring the ${d.n} × ${d.n} matrix takes about ${fmt3(d.ops)} arithmetic steps; ${fmtNum(d.big)} points would take ${fmt3(d.opsBig)}.`, `分解 ${d.n} × ${d.n} 的矩阵大约需要 ${fmt3(d.ops)} 步运算；${fmtNum(d.big)} 个点则需要 ${fmt3(d.opsBig)} 步。`)
		: d.gw !== undefined ? I(`The widest gap runs from x = ${fmt3(d.ga)} to ${fmt3(d.gb)}: the band is ±${fmt3(2 * d.sdA)} at its left end, ±${fmt3(2 * d.sdMid)} in its middle and ±${fmt3(2 * d.sdB)} at its right end. The mean peaks at ${fmt3(d.peak)}.`, `最宽的空白从 x = ${fmt3(d.ga)} 到 ${fmt3(d.gb)}：误差带在左端是 ±${fmt3(2 * d.sdA)}，中间是 ±${fmt3(2 * d.sdMid)}，右端是 ±${fmt3(2 * d.sdB)}。均值最高到 ${fmt3(d.peak)}。`)
		: I('The posterior mean and its band.', '后验均值和它的误差带。');
	return I(head.en + tail.en, `${head.zh}${tail.zh}`);
}

const meta = {
	key: 'gaussian-process/plot',
	title: I('The points, the posterior and the kernel', '数据点、后验和核函数'),
	kinds: ['process'],
	describe: describePlot,
	async demo() {
		const tape = record(gp, dataOf(START), {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'gaussian-process', input: START, tapes: { gp: tape } });
		return { rt, source: 'gp', at: atFor(rt, 'gp', tape.find({ phase: 'done' })), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'gap' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
