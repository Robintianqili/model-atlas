// hmm view 'lattice': the days across, the two weather states down, one cell per day and state. props.show picks the
// tape and what is drawn: guess (the forward tape's first frame: one guessed path and its product), forward / fill (the
// forward tape: α cells filling in, the current cell's two ways in and its formula; fill also the last column's sum),
// viterbi / back (the Viterbi tape: δ cells, back-pointers, the trace back, the decoded path against the activity-alone
// row), task (the Viterbi tape at its end: your path against the starting tables' path, and the tables to edit).
// Pure helpers here, the drawing is View.svelte.
import type { I18n, StageProps, Tape, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ACTS, dataOf, I, K, START, STATES } from '../../hmm.ts';
import { forward } from '../../process.ts';

export type Show = 'guess' | 'forward' | 'fill' | 'viterbi' | 'back' | 'task';
export interface Lat { n: number; obs: number[]; val: number[][]; psi: number[][]; path: number[]; naive: number[]; path0: number[]; guess: number[] }
const arr = (x: unknown): number[] => Array.from(((x as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);
const rows = (x: unknown, n: number): number[][] => {
	const a = arr(x);
	return Array.from({ length: n }, (_, t) => a.slice(t * K, t * K + K));
};
/** everything the lattice draws, from the state of frame i of a forward or Viterbi tape */
export function latOf(tape: Tape | undefined, i: number): Lat {
	const s = (tape && tape.length && i >= 0 ? tape.state(Math.min(i, tape.length - 1)) : {}) as Record<string, unknown>;
	const obs = arr(s.obs), n = obs.length;
	return { n, obs, val: rows(s.alpha ?? s.delta, n), psi: rows(s.psi, n), path: arr(s.path), naive: arr(s.naive), path0: arr(s.path0), guess: arr(s.guess) };
}
/** a state's name, and a short one for chips */
export const stateName = (j: number): I18n => STATES[j] ?? I('?', '？');
/** the decoded path as words: "Sunny, Sunny, Rainy" */
export const pathText = (p: number[]): I18n => I(p.map((s) => STATES[s]?.en ?? '?').join(', '), p.map((s) => STATES[s]?.zh ?? '？').join('、'));
/** the days on which two paths differ, as "3 and 6" */
export function daysText(a: number[], b: number[]): I18n {
	const d = a.map((v, t) => (v !== b[t] ? t : -1)).filter((t) => t >= 0);
	const en = d.length < 2 ? d.join('') : `${d.slice(0, -1).join(', ')} and ${d[d.length - 1]}`;
	return I(en, d.join('、'));
}

export function describeLattice(p: Omit<StageProps, 'ctx'>): I18n {
	const f = p.at.frame, m = f?.metrics ?? {}, L = latOf(p.at.tape, p.at.index), show = String(p.props?.show ?? 'forward');
	if (!f || !L.n) return I('The lattice: nothing to show yet.', '网格：还没有可显示的内容。');
	const acts = I(L.obs.map((o) => ACTS[o].en).join(', '), L.obs.map((o) => ACTS[o].zh).join('、'));
	const head = I(`${L.n} days (${acts.en}) by 2 weather states.`, `${L.n} 天（${acts.zh}）乘 2 种天气。`);
	const S = (j: number) => stateName(j);
	const tail: I18n =
		f.phase === 'guess' ? I(`One guessed path, ${pathText(L.guess).en}: probability ${fmt3(m.joint)} together with her activities.`, `一条猜测的路径（${pathText(L.guess).zh}）：它与她的活动同时出现的概率是 ${fmt3(m.joint)}。`)
		: f.phase === 'init' ? I(`Day 0, ${S(m.j).en}: ${fmt3(m.pi)} × ${fmt3(m.b)} = ${fmt3(m.alpha ?? m.delta)}.`, `第 0 天，${S(m.j).zh}：${fmt3(m.pi)} × ${fmt3(m.b)} = ${fmt3(m.alpha ?? m.delta)}。`)
		: f.phase === 'cell' && m.alpha !== undefined ? I(`Day ${m.t}, ${S(m.j).en}: (${fmt3(m.p0)} × ${fmt3(m.a0)} + ${fmt3(m.p1)} × ${fmt3(m.a1)}) × ${fmt3(m.b)} = ${fmt3(m.alpha)}.`, `第 ${m.t} 天，${S(m.j).zh}：(${fmt3(m.p0)} × ${fmt3(m.a0)} + ${fmt3(m.p1)} × ${fmt3(m.a1)}) × ${fmt3(m.b)} = ${fmt3(m.alpha)}。`)
		: f.phase === 'cell' ? I(`Day ${m.t}, ${S(m.j).en}: the larger of ${fmt3(m.c0)} and ${fmt3(m.c1)}, times ${fmt3(m.b)}, is ${fmt3(m.delta)}; the back-pointer points to ${S(m.from).en}.`, `第 ${m.t} 天，${S(m.j).zh}：${fmt3(m.c0)} 和 ${fmt3(m.c1)} 中较大的一个乘 ${fmt3(m.b)}，得 ${fmt3(m.delta)}；回溯指针指向${S(m.from).zh}。`)
		: f.phase === 'end' && m.evidence !== undefined && m.last === undefined ? I(`The last column sums to ${fmt3(m.evidence)}, the probability of all her activities.`, `最后一列之和是 ${fmt3(m.evidence)}，也就是她全部活动的概率。`)
		: L.path.length && L.path.every((s) => s >= 0) ? I(`The decoded weather: ${pathText(L.path).en}.${show === 'task' && L.path0.length && L.path0[0] >= 0 ? ` With the starting tables: ${pathText(L.path0).en}.` : ''}`, `解码出的天气：${pathText(L.path).zh}。${show === 'task' && L.path0.length && L.path0[0] >= 0 ? `用起始表格：${pathText(L.path0).zh}。` : ''}`)
		: I(`Tracing back: day ${m.t ?? L.n - 1} is ${S(m.s ?? m.last ?? 0).en}.`, `正在回溯：第 ${m.t ?? L.n - 1} 天是${S(m.s ?? m.last ?? 0).zh}。`);
	return I(`${head.en} ${tail.en}`, `${head.zh}${tail.zh}`);
}

const meta = {
	key: 'hmm/lattice',
	title: I('The lattice: days by weather', '网格：天数乘天气'),
	kinds: ['process'],
	describe: describeLattice,
	async demo() {
		const tape = record(forward, dataOf(START), {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'hmm', input: START, tapes: { fwd: tape } });
		return { rt, source: 'fwd', at: atFor(rt, 'fwd', tape.find({ phase: 'cell', step: 1, sub: 1 })), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'forward' } };
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
