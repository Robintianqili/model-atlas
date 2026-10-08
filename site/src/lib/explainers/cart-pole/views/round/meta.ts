// cart-pole view 'round': the steps of a round of episodes as rows of cells, one row per episode. On the 'episode' tape:
// one row, each step's discounted return G as it is computed backwards. On a fine train tape (step1): every episode of
// the round, colored by props.show: 'A' the normalized advantage, 'pull' each step's pull on the output score,
// A·(p(left) − y)/N (y = 1 for a left push), whose sum is the gradient of the output bias. Orange: positive, blue: negative.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { trainingProcess } from '../../../../atlas/train/trainer.ts';
import { reinforce } from '../../trainer.ts';
import { meanStd } from '../../sim.ts';

export type Mode = 'G' | 'A' | 'pull';
export interface Cell { G: number; A: number; p: number; left: number; pull: number }
export interface RoundData { mode: Mode; rows: Cell[][]; N: number; maxLen: number; maxAbs: number; focus: { ep: number; t: number } | null; sum: number; mean?: number; sd?: number }
type Batch = Record<string, Tensor>;

/** The round of a fine train frame, by episode (the batch tensors the trainer wrote). */
export function cellsOf(b: Batch): Cell[][] {
	const N = b.left?.data.length ?? 0, rows: Cell[][] = [];
	for (let i = 0; i < N; i++) {
		const e = b.ep.data[i], A = b.adv.data[i], p = b.p.data[i], left = b.left.data[i];
		(rows[e] ??= []).push({ G: b.ret.data[i], A, p, left, pull: (A * (p - left)) / N });
	}
	return rows;
}

export function roundOf(p: Omit<StageProps, 'ctx'>): RoundData | null {
	const tape = p.at.tape;
	// a new input restarts the sources: until the lesson re-resolves the step, its frame index may not be recorded yet
	if (!tape || p.at.index < 0 || p.at.index >= tape.length) return null;
	const st = tape.state(p.at.index) as { G?: Tensor; batch?: Batch };
	const f = p.at.frame;
	if (st.batch?.adv) {
		const rows = cellsOf(st.batch), mode = (p.props.show === 'pull' ? 'pull' : 'A') as Mode;
		const vals = rows.flat().map((c) => c[mode]);
		const fo = (p.props.focus as { ep: number; t: number } | undefined) ?? { ep: 0, t: 0 };
		const { mean, std } = meanStd(rows.flat().map((c) => c.G));
		return { mode, rows, N: vals.length, maxLen: Math.max(...rows.map((r) => r.length)), maxAbs: Math.max(1e-12, ...vals.map(Math.abs)), focus: fo, sum: rows.flat().reduce((s, c) => s + c.pull, 0), mean, sd: std };
	}
	if (!st.G) return null;
	const len = st.G.shape[0], done = f?.phase === 'return' ? len - (f.metrics?.t ?? len) : 0;
	const row = Array.from(st.G.data, (G, t) => ({ G: t >= len - done ? G : NaN, A: NaN, p: NaN, left: NaN, pull: NaN }));
	return { mode: 'G', rows: [row], N: len, maxLen: len, maxAbs: Math.max(1, ...row.map((c) => (Number.isFinite(c.G) ? c.G : 0))), focus: f?.phase === 'return' ? { ep: 0, t: f.metrics!.t } : null, sum: NaN };
}

export function describeRound(p: Omit<StageProps, 'ctx'>): I18n {
	const d = roundOf(p);
	if (!d) return { en: 'The round: nothing to show yet.', zh: '这一轮：还没有可显示的内容。' };
	const c = d.focus ? d.rows[d.focus.ep]?.[d.focus.t] : undefined;
	if (d.mode === 'G') {
		const n = d.rows[0].filter((x) => Number.isFinite(x.G)).length;
		return { en: `One episode of ${d.N} steps; ${n} returns computed so far, from the last step back${c ? `; step ${d.focus!.t}: G = ${fmt3(c.G)}` : ''}.`, zh: `一个 ${d.N} 步的回合；已经从最后一步往回算了 ${n} 个回报${c ? `；第 ${d.focus!.t} 步：G = ${fmt3(c.G)}` : ''}。` };
	}
	const lens = d.rows.map((r) => r.length).join(', ');
	const pos = d.rows.flat().filter((x) => x.A > 0).length;
	const head = { en: `A round of ${d.rows.length} episodes (lengths ${lens}), ${d.N} steps, mean return ${fmt3(d.mean)}, standard deviation ${fmt3(d.sd)}; ${pos} steps have a positive advantage.`, zh: `一轮 ${d.rows.length} 个回合（长度 ${lens}），共 ${d.N} 步，回报的平均值 ${fmt3(d.mean)}，标准差 ${fmt3(d.sd)}；其中 ${pos} 步的优势为正。` };
	if (d.mode === 'pull') return { en: `${head.en} The pulls on the output bias add up to ${fmt3(d.sum)}.`, zh: `${head.zh}对输出偏置的拉力加起来是 ${fmt3(d.sum)}。` };
	return head;
}

const meta = {
	key: 'cart-pole/round',
	title: { en: 'A round of episodes, step by step', zh: '一轮回合，逐步展开' },
	kinds: ['process', 'train'],
	describe: describeRound,
	async demo() {
		const input = { start: [0, 0, 0.05, 0] };
		const tape = record(trainingProcess(reinforce), input, { optimizer: 'sgd', lr: 0.5, steps: 1 }, { seed: 1, detail: 'fine' });
		const rt = fakeRuntime({ slug: 'cart-pole', input, tapes: { step1: tape } });
		return { rt, source: 'step1', at: atFor(rt, 'step1', 3), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'pull' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
