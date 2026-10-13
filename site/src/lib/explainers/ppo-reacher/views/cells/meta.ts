// ppo-reacher view 'cells': steps as rows of colored cells. On the 'episode' tape: one episode's rewards r, values V(s),
// TD errors δ and advantages A (δ and A appear as the GAE pass computes them, from the last step back), the ringed column
// being the frame's step. On a fine train tape at an iteration's first update: the whole batch, one row per episode,
// colored by its normalized advantage. Orange: above 0, blue: below 0, stronger = larger.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { episode } from '../../process.ts';

export interface Row { label: I18n; v: number[] }
export interface Cells { mode: 'episode' | 'batch'; rows: Row[]; T: number; focus: number | null; maxAbs: number[] }
const I = (en: string, zh: string): I18n => ({ en, zh });

export function cellsOf(p: Omit<StageProps, 'ctx'>): Cells | null {
	const tape = p.at.tape;
	// a new input restarts the sources: until the lesson re-resolves the step, its frame index may not be recorded yet
	if (!tape || p.at.index < 0 || p.at.index >= tape.length) return null;
	const b = (tape.state(p.at.index) as { batch?: Record<string, Tensor> }).batch;
	if (b?.allAdv) {
		const E = b.epRet.data.length, T = b.allAdv.data.length / E, all = Array.from(b.allAdv.data);
		const rows = Array.from({ length: E }, (_, e) => ({ label: I(`${e}`, `${e}`), v: all.slice(e * T, (e + 1) * T) }));
		const m = Math.max(1e-12, ...all.map(Math.abs));
		return { mode: 'batch', rows, T, focus: null, maxAbs: rows.map(() => m) };
	}
	const r: number[] = [], v: number[] = [], d: number[] = [], A: number[] = [];
	let focus: number | null = null;
	for (let i = 0; i <= p.at.index; i++) {
		const f = tape.frame(i), m = f.metrics ?? {};
		if (f.phase === 'act') v[m.t] = m.v;
		if (f.phase === 'move') r[m.t] = m.r;
		if (f.phase === 'gae') (d[m.t] = m.delta), (A[m.t] = m.A), i === p.at.index && (focus = m.t);
	}
	const T = r.length;
	if (!T) return null;
	const fill = (x: number[]) => Array.from({ length: T }, (_, t) => x[t] ?? NaN);
	const rows = ([['r', r], ['V(s)', v], ['δ', d], ['A', A]] as [string, number[]][]).map(([k, x]) => ({ label: I(k, k), v: fill(x) }));
	return { mode: 'episode', rows, T, focus, maxAbs: rows.map((x) => Math.max(1e-12, ...x.v.filter(Number.isFinite).map(Math.abs))) };
}

export function describeCells(p: Omit<StageProps, 'ctx'>): I18n {
	const c = cellsOf(p);
	if (!c) return { en: 'The steps: nothing to show yet.', zh: '每一步：还没有可显示的内容。' };
	if (c.mode === 'batch') {
		const pos = c.rows.flatMap((r) => r.v).filter((x) => x > 0).length;
		return { en: `A batch of ${c.rows.length} episodes of ${c.T} steps; ${pos} of the ${c.rows.length * c.T} steps have an advantage above 0.`, zh: `一批 ${c.rows.length} 个回合，每个 ${c.T} 步；${c.rows.length * c.T} 步里有 ${pos} 步的优势大于 0。` };
	}
	const n = c.rows[3].v.filter(Number.isFinite).length, t = c.focus;
	const at = t === null ? { en: '', zh: '' } : { en: ` Step ${t}: r = ${fmt3(c.rows[0].v[t])}, V = ${fmt3(c.rows[1].v[t])}, δ = ${fmt3(c.rows[2].v[t])}, A = ${fmt3(c.rows[3].v[t])}.`, zh: `第 ${t} 步：r = ${fmt3(c.rows[0].v[t])}，V = ${fmt3(c.rows[1].v[t])}，δ = ${fmt3(c.rows[2].v[t])}，A = ${fmt3(c.rows[3].v[t])}。` };
	return { en: `One episode of ${c.T} steps: rewards and values of every step; ${n} advantages worked out so far, from the last step back.${at.en}`, zh: `一个 ${c.T} 步的回合：每一步的奖励和价值；已经从最后一步往回算出 ${n} 个优势。${at.zh}` };
}

const meta = {
	key: 'ppo-reacher/cells',
	title: { en: 'Steps as rows of cells', zh: '用一行行格子表示每一步' },
	kinds: ['process', 'train'],
	describe: describeCells,
	async demo() {
		const input = { target: [-0.3, 0.6], eps: 0.2 };
		const tape = record(episode, input, {}, { seed: 1, recordDraws: true });
		const rt = fakeRuntime({ slug: 'ppo-reacher', input, tapes: { episode: tape } });
		return { rt, source: 'episode', at: atFor(rt, 'episode', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
