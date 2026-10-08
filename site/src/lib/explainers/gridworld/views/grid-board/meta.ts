// gridworld view 'grid-board': the maze of a 'q' or 'vi' tape with its Q-table (four triangles per cell), values,
// arrows, the agent and the path of an episode. Pure helpers here (shared with View.svelte); the drawing is View.svelte.
import type { I18n, StageProps, Tape, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { arrowOf, DEFAULTS, DIR_NAMES, type World } from '../../world.ts';
import { q } from '../../process.ts';

export interface Board extends World { gamma: number; Q?: ArrayLike<number>; V?: ArrayLike<number>; pos?: number }
/** The board at frame i of a q or vi tape (the start and γ are metrics of its first frame). */
export function boardAt(tape: Tape | null | undefined, i: number): Board | null {
	if (!tape || i < 0 || i >= tape.length) return null;
	const st = tape.state(i) as { board?: Tensor; Q?: Tensor; V?: Tensor; pos?: number };
	const m0 = tape.frame(0).metrics ?? {};
	if (!st.board) return null;
	return { w: st.board.shape[1], h: st.board.shape[0], cell: Array.from(st.board.data), start: m0.start ?? 0, gamma: m0.gamma ?? 0.9, Q: st.Q?.data, V: st.V?.data, pos: st.pos };
}
/** A cell's value: V on a vi tape, else its best Q. */
export const valueOf = (b: Board, s: number) => (b.V ? b.V[s] : b.Q ? Math.max(b.Q[s * 4], b.Q[s * 4 + 1], b.Q[s * 4 + 2], b.Q[s * 4 + 3]) : 0);
const cr = (b: Board, s: number) => [s % b.w, Math.floor(s / b.w)];

export function describeBoard(p: Omit<StageProps, 'ctx'>): I18n {
	const b = boardAt(p.at.tape, p.at.index);
	if (!b) return { en: 'The maze: nothing to show yet.', zh: '迷宫：还没有可显示的内容。' };
	const m = p.at.frame?.metrics ?? {};
	const [x, y] = cr(b, b.start);
	const head = { en: `A maze of ${b.w} by ${b.h} cells with the start at column ${x}, row ${y} (counting from 0).`, zh: `一个 ${b.w}×${b.h} 的迷宫，起点在第 ${x} 列、第 ${y} 行（从 0 数起）。` };
	const ph = p.at.frame?.phase;
	if (ph === 'act')
		return { en: `${head.en} The agent at column ${m.x}, row ${m.y} draws u = ${fmt3(m.u)} against ε = ${fmt3(m.eps)}, ${m.explore ? 'explores' : 'exploits'}, moves ${DIR_NAMES[m.a].en} to column ${m.x2}, row ${m.y2} and gets r = ${m.r}.`, zh: `${head.zh}智能体在第 ${m.x} 列、第 ${m.y} 行，抽到 u = ${fmt3(m.u)}，与 ε = ${fmt3(m.eps)} 比较后选择${m.explore ? '探索' : '利用'}，向${DIR_NAMES[m.a].zh}走到第 ${m.x2} 列、第 ${m.y2} 行，得到 r = ${m.r}。` };
	if (ph === 'update')
		return { en: `${head.en} Target ${fmt3(m.r)} + ${fmt3(m.gamma)} × ${fmt3(m.maxNext)} = ${fmt3(m.target)}; Q goes from ${fmt3(m.old)} to ${fmt3(m.new)}.`, zh: `${head.zh}目标值 ${fmt3(m.r)} + ${fmt3(m.gamma)} × ${fmt3(m.maxNext)} = ${fmt3(m.target)}；Q 从 ${fmt3(m.old)} 变成 ${fmt3(m.new)}。` };
	const arrows = b.Q ? b.cell.filter((_, s) => arrowOf(b.Q!, s) >= 0).length : 0;
	return { en: `${head.en} The start's value is ${fmt3(valueOf(b, b.start))}; ${arrows} cells have an arrow.`, zh: `${head.zh}起点的价值是 ${fmt3(valueOf(b, b.start))}；有 ${arrows} 个格子有箭头。` };
}

const meta = {
	key: 'gridworld/grid-board',
	title: { en: 'The maze and its Q-table', zh: '迷宫和它的 Q 表' },
	kinds: ['process'],
	describe: describeBoard,
	async demo() {
		const tape = record(q, DEFAULTS, {}, { seed: 12 });
		const rt = fakeRuntime({ slug: 'gridworld', input: DEFAULTS, tapes: { q: tape } });
		return { rt, source: 'q', at: atFor(rt, 'q', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'heat,arrows' } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
