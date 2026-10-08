// classic-ml view 'canvas': the learner's points on the plane with what one algorithm does to them. props.show picks the
// layer: knn (the new point, its distances, the k nearest and their vote), map (kNN's decision map), gini (training and
// held-out points), split (one node's best cut and every cut it tried), grow (the tree's boxes and its diagram), depth
// (the tree cut at the learner's depth, accuracy by depth), km (clusters and centroids, round by round), rings (k-means
// on the ring, and the total squared distance for k = 1..6). Pure helpers here, the drawing is View.svelte.
import type { Frame, I18n, StageProps, Tape, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { CENTROIDS, dataOf, majority, R, rawOf, START } from '../../ml.ts';
import { knn } from '../../process.ts';

export type Show = 'knn' | 'map' | 'gini' | 'split' | 'grow' | 'depth' | 'km' | 'rings';
export interface Pts { n: number; xy: ArrayLike<number>; label: ArrayLike<number>; test?: ArrayLike<number> }
export interface TNode { id: number; depth: number; box: number[]; pred: number; axis: number; thr: number; m: number; kids: number[] }

type St = { xy?: Tensor; label?: Tensor; test?: Tensor; q?: Tensor; d?: Tensor; nn?: Tensor; grid?: Tensor; cand?: Tensor; C?: Tensor; assign?: Tensor };
export const stateAt = (tape: Tape | undefined, i: number): St => (tape && i >= 0 && i < tape.length ? (tape.state(i) as St) : {});
export function pointsAt(tape: Tape | undefined, i: number): Pts | null {
	const s = stateAt(tape, i);
	return s.xy && s.label ? { n: s.xy.shape[0], xy: s.xy.data, label: s.label.data, test: s.test?.data } : null;
}
export const framesOf = (tape: Tape | undefined, phase: string, upTo = Infinity): Frame[] => {
	const out: Frame[] = [];
	for (let j = 0; tape && j < tape.length && j <= upTo; j++) if (tape.frame(j).phase === phase) out.push(tape.frame(j));
	return out;
};
const counts = (m: Record<string, number>, pre: string) => {
	const c: number[] = [];
	for (let k = 0; `${pre}${k}` in m; k++) c.push(m[`${pre}${k}`]);
	return c;
};
/** the tree as far as frame i of a tree tape: every node examined so far, and the two boxes each split made */
export function treeAt(tape: Tape, i: number): Map<number, TNode> {
	const out = new Map<number, TNode>();
	for (const f of framesOf(tape, 'node', i)) {
		const m = f.metrics!, b = [m.x0, m.x1, m.y0, m.y1];
		out.set(m.node, { id: m.node, depth: m.depth, box: b, pred: m.pred, axis: m.axis, thr: m.thr ?? 0, m: m.m, kids: m.axis >= 0 ? [m.left, m.right] : [] });
		if (m.axis < 0) continue;
		const boxes = m.axis === 0 ? [[b[0], m.thr, b[2], b[3]], [m.thr, b[1], b[2], b[3]]] : [[b[0], b[1], b[2], m.thr], [b[0], b[1], m.thr, b[3]]];
		[m.left, m.right].forEach((id, s) => out.set(id, { id, depth: m.depth + 1, box: boxes[s], pred: majority(counts(m, s ? 'r' : 'l')), axis: -1, thr: 0, m: s ? m.nR : m.nL, kids: [] }));
	}
	return out;
}
/** grid rows as runs of one value (fewer rectangles to draw) */
export function runs(g: ArrayLike<number>): { r: number; c: number; len: number; v: number }[] {
	const out: { r: number; c: number; len: number; v: number }[] = [];
	for (let r = 0; r < R; r++)
		for (let c = 0; c < R; ) {
			let e = c;
			while (e < R && g[r * R + e] === g[r * R + c]) e++;
			out.push({ r, c, len: e - c, v: g[r * R + c] });
			c = e;
		}
	return out;
}
/** the question a split asks: "x ≤ 0.148" */
export const question = (axis: number, thr: number) => `${axis === 0 ? 'x' : 'y'} ≤ ${fmt3(thr)}`;
const I = (en: string, zh: string): I18n => ({ en, zh });

const depthText = (g: Record<string, number>) =>
	I(` The tree cut at depth ${g.d} gets ${fmtPct(g.trainAcc)} of the training points right${g.testAcc >= 0 ? ` and ${fmtPct(g.testAcc)} of the held-out points` : ''}.`, `截到深度 ${g.d} 的树答对 ${fmtPct(g.trainAcc)} 的训练点${g.testAcc >= 0 ? `、${fmtPct(g.testAcc)} 的留出点` : ''}。`);
export function describeCanvas(p: Omit<StageProps, 'ctx'>): I18n {
	const f = p.at.frame, m = f?.metrics ?? {}, pts = pointsAt(p.at.tape, p.at.index);
	if (!f || !pts) return I('The plane: nothing to show yet.', '平面：还没有可显示的内容。');
	const head = I(`${pts.n} points on the plane.`, `平面上有 ${pts.n} 个点。`);
	const tail: I18n =
		f.phase === 'distances' ? I(` The new point is at (${fmt3(m.qx)}, ${fmt3(m.qy)}); the nearest point, number ${m.i0}, is ${fmt3(m.d0)} away.`, `新点在 (${fmt3(m.qx)}, ${fmt3(m.qy)})；最近的点是第 ${m.i0} 号，距离 ${fmt3(m.d0)}。`)
		: f.phase === 'vote' ? I(` The ${m.k} nearest points lie within ${fmt3(m.r)} of the new point; the vote says class ${m.pred}.`, `最近的 ${m.k} 个点都在新点 ${fmt3(m.r)} 以内；投票结果是第 ${m.pred} 类。`)
		: f.phase === 'map' ? I(` Colored by kNN's answer with k = ${m.k}: ${m.patches} separate patches.`, `按 k = ${m.k} 时 kNN 的回答上色：共 ${m.patches} 块互不相连的区域。`)
		: f.phase === 'gini' ? I(` ${m.m} training points, Gini impurity ${fmt3(m.gini)}.`, `${m.m} 个训练点，基尼不纯度 ${fmt3(m.gini)}。`)
		: f.phase === 'node' ? (m.axis >= 0 ? I(` Node ${m.node} asks ${question(m.axis, m.thr)}: weighted Gini ${fmt3(m.w)}, down from ${fmt3(m.gini)}.`, `第 ${m.node} 个节点问 ${question(m.axis, m.thr)}：加权基尼 ${fmt3(m.w)}，原来是 ${fmt3(m.gini)}。`) : I(` Node ${m.node} is a leaf answering class ${m.pred}.`, `第 ${m.node} 个节点是叶子，回答第 ${m.pred} 类。`))
		: f.phase === 'depth' ? depthText(framesOf(p.at.tape, 'depth').find((x) => x.step === rawOf(p.rt.input).depth)?.metrics ?? m)
		: f.phase === 'sweep' ? I(` The total squared distance with k = ${m.k} is ${fmt3(m.inertia)}.`, `k = ${m.k} 时总平方距离是 ${fmt3(m.inertia)}。`)
		: f.phase === 'assign' ? I(` Every point joined its nearest centroid; total squared distance ${fmt3(m.inertia)}.`, `每个点都归入了最近的质心；总平方距离 ${fmt3(m.inertia)}。`)
		: f.phase === 'update' ? I(` Every centroid moved to the mean of its points; total squared distance ${fmt3(m.inertia)}.`, `每个质心都移到了它那组点的平均位置；总平方距离 ${fmt3(m.inertia)}。`)
		: f.phase === 'done' ? I(` No point changed its centroid; total squared distance ${fmt3(m.inertia)}.`, `没有点换质心；总平方距离 ${fmt3(m.inertia)}。`)
		: I(` ${m.k ?? ''} centroids start on random points.`, `${m.k ?? ''} 个质心从随机挑的点出发。`);
	return { en: head.en + tail.en, zh: (head.zh ?? '') + (tail.zh ?? '') };
}

const meta = {
	key: 'classic-ml/canvas',
	title: I('Your points, and what an algorithm does with them', '你的点，以及算法怎样处理它们'),
	kinds: ['process'],
	describe: describeCanvas,
	async demo() {
		const tape = record(knn, dataOf(START), {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'classic-ml', input: START, tapes: { knn: tape } });
		return { rt, source: 'knn', at: atFor(rt, 'knn', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'knn' } };
	},
	minHeight: 420
} satisfies ViewMeta;

export default meta;
