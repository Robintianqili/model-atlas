// classic-ml: the processes (kind C). Each reads MlData (ml.ts dataOf: plain numbers) and yields one frame per event;
// pipeline/classic-ml/reference.py repeats each one in Python, frame by frame. Ties go to the smaller index everywhere
// (neighbours at equal distance, splits with equal impurity, equal votes, equally near centroids).
//   knn     one new point: query -> distances (to every point) -> vote (the k nearest)
//   knnMap  kNN's answer at every cell of the grid: one 'map' frame per k of KS (step = k)
//   tree    CART with Gini: holdout (one point in four held back) -> gini (the root) -> node (every node above depth
//           DMAX, breadth first; step = node id; a split or a leaf) -> depth (the tree cut at depth d = 1..DMAX)
//   kmeans  Lloyd's algorithm: init (k random points) -> assign -> update -> ... -> done (no point changed);
//           opt.sweep: then k-means again for k = 1..sweep, each from its own random start (sweep frames)
import type { Frame, ProcessDef, ProcessRun, Rng, Tensor } from '../../atlas/types/index.ts';
import { B, cellX, cellY, DMAX, gini, I, KS, majority, patches, R, type MlData } from './ml.ts';

const T = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(data) });
const per = (pre: string, v: number[]) => Object.fromEntries(v.map((x, k) => [`${pre}${k}`, x]));
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const pts = (d: MlData) => ({ xy: T([d.label.length, 2], d.xy), label: T([d.label.length], d.label) });

// ------------------------------------------------------------------ k nearest neighbours
export type KnnState = { xy: Tensor; label: Tensor; q: Tensor; d: Tensor; nn: Tensor };
export const knn: ProcessDef<MlData, Record<string, never>, KnnState> = {
	id: 'knn', kind: 'algorithm', title: I('k nearest neighbours: one new point', 'k 近邻：一个新点'),
	phases: {
		query: { label: I('A new point', '一个新点'), kind: 'input' },
		distances: { label: I('Its distance to every point', '它到每个点的距离'), kind: 'forward' },
		vote: { label: I('The k nearest vote', '最近的 k 个点投票'), kind: 'output' }
	},
	schema: {
		xy: { kind: 'tensor', label: I('Your points (x, y)', '你的点（x, y）'), tol: 'exact' },
		label: { kind: 'tensor', label: I('Class of each point', '每个点的类别'), tol: 'exact' },
		q: { kind: 'tensor', label: I('The new point', '新点'), tol: 'exact' },
		d: { kind: 'tensor', label: I('Distance to each point', '到每个点的距离'), tol: 1e-12 },
		nn: { kind: 'tensor', label: I('The k nearest points, nearest first', '最近的 k 个点，由近到远'), tol: 'exact' }
	},
	defaults: {},
	*run(input): ProcessRun<KnnState, Record<string, never>> {
		const n = input.label.length, [qx, qy] = input.q, C = input.classes;
		if (!n) return;
		yield { phase: 'query', state: { ...pts(input), q: T([2], input.q) }, metrics: { n, qx, qy, classes: C } };
		const d = range(n).map((i) => {
			const dx = input.xy[2 * i] - qx, dy = input.xy[2 * i + 1] - qy;
			return Math.sqrt(dx * dx + dy * dy);
		});
		const order = range(n).sort((a, b) => d[a] - d[b] || a - b), i0 = order[0];
		yield { phase: 'distances', state: { d: T([n], d) }, metrics: { n, qx, qy, i0, x0: input.xy[2 * i0], y0: input.xy[2 * i0 + 1], d0: d[i0], c0: input.label[i0] } };
		const k = Math.min(input.k, n), votes = new Array<number>(C).fill(0);
		for (let j = 0; j < k; j++) votes[input.label[order[j]]]++;
		const top = Math.max(...votes);
		yield { phase: 'vote', state: { nn: T([k], order.slice(0, k)) }, metrics: { k, kAsked: input.k, r: d[order[k - 1]], ...per('v', votes), pred: majority(votes), tie: votes.filter((v) => v === top).length > 1 ? 1 : 0 } };
	}
};

export type MapState = { grid: Tensor };
export const knnMap: ProcessDef<MlData, Record<string, never>, MapState> = {
	id: 'knnMap', kind: 'algorithm', title: I('kNN’s answer everywhere, for several k', '不同 k 下 kNN 在每一处的回答'),
	phases: { map: { label: I('The decision map', '决策图'), kind: 'output' } },
	schema: { grid: { kind: 'tensor', label: I('The answer at each cell of the plane', '平面上每个格子的回答'), scale: 'categorical', tol: 'exact' } },
	defaults: {},
	*run(input): ProcessRun<MapState, Record<string, never>> {
		const n = input.label.length, ks = KS.filter((k) => k <= n), C = input.classes;
		if (!n) return;
		const kmax = ks[ks.length - 1], grids = ks.map(() => new Float64Array(R * R)), bi = new Int32Array(kmax), bd = new Float64Array(kmax);
		for (let r = 0; r < R; r++)
			for (let c = 0; c < R; c++) {
				const px = cellX(c), py = cellY(r);
				let m = 0;
				for (let i = 0; i < n; i++) {
					const dx = input.xy[2 * i] - px, dy = input.xy[2 * i + 1] - py, v = dx * dx + dy * dy;
					if (m === kmax && v >= bd[m - 1]) continue;
					let j = m < kmax ? m++ : m - 1;
					for (; j > 0 && bd[j - 1] > v; j--) (bd[j] = bd[j - 1]), (bi[j] = bi[j - 1]);
					(bd[j] = v), (bi[j] = i);
				}
				const votes = new Array<number>(C).fill(0);
				for (let j = 0, s = 0; j < kmax; j++) {
					votes[input.label[bi[j]]]++;
					if (j + 1 === ks[s]) grids[s++][r * R + c] = majority(votes);
				}
			}
		for (let s = 0; s < ks.length; s++) yield { phase: 'map', step: ks[s], state: { grid: T([R, R], grids[s]) }, metrics: { k: ks[s], patches: patches(grids[s]) }, caption: I(`k = ${ks[s]}`, `k = ${ks[s]}`) };
	}
};

// ------------------------------------------------------------------ decision tree (CART, Gini)
export type TreeState = { xy: Tensor; label: Tensor; test: Tensor; cand: Tensor; grid: Tensor };
interface Node { depth: number; idx: number[]; box: number[]; pred: number; axis: number; thr: number; kids: number[] }
export const tree: ProcessDef<MlData, Record<string, never>, TreeState> = {
	id: 'tree', kind: 'algorithm', title: I('A decision tree grows on your points', '一棵决策树在你的点上生长'),
	phases: {
		holdout: { label: I('Hold back one point in four', '每四个点留出一个'), kind: 'input' },
		gini: { label: I('Impurity of the training points', '训练点的不纯度'), kind: 'loss' },
		node: { label: I('Split a box, or stop', '切开一个方框，或者停下'), kind: 'update' },
		depth: { label: I('The tree cut at depth d', '截到深度 d 的树'), kind: 'observe' }
	},
	schema: {
		xy: { kind: 'tensor', label: I('Your points (x, y)', '你的点（x, y）'), tol: 'exact' },
		label: { kind: 'tensor', label: I('Class of each point', '每个点的类别'), tol: 'exact' },
		test: { kind: 'tensor', label: I('Held out (1) or training point (0)', '留出点（1）或训练点（0）'), tol: 'exact' },
		cand: { kind: 'tensor', label: I('Every split tried here: axis, threshold, weighted Gini', '这里试过的每个切分：轴、阈值、加权基尼'), tol: 1e-12 },
		grid: { kind: 'tensor', label: I('The tree’s answer at each cell', '树在每个格子的回答'), scale: 'categorical', tol: 'exact' }
	},
	defaults: {},
	*run(input, ctx): ProcessRun<TreeState, Record<string, never>> {
		const n = input.label.length, C = input.classes, at = (i: number, axis: number) => input.xy[2 * i + axis];
		if (!n) return;
		const perm = range(n);
		for (let i = n - 1; i > 0; i--) {
			const j = ctx.rng.int(i + 1);
			[perm[i], perm[j]] = [perm[j], perm[i]];
		}
		const nTest = n >= 2 ? Math.max(1, Math.floor(n / 4)) : 0, test = new Array<number>(n).fill(0);
		for (let i = 0; i < nTest; i++) test[perm[i]] = 1;
		const train = range(n).filter((i) => !test[i]);
		const count = (idx: number[]) => idx.reduce((c, i) => (c[input.label[i]]++, c), new Array<number>(C).fill(0));
		yield { phase: 'holdout', state: { ...pts(input), test: T([n], test) }, metrics: { n, nTrain: train.length, nTest, classes: C } };
		const root = count(train);
		yield { phase: 'gini', state: {}, metrics: { m: train.length, gini: gini(root, train.length), ...per('c', root), pred: majority(root) } };
		const nodes: Node[] = [{ depth: 0, idx: train, box: [-B, B, -B, B], pred: majority(root), axis: -1, thr: 0, kids: [] }];
		for (let id = 0; id < nodes.length; id++) {
			const nd = nodes[id], m = nd.idx.length, counts = count(nd.idx), g = gini(counts, m);
			if (nd.depth >= DMAX) continue;
			const cand: number[] = [];
			let best = { w: Infinity, axis: -1, thr: 0, cut: 0, order: [] as number[] };
			for (const axis of g > 0 ? [0, 1] : []) {
				const order = [...nd.idx].sort((a, b) => at(a, axis) - at(b, axis) || a - b), left = new Array<number>(C).fill(0);
				for (let i = 1; i < m; i++) {
					left[input.label[order[i - 1]]]++;
					if (at(order[i - 1], axis) === at(order[i], axis)) continue;
					const w = (i / m) * gini(left, i) + ((m - i) / m) * gini(counts.map((c, k) => c - left[k]), m - i), thr = (at(order[i - 1], axis) + at(order[i], axis)) / 2;
					cand.push(axis, thr, w);
					if (w < best.w) best = { w, axis, thr, cut: i, order };
				}
			}
			const base = { node: id, depth: nd.depth, m, gini: g, ...per('c', counts), pred: nd.pred, x0: nd.box[0], x1: nd.box[1], y0: nd.box[2], y1: nd.box[3] };
			if (best.axis < 0 || !(g - best.w > 1e-12)) {
				yield { phase: 'node', step: id, state: {}, metrics: { ...base, axis: -1 } };
				continue;
			}
			const { axis, thr } = best, sides = [best.order.slice(0, best.cut), best.order.slice(best.cut)], cs = sides.map(count);
			const boxes = axis === 0 ? [[nd.box[0], thr, nd.box[2], nd.box[3]], [thr, nd.box[1], nd.box[2], nd.box[3]]] : [[nd.box[0], nd.box[1], nd.box[2], thr], [nd.box[0], nd.box[1], thr, nd.box[3]]];
			Object.assign(nd, { axis, thr, kids: [nodes.length, nodes.length + 1] });
			sides.forEach((idx, s) => nodes.push({ depth: nd.depth + 1, idx, box: boxes[s], pred: majority(cs[s]), axis: -1, thr: 0, kids: [] }));
			const [nL, nR] = [sides[0].length, sides[1].length], [gL, gR] = [gini(cs[0], nL), gini(cs[1], nR)];
			yield { phase: 'node', step: id, state: { cand: T([cand.length / 3, 3], cand) }, metrics: { ...base, axis, thr, nL, nR, gL, gR, w: best.w, gain: g - best.w, left: nd.kids[0], right: nd.kids[1], ...per('l', cs[0]), ...per('r', cs[1]) } };
		}
		const leafOf = (x: number, y: number, d: number) => {
			let nd = nodes[0];
			while (nd.depth < d && nd.kids.length) nd = nodes[(nd.axis === 0 ? x : y) <= nd.thr ? nd.kids[0] : nd.kids[1]];
			return nd;
		};
		for (let d = 1; d <= DMAX; d++) {
			const grid = new Float64Array(R * R), ok = [0, 0];
			for (let r = 0; r < R; r++) for (let c = 0; c < R; c++) grid[r * R + c] = leafOf(cellX(c), cellY(r), d).pred;
			for (let i = 0; i < n; i++) if (leafOf(at(i, 0), at(i, 1), d).pred === input.label[i]) ok[test[i]]++;
			const leaves = nodes.filter((nd) => nd.depth === d || (nd.depth < d && !nd.kids.length)).length;
			yield { phase: 'depth', step: d, state: { grid: T([R, R], grid) }, metrics: { d, trainAcc: ok[0] / train.length, testAcc: nTest ? ok[1] / nTest : -1, leaves, nTrain: train.length, nTest } };
		}
	}
};

// ------------------------------------------------------------------ k-means (Lloyd)
export type KmOpt = { sweep: number };
export type KmState = { xy: Tensor; label: Tensor; C: Tensor; assign: Tensor };
const MAXIT = 100;
/** Lloyd's algorithm from k random points of d; pushes its frames into `out` when given */
function lloyd(d: MlData, k: number, rng: Rng, out?: Frame<KmState>[]): { inertia: number; rounds: number } {
	const n = d.label.length, X = (i: number) => d.xy[2 * i], Y = (i: number) => d.xy[2 * i + 1];
	const pool = range(n), pick: number[] = [];
	for (let j = 0; j < k; j++) {
		const r = rng.int(n - j);
		pick.push(pool[r]);
		pool[r] = pool[n - 1 - j];
	}
	const C = pick.flatMap((i) => [X(i), Y(i)]), assign = new Array<number>(n).fill(-1);
	const d2 = (i: number, j: number) => (X(i) - C[2 * j]) * (X(i) - C[2 * j]) + (Y(i) - C[2 * j + 1]) * (Y(i) - C[2 * j + 1]);
	const inertia = () => range(n).reduce((s, i) => s + d2(i, assign[i]), 0);
	// the worked example: the point nearest the middle of the starting centroids
	const mx = range(k).reduce((s, j) => s + C[2 * j], 0) / k, my = range(k).reduce((s, j) => s + C[2 * j + 1], 0) / k;
	const mid = (i: number) => (X(i) - mx) * (X(i) - mx) + (Y(i) - my) * (Y(i) - my);
	let p = 0;
	for (let i = 1; i < n; i++) if (mid(i) < mid(p)) p = i;
	out?.push({ phase: 'init', step: 0, state: { ...pts(d), C: T([k, 2], C), assign: T([n], assign) }, metrics: { k, n, p, ...per('p', pick) } });
	for (let t = 0; t < MAXIT; t++) {
		let changed = 0;
		for (let i = 0; i < n; i++) {
			let b = 0;
			for (let j = 1; j < k; j++) if (d2(i, j) < d2(i, b)) b = j;
			if (b !== assign[i]) changed++;
			assign[i] = b;
		}
		const e = inertia();
		out?.push({ phase: 'assign', step: t, state: { assign: T([n], assign) }, metrics: { t, changed, inertia: e, p, px: X(p), py: Y(p), j: assign[p], ...per('d', range(k).map((j) => Math.sqrt(d2(p, j)))) } });
		if (!changed || t === MAXIT - 1) {
			const mix = range(k).flatMap((j) => range(d.classes).map((c) => range(n).filter((i) => assign[i] === j && d.label[i] === c).length));
			out?.push({ phase: 'done', step: t, state: {}, metrics: { k, rounds: t, inertia: e, converged: changed ? 0 : 1, ...per('m', mix) } });
			return { inertia: e, rounds: t };
		}
		const old = C.slice();
		let moved = 0;
		const sums = range(k).map((j) => {
			let sx = 0, sy = 0, c = 0;
			for (let i = 0; i < n; i++) if (assign[i] === j) (sx += X(i)), (sy += Y(i)), c++;
			if (c) (C[2 * j] = sx / c), (C[2 * j + 1] = sy / c);
			const dx = C[2 * j] - old[2 * j], dy = C[2 * j + 1] - old[2 * j + 1];
			moved = Math.max(moved, Math.sqrt(dx * dx + dy * dy));
			return { sx, sy, c };
		});
		out?.push({ phase: 'update', step: t, state: { C: T([k, 2], C) }, metrics: { t, moved, inertia: inertia(), cnt: sums[0].c, sx: sums[0].sx, sy: sums[0].sy, ox: old[0], oy: old[1], nx: C[0], ny: C[1] } });
	}
	return { inertia: NaN, rounds: MAXIT };
}

export const kmeans: ProcessDef<MlData, KmOpt, KmState> = {
	id: 'kmeans', kind: 'algorithm', title: I('k-means finds groups by itself', 'k-means 自己找出分组'),
	phases: {
		init: { label: I('k random points become centroids', '随机挑 k 个点当质心'), kind: 'input' },
		assign: { label: I('Assign: every point joins its nearest centroid', '分配：每个点归到最近的质心'), kind: 'aggregate' },
		update: { label: I('Update: each centroid moves to the mean of its points', '更新：质心移到它那组点的平均位置'), kind: 'update' },
		done: { label: I('No point changed: done', '没有点换组：结束'), kind: 'output' },
		sweep: { label: I('k-means again with another k', '换一个 k 再做一次'), kind: 'observe' }
	},
	schema: {
		xy: { kind: 'tensor', label: I('Your points (x, y)', '你的点（x, y）'), tol: 'exact' },
		label: { kind: 'tensor', label: I('Class of each point (k-means never reads it)', '每个点的类别（k-means 不看它）'), tol: 'exact' },
		C: { kind: 'tensor', label: I('The centroids (x, y)', '质心（x, y）'), tol: 1e-12 },
		assign: { kind: 'tensor', label: I('The centroid each point joined', '每个点归入的质心'), scale: 'categorical', tol: 'exact' }
	},
	defaults: { sweep: 0 },
	*run(input, ctx): ProcessRun<KmState, KmOpt> {
		const n = input.label.length;
		if (!n) return;
		const out: Frame<KmState>[] = [];
		lloyd(input, Math.min(input.kc, n), ctx.rng, out);
		yield* out;
		for (let k = 1; k <= ctx.opt.sweep; k++) {
			const r = lloyd(input, Math.min(k, n), ctx.rng.fork(`k${k}`));
			yield { phase: 'sweep', step: k, state: {}, metrics: { k, inertia: r.inertia, rounds: r.rounds } };
		}
	}
};
