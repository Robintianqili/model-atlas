// two-tower processes (kind C):
//   logs     the training logs, made by world.ts. Frame 0: the 36 movies' features and the features of the fixed inputs
//            (presets and edges: export.py's reference cases, so parity runs on the bytes the page computes); then one
//            frame per chunk of viewers: their features [c, 15], the movies they liked besides their history (pos, with
//            owner = the viewer's row) and their history (hist, −1 pads). Viewer u depends only on u (world.ts logUser).
//            `npm run atlas -- run two-tower logs --opt '{"n":6000}' --save` writes what pipeline/two-tower/train.py reads.
//   inbatch  B viewers and the movie each liked (recs.ts batchOf, through the exported towers) -> the B×B scores (every
//            viewer against every movie of the batch), the softmax along each row, the loss (cross-entropy of the true
//            movie, averaged over the rows) and its gradient on the scores (kit autograd). Python twin:
//            pipeline/two-tower/reference.py (torch float64).
import type { Frame, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { backward, input as inputVar } from '../../atlas/autograd/var.ts';
import { infoNCE } from '../../atlas/autograd/loss.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { FIXED, MAX_PICKS, MOVIES, N_ITEM_F, N_USER_F, itemFeatures, logUser, userFeatures } from './world.ts';

const I18 = (en: string, zh: string) => ({ en, zh });
type I32 = { shape: number[]; data: Int32Array; dtype: 'i32' };
export type LogsOpt = { n: number; chunk: number };
type LogsState = { items: Tensor; fixedUsers: Tensor; fixedItems: Tensor; fixedPicks: I32; users: Tensor; pos: I32; owner: I32; hist: I32 };
const sum = (a: ArrayLike<number>) => Array.from(a).reduce((x, y) => x + y, 0);
export const HIST_MAX = 6;

export const logs: ProcessDef<unknown, LogsOpt, LogsState> = {
	id: 'logs',
	kind: 'algorithm',
	title: I18('Write the training logs', '生成训练日志'),
	phases: { fixed: { label: I18('The movies and the fixed inputs', '电影和固定输入'), kind: 'input' }, chunk: { label: I18('A chunk of viewers', '一批观众'), kind: 'input' } },
	schema: {
		items: { kind: 'tensor', label: I18('Movie features', '电影特征'), role: 'data', tol: 'exact' },
		fixedUsers: { kind: 'tensor', label: I18('User features of the fixed inputs', '固定输入的用户特征'), role: 'data', tol: 'exact' },
		fixedItems: { kind: 'tensor', label: I18('New-movie features of the fixed inputs', '固定输入的新电影特征'), role: 'data', tol: 'exact' },
		fixedPicks: { kind: 'tensor', label: I18('Picked movies of the fixed inputs (−1 pads)', '固定输入所选的电影（−1 是填充）'), role: 'data', tol: 'exact' },
		users: { kind: 'tensor', label: I18('Viewer features', '观众特征'), role: 'data', tol: 'exact' },
		pos: { kind: 'tensor', label: I18('Movies liked (targets)', '喜欢的电影（目标）'), role: 'target', tol: 'exact' },
		owner: { kind: 'tensor', label: I18('Whose like (row)', '是谁喜欢的（行）'), role: 'data', tol: 'exact' },
		hist: { kind: 'tensor', label: I18('History (−1 pads)', '历史（−1 是填充）'), role: 'data', tol: 'exact' }
	},
	defaults: { n: 24, chunk: 1000 },
	*run(_input, ctx): ProcessRun<LogsState, LogsOpt> {
		const items = new Float32Array(MOVIES.length * N_ITEM_F);
		MOVIES.forEach((m, i) => items.set(itemFeatures(m), i * N_ITEM_F));
		const F = FIXED.length, fu = new Float32Array(F * N_USER_F), fi = new Float32Array(F * N_ITEM_F), fp = new Int32Array(F * MAX_PICKS).fill(-1);
		FIXED.forEach((f, k) => {
			fu.set(userFeatures(f.raw.genres, f.raw.picks), k * N_USER_F);
			fi.set(itemFeatures(f.raw.item), k * N_ITEM_F);
			fp.set(f.raw.picks, k * MAX_PICKS);
		});
		const fixed = { items: { shape: [MOVIES.length, N_ITEM_F], data: items }, fixedUsers: { shape: [F, N_USER_F], data: fu }, fixedItems: { shape: [F, N_ITEM_F], data: fi }, fixedPicks: { shape: [F, MAX_PICKS], data: fp, dtype: 'i32' } };
		yield { phase: 'fixed', state: fixed as LogsState, metrics: { count: F, sum: sum(items) + sum(fu) + sum(fi) } };
		const { n, chunk } = ctx.opt;
		for (let from = 0, k = 0; from < n && !ctx.signal.cancelled; from += chunk, k++) {
			const c = Math.min(chunk, n - from);
			const users = new Float32Array(c * N_USER_F), hist = new Int32Array(c * HIST_MAX).fill(-1), pos: number[] = [], owner: number[] = [];
			for (let r = 0; r < c; r++) {
				const u = logUser(from + r);
				users.set(userFeatures(u.genres, u.history), r * N_USER_F);
				hist.set(u.history, r * HIST_MAX);
				for (const i of u.liked) pos.push(i), owner.push(r);
			}
			yield {
				phase: 'chunk',
				step: k,
				state: { users: { shape: [c, N_USER_F], data: users }, pos: { shape: [pos.length], data: Int32Array.from(pos), dtype: 'i32' }, owner: { shape: [pos.length], data: Int32Array.from(owner), dtype: 'i32' }, hist: { shape: [c, HIST_MAX], data: hist, dtype: 'i32' } } as LogsState,
				metrics: { from, count: c, pairs: pos.length, sum: sum(users) }
			};
		}
	}
};

/** inbatch input: B user vectors and B movie vectors (row i of each is pair i), D numbers each */
export type BatchIn = { U: number[]; V: number[]; B: number; D: number };
type BatchState = { S: Tensor; P: Tensor; G: Tensor };
const mat = (B: number, d: ArrayLike<number>): Tensor => ({ shape: [B, B], data: Float64Array.from(d) });

export const inbatch: ProcessDef<BatchIn | null, Record<string, never>, BatchState> = {
	id: 'inbatch',
	kind: 'training',
	title: I18('Score a batch against itself', '让一批数据自己给自己打分'),
	phases: {
		scores: { label: I18('Every viewer against every movie of the batch', '批里每位观众对每部电影'), kind: 'forward' },
		softmax: { label: I18('Softmax along each row, and the loss', '沿每一行做 softmax，以及损失'), kind: 'loss' },
		grad: { label: I18('Gradient on the scores', '分数上的梯度'), kind: 'backward' }
	},
	schema: {
		S: { kind: 'tensor', label: I18('Scores (dot products)', '分数（点积）'), role: 'activation', tol: 1e-12 },
		P: { kind: 'tensor', label: I18('Row softmax: which movie of the batch fits each viewer', '行 softmax：批里哪部电影最配每位观众'), role: 'output', tol: 1e-12 },
		G: { kind: 'tensor', label: I18('Gradient of the loss on each score', '损失对每个分数的梯度'), role: 'grad', tol: 1e-12 }
	},
	defaults: {},
	*run(b): ProcessRun<BatchState, Record<string, never>> {
		if (!b || b.B < 1) return;
		const { B, D } = b;
		const s = new Float64Array(B * B);
		for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
			let a = 0;
			for (let d = 0; d < D; d++) a += b.U[i * D + d] * b.V[j * D + d];
			s[i * B + j] = a;
		}
		const S = inputVar({ shape: [B, B], data: s }, true);
		const diag = (a: ArrayLike<number>) => Array.from({ length: B }, (_, i) => a[i * B + i]);
		const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
		const off = Array.from(s).filter((_, k) => k % (B + 1) !== 0);
		yield { phase: 'scores', state: { S: mat(B, s) }, metrics: { B, D, s00: s[0], diagMean: mean(diag(s)), offMean: off.length ? mean(off) : 0 } } as Frame<BatchState>;
		const P = op.softmax(S, 1).data;
		const loss = infoNCE(S, { symmetric: false });
		const hits = Array.from({ length: B }, (_, i): number => diag(s)[i] >= Math.max(...Array.from(s.subarray(i * B, i * B + B))) ? 1 : 0).reduce((x, y) => x + y, 0);
		yield { phase: 'softmax', state: { P: mat(B, P) }, metrics: { p00: P[0], loss0: -Math.log(P[0]), loss: loss.data[0], chance: Math.log(B), hits } } as Frame<BatchState>;
		backward(loss);
		const g = S.grad!.data;
		const offG = Array.from(g).filter((_, k) => k % (B + 1) !== 0);
		yield { phase: 'grad', state: { G: mat(B, g) }, metrics: { g00: g[0], g01: B > 1 ? g[1] : 0, gMin: Math.min(...diag(g)), gOffMax: offG.length ? Math.max(...offG) : 0, gSum: Array.from(g).reduce((x, y) => x + y, 0) } } as Frame<BatchState>;
	}
};
