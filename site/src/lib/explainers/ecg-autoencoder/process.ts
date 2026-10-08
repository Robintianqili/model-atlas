// ecg-autoencoder: the processes (kind C).
//   data   the synthetic beats: training, validation and test sets, one frame per chunk; beat i of a set draws only from
//          ctx.rng.fork(`<set><i>`), so the first beats are the same whatever the sizes. Validation and test: nVal (nTest)
//          normal beats, then nOdd of each anomaly kind (the training set is normal only). `npm run atlas -- run
//          ecg-autoencoder data --opt '{"nTrain":4000}' --save` writes the sets pipeline/ecg-autoencoder/train.py reads
//          (it checks every chunk's CRC-32 on the bytes it read). No Python reference: TS makes the data.
//   sweep  the threshold slides down over the test set's anomaly scores (precomputed by PyTorch, testset.json): each step
//          flags one more beat, counted as caught (odd) or a false alarm (normal); the counts trace the ROC curve, and the
//          area under it is added up by the trapezoid rule. Python reference: pipeline/ecg-autoencoder/reference.py.
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { I, KINDS, N, beat, crcBytes, testKinds } from './beats.ts';

type I32 = { shape: number[]; data: Int32Array; dtype: 'i32' };
export type DataOpt = { nTrain: number; nVal: number; nTest: number; nOdd: number; chunk: number };
type DataState = { beats: Tensor; kinds: I32 };
const SETS = ['train', 'val', 'test'] as const;

export const data: ProcessDef<unknown, DataOpt, DataState> = {
	id: 'data',
	kind: 'algorithm',
	title: I('Make the synthetic heartbeats', '生成合成心跳'),
	phases: { chunk: { label: I('A batch of beats', '一批心跳'), kind: 'input' } },
	schema: {
		beats: { kind: 'tensor', label: I('Beats (140 numbers each)', '心跳（每个 140 个数）'), role: 'data', tol: 'exact' },
		kinds: { kind: 'tensor', label: I('Kind of each beat (0 is normal)', '每个心跳的类型（0 是正常）'), role: 'target', tol: 'exact' }
	},
	defaults: { nTrain: 12, nVal: 0, nTest: 0, nOdd: 0, chunk: 500 },
	*run(_input, ctx): ProcessRun<DataState, DataOpt> {
		const { nTrain, nVal, nTest, nOdd, chunk } = ctx.opt;
		const kinds = { train: Array(nTrain).fill(0), val: testKinds(nVal, nOdd), test: testKinds(nTest, nOdd) };
		let k = 0;
		for (const [s, set] of SETS.entries()) {
			const ks = kinds[set];
			for (let from = 0; from < ks.length && !ctx.signal.cancelled; from += chunk) {
				const c = Math.min(chunk, ks.length - from);
				const beats = new Float32Array(c * N), kk = new Int32Array(c);
				for (let i = 0; i < c; i++) {
					kk[i] = ks[from + i];
					beats.set(beat(KINDS[kk[i]], ctx.rng.fork(`${set}${from + i}`)), i * N);
				}
				yield { phase: 'chunk', step: k++, state: { beats: { shape: [c, N], data: beats }, kinds: { shape: [c], data: kk, dtype: 'i32' } }, metrics: { set: s, from, count: c, hash: crcBytes(beats) } };
			}
		}
	}
};

/** sweep input: the test set's anomaly scores and kinds (0 normal, 1-4 the anomaly kinds), testset.json. */
export type SweepIn = { scores: number[]; kinds: number[] };
type SweepState = { point: Tensor };

export const sweep: ProcessDef<SweepIn | null, Record<string, never>, SweepState> = {
	id: 'sweep',
	kind: 'algorithm',
	title: I('Slide the threshold down: the ROC curve', '把阈值往下滑：ROC 曲线'),
	phases: {
		start: { label: I('Threshold above every score: nothing flagged', '阈值高于所有分数：什么都不标记'), kind: 'input' },
		flag: {
			label: I('Lower the threshold to the next score: one more beat flagged', '把阈值降到下一个分数：多标记一个心跳'),
			kind: 'other',
			narration: { what: I('The threshold drops to {frame.thr}: {frame.tp} odd beats caught, {frame.fp} false alarms.', '阈值降到 {frame.thr}：抓到 {frame.tp} 个异常心跳，误报 {frame.fp} 个。') }
		},
		area: {
			label: I('The area under the curve', '曲线下的面积'),
			kind: 'output',
			formula: { tex: '\\text{AUC} = \\sum \\tfrac{1}{2}(\\text{TPR}_{k-1} + \\text{TPR}_k)\\,(\\text{FPR}_k - \\text{FPR}_{k-1}) = {{auc}}', values: (f) => ({ auc: f.metrics!.auc }) }
		}
	},
	schema: { point: { kind: 'tensor', label: I('Where the curve is: false-alarm rate, catch rate', '曲线当前的位置：误报率、检出率'), tol: 1e-12 } },
	defaults: {},
	*run(d): ProcessRun<SweepState, Record<string, never>> {
		if (!d || !d.scores.length) return;
		const n = d.scores.length, odd = d.kinds.map((k) => (k > 0 ? 1 : 0));
		const nPos = odd.reduce((a: number, b) => a + b, 0), nNeg = n - nPos;
		// highest score first; ties: the smaller index first
		const order = d.scores.map((_, i) => i).sort((a, b) => d.scores[b] - d.scores[a] || a - b);
		const pt = (x: number, y: number): Tensor => ({ shape: [2], data: Float64Array.of(x, y) });
		yield { phase: 'start', step: 0, state: { point: pt(0, 0) }, metrics: { n, nPos, nNeg, tp: 0, fp: 0 } };
		let tp = 0, fp = 0, auc = 0, x0 = 0, y0 = 0;
		for (let k = 0; k < n; k++) {
			const i = order[k];
			if (odd[i]) tp++;
			else fp++;
			const x = nNeg ? fp / nNeg : 0, y = nPos ? tp / nPos : 0;
			auc += 0.5 * (y0 + y) * (x - x0);
			x0 = x;
			y0 = y;
			const prec = tp / (tp + fp);
			yield { phase: 'flag', step: k + 1, state: { point: pt(x, y) }, metrics: { k: k + 1, beat: i, kind: d.kinds[i], thr: d.scores[i], tp, fp, fpr: x, tpr: y, precision: prec, recall: y } };
		}
		yield { phase: 'area', step: n + 1, state: {}, metrics: { auc, nPos, nNeg } };
	}
};
