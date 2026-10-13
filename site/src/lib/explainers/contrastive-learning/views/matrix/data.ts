// contrastive-learning view 'matrix': what each mode draws, read from the step's frame (pure TS: the view, its describe()
// and the lesson's narration use the same numbers).
import type { I18n, StageProps, Tensor } from '../../../../atlas/types/index.ts';
import { pairsOf, tauOf } from '../../adapters/batch/adapter.ts';
import { render } from '../../scenes.ts';
import { field, lossAt, met, tauCurve } from '../../data.ts';
import { partner } from '../../process.ts';

type P = Pick<StageProps, 'rt' | 'at' | 'props'>;
export type Show = 'cos' | 'P' | 'Q' | 'G';
const WHAT: Record<string, I18n> = {
	cos: { en: 'cosines: drawing i (row) with caption j (column)', zh: '余弦：画 i（行）与描述 j（列）' },
	P: { en: 'row softmax of cos / τ: which caption fits each drawing', zh: 'cos / τ 的行 softmax：每幅画配哪句描述' },
	Q: { en: 'column softmax of cos / τ: which drawing fits each caption', zh: 'cos / τ 的列 softmax：每句描述配哪幅画' },
	G: { en: 'gradient of the loss on each score', zh: '损失对每个分数的梯度' },
	progress: { en: 'cosines at this checkpoint of the training run', zh: '训练到这个检查点时的余弦' },
	simclr: { en: 'cosines of every two views: view i (row) with view j (column)', zh: '每两个视图的余弦：视图 i（行）与视图 j（列）' }
};
const FIELD: Record<Show, string> = { cos: 'C', P: 'P', Q: 'Q', G: 'G' };
const arr = (t: Tensor | undefined) => (t ? Array.from(t.data) : null);
const thumbs = (V: Tensor | undefined, n: number) => Array.from({ length: n }, (_, k) => (V ? V.data.subarray(k * 784, (k + 1) * 784) : new Float32Array(784)));

export type ViewData =
	| { kind: 'table'; show: Show | 'progress' | 'simclr'; what: I18n; M: number[] | null; rows: ArrayLike<number>[]; captions: string[] | null; marks: [number, number][]; blank: boolean; domain: 'unit' | 'share' | 'signed' }
	| { kind: 'pair'; a: number[]; b: number[]; prods: number[]; cos: number }
	| { kind: 'curve'; taus: number[]; loss: number[]; tau: number; now: number; best: { tau: number; loss: number; atEdge: boolean }; chance: number }
	| { kind: 'shift'; images: ArrayLike<number>[]; jitter: number; shift: number };

export function viewData(p: P, toggled?: Show): ViewData {
	const mode = String(p.props.mode ?? 'cos');
	const pairs = pairsOf(p.rt.input);
	const N = pairs.length;
	const images = () => {
		const d = p.rt.pre?.tensors.images?.data as Float32Array | undefined;
		return pairs.map((q, i) => d?.subarray(i * 784, (i + 1) * 784) ?? render(q.strokes));
	};
	const diag = (n: number): [number, number][] => Array.from({ length: n }, (_, i) => [i, i]);
	if (mode === 'pair') {
		const I = field(p.at, 'I'), T = field(p.at, 'T');
		const D = I?.shape[1] ?? 0;
		const a = I ? Array.from(I.data.subarray(0, D)) : [], b = T ? Array.from(T.data.subarray(0, D)) : [];
		const prods = a.map((x, d) => x * b[d]);
		return { kind: 'pair', a, b, prods, cos: prods.reduce((s, x) => s + x, 0) };
	}
	if (mode === 'tau') {
		const C = field(p.at, 'C');
		const n = C?.shape[0] ?? 0;
		const c = C ? tauCurve(C.data, n) : { taus: [], loss: [], best: { tau: NaN, loss: NaN, atEdge: false } };
		const tau = tauOf(p.rt.input);
		const now = C ? lossAt(C.data, n, tau) : NaN;
		return { kind: 'curve', taus: c.taus, loss: c.loss, tau, now, best: c.best, chance: Math.log(n) };
	}
	if (mode === 'progress') {
		return { kind: 'table', show: 'progress', what: WHAT.progress, M: arr(field(p.at, 'C')), rows: images(), captions: pairs.map((q) => q.caption), marks: diag(N), blank: false, domain: 'unit' };
	}
	if (mode === 'simclr') {
		const S = field(p.at, 'S');
		const M = S?.shape[0] ?? 0;
		return { kind: 'table', show: 'simclr', what: WHAT.simclr, M: arr(S), rows: thumbs(field(p.at, 'V'), M), captions: null, marks: Array.from({ length: M }, (_, k) => [k, partner(k)] as [number, number]), blank: true, domain: 'unit' };
	}
	if (mode === 'shift') {
		const V = field(p.at, 'V');
		return { kind: 'shift', images: [field(p.at, 'A')?.data ?? new Float32Array(784), thumbs(V, 1)[0], field(p.at, 'B')?.data ?? new Float32Array(784)], jitter: met(p.at, 'jitter0'), shift: met(p.at, 'shift0') };
	}
	const show = (toggled ?? (mode as Show)) in FIELD ? (toggled ?? (mode as Show)) : 'cos';
	return { kind: 'table', show, what: WHAT[show], M: arr(field(p.at, FIELD[show])), rows: images(), captions: pairs.map((q) => q.caption), marks: diag(N), blank: false, domain: show === 'cos' ? 'unit' : show === 'G' ? 'signed' : 'share' };
}

