// ecg-autoencoder: local input adapter 'series' (it replaces the kit's for this package; the kit's Capture.svelte is kept:
// chips, drawing a stretch of the curve, arrow keys). The learner's raw input is a BeatRaw: the 140 heights `y` (a
// preset's or drawn ones), the preset it came from (`beat`, only to name it) and the threshold `thr` set on the stage.
// It becomes the model input `input` [140] (float32); another length is stretched to 140 points (linear interpolation).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { I, KINDS, N, beatOf, type Kind } from '../../beats.ts';

export interface BeatRaw { y?: number[]; beat?: { kind: Kind; seed: number }; thr?: number }
export const beatRaw = (kind: Kind, seed: number): BeatRaw => ({ beat: { kind, seed }, y: Array.from(beatOf(kind, seed), (v) => Math.round(v * 1e5) / 1e5) });
export const rawOf = (raw: unknown): BeatRaw => (raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as BeatRaw) : {});

/** The 140 numbers the model reads (empty when nothing is drawn). */
export function valuesOf(raw: unknown): Float32Array {
	const y = (rawOf(raw).y ?? []).map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0));
	if (y.length === N || y.length <= 1) return y.length ? (y.length === N ? Float32Array.from(y) : new Float32Array(N).fill(y[0])) : new Float32Array(0);
	const out = new Float32Array(N);
	for (let i = 0; i < N; i++) {
		const x = (i * (y.length - 1)) / (N - 1), a = Math.floor(x), b = Math.min(y.length - 1, a + 1);
		out[i] = y[a] + (x - a) * (y[b] - y[a]);
	}
	return out;
}

/** The preset a raw input still equals (its kind), or null when the learner drew on it. */
export function presetKind(raw: unknown): Kind | null {
	const r = rawOf(raw);
	if (!r.beat || !KINDS.includes(r.beat.kind)) return null;
	const y = valuesOf(raw), ref = beatOf(r.beat.kind, r.beat.seed);
	return y.length === N && y.every((v, i) => Math.abs(v - ref[i]) < 1e-4) ? r.beat.kind : null;
}

/** The input widget's beats (manifest input.presetsFromAdapter: no presets file). Seed 9's shape draws give a typical
 * normal beat and a strong version of every anomaly (its k is 0.94); the claim tests check what the lesson says of them. */
export const PRESETS: { id: string; kind: Kind; seed: number; en: string; zh: string }[] = [
	{ id: 'normal', kind: 'normal', seed: 9, en: 'Normal beat', zh: '正常心跳' },
	{ id: 'normal-2', kind: 'normal', seed: 21, en: 'Another normal beat', zh: '另一个正常心跳' },
	{ id: 'noP', kind: 'noP', seed: 9, en: 'P wave missing', zh: '缺 P 波' },
	{ id: 'wideQRS', kind: 'wideQRS', seed: 9, en: 'Wide QRS', zh: 'QRS 增宽' },
	{ id: 'stUp', kind: 'stUp', seed: 9, en: 'Raised ST segment', zh: 'ST 段抬高' },
	{ id: 'noisy', kind: 'noisy', seed: 9, en: 'Noisy recording', zh: '噪声大的记录' }
];

const adapter: InputAdapter<BeatRaw> = {
	id: 'series',
	kind: 'series',
	preprocess(raw): Preprocessed {
		const x = valuesOf(raw);
		const notes = x.length ? [] : [I('There is no beat yet: pick one, or draw one.', '还没有心跳：选一个，或者画一个。')];
		return { tensors: { input: { shape: [x.length], data: x } }, steps: [{ id: 'beat', label: I('Your beat: 140 heights in time order', '你的心跳：按时间排列的 140 个高度'), tensor: { shape: [x.length], data: x } }], empty: !x.length, notes };
	},
	presets: () => PRESETS.map((p) => ({ id: p.id, label: I(p.en, p.zh), raw: beatRaw(p.kind, p.seed) })),
	edgeFixtures: () => [
		{ id: 'empty', label: I('No beat', '没有心跳'), raw: { y: [] } },
		{ id: 'flat', label: I('A flat line', '一条平线'), raw: { y: Array(N).fill(0) } },
		{ id: 'short', label: I('A drawing of 5 values, stretched to 140', '画了 5 个值，拉伸成 140 个'), raw: { y: [0, 0.2, 1.2, -0.3, 0.3] } },
		{ id: 'top', label: I('Pinned at the top of the box, threshold 0', '贴着框的顶部，阈值为 0'), raw: { y: Array(N).fill(1.4), thr: 0 } }
	],
	layer0: { view: 'ecg-autoencoder/beat', narration: { what: I('A beat is {n|int} numbers.', '一次心跳是 {n|int} 个数。') } }
};

export default adapter;
