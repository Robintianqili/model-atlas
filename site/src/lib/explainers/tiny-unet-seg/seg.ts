// tiny-unet-seg helpers (pure TS, shared by the views, the lesson and the claim tests): read the traces of the U-Net
// (source fwd) and of the same network without skip connections (source noskip); the class map (argmax of the 4 shares
// of each pixel), IoU per class against the true classes of the learner's scene, the border pixels, and the window of
// input pixels an output pixel's answer is computed from along each path of the U (the receptive field).
import type { AtlasSpec, Box, ExplainerRuntime, GraphModel, Tensor } from '../../atlas/types/index.ts';
import { mergeParams } from '../../atlas/ops/registry.ts';
import { fakeRuntime } from '../../atlas/runtime/fake.ts';
import { truthOf } from './adapters/shapes32/adapter.ts';
import { render, S } from './scenes.ts';

type RT = ExplainerRuntime;
export const K = 4;
export const T = (rt: RT, id: string, src = 'fwd'): Tensor | undefined => rt.sources[src]?.trace?.t[id];

/** The class of every pixel (the largest of its 4 shares) and that share. */
export function classMap(rt: RT, src = 'fwd'): { k: Uint8Array; p: Float32Array } | null {
	const pr = T(rt, 'probs', src);
	if (!pr) return null;
	const n = S * S, k = new Uint8Array(n), p = new Float32Array(n);
	for (let i = 0; i < n; i++) {
		let b = 0;
		for (let c = 1; c < K; c++) if (pr.data[c * n + i] > pr.data[b * n + i]) b = c;
		(k[i] = b), (p[i] = pr.data[b * n + i]);
	}
	return { k, p };
}

/** True for pixels whose class differs from one of their 4 neighbours (both sides of every border), as train.py. */
export function borders(lab: Uint8Array): Uint8Array {
	const e = new Uint8Array(S * S);
	for (let i = 0; i < S * S; i++) {
		const x = i % S;
		if (x < S - 1 && lab[i] !== lab[i + 1]) e[i] = e[i + 1] = 1;
		if (i + S < S * S && lab[i] !== lab[i + S]) e[i] = e[i + S] = 1;
	}
	return e;
}

export type ClassScore = { k: number; inter: number; union: number; truth: number; pred: number; iou: number };
/** IoU per class on the learner's scene (classes that are in the truth or in the answer), their mean, accuracy. */
export function scoreOf(rt: RT, src = 'fwd') {
	const m = classMap(rt, src);
	if (!m) return null;
	const y = truthOf(rt.input), e = borders(y);
	const per: ClassScore[] = [];
	for (let k = 0; k < K; k++) {
		let inter = 0, union = 0, truth = 0, pred = 0;
		for (let i = 0; i < S * S; i++) {
			const a = m.k[i] === k, b = y[i] === k;
			inter += +(a && b), union += +(a || b), truth += +b, pred += +a;
		}
		if (union) per.push({ k, inter, union, truth, pred, iou: inter / union });
	}
	let right = 0, edge = 0, edgeRight = 0;
	for (let i = 0; i < S * S; i++) (right += +(m.k[i] === y[i])), (edge += e[i]), (edgeRight += +(e[i] && m.k[i] === y[i]));
	const shapes = per.filter((c) => c.k > 0);
	const worst = shapes.length ? shapes.reduce((a, b) => (b.iou < a.iou ? b : a)) : null;
	return { per, miou: per.reduce((s, c) => s + c.iou, 0) / per.length, acc: right / (S * S), wrong: S * S - right, edge, edgeAcc: edge ? edgeRight / edge : NaN, worst, map: m, truth: y };
}

/** How many pixels have a top share below `sure` (default 90%): the soft, unsure band along the edges. */
export const SURE = 0.9;
export function unsure(rt: RT, src = 'fwd', sure = SURE): number {
	const m = classMap(rt, src);
	return m ? m.p.filter((p) => p < sure).length : NaN;
}

/** The pixel the lesson probes: the centre of the true shape pixels of the largest class present, else the middle. */
export function probe(rt: RT): [number, number] {
	const y = truthOf(rt.input);
	const n = [0, 0, 0, 0], sx = [0, 0, 0, 0], sy = [0, 0, 0, 0];
	for (let i = 0; i < S * S; i++) (n[y[i]] += 1), (sx[y[i]] += i % S), (sy[y[i]] += Math.floor(i / S));
	let k = 1;
	for (let c = 2; c < K; c++) if (n[c] > n[k]) k = c;
	if (!n[k]) return [S / 2, S / 2];
	let best = -1, bd = Infinity; // the pixel of class k nearest to the class's mean position
	for (let i = 0; i < S * S; i++) {
		const d = Math.hypot((i % S) - sx[k] / n[k], Math.floor(i / S) - sy[k] / n[k]);
		if (y[i] === k && d < bd) (bd = d), (best = i);
	}
	return [Math.floor(best / S), best % S];
}

/** The least sure pixel of the U-Net's answer (smallest top share; ties: the first). */
export function leastSure(rt: RT): [number, number] {
	const m = classMap(rt);
	if (!m) return [0, 0];
	let b = 0;
	for (let i = 1; i < S * S; i++) if (m.p[i] < m.p[b]) b = i;
	return [Math.floor(b / S), b % S];
}

/** The window of input pixels that output pixel (y, x) is computed from along one path: at each concat take input
 *  `picks[j]` (0 = the copy across, 1 = the path from below). Paths: [0] skip 1, [1, 0] skip 2, [1, 1] the bottleneck. */
export function windowOf(model: GraphModel | null, y: number, x: number, picks: number[]): Box | null {
	if (!model) return null;
	const spec: AtlasSpec = model.spec;
	const byId = new Map(spec.layers.map((l) => [l.id, l]));
	let L = byId.get('logits'), b: Box = { y0: y, x0: x, y1: y, x1: x }, j = 0;
	while (L && L.op !== 'input') {
		const next = L.op === 'concat' ? L.in[picks[j++] ?? 0] : L.in[0];
		if (L.op !== 'concat') {
			const r = model.ops.get(L.op);
			b = r.def.receptive ? r.def.receptive(b, mergeParams(r.def, r.params, L.params) as never) : b;
		}
		const sh = byId.get(next)?.outShape ?? [1, S, S];
		const H = Number(sh[1]), W = Number(sh[2]);
		b = { y0: Math.max(0, b.y0), x0: Math.max(0, b.x0), y1: Math.min(H - 1, b.y1), x1: Math.min(W - 1, b.x1) };
		L = byId.get(next);
	}
	return b;
}
export const PATHS = [[0], [1, 0], [1, 1]];
/** Width of the window along each path for a pixel far from the border (structure, not this input). */
export const pathWidths = (model: GraphModel | null) => PATHS.map((p) => windowOf(model, S / 2, S / 2, p)).map((b) => (b ? b.x1 - b.x0 + 1 : NaN));

/** Made-up traces for the view demos (gallery, Lesson Lab): a circle and a square; the shares say 0.9 for the true class
 *  (the made-up 'noskip' answer misses the border pixels); every map is the picture averaged down, sign alternating. */
export function demoRt(): ExplainerRuntime {
	const raw = { shapes: [{ k: 1, x: 10, y: 11, r: 6 }, { k: 2, x: 22, y: 21, r: 5 }], seed: 3 };
	const { image, label } = render(raw), n = S * S, e = borders(label);
	const probs = (miss: boolean) => Float32Array.from({ length: 4 * n }, (_, j) => ((miss && e[j % n] ? 0 : label[j % n]) === Math.floor(j / n) ? 0.9 : 0.1 / 3));
	const maps = (C: number, H: number) => Float32Array.from({ length: C * H * H }, (_, j) => {
		const f = S / H, y = Math.floor((j % (H * H)) / H) * f, x = (j % H) * f;
		return (Math.floor(j / (H * H)) % 2 ? -1 : 1) * image[Math.floor(y) * S + Math.floor(x)];
	});
	const t = (shape: number[], data: Float32Array): Tensor => ({ shape, data });
	const lv: [string, number, number][] = [['enc1', 8, 32], ['enc2', 16, 16], ['bott', 32, 8], ['up2', 32, 16], ['dec2', 16, 16], ['up1', 16, 32], ['dec1', 8, 32]];
	const tr = (miss: boolean) => ({ order: ['input', ...lv.map((l) => l[0]), 'probs'], t: { input: t([1, S, S], image), ...Object.fromEntries(lv.map(([id, C, H]) => [id, t([C, H, H], maps(C, H))])), probs: t([4, S, S], probs(miss)) } });
	return fakeRuntime({ slug: 'tiny-unet-seg', trace: tr(false), input: raw, sources: { noskip: { kind: 'trace', status: 'ready', provenance: 'illustrative', trace: tr(true) } } });
}
