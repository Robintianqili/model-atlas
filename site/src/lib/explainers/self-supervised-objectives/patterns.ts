// self-supervised-objectives: the illustrative objectives, drawn as mask patterns on the learner's sentence and on one
// drawing's 4 x 4 patch grid. No model on this site was trained with these objectives: each function only decides, with
// the kit's seeded rng (never Math.random), which tokens or patches are hidden and what the target would be, using the
// settings reported in each method's paper (constants below; claims.test.ts pins them).
import type { Rng } from '../../atlas/types/index.ts';
import { createRng, sampleIndices } from '../../atlas/core/rng.ts';
import { DRAWING } from './source_data.ts';

/** T5 (Raffel et al. 2020): 15% of tokens corrupted, in spans of mean length 3. */
export const T5 = { rate: 0.15, meanSpan: 3 };
/** BART (Lewis et al. 2020), text infilling: 30% of tokens masked, span lengths drawn from Poisson(λ = 3). */
export const BART = { rate: 0.3, lambda: 3 };
/** MAE (He et al. 2022): 75% of the image patches hidden. */
export const MAE = { rate: 0.75 };
/** TinyViT's patches: a 28 x 28 drawing cut into 4 x 4 patches of 7 x 7 pixels. */
export const GRID = { side: 28, patch: 7, n: 4 };
export const SENTINELS = ['X', 'Y', 'Z', 'W', 'V', 'U', 'T', 'S'];
const SEED = 7;

/** Split `items` into `segs` non-empty runs at random (T5's _random_segmentation): their lengths, in order. */
function segment(rng: Rng, items: number, segs: number): number[] {
	const cuts = sampleIndices(rng, items - 1, segs - 1).map((i) => i + 1).sort((a, b) => a - b);
	const edges = [0, ...cuts, items];
	return edges.slice(1).map((e, i) => e - edges[i]);
}

/** T5 span corruption of n tokens: span[i] = the span number of token i (counting from 0), or -1 for a kept token.
 *  T5's random_spans_noise_mask with random_roll: kept and hidden runs alternate, then the whole mask is rotated by a
 *  random offset, so the last token is not always hidden. */
export function t5Spans(n: number, seed = SEED) {
	const span = new Array<number>(n).fill(-1);
	if (n < 2) return { span, spans: 0, noise: 0, inLen: n, outLen: 0 };
	const rng = createRng(seed).fork('t5');
	const noise = Math.min(Math.max(Math.round(n * T5.rate), 1), n - 1);
	const parts = Math.max(Math.round(noise / T5.meanSpan), 1);
	const noiseLens = segment(rng, noise, parts);
	const keepLens = segment(rng, n - noise, parts);
	const hid = new Array<boolean>(n).fill(false);
	let i = 0;
	for (let s = 0; s < parts; s++) {
		i += keepLens[s];
		for (let k = 0; k < noiseLens[s]; k++) hid[i++] = true;
	}
	const roll = rng.int(n);
	const rolled = hid.map((_, j) => hid[(j - roll + n) % n]);
	let spans = 0;
	for (let j = 0; j < n; j++) if (rolled[j]) span[j] = j > 0 && rolled[j - 1] ? spans - 1 : spans++;
	// the input keeps the kept tokens and one sentinel per span; the target lists each sentinel and its span, then a last sentinel
	return { span, spans, noise, inLen: n - noise + spans, outLen: noise + spans + 1 };
}

/** One Poisson(λ) draw (Knuth). */
function poisson(rng: Rng, lambda: number): number {
	const L = Math.exp(-lambda);
	let k = 0;
	let p = rng.next();
	while (p > L) {
		k++;
		p *= rng.next();
	}
	return k;
}

/** BART text infilling of n tokens: masked[i]; every run of masked tokens becomes one [MASK]; the target is all n tokens.
 *  Spans start at random tokens; a zero-length draw (an inserted [MASK] in BART) is left out of this drawing. */
export function bartSpans(n: number, seed = SEED) {
	const masked = new Array<boolean>(n).fill(false);
	if (n < 2) return { masked, runs: 0, hidden: 0, inLen: n, outLen: n };
	const rng = createRng(seed).fork('bart');
	const goal = Math.max(Math.round(n * BART.rate), 1);
	let hidden = 0;
	for (let tries = 0; hidden < goal && tries < 100; tries++) {
		const len = Math.min(poisson(rng, BART.lambda), goal - hidden);
		const start = rng.int(n);
		for (let i = start; i < Math.min(n, start + len); i++) if (!masked[i]) ((masked[i] = true), hidden++);
	}
	let runs = 0;
	for (let i = 0; i < n; i++) if (masked[i] && (i === 0 || !masked[i - 1])) runs++;
	return { masked, runs, hidden, inLen: n - hidden + runs, outLen: n };
}

/** MAE on the 16 patches: which are hidden (true). */
export function maePatches(seed = SEED) {
	const N = GRID.n * GRID.n;
	const k = Math.round(N * MAE.rate);
	const hidden = new Array<boolean>(N).fill(false);
	for (const i of sampleIndices(createRng(seed).fork('mae'), N, k)) hidden[i] = true;
	return { hidden, k, N };
}

/** Two random crops of the drawing (contrastive views): top-left corner and side in pixels; the second is mirrored. */
export function cropViews(seed = SEED) {
	const rng = createRng(seed).fork('crop');
	const size = 2 * GRID.patch * 1.5; // 21 pixels: three quarters of the side
	const at = () => ({ x: rng.int(GRID.side - size + 1), y: rng.int(GRID.side - size + 1) });
	return [{ ...at(), size, flip: false }, { ...at(), size, flip: true }];
}

/** I-JEPA (Assran et al. 2023) hides blocks of neighbouring patches, not scattered ones: here one block of 2 x 2 patches
 *  at a random place on TinyViT's 4 x 4 grid (a simplification of the paper's four target blocks). */
export const JEPA = { block: 2 };
export function jepaBlock(seed = SEED) {
	const rng = createRng(seed).fork('jepa');
	const r0 = rng.int(GRID.n - JEPA.block + 1);
	const c0 = rng.int(GRID.n - JEPA.block + 1);
	const inside = (v: number, lo: number) => v >= lo && v < lo + JEPA.block;
	const hidden = Array.from({ length: GRID.n * GRID.n }, (_, i) => inside(Math.floor(i / GRID.n), r0) && inside(i % GRID.n, c0));
	return { hidden, k: JEPA.block * JEPA.block, row: r0, col: c0 };
}

/** The drawing's 784 pixel values in [0, 1] (row by row). */
export function pixels(): Float32Array {
	const s = atob(DRAWING.b64);
	return Float32Array.from({ length: s.length }, (_, i) => s.charCodeAt(i) / 255);
}

/** The same drawing with Gaussian noise of standard deviation sigma on every pixel (a denoising objective's input). */
export function noisy(sigma: number, seed = SEED): Float32Array {
	const rng = createRng(seed).fork('noise');
	return pixels().map((v) => v + sigma * rng.normal());
}
