// tinyclip: the synthetic world (design §7). Shapes × size × place, drawn as strokes in a unit square (y down) and rendered
// by the kit's drawing28 path (TinyCNN's preprocessStrokes), with template captions over a 23-word vocabulary.
// preprocessStrokes crops to the ink and centres it, which would erase size and place; every picture therefore has a
// frame around the whole square (FRAME), so the crop is always the square and a shape keeps its size and place.
// The training set (process `synth`), the presets, the page's partner pairs and the learner's drawing all go through
// render() here, so Python trains on exactly the bytes the page computes (pipeline/tinyclip/train.py checks the hash).
import type { Rng } from '../../atlas/types/index.ts';
import { drawingInput } from '../../atlas/adapters/drawing28/adapter.ts';
import { createRng, fnv1a32 } from '../../atlas/core/rng.ts';

/** id 0 pads a short caption; it never counts in the average */
export const VOCAB = ['<pad>', 'a', 'the', 'small', 'big', 'large', 'circle', 'square', 'triangle', 'cross', 'star', 'on', 'at', 'in', 'left', 'right', 'top', 'bottom', 'middle', 'center', 'above', 'below', 'of', 'and'];
export const MAX_WORDS = 8;
export const SHAPES = ['circle', 'square', 'triangle', 'cross', 'star'] as const;
export const PLACES = ['left', 'right', 'top', 'bottom', 'middle'] as const;
const AT: Record<string, [number, number]> = { left: [0.29, 0.5], right: [0.71, 0.5], top: [0.5, 0.29], bottom: [0.5, 0.71], middle: [0.5, 0.5] };
const PHRASE: Record<string, string[]> = { left: ['on the left'], right: ['on the right'], top: ['at the top'], bottom: ['at the bottom'], middle: ['in the middle', 'in the center'] };

export type Pt = { x: number; y: number };
/** one: a shape (index into SHAPES), size 0 small / 1 big, place (index into PLACES); two: shape a above b (rel 0) or left of b (rel 1) */
export type Scene = { kind: 'one'; shape: number; size: number; place: number } | { kind: 'two'; a: number; b: number; rel: number };
export const FRAME: Pt[] = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }, { x: 0, y: 0 }];

/** The shape's strokes around (0, 0) with radius 1. */
function outline(shape: number): [number, number][][] {
	const ring = (n: number, r: (k: number) => number) => Array.from({ length: n + 1 }, (_, k) => {
		const a = -Math.PI / 2 + (2 * Math.PI * k) / n;
		return [r(k) * Math.cos(a), r(k) * Math.sin(a)] as [number, number];
	});
	switch (SHAPES[shape]) {
		case 'circle': return [ring(20, () => 1)];
		case 'square': return [[[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8], [-0.8, -0.8]]];
		case 'triangle': return [[[0, -1], [0.9, 0.75], [-0.9, 0.75], [0, -1]]];
		case 'cross': return [[[-1, 0], [1, 0]], [[0, -1], [0, 1]]];
		default: return [ring(10, (k) => (k % 2 ? 0.45 : 1))];
	}
}

/** A hand-drawn copy: jittered centre, size and angle, and every point nudged a little. */
function draw(shape: number, cx: number, cy: number, r: number, rng: Rng): Pt[][] {
	const j = (s: number) => (rng.next() * 2 - 1) * s;
	const x0 = cx + j(0.04), y0 = cy + j(0.04), rr = r * (1 + j(0.15)), rot = SHAPES[shape] === 'circle' ? 0 : j(0.15);
	const c = Math.cos(rot), s = Math.sin(rot);
	return outline(shape).map((line) => line.map(([u, v]) => ({ x: x0 + rr * (c * u - s * v) + j(0.008), y: y0 + rr * (s * u + c * v) + j(0.008) })));
}

export function strokesOf(sc: Scene, rng: Rng): Pt[][] {
	if (sc.kind === 'one') {
		const [cx, cy] = AT[PLACES[sc.place]];
		return draw(sc.shape, cx, cy, sc.size ? 0.2 : 0.11, rng);
	}
	const [p, q]: [number, number][] = sc.rel === 0 ? [[0.5, 0.3], [0.5, 0.7]] : [[0.3, 0.5], [0.7, 0.5]];
	return [...draw(sc.a, p[0], p[1], 0.13, rng), ...draw(sc.b, q[0], q[1], 0.13, rng)];
}

const pick = <T>(rng: Rng, xs: readonly T[]): T => xs[rng.int(xs.length)];

/** A caption the way the training set words it: full, partial or the shape alone; two shapes in either order. */
export function captionOf(sc: Scene, rng: Rng): string {
	if (sc.kind === 'two') {
		const [A, B] = [SHAPES[sc.a], SHAPES[sc.b]];
		const u = rng.next();
		if (u < 0.2) return `a ${A} and a ${B}`;
		if (sc.rel === 0) return u < 0.6 ? `a ${A} above a ${B}` : `a ${B} below a ${A}`;
		return u < 0.6 ? `a ${A} left of a ${B}` : `a ${B} right of a ${A}`;
	}
	const S = SHAPES[sc.shape];
	const Z = sc.size ? pick(rng, ['big', 'large']) : 'small';
	const P = pick(rng, PHRASE[PLACES[sc.place]]);
	const u = rng.next();
	return u < 0.5 ? `a ${Z} ${S} ${P}` : u < 0.7 ? `a ${S} ${P}` : u < 0.85 ? `a ${Z} ${S}` : `a ${S}`;
}

export function randomScene(rng: Rng, twoShare = 0.25): Scene {
	if (rng.next() < twoShare) {
		const a = rng.int(SHAPES.length);
		const b = (a + 1 + rng.int(SHAPES.length - 1)) % SHAPES.length;
		return { kind: 'two', a, b, rel: rng.int(2) };
	}
	return { kind: 'one', shape: rng.int(SHAPES.length), size: rng.int(2), place: rng.int(PLACES.length) };
}

/** The 784 numbers the image tower reads: the frame plus the strokes, through drawing28 (preprocessStrokes). */
export function render(strokes: readonly (readonly Pt[])[]): Float32Array {
	return drawingInput({ strokes: [FRAME, ...strokes.map((s) => s.map((p) => ({ x: p.x, y: p.y })))] }).input;
}

/** Words of a caption: known ones get their id, the others are skipped (and shown as skipped). At most MAX_WORDS ids. */
export function tokenize(text: string): { ids: number[]; words: { w: string; id: number }[]; cut: boolean } {
	const words = (text.toLowerCase().match(/[a-z]+/g) ?? []).map((w) => ({ w, id: VOCAB.indexOf(w) > 0 ? VOCAB.indexOf(w) : -1 }));
	const ids = words.filter((x) => x.id > 0).map((x) => x.id);
	return { ids: ids.slice(0, MAX_WORDS), words, cut: ids.length > MAX_WORDS };
}

/** CRC-32 (IEEE, as zlib.crc32) over the bytes of a float32 array: the training-set check Python repeats on the bytes it read. */
const CRC = Array.from({ length: 256 }, (_, n) => {
	let c = n;
	for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
	return c >>> 0;
});
export function crcBytes(a: Float32Array): number {
	const b = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
	let c = 0xffffffff;
	for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}

// ------------------------------------------------------------------ fixed drawings (presets, the page's partner pairs, edge cases)
export type Fixed = { id: string; label: { en: string; zh: string }; scene: Scene | null; strokes?: Pt[][]; captions: string[] };
const one = (shape: number, size: number, place: number): Scene => ({ kind: 'one', shape, size, place });
/** The learner's presets: a drawing and three captions to start from (the first true, the last false). */
export const PRESETS: Fixed[] = [
	{ id: 'circle', label: { en: 'big circle, middle', zh: '大圆，中间' }, scene: one(0, 1, 4), captions: ['a big circle in the middle', 'a small circle', 'a square on the left'] },
	{ id: 'square', label: { en: 'small square, left', zh: '小正方形，左边' }, scene: one(1, 0, 0), captions: ['a small square on the left', 'a square', 'a big star at the top'] },
	{ id: 'triangle', label: { en: 'big triangle, bottom', zh: '大三角形，下面' }, scene: one(2, 1, 3), captions: ['a big triangle at the bottom', 'a triangle at the top', 'a cross'] },
	{ id: 'star', label: { en: 'small star, right', zh: '小星星，右边' }, scene: one(4, 0, 1), captions: ['a small star on the right', 'a big star on the right', 'a circle on the right'] },
	{ id: 'cross', label: { en: 'big cross, top', zh: '大十字，上面' }, scene: one(3, 1, 2), captions: ['a big cross at the top', 'a cross at the bottom', 'a large cross'] },
	{ id: 'stack', label: { en: 'circle above square', zh: '圆在正方形上面' }, scene: { kind: 'two', a: 0, b: 1, rel: 0 }, captions: ['a circle above a square', 'a square above a circle', 'a square below a circle'] }
];
/** The five other pairs of the page's batch (rows and columns 1 to 5 of the matrix; the learner's pair is 0). */
export const PARTNERS: Fixed[] = [
	{ id: 'p-triangle', label: { en: 'small triangle, top', zh: '小三角形，上面' }, scene: one(2, 0, 2), captions: ['a small triangle at the top'] },
	{ id: 'p-square', label: { en: 'big square, right', zh: '大正方形，右边' }, scene: one(1, 1, 1), captions: ['a big square on the right'] },
	{ id: 'p-star', label: { en: 'big star, bottom', zh: '大星星，下面' }, scene: one(4, 1, 3), captions: ['a star at the bottom'] },
	{ id: 'p-cross', label: { en: 'small cross, left', zh: '小十字，左边' }, scene: one(3, 0, 0), captions: ['a small cross on the left'] },
	{ id: 'p-pair', label: { en: 'star left of cross', zh: '星星在十字左边' }, scene: { kind: 'two', a: 4, b: 3, rel: 1 }, captions: ['a star left of a cross'] }
];
/** Edge cases (also reference cases for parity): only the frame, one dot, a tiny stroke with a caption longer than 8 words. */
export const EDGES: Fixed[] = [
	{ id: 'blank', label: { en: 'nothing drawn', zh: '什么都没画' }, scene: null, strokes: [], captions: ['circle'] },
	{ id: 'dot', label: { en: 'a single dot', zh: '一个点' }, scene: null, strokes: [[{ x: 0.5, y: 0.5 }]], captions: ['a small circle in the middle', 'a big star'] },
	{ id: 'small', label: { en: 'a tiny stroke, a long caption', zh: '很小的一笔，很长的描述' }, scene: null, strokes: [[{ x: 0.47, y: 0.47 }, { x: 0.5, y: 0.52 }, { x: 0.53, y: 0.5 }]], captions: ['a big circle on the left and a small square'] }
];
/** The strokes of a fixed drawing (seeded by its id, so it never changes). */
export function strokesFor(f: Fixed): Pt[][] {
	return f.strokes ?? strokesOf(f.scene!, createRng(fnv1a32(f.id)));
}
export const FIXED: Fixed[] = [...PRESETS, ...PARTNERS, ...EDGES];
