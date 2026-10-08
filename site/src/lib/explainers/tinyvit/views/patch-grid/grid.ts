// What the patch-grid view draws, from StageProps (pure TS: meta.ts, View.svelte, lesson.ts and claims.test.ts use it).
//   mode 'patches'    the 28×28 drawing cut into 4×4 patches of 7×7 pixels; the current patch comes from the stage's
//                     unit (a scan over the 'patch' convolution: [channel, row, col]) or props.patch
//   mode 'attention'  one attention row (props.query, default 0: the [CLS] row) of block props.block (1-based, as the
//                     labels say "Block 1") and head props.head (from 0), drawn over the patches it weighs
import type { StageProps, Tensor, Trace } from '../../../../atlas/types/index.ts';

export const GRID = 4;
export const PATCH = 7;
export const SIDE = GRID * PATCH;
/** a patch "has ink" when its brightest pixel is >= INK (pipeline/tinyvit/stats.py uses the same rule) */
export const INK = 0.5;

export interface PatchGrid {
	mode: 'patches' | 'attention';
	img: ArrayLike<number>;
	inked: boolean[];
	/** patches mode: the patch shown (row-major, 0..15), its 49 pixels */
	patch: number;
	pixels: number[];
	/** attention mode */
	block: number;
	blocks: number;
	head: number;
	heads: number;
	query: number;
	/** weight on each of the 16 patches, and on row 0 ([CLS]) */
	w: number[];
	self: number;
	inkShare: number;
	top: number;
}

export type GridProps = Pick<StageProps, 'rt' | 'source' | 'at' | 'props'>;

export function traceOf(p: GridProps): Trace | null {
	return p.rt.sources[p.source]?.trace ?? p.rt.sources[typeof p.props.trace === 'string' ? p.props.trace : 'fwd']?.trace ?? null;
}

export function inkedPatches(img: ArrayLike<number>): boolean[] {
	return Array.from({ length: GRID * GRID }, (_, k) => patchPixels(img, k).some((v) => v >= INK));
}

export function patchPixels(img: ArrayLike<number>, k: number): number[] {
	const r0 = Math.floor(k / GRID) * PATCH;
	const c0 = (k % GRID) * PATCH;
	const out: number[] = [];
	for (let r = 0; r < PATCH; r++) for (let c = 0; c < PATCH; c++) out.push(img[(r0 + r) * SIDE + c0 + c]);
	return out;
}

/** The [CLS] row (or row `query`) of block `block` (1-based), head `head`: weights on row 0 and the 16 patches. */
export function attnRow(tr: Trace, block: number, head: number, query = 0): { w: number[]; self: number; heads: number } | null {
	const a: Tensor | undefined = tr.t[`B${block}.attn`];
	if (!a) return null;
	const [H, Tq, Tk] = a.shape;
	const h = Math.max(0, Math.min(H - 1, head));
	const row = Array.from(a.data.subarray((h * Tq + query) * Tk, (h * Tq + query + 1) * Tk));
	return { w: row.slice(1), self: row[0], heads: H };
}

export function gridOf(p: GridProps, pick: { head?: number | null; block?: number | null } = {}): PatchGrid | null {
	const tr = traceOf(p);
	const img = tr?.t.input?.data;
	if (!tr || !img || img.length !== SIDE * SIDE) return null;
	const inked = inkedPatches(img);
	const unit = p.at.unit?.layer === 'patch' ? p.at.unit.unit : undefined;
	const patch = unit ? unit[1] * GRID + unit[2] : typeof p.props.patch === 'number' ? p.props.patch : 0;
	const blocks = Object.keys(tr.t).filter((k) => /^B\d+\.attn$/.test(k)).length;
	const block = Math.max(1, Math.min(blocks, pick.block ?? (typeof p.props.block === 'number' ? p.props.block : 1)));
	const head = pick.head ?? (typeof p.props.head === 'number' ? p.props.head : 0);
	const query = typeof p.props.query === 'number' ? p.props.query : 0;
	const r = attnRow(tr, block, head, query);
	const w = r?.w ?? [];
	const inkShare = w.reduce((s, v, k) => s + (inked[k] ? v : 0), 0);
	const top = w.reduce((b, v, k) => (v > w[b] ? k : b), 0);
	return {
		mode: p.props.mode === 'patches' ? 'patches' : 'attention',
		img,
		inked,
		patch,
		pixels: patchPixels(img, patch),
		block,
		blocks,
		head: Math.max(0, Math.min((r?.heads ?? 1) - 1, head)),
		heads: r?.heads ?? 0,
		query,
		w,
		self: r?.self ?? NaN,
		inkShare,
		top
	};
}
