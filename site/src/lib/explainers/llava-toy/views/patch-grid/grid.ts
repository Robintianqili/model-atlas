// What the patch-grid view draws, from StageProps (pure TS: meta.ts, View.svelte, lesson.ts and claims.test.ts use it).
// Adapted from the tinyvit package's view of the same name: one attention row of the language model (the answer row,
// props.query, default the last row) in block props.block (1-based, as the labels say "LM block 1"), for head props.head
// (from 0) or the mean of the heads (the default, -1). Its weights on the 16 image tokens are drawn over the patches of
// the drawing they come from; its weights on the question's words are listed beside them.
import type { StageProps, Tensor, Trace } from '../../../../atlas/types/index.ts';

export const GRID = 4;
export const PATCH = 7;
export const SIDE = GRID * PATCH;
export const N_IMG = GRID * GRID;
/** a patch "has ink" when its brightest pixel is >= INK (pipeline/llava-toy/stats.py uses the same rule) */
export const INK = 0.5;

export interface PatchGrid {
	img: ArrayLike<number>;
	inked: boolean[];
	block: number;
	blocks: number;
	/** -1: the mean of the heads */
	head: number;
	heads: number;
	query: number;
	/** weight on each of the 16 image tokens (patches, row-major) */
	w: number[];
	/** the rows after the image tokens, up to the query: name and weight */
	words: { name: string; w: number }[];
	imgShare: number;
	inkShare: number;
	top: number;
}

export type GridProps = Pick<StageProps, 'rt' | 'source' | 'at' | 'props'>;

export function traceOf(p: GridProps): Trace | null {
	return p.rt.sources[p.source]?.trace ?? p.rt.sources.fwd?.trace ?? null;
}

export function inkedPatches(img: ArrayLike<number>): boolean[] {
	return Array.from({ length: N_IMG }, (_, k) => {
		const r0 = Math.floor(k / GRID) * PATCH;
		const c0 = (k % GRID) * PATCH;
		for (let r = 0; r < PATCH; r++) for (let c = 0; c < PATCH; c++) if (img[(r0 + r) * SIDE + c0 + c] >= INK) return true;
		return false;
	});
}

/** Row `query` (default: the last) of `lm.L{block}.attn`, head `head` or the mean of the heads (head < 0). */
export function attnRow(tr: Trace, block: number, head: number, query = -1): { w: number[]; heads: number; query: number } | null {
	const a: Tensor | undefined = tr.t[`lm.L${block}.attn`];
	if (!a) return null;
	const [H, Tq, Tk] = a.shape;
	const q = query < 0 ? Tq + query : Math.min(query, Tq - 1);
	const hs = head < 0 ? Array.from({ length: H }, (_, h) => h) : [Math.min(H - 1, head)];
	const w = Array.from({ length: Tk }, (_, k) => hs.reduce((s, h) => s + a.data[(h * Tq + q) * Tk + k], 0) / hs.length);
	return { w, heads: H, query: q };
}

export function gridOf(p: GridProps, pick: { head?: number | null; block?: number | null } = {}): PatchGrid | null {
	const tr = traceOf(p);
	const img = tr?.t.image?.data;
	if (!tr || !img || img.length !== SIDE * SIDE) return null;
	const inked = inkedPatches(img);
	const blocks = Object.keys(tr.t).filter((k) => /^lm\.L\d+\.attn$/.test(k)).length;
	const block = Math.max(1, Math.min(blocks, pick.block ?? (typeof p.props.block === 'number' ? p.props.block : 1)));
	const head = pick.head ?? (typeof p.props.head === 'number' ? p.props.head : -1);
	const r = attnRow(tr, block, head, typeof p.props.query === 'number' ? p.props.query : -1);
	if (!r) return null;
	const names = (p.rt.pre as { tokens?: string[] } | null)?.tokens ?? [];
	const w = r.w.slice(0, N_IMG);
	const words = r.w.slice(N_IMG, r.query + 1).map((x, j) => ({ name: names[N_IMG + j] ?? `${N_IMG + j}`, w: x }));
	return {
		img,
		inked,
		block,
		blocks,
		head: Math.min(head, r.heads - 1),
		heads: r.heads,
		query: r.query,
		w,
		words,
		imgShare: w.reduce((s, v) => s + v, 0),
		inkShare: w.reduce((s, v, k) => s + (inked[k] ? v : 0), 0),
		top: w.reduce((b, v, k) => (v > w[b] ? k : b), 0)
	};
}
