// vq-vae local op 'pixel_shuffle': [C·r·r, H, W] -> [C, H·r, W·r] as torch.nn.functional.pixel_shuffle. Channel
// c·r·r + a·r + b of cell (i, j) goes to row i·r + a, column j·r + b: the decoder's r·r scores of each cell become
// that cell's r × r block of the picture. Values are copied, never computed, so parity is exact.
// Python mirror: pipeline/vq-vae/ops_pixel_shuffle.py.
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, t, unitBase } from '../../../atlas/ops/common.ts';

type ShuffleP = { factor: number };

export const pixelShuffle: OpDef<ShuffleP> = {
	op: 'pixel_shuffle',
	version: 1,
	attrs: { factor: { type: 'int', required: true, doc: 'block side r' } },
	category: 'shape',
	label: t('Lay out as blocks', '排成小块'),
	inferShape(ins, p) {
		const [C, H, W] = ins[0], r = p.factor;
		if (ins[0].length !== 3 || typeof C !== 'number' || C % (r * r)) throw new Error(`pixel_shuffle: channels must be a multiple of ${r * r}`);
		return [C / (r * r), typeof H === 'number' ? H * r : H, typeof W === 'number' ? W * r : W];
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs, p) {
		const x = xs[0], r = p.factor, [C, H, W] = x.shape;
		const out = outArray(x.data.length, x);
		for (let ch = 0; ch < C; ch++) {
			const c = Math.floor(ch / (r * r)), a = Math.floor((ch % (r * r)) / r), b = ch % r;
			for (let i = 0; i < H; i++)
				for (let j = 0; j < W; j++) out[(c * H * r + i * r + a) * W * r + j * r + b] = x.data[(ch * H + i) * W + j];
		}
		return { shape: [C / (r * r), H * r, W * r], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit), r = q.p.factor, [, H, W] = q.ins[0].shape;
		const [c, row, col] = q.unit, i = Math.floor(row / r), j = Math.floor(col / r);
		const ch = c * r * r + (row % r) * r + (col % r), src = (ch * H + i) * W + j;
		const e = unitBase('index-map', q.layer, q.unit, q.ins[0].data[src], q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { ch, i, j, row, col, v: q.ins[0].data[src] };
		return e;
	},
	formula: `${sym('out', 'y')}[{{row}}, {{col}}] = ${sym('in', 's')}[{{ch}}, {{i}}, {{j}}] = {{v}}`,
	narration: {
		what: t(
			'Each cell of the grid carries {block|int} scores; they are laid out as its {r|int}×{r|int} block of the picture, row by row. No number changes.',
			'网格的每个格子带着 {block|int} 个分数；它们被逐行排成这个格子在图中的 {r|int}×{r|int} 小块。没有任何数改变。'
		),
		how: t('Score a·{r|int} + b of a cell goes to row a, column b of its block.', '一个格子的第 a·{r|int} + b 个分数放到它那一块的第 a 行第 b 列。'),
		why: t('The decoder works cell by cell; this turns its scores back into a picture the size of your drawing.', '解码器一个格子一个格子地算；这一步把它的分数变回和你的画一样大的图。'),
		look: t('Each block of the picture comes from one cell of the grid.', '图中的每一小块都来自网格里的一个格子。'),
		unit: t('Row {row|int}, column {col|int} (counting from 0) is score {ch|int} of cell ({i|int}, {j|int}): {v}.', '第 {row|int} 行第 {col|int} 列（从 0 数起）是格子 ({i|int}, {j|int}) 的第 {ch|int} 个分数：{v}。')
	},
	vars(layer, spec) {
		const r = Number((layer.params as ShuffleP | undefined)?.factor ?? NaN);
		return { ...baseVars(layer, spec), r, block: r * r };
	},
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: (layer) => `F.pixel_shuffle(x, ${(layer.params as ShuffleP | undefined)?.factor ?? 'r'})`
};

export default pixelShuffle;
