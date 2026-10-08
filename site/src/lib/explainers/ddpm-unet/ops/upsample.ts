// ddpm-unet local op 'upsample': nearest-neighbour upsampling of a [C, H, W] map by an integer scale (default 2): every
// number is copied into a scale × scale block, so [C, H, W] -> [C, 2H, 2W]. Values are copied, never computed, so
// parity is exact. Python mirror: pipeline/ddpm-unet/ops_upsample.py (F.interpolate(..., mode='nearest')).
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, t, unitBase } from '../../../atlas/ops/common.ts';

type P = { scale: number };

export const upsample: OpDef<P> = {
	op: 'upsample',
	version: 1,
	attrs: { scale: { type: 'int', default: 2, doc: 'each number becomes a scale × scale block' } },
	category: 'shape',
	label: t('Upsample (copy each number into a 2×2 block)', '上采样（每个数复制成 2×2 的块）'),
	inferShape(ins, p) {
		const [c, h, w] = ins[0];
		const s = p.scale ?? 2;
		if (ins[0].length !== 3 || typeof h !== 'number' || typeof w !== 'number') throw new Error('upsample: input must be [C, H, W]');
		return [c, h * s, w * s];
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs, p) {
		const x = xs[0], s = p.scale ?? 2;
		const [C, H, W] = x.shape;
		const y = outArray(C * H * W * s * s, x);
		for (let c = 0; c < C; c++)
			for (let r = 0; r < H * s; r++)
				for (let k = 0; k < W * s; k++) y[(c * H * s + r) * W * s + k] = x.data[(c * H + Math.floor(r / s)) * W + Math.floor(k / s)];
		return { shape: [C, H * s, W * s], data: y };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const s = q.p.scale ?? 2;
		const [c, r, k] = q.unit;
		const [, H, W] = q.ins[0].shape;
		const src = (c * H + Math.floor(r / s)) * W + Math.floor(k / s);
		const y = q.ins[0].data[src];
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { c, row: r, col: k, srow: Math.floor(r / s), scol: Math.floor(k / s), y };
		return e;
	},
	formula: `${sym('out', 'y')}_{ {{c}},{{row}},{{col}} } = ${sym('in', 'x')}_{ {{c}},\\lfloor {{row}}/2 \\rfloor,\\lfloor {{col}}/2 \\rfloor} = ${sym('in', 'x')}_{ {{c}},{{srow}},{{scol}} } = {{y}}`,
	narration: {
		what: t('Upsampling doubles the size of every map, {inShape} → {outShape}: each number is copied into a 2×2 block. Nothing is learned here.', '上采样把每张图的尺寸翻倍，{inShape} → {outShape}：每个数被复制成一个 2×2 的块。这里没有要学的东西。'),
		how: t('Output row r, column k reads input row ⌊r/2⌋, column ⌊k/2⌋.', '输出的第 r 行第 k 列读取输入的第 ⌊r/2⌋ 行第 ⌊k/2⌋ 列。'),
		why: t('The way up the U must return to the drawing’s size, one halving undone at a time.', 'U 形的上行路径要回到画的尺寸，每次撤销一次减半。'),
		look: t('Blocky maps: every 2×2 block holds one number.', '块状的图：每个 2×2 的块里是同一个数。'),
		unit: t('Row {row|int}, column {col|int} (counting from 0) copies row {srow|int}, column {scol|int}: {y}.', '第 {row|int} 行第 {col|int} 列（从 0 数起）复制第 {srow|int} 行第 {scol|int} 列：{y}。')
	},
	vars: (layer, spec) => baseVars(layer, spec),
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => "F.interpolate(h, scale_factor=2, mode='nearest')",
	terms: ['upsampling']
};

export default upsample;
