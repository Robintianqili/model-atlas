// tiny-unet-seg local op 'upsample': bilinear upsampling of [C, H, W] maps by an integer scale, as
// F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False). Each output cell reads the 2×2 input cells
// around its centre, mapped back: src = (dst + 0.5) / s − 0.5 (0 below 0), i0 = floor(src), i1 = min(i0 + 1, H − 1),
// λ = src − i0; y = (1−λy)·[(1−λx)·a + λx·b] + λy·[(1−λx)·c + λx·d] (PyTorch's order, f64, stored f32).
// Python mirror: pipeline/tiny-unet-seg/ops_upsample.py.
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, needRank, outArray, store, t, unitBase } from '../../../atlas/ops/common.ts';

type UpP = { scale: number };

/** The two source rows (or columns) of output index o and the weight of the second. */
export function taps(o: number, n: number, s: number): [number, number, number] {
	const src = Math.max(0, (o + 0.5) / s - 0.5);
	const i0 = Math.floor(src);
	return [i0, Math.min(i0 + 1, n - 1), src - i0];
}

export const upsample: OpDef<UpP> = {
	op: 'upsample',
	version: 1,
	attrs: { scale: { type: 'int', default: 2, doc: 'the integer factor of both sides' } },
	category: 'shape',
	label: t('Upsample (bilinear)', '上采样（双线性）'),
	inferShape(ins, p) {
		const x = ins[0], s = p.scale ?? 2;
		if (x.length !== 3) throw new Error(`upsample: input must be [C, H, W], got [${x}]`);
		return [x[0], typeof x[1] === 'number' ? x[1] * s : x[1], typeof x[2] === 'number' ? x[2] * s : x[2]];
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs, p) {
		const src = xs[0];
		needRank(src, 3, 'upsample input');
		const [C, H, W] = src.shape, s = p.scale ?? 2, OH = H * s, OW = W * s, x = src.data;
		const out = outArray(C * OH * OW, src);
		for (let c = 0; c < C; c++) {
			for (let oy = 0; oy < OH; oy++) {
				const [y0, y1, ly] = taps(oy, H, s);
				for (let ox = 0; ox < OW; ox++) {
					const [x0, x1, lx] = taps(ox, W, s);
					const r0 = (c * H + y0) * W, r1 = (c * H + y1) * W;
					out[(c * OH + oy) * OW + ox] = (1 - ly) * ((1 - lx) * x[r0 + x0] + lx * x[r0 + x1]) + ly * ((1 - lx) * x[r1 + x0] + lx * x[r1 + x1]);
				}
			}
		}
		return { shape: [C, OH, OW], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'raster', pick: 'strongestChannel' },
	explain(q): UnitExplanation {
		const [c, oy, ox] = q.unit;
		const src = q.ins[0], [, H, W] = src.shape, s = q.p.scale ?? 2;
		const [y0, y1, ly] = taps(oy, H, s), [x0, x1, lx] = taps(ox, W, s);
		const at = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]].map(([yy, xx]) => (c * H + yy) * W + xx);
		const v = at.map((i) => src.data[i]);
		const w = [(1 - ly) * (1 - lx), (1 - ly) * lx, ly * (1 - lx), ly * lx];
		const y = store((1 - ly) * ((1 - lx) * v[0] + lx * v[1]) + ly * ((1 - lx) * v[2] + lx * v[3]), q.out);
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [{ tensor: q.layer.in[0], cells: at, box: { y0, x0, y1, x1 } }];
		e.terms = v.map((x, i) => ({ x, w: w[i], p: x * w[i], xAt: at[i] }));
		e.values = { n: 4, s: y, b: 0, y, ch: c, row: oy, col: ox, wNear: Math.max(...w), wFar: Math.min(...w) };
		return e;
	},
	receptive(box, p) {
		const s = p.scale ?? 2, lo = (o: number) => Math.floor(Math.max(0, (o + 0.5) / s - 0.5));
		return { y0: lo(box.y0), x0: lo(box.x0), y1: lo(box.y1) + 1, x1: lo(box.x1) + 1 };
	},
	formula: `${sym('out', 'y')} = \\sum_{i=1}^{4} ${sym('param', 'w')}_i\\,${sym('in', 'x')}_i = {{y}}`,
	narration: {
		what: t(
			'Upsampling makes each map {scale|int} times as wide and as tall, from {inShape} to {outShape}. Every new cell is a mix of the 4 old cells nearest to its centre, with weights that add up to 1: the nearest gets the most.',
			'上采样把每张图的宽和高都放大 {scale|int} 倍，从 {inShape} 变成 {outShape}。每个新格子都是离它中心最近的 4 个旧格子的混合，权重加起来是 1：最近的那个分得最多。'
		),
		how: t('Nothing is learned here: the weights come only from where the new cell sits between the old ones.', '这里没有要学习的东西：权重只取决于新格子落在旧格子之间的什么位置。'),
		why: t('It brings the coarse maps back to the size of the picture, but it cannot add detail the coarse maps do not have.', '它把粗糙的图放大回图片的尺寸，但加不出粗糙的图里本来就没有的细节。'),
		look: t('The bigger maps look smooth and soft: every block of new cells blends between its old neighbours.', '放大后的图看起来平滑而柔和：每一块新格子都在旧邻居之间过渡。'),
		unit: t('The 4 weighted cells add up to {y}.', '4 个加权格子加起来是 {y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), scale: Number((layer.params ?? {}).scale ?? 2) }),
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: (layer) => `F.interpolate(x, scale_factor=${Number((layer.params ?? {}).scale ?? 2)}, mode="bilinear", align_corners=False)`,
	terms: ['upsampling', 'feature-map']
};

export default upsample;
