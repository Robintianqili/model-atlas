// View 'tiny-unet-seg/upmix': one cell of an upsampling layer worked out with the op's own arithmetic (ops/upsample.ts
// taps): the old map (one channel) with the 2×2 cells it reads, the new map with the cell, and value × weight = product
// for the 4 cells, summed. props.layer (default 'up2'); the cell is the step's selection; hover or tap the new map to
// work out another cell of the same map.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtSetting, paren } from '../../../../atlas/core/format.ts';
import { taps } from '../../ops/upsample.ts';
import { T, demoRt } from '../../seg.ts';

type P = Omit<StageProps, 'ctx'>;
const IN: Record<string, string> = { up2: 'bott', up1: 'dec2' };
const FROM: Record<string, { en: string; zh: string }> = { bott: { en: 'the bottleneck', zh: '瓶颈' }, dec2: { en: 'up 2', zh: '上行 2' } };
export const unitOf = (p: P) => (p.sel.unit as number[] | undefined) ?? [0, 5, 6];

export function mixOf(p: P, u: number[]) {
	const layer = String(p.props.layer ?? 'up2'), x = T(p.rt, IN[layer]), y = T(p.rt, layer);
	if (!x || !y) return null;
	const [, H] = x.shape, OH = y.shape[1], [c, oy, ox] = u;
	const [y0, y1, ly] = taps(oy, H, OH / H), [x0, x1, lx] = taps(ox, H, OH / H);
	const cells = [[y0, x0, (1 - ly) * (1 - lx)], [y0, x1, (1 - ly) * lx], [y1, x0, ly * (1 - lx)], [y1, x1, ly * lx]].map(([r, q, w]) => ({ r, q, w, v: x.data[(c * H + r) * H + q] }));
	const inMap = x.data.subarray(c * H * H, (c + 1) * H * H), outMap = y.data.subarray(c * OH * OH, (c + 1) * OH * OH);
	const top = Math.max(1e-6, ...Array.from(inMap, Math.abs), ...Array.from(outMap, Math.abs));
	return { layer, from: FROM[IN[layer]], c, H, OH, cells, inMap, outMap, dom: [-top, top] as [number, number], box: { y0, x0, y1, x1 }, y: y.data[(c * OH + oy) * OH + ox], ly, lx };
}

export function describeMix(p: P): I18n {
	const m = mixOf(p, unitOf(p)), u = unitOf(p);
	if (!m) return { en: 'One upsampled cell: nothing to show yet.', zh: '一个上采样格子：还没有内容。' };
	const sum = m.cells.map((q) => `${fmtSetting(q.w)} × ${paren(q.v)}`).join(' + ');
	return { en: `Map ${m.c}, new cell row ${u[1]}, column ${u[2]} (counting from 0): ${sum} = ${fmt3(m.y)}.`, zh: `第 ${m.c} 张图，新格子第 ${u[1]} 行第 ${u[2]} 列（从 0 数起）：${sum} = ${fmt3(m.y)}。` };
}

const meta = {
	key: 'tiny-unet-seg/upmix',
	title: { en: 'One upsampled cell, worked out', zh: '算出一个上采样格子' },
	kinds: ['trace'],
	describe: describeMix,
	async demo() {
		return { rt: demoRt(), source: 'fwd', at: { index: 0 }, focus: [], sel: { layer: 'up2', unit: [0, 5, 6] }, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 420
} satisfies ViewMeta;

export default meta;
