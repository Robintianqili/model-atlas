// View 'vq-vae/grid': the learner's drawing cut into the G × G patches, with props.mode
//   'patches'  the grid over the drawing; a click outlines a patch (the raw input's `cell`, read by the other steps)
//   'codes'    the code index of every patch, over the drawing and as a G × G table
//   'redraw'   (default) the drawing, its code indices and the redrawing the decoder makes from them
//   'swap'     pick a patch and a code: the redrawing before and after, and what changed (the raw input's `swap`)
//   'limits'   the drawing, the redrawing and the redrawing error of every pixel (binary cross-entropy)
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { cellOf, codesOf, swapOf } from '../../vq.ts';

export function describeGrid(p: Omit<StageProps, 'ctx'>): I18n {
	const codes = codesOf(p.rt);
	if (!codes.length) return { en: 'Your drawing cut into patches: nothing drawn yet.', zh: '你的画切成的图块：还没有画。' };
	const c = cellOf(p.rt), G = Math.round(Math.sqrt(codes.length));
	const base = {
		en: `Your drawing as ${codes.length} code indices, row by row: ${codes.join(', ')}. Outlined: row ${Math.floor(c / G)}, column ${c % G} (counting from 0).`,
		zh: `你的画变成 ${codes.length} 个码字编号，逐行是：${codes.join('、')}。框出的是第 ${Math.floor(c / G)} 行第 ${c % G} 列（从 0 数起）。`
	};
	const s = p.props.mode === 'swap' ? swapOf(p.rt) : null;
	if (!s) return base;
	return {
		en: `${base.en} Swapped: the patch at row ${Math.floor(s.cell / G)}, column ${s.cell % G} from code ${s.from} to code ${s.to}; in that block, the pixels above half ink went from ${s.ink0} to ${s.ink1}.`,
		zh: `${base.zh} 已替换：第 ${Math.floor(s.cell / G)} 行第 ${s.cell % G} 列的图块从码字 ${s.from} 换成码字 ${s.to}；这一块里墨量超过一半的像素从 ${s.ink0} 个变成 ${s.ink1} 个。`
	};
}

const meta = {
	key: 'vq-vae/grid',
	title: { en: 'Your drawing as a grid of codes', zh: '你的画变成码字网格' },
	kinds: ['trace'],
	describe: describeGrid,
	async demo() {
		// made-up data: a ring, a softer copy of it as the "redrawing" and made-up code indices
		const ring = new Float32Array(784), soft = new Float32Array(784);
		for (let i = 0; i < 784; i++) {
			const d = Math.abs(Math.hypot((i % 28) - 13.5, Math.floor(i / 28) - 13.5) - 8);
			ring[i] = d < 1.2 ? 1 : 0;
			soft[i] = Math.exp(-d * d / 3);
		}
		const idx = Float32Array.from([3, 7, 7, 3, 9, 1, 1, 9, 9, 1, 1, 9, 3, 7, 7, 3]);
		const t = (shape: number[], data: Float32Array) => ({ shape, data });
		const trace = { order: ['input', 'zq', 'out'], t: { input: t([1, 28, 28], ring), 'zq#idx': t([4, 4], idx), out: t([1, 28, 28], soft) } };
		const rt = fakeRuntime({ slug: 'vq-vae', trace });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'codes' } };
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
