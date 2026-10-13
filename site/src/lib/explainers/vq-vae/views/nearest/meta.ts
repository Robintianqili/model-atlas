// View 'vq-vae/nearest': one patch (the outlined one, vq.ts cellOf) and its nearest code, with props.mode
//   'snap'   (default) the patch, its numbers z_e, the squared distance to every code (nearest first) and the sum
//            behind the nearest one, term by term
//   'train'  the straight-through estimator for this patch: z_e, the code e_k the decoder saw, and the gradient of the
//            training loss at z_q, which training copies to z_e unchanged (vq.ts gradAt)
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { T, cellOf, cellVec, distOf, gradAt, rankOf } from '../../vq.ts';

export function nearestOf(p: Omit<StageProps, 'ctx'>) {
	const ze = T(p.rt, 'ze');
	if (!ze || !T(p.rt, 'zq#dist')) return null;
	const c = cellOf(p.rt), G = ze.shape[2], rank = rankOf(distOf(p.rt, c));
	return { c, row: Math.floor(c / G), col: c % G, G, ze: cellVec(ze, c), zq: cellVec(T(p.rt, 'zq'), c), rank, grad: p.props.mode === 'train' ? gradAt(p.rt, c) : [] };
}

export function describeNearest(p: Omit<StageProps, 'ctx'>): I18n {
	const m = nearestOf(p);
	if (!m) return { en: 'The nearest code of one patch: nothing drawn yet.', zh: '一个图块最近的码字：还没有画。' };
	const [a, b] = m.rank, v = (xs: number[]) => xs.map(fmt3).join(', ');
	const base = {
		en: `Patch at row ${m.row}, column ${m.col} (counting from 0): z_e = (${v(m.ze)}). Nearest code ${a.k} at squared distance ${fmt3(a.d)}; next code ${b.k} at ${fmt3(b.d)}.`,
		zh: `第 ${m.row} 行第 ${m.col} 列（从 0 数起）的图块：z_e = (${v(m.ze)})。最近的码字 ${a.k}，距离平方 ${fmt3(a.d)}；其次是码字 ${b.k}，${fmt3(b.d)}。`
	};
	if (!m.grad.length) return base;
	return { en: `${base.en} Gradient at z_q, copied to z_e: (${v(m.grad)}).`, zh: `${base.zh} z_q 处的梯度，原样复制给 z_e：(${v(m.grad)})。` };
}

const meta = {
	key: 'vq-vae/nearest',
	title: { en: 'A patch snaps to its nearest code', zh: '图块吸附到最近的码字' },
	kinds: ['trace'],
	describe: describeNearest,
	async demo() {
		// made-up data: a 2 × 2 grid of patches, 2 numbers each, 3 codes
		const t = (shape: number[], data: number[]) => ({ shape, data: Float32Array.from(data) });
		const input = new Float32Array(784);
		for (let i = 0; i < 14; i++) input[i * 28 + 3] = 1;
		const trace = { order: ['input', 'ze', 'zq'], t: { input: t([1, 28, 28], Array.from(input)), ze: t([2, 2, 2], [0.4, 1.2, 0.1, 0.2, -0.3, 0.5, 0.3, 0.1]), zq: t([2, 2, 2], [0.5, 1, 0, 0, -0.5, 0.5, 0, 0]), 'zq#dist': t([3, 2, 2], [0.05, 0.04, 0.1, 0.05, 1.3, 0.1, 1.2, 1.1, 2.9, 3.1, 2.5, 2.4]) } };
		const rt = fakeRuntime({ slug: 'vq-vae', trace });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'snap' } };
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
