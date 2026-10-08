// View 'tinyvit/patch-grid': the drawing cut into 16 patches; one patch at a time, or one attention row over them.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { GRID, SIDE, gridOf } from './grid.ts';

const meta = {
	key: 'tinyvit/patch-grid',
	title: { en: 'Patches of the drawing', zh: '画的图块' },
	kinds: ['trace'],
	describe: (p: Omit<StageProps, 'ctx'>) => {
		const g = gridOf(p);
		if (!g) return { en: 'The patch grid: nothing drawn yet.', zh: '图块网格：还没有画。' };
		const ink = g.inked.filter(Boolean).length;
		if (g.mode === 'patches')
			return {
				en: `The drawing cut into ${GRID * GRID} patches of 7 × 7 pixels; ${ink} have ink. Patch ${g.patch} is shown; its brightest pixel is ${Math.max(...g.pixels).toFixed(2)}.`,
				zh: `画被切成 ${GRID * GRID} 个 7 × 7 像素的图块，其中 ${ink} 个有墨。现在显示第 ${g.patch} 个图块，它最亮的像素是 ${Math.max(...g.pixels).toFixed(2)}。`
			};
		return {
			en: `Block ${g.block}, head ${g.head}: the [CLS] row gives ${fmtPct(g.self)} to itself and ${fmtPct(g.inkShare)} to the ${ink} patches with ink; the largest share, ${fmtPct(g.w[g.top])}, goes to patch ${g.top}.`,
			zh: `第 ${g.block} 块、头 ${g.head}：[CLS] 行给它自己 ${fmtPct(g.self)}，给 ${ink} 个有墨的图块共 ${fmtPct(g.inkShare)}；最大的份额 ${fmtPct(g.w[g.top])} 给了第 ${g.top} 个图块。`
		};
	},
	async demo() {
		// made-up data: a square outline and an attention row that favours its corners
		const img = new Float32Array(SIDE * SIDE);
		for (let i = 4; i < 24; i++) for (const [r, c] of [[4, i], [23, i], [i, 4], [i, 23]]) img[r * SIDE + c] = 1;
		const raw = Array.from({ length: 17 }, (_, k) => (k === 0 ? 0.5 : [1, 4, 13, 16].includes(k) ? 3 : 1));
		const sum = raw.reduce((a, b) => a + b, 0);
		const attn = new Float32Array(4 * 17 * 17);
		for (let h = 0; h < 4; h++) for (let q = 0; q < 17; q++) raw.forEach((v, k) => (attn[(h * 17 + q) * 17 + k] = v / sum));
		const rt = fakeRuntime({ slug: 'tinyvit', trace: { order: ['input', 'B1.attn'], t: { input: { shape: [1, SIDE, SIDE], data: img }, 'B1.attn': { shape: [4, 17, 17], data: attn } } } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'attention', block: 1, head: 0 } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
