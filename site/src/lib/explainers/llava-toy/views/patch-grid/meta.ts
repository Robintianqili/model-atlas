// View 'llava-toy/patch-grid': the answer row's attention, drawn over the patches its image tokens come from.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { N_IMG, SIDE, gridOf } from './grid.ts';

const meta = {
	key: 'llava-toy/patch-grid',
	title: { en: 'Where the answer row looks', zh: '写答案的那一行看向哪里' },
	kinds: ['trace'],
	describe: (p: Omit<StageProps, 'ctx'>) => {
		const g = gridOf(p);
		if (!g) return { en: 'The answer row’s attention: nothing to show yet.', zh: '写答案那一行的注意力：还没有可显示的内容。' };
		const ink = g.inked.filter(Boolean).length;
		const head = g.head < 0 ? { en: 'the mean of the heads', zh: '各个头的平均' } : { en: `head ${g.head}`, zh: `头 ${g.head}` };
		const words = g.words.map((x) => `“${x.name}” ${fmtPct(x.w)}`).join(', ');
		return {
			en: `LM block ${g.block}, ${head.en}: the answer row gives ${fmtPct(g.imgShare)} to the ${N_IMG} image tokens, ${fmtPct(g.inkShare)} of it to the ${ink} patches with ink; the most, ${fmtPct(g.w[g.top])}, goes to image token ${g.top}. The words get ${words}.`,
			zh: `语言模型第 ${g.block} 块，${head.zh}：写答案的那一行给 ${N_IMG} 个图像词元 ${fmtPct(g.imgShare)}，其中 ${fmtPct(g.inkShare)} 给了 ${ink} 个有墨的图块；最多的 ${fmtPct(g.w[g.top])} 给了第 ${g.top} 个图像词元。各个词得到 ${words}。`
		};
	},
	async demo() {
		// made-up data: a square outline, 2 words, an answer row that favours the square's corners
		const img = new Float32Array(SIDE * SIDE);
		for (let i = 4; i < 24; i++) for (const [r, c] of [[4, i], [23, i], [i, 4], [i, 23]]) img[r * SIDE + c] = 1;
		const T = N_IMG + 2;
		const raw = Array.from({ length: T }, (_, k) => (k >= N_IMG ? 1 : [0, 3, 12, 15].includes(k) ? 3 : 1));
		const sum = raw.reduce((a, b) => a + b, 0);
		const attn = new Float32Array(4 * T * T);
		for (let h = 0; h < 4; h++) for (let q = 0; q < T; q++) for (let k = 0; k <= q; k++) attn[(h * T + q) * T + k] = q === T - 1 ? raw[k] / sum : 1 / (q + 1);
		const rt = fakeRuntime({ slug: 'llava-toy', trace: { order: ['image', 'lm.L1.attn'], t: { image: { shape: [1, SIDE, SIDE], data: img }, 'lm.L1.attn': { shape: [4, T, T], data: attn } } } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { block: 1 } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
