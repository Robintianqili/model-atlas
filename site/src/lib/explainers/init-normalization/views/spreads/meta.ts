// View 'init-normalization/spreads': the frame of a process 'inits' source, the same network under the five named
// initializations. Two plots on log axes, layer 1 to 10 across: the spread of each layer's activations, and the spread of
// the gradient that flows back into each layer; one line per initialization, in its categorical color, named in the
// legend. A run whose numbers are all 0 (zero initialization) cannot be drawn on a log axis: the legend says so.
// Props:
//   grads?: boolean   the gradient plot (default true)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ACT_NAME, DEPTH, INITS, INIT_NAME, NORM_NAME, actOfCode, normOfCode, type InitId } from '../../data.ts';

export interface SpreadsPic {
	ids: InitId[];
	/** [init][layer] */
	act: number[][];
	grad: number[][];
	actName: I18n;
	normName: I18n;
}

export function spreadsOf(p: Pick<StageProps, 'at'>): SpreadsPic | null {
	const s = (p.at?.state ?? {}) as { sa?: { data: ArrayLike<number> }; sg?: { data: ArrayLike<number> } };
	const m = p.at?.frame?.metrics;
	if (!s.sa || !s.sg || !m) return null;
	const rows = (d: ArrayLike<number>) => INITS.map((_, k) => Array.from(d).slice(k * DEPTH, (k + 1) * DEPTH));
	return { ids: INITS, act: rows(s.sa.data), grad: rows(s.sg.data), actName: ACT_NAME[actOfCode(m.act)], normName: NORM_NAME[normOfCode(m.norm)] };
}

export function describeSpreads(pic: SpreadsPic | null): I18n {
	if (!pic) return { en: 'The five runs have not finished yet.', zh: '五次运行还没有完成。' };
	const line = (k: number, lang: 'en' | 'zh') => `${INIT_NAME[pic.ids[k]][lang]}: ${fmt3(pic.act[k][0])} → ${fmt3(pic.act[k][DEPTH - 1])}, ${fmt3(pic.grad[k][DEPTH - 1])} → ${fmt3(pic.grad[k][0])}`;
	return {
		en: `The same 10-layer network (${pic.actName.en}, ${pic.normName.en}) under five initializations. For each: the activations' spread at layer 1 → layer 10, then the gradient's spread at layer 10 → layer 1. ${pic.ids.map((_, k) => line(k, 'en')).join('; ')}.`,
		zh: `同一个 10 层网络（${pic.actName.zh}，${pic.normName.zh}）的五种初始化。每种依次给出：激活值的标准差从第 1 层到第 10 层，梯度的标准差从第 10 层到第 1 层。${pic.ids.map((_, k) => line(k, 'zh')).join('；')}。`
	};
}

const meta = {
	key: 'init-normalization/spreads',
	title: { en: 'Five initializations, layer by layer', zh: '五种初始化，逐层对比' },
	kinds: ['process'],
	describe: (p: StageProps) => describeSpreads(spreadsOf(p)),
	demo: async () => {
		const [{ record }, { inits }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { act: 'relu', norm: 'none' };
		const tape = record(inits, input, {});
		const rt = fakeRuntime({ slug: 'init-normalization', input, tapes: { relus: tape } });
		return { rt, source: 'relus', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
