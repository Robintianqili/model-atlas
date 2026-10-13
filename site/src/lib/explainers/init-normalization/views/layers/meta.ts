// View 'init-normalization/layers': the toy network layer by layer at the frame of a process 'mlp' source. One row per
// hidden layer (layer 1 at the top): the histogram of its 16,384 activations (each row scaled to its own tallest bar,
// one x range for every row: [−1, 1] for tanh, [0, 4 × the largest spread] for ReLU) and their spread on a log axis
// shared by the rows; at a 'backward' frame the same for the gradient dL/dz that flows back into each layer. Props:
//   grads?: boolean   the gradient panel (default: when the frame has it)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ACT_NAME, BINS, DEPTH, NORM_NAME, actOfCode, normOfCode, stdText } from '../../data.ts';

export interface Panel {
	/** [DEPTH][BINS] counts and [DEPTH] spreads */
	hist: number[][];
	spread: number[];
	lo: number;
	hi: number;
}
export interface LayersPic {
	act: Panel;
	grad: Panel | null;
	std: number;
	actName: I18n;
	normName: I18n;
}

const rows = (t: unknown, n: number): number[][] => {
	const d = Array.from(((t as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);
	return Array.from({ length: DEPTH }, (_, l) => d.slice(l * n, (l + 1) * n));
};
const vec = (t: unknown): number[] => Array.from(((t as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);

/** What the view draws (pure; also the text form). Null before the frame exists. */
export function layersOf(p: Pick<StageProps, 'at' | 'props'>): LayersPic | null {
	const s = (p.at?.state ?? {}) as Record<string, unknown>;
	const m = p.at?.frame?.metrics;
	if (!s.ha || !s.sa || !m) return null;
	const act: Panel = { hist: rows(s.ha, BINS), spread: vec(s.sa), lo: m.aLo, hi: m.aHi };
	const want = p.props?.grads !== false;
	const grad: Panel | null = want && s.hg && s.sg ? { hist: rows(s.hg, BINS), spread: vec(s.sg), lo: m.gLo, hi: m.gHi } : null;
	return { act, grad, std: m.std, actName: ACT_NAME[actOfCode(m.act)], normName: NORM_NAME[normOfCode(m.norm)] };
}

export function describeLayers(pic: LayersPic | null): I18n {
	if (!pic) return { en: 'The network has not run yet.', zh: '网络还没有运行。' };
	const list = (a: number[]) => a.map((v, l) => `${l + 1}: ${fmt3(v)}`).join('; ');
	const g = pic.grad;
	return {
		en: `A 10-layer network, ${pic.actName.en}, σ = ${stdText(pic.std)}, ${pic.normName.en}. Spread of the activations per layer: ${list(pic.act.spread)}.${g ? ` Spread of the gradient reaching each layer: ${list(g.spread)}.` : ''}`,
		zh: `一个 10 层网络，${pic.actName.zh}，σ = ${stdText(pic.std)}，${pic.normName.zh}。各层激活值的标准差：${list(pic.act.spread)}。${g ? `传回各层的梯度的标准差：${list(g.spread)}。` : ''}`
	};
}

const meta = {
	key: 'init-normalization/layers',
	title: { en: 'Each layer as a histogram', zh: '把每一层画成直方图' },
	kinds: ['process'],
	describe: (p: StageProps) => describeLayers(layersOf(p)),
	demo: async () => {
		const [{ record }, { mlp }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { act: 'tanh', std: 1, norm: 'none' };
		const tape = record(mlp, input, {});
		const rt = fakeRuntime({ slug: 'init-normalization', input, tapes: { large: tape } });
		return { rt, source: 'large', at: { index: 1, frame: tape.frame(1), state: tape.state(1), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
