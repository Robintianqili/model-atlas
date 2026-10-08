// View 'loss-gradient-descent/fit': the eight points, the neuron's line y = w·x + b at the frame's dials (state.w,
// state.b), each point's error (prediction minus answer) as a vertical segment, and the squared errors as bars with their
// mean, the error score. Props:
//   squares?: boolean      the bars of squared errors (default true)
//   mark?: 'worst' | number   ring one point (the one the line misses most, or point i counting from 0)
//   stepButton?: boolean   a button that sets the dials to the step source's first update (its frame 'update', step 0):
//                          "let gradient descent turn the dials once" (needs setInput)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { N, XS, YS, errors, lineText, scoreOf, worstPoint } from '../../data.ts';

export interface FitPic {
	w: number;
	b: number;
	errs: number[];
	sq: number[];
	score: number;
	mark: number | null;
}

/** What the view draws (pure; also the text form). Null when the frame has no dials. */
export function fitOf(p: Pick<StageProps, 'at' | 'props'>): FitPic | null {
	const s = (p.at?.state ?? {}) as { w?: unknown; b?: unknown };
	if (typeof s.w !== 'number' || typeof s.b !== 'number') return null;
	const { w, b } = s as { w: number; b: number };
	const errs = errors(w, b);
	const m = p.props?.mark;
	const mark = m === 'worst' ? worstPoint(w, b).i : typeof m === 'number' && m >= 0 && m < N ? m : null;
	return { w, b, errs, sq: errs.map((e) => e * e), score: scoreOf(w, b), mark };
}

export function describeFit(pic: FitPic | null): I18n {
	if (!pic) return { en: 'No dials to draw yet.', zh: '还没有可画的旋钮数值。' };
	const list = pic.errs.map((e, i) => `(${fmt3(XS[i])}, ${fmt3(YS[i])}) ${fmt3(e)}`).join('; ');
	return {
		en: `The line y = ${lineText(pic.w, pic.b)} and the eight points. Each point's error, prediction minus answer: ${list}. The mean of the squared errors, the error score, is ${fmt3(pic.score)}.`,
		zh: `直线 y = ${lineText(pic.w, pic.b)} 和八个点。每个点的误差（预测减答案）：${list}。误差平方的平均值，也就是误差分数，是 ${fmt3(pic.score)}。`
	};
}

const meta = {
	key: 'loss-gradient-descent/fit',
	title: { en: 'A line, its errors and its score', zh: '一条直线、它的误差和分数' },
	kinds: ['process'],
	describe: (p: StageProps) => describeFit(fitOf(p)),
	demo: async () => {
		const [{ record }, { gd }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { w: 0.5, b: 1 };
		const tape = record(gd, input, { lr: 0.1, steps: 2 }, { detail: 'fine' });
		const rt = fakeRuntime({ slug: 'loss-gradient-descent', input, tapes: { gd: tape } });
		return { rt, source: 'gd', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { mark: 'worst' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
