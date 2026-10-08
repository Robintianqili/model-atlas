// View 'loss-gradient-descent/bowl': the error score over every setting of the two dials, a landscape seen from above
// (w across, b up; the kit's sequential scale on the log of the score), with the lowest point (the best line), the dials at the
// frame (state.w, state.b) and the path gradient descent has taken so far (state.path). Props:
//   arrow?: boolean      at a frame with a gradient (state.gw, state.gb): the step the update takes, −η·gradient
//                        (η = the frame's metrics.lr)
//   others?: { source: string; label: I18n }[]   other runs' paths at the same step count (their last frame if shorter)
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { B_RANGE, W_RANGE, bestLine, scoreOf } from '../../data.ts';

export type Other = { source: string; label: I18n };
export interface BowlPic {
	w: number;
	b: number;
	score: number;
	path: number[][];
	arrow: { dw: number; db: number } | null;
	best: { w: number; b: number; score: number };
	others: { label: I18n; path: number[][] }[];
}

const rows = (t: Tensor | undefined): number[][] => {
	if (!t || t.shape.length !== 2 || t.shape[1] !== 2) return [];
	return Array.from({ length: t.shape[0] }, (_, i) => [t.data[2 * i], t.data[2 * i + 1]]);
};

/** What the view draws (pure; also the text form). Null when the frame has no dials. */
export function bowlOf(p: Pick<StageProps, 'rt' | 'at' | 'props'>): BowlPic | null {
	const s = (p.at?.state ?? {}) as { w?: unknown; b?: unknown; gw?: unknown; gb?: unknown; path?: Tensor };
	if (typeof s.w !== 'number' || typeof s.b !== 'number') return null;
	const lr = p.at?.frame?.metrics?.lr;
	const arrow = p.props?.arrow && typeof s.gw === 'number' && typeof s.gb === 'number' && typeof lr === 'number' ? { dw: -lr * s.gw, db: -lr * s.gb } : null;
	const others: BowlPic['others'] = [];
	const step = p.at?.frame?.step;
	for (const o of (p.props?.others ?? []) as Other[]) {
		const tape = p.rt.sources[o.source]?.tape;
		if (!tape || !tape.length) continue;
		let i = typeof step === 'number' ? tape.find({ step }) : -1;
		if (i < 0) i = tape.length - 1;
		others.push({ label: o.label, path: rows((tape.state(i) as { path?: Tensor }).path) });
	}
	return { w: s.w, b: s.b, score: scoreOf(s.w, s.b), path: rows(s.path), arrow, best: bestLine(), others };
}

const inside = (q: number[]) => q[0] >= W_RANGE[0] && q[0] <= W_RANGE[1] && q[1] >= B_RANGE[0] && q[1] <= B_RANGE[1];

export function describeBowl(pic: BowlPic | null): I18n {
	if (!pic) return { en: 'No dials to draw yet.', zh: '还没有可画的旋钮数值。' };
	const arrow = pic.arrow
		? { en: ` The step the update takes moves w by ${fmt3(pic.arrow.dw)} and b by ${fmt3(pic.arrow.db)}.`, zh: `这一步更新让 w 变化 ${fmt3(pic.arrow.dw)}，b 变化 ${fmt3(pic.arrow.db)}。` }
		: { en: '', zh: '' };
	const out = pic.others.map((o) => ({ o, n: o.path.length, gone: o.path.some((q) => !inside(q)) }));
	const others = {
		en: out.map((x) => ` ${x.o.label.en}: ${fmtInt(x.n)} points${x.gone ? ', some outside the window' : ''}.`).join(''),
		zh: out.map((x) => `${x.o.label.zh ?? x.o.label.en}：${fmtInt(x.n)} 个点${x.gone ? '，有的在窗口之外' : ''}。`).join('')
	};
	return {
		en: `The error score over every dial setting, w from ${W_RANGE[0]} to ${W_RANGE[1]} and b from ${B_RANGE[0]} to ${B_RANGE[1]}. The dials are at w = ${fmt3(pic.w)}, b = ${fmt3(pic.b)}, score ${fmt3(pic.score)}; the lowest score, ${fmt3(pic.best.score)}, is at w = ${fmt3(pic.best.w)}, b = ${fmt3(pic.best.b)}. The path so far has ${fmtInt(pic.path.length)} points.${arrow.en}${others.en}`,
		zh: `误差分数在所有旋钮设置上的取值：w 从 ${W_RANGE[0]} 到 ${W_RANGE[1]}，b 从 ${B_RANGE[0]} 到 ${B_RANGE[1]}。旋钮位于 w = ${fmt3(pic.w)}、b = ${fmt3(pic.b)}，分数 ${fmt3(pic.score)}；最低分数 ${fmt3(pic.best.score)} 位于 w = ${fmt3(pic.best.w)}、b = ${fmt3(pic.best.b)}。目前的路径有 ${fmtInt(pic.path.length)} 个点。${arrow.zh}${others.zh}`
	};
}

const meta = {
	key: 'loss-gradient-descent/bowl',
	title: { en: 'The error landscape of two dials', zh: '两个旋钮的误差地形' },
	kinds: ['process'],
	describe: (p: StageProps) => describeBowl(bowlOf(p)),
	demo: async () => {
		const [{ record }, { gd }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { w: -1, b: 2 };
		const tape = record(gd, input, { lr: 0.1, steps: 6 }, { detail: 'fine' });
		const rt = fakeRuntime({ slug: 'loss-gradient-descent', input, tapes: { gd: tape } });
		const i = tape.find({ phase: 'grad', step: 5 });
		return { rt, source: 'gd', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { arrow: true } };
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
