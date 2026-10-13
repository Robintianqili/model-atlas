// View 'regularization/field': the plane with the training points (filled dots), the test points (rings), the network's
// answer everywhere (the share of class B as a tint of each class's color, from the frame's weights, state.params) and
// its border (where the share is one half), and a ring around every test point the network gets wrong. Below it the
// frame's numbers: both losses and both accuracies. Props:
//   field?: boolean   draw the network's answer, its border and the wrong-answer rings (default true; false: only the points)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
import { knobsOf, splitOf, START } from '../../data.ts';
import { p1Of, type Params } from '../../net.ts';
import type { Points2D } from '../../../../atlas/data/points2d.ts';

export const CLASS_NAMES: I18n[] = [
	{ en: 'moon A', zh: '月牙 A' },
	{ en: 'moon B', zh: '月牙 B' }
];

export interface FieldPic {
	train: Points2D;
	test: Points2D;
	params: Params | null;
	/** per test point: the network's answer is wrong */
	wrong: boolean[];
	m: Record<string, number>;
}

/** What the view draws (pure; also the text form). The data come from the frame's own knobs (its metrics). */
export function fieldPic(p: Pick<StageProps, 'at'>): FieldPic | null {
	const m = p.at?.frame?.metrics;
	if (!m || typeof m.noise !== 'number') return null;
	const k = knobsOf({ ...START, noise: m.noise, ratio: m.ratio });
	const { train, test } = splitOf(k.noise, k.ratio);
	const s = (p.at.state ?? {}) as { params?: Params };
	const params = s.params ?? null;
	const wrong: boolean[] = new Array(test.n).fill(false);
	if (params) {
		const p1 = p1Of(params, test.xy, test.n);
		for (let i = 0; i < test.n; i++) wrong[i] = (p1[i] > 0.5 ? 1 : 0) !== test.label[i];
	}
	return { train, test, params, wrong, m };
}

export function describeField(pic: FieldPic | null, rings = true): I18n {
	if (!pic) return { en: 'No run to draw yet.', zh: '还没有可画的训练。' };
	const nWrong = pic.wrong.filter(Boolean).length;
	const w = rings
		? { en: ` ${fmtInt(nWrong)} test points are on the wrong side of its border (marked with a second ring).`, zh: `有 ${fmtInt(nWrong)} 个测试点落在边界错误的一侧（外面多画了一个圈）。` }
		: { en: '', zh: '' };
	const m = pic.m;
	return {
		en: `${fmtInt(pic.train.n)} training points (filled dots) and ${fmtInt(pic.test.n)} test points (rings) of two moons. After ${fmtInt(m.step)} updates the network's training loss is ${fmt3(m.trainLoss)} and its test loss ${fmt3(m.testLoss)}; it sorts ${fmtPct(m.trainAcc)} of the training points and ${fmtPct(m.testAcc)} of the test points correctly.${w.en}`,
		zh: `两个月牙的 ${fmtInt(pic.train.n)} 个训练点（实心点）和 ${fmtInt(pic.test.n)} 个测试点（空心圈）。更新 ${fmtInt(m.step)} 次之后，网络的训练损失是 ${fmt3(m.trainLoss)}，测试损失是 ${fmt3(m.testLoss)}；它把 ${fmtPct(m.trainAcc)} 的训练点和 ${fmtPct(m.testAcc)} 的测试点分对了。${w.zh}`
	};
}

const meta = {
	key: 'regularization/field',
	title: { en: 'Training points, test points and the border', zh: '训练点、测试点和分界线' },
	kinds: ['process'],
	describe: (p: StageProps) => describeField(fieldPic(p), p.props?.field !== false),
	demo: async () => {
		const [{ record }, { fit }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { ...START };
		const tape = record(fit, input, { steps: 60 });
		const i = tape.length - 1;
		const rt = fakeRuntime({ slug: 'regularization', input, tapes: { run: tape } });
		return { rt, source: 'run', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
