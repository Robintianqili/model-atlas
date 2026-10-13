// View 'optimizers/range': the learning-rate range test (process 'range') as the loss against the learning rate, both on
// log scales, as far as the frame. At the 'end' frame it marks what the test reads off: the steepest fall, the lowest
// loss and where the loss climbs back above its start (the blow-up).
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';

export interface RangePic {
	lr: number[];
	loss: number[];
	marks: { key: 'steep' | 'low' | 'blow'; lr: number; loss: number }[];
}

export function rangePic(p: Pick<StageProps, 'at'>): RangePic | null {
	const tape = p.at?.tape;
	if (!tape || !tape.length) return null;
	const n = Math.min(p.at.index + 1, tape.length);
	const lr: number[] = [];
	const loss: number[] = [];
	for (let i = 0; i < n; i++) {
		const m = tape.frame(i).metrics ?? {};
		if (Number.isFinite(m.lr) && Number.isFinite(m.loss)) {
			lr.push(m.lr);
			loss.push(m.loss);
		}
	}
	const e = p.at.frame?.metrics ?? {};
	const marks: RangePic['marks'] = [];
	for (const key of ['steep', 'low', 'blow'] as const) {
		const a = e[`${key}Lr`];
		const b = e[`${key}Loss`];
		if (Number.isFinite(a) && Number.isFinite(b)) marks.push({ key, lr: a, loss: b });
	}
	return lr.length ? { lr, loss, marks } : null;
}

export const MARK_NAMES: Record<'steep' | 'low' | 'blow', I18n> = {
	steep: { en: 'steepest fall', zh: '下降最快' },
	low: { en: 'lowest loss', zh: '损失最低' },
	blow: { en: 'blows up', zh: '开始崩溃' }
};

export function describeRange(pic: RangePic | null): I18n {
	if (!pic) return { en: 'The range test has not run yet.', zh: '区间测试还没有运行。' };
	const n = pic.lr.length;
	const marks = pic.marks.map((m) => ({ en: ` ${MARK_NAMES[m.key].en}: η = ${fmt3(m.lr)}, loss ${fmt3(m.loss)}.`, zh: `${MARK_NAMES[m.key].zh}：η = ${fmt3(m.lr)}，损失 ${fmt3(m.loss)}。` }));
	return {
		en: `The loss against the learning rate η over ${fmtInt(n)} training steps, both axes on log scales: η from ${fmt3(pic.lr[0])} to ${fmt3(pic.lr[n - 1])}, the loss from ${fmt3(pic.loss[0])} to ${fmt3(pic.loss[n - 1])}.${marks.map((m) => m.en).join('')}`,
		zh: `${fmtInt(n)} 个训练步中损失随学习率 η 的变化，两个轴都是对数刻度：η 从 ${fmt3(pic.lr[0])} 到 ${fmt3(pic.lr[n - 1])}，损失从 ${fmt3(pic.loss[0])} 到 ${fmt3(pic.loss[n - 1])}。${marks.map((m) => m.zh).join('')}`
	};
}

const meta = {
	key: 'optimizers/range',
	title: { en: 'A learning-rate range test', zh: '学习率区间测试' },
	kinds: ['process'],
	describe: (p: StageProps) => describeRange(rangePic(p)),
	demo: async () => {
		const [{ record }, { range }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const tape = record(range, null, { steps: 100 });
		const rt = fakeRuntime({ slug: 'optimizers', input: null, tapes: { range: tape } });
		const i = tape.length - 1;
		return { rt, source: 'range', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 280
} satisfies ViewMeta;

export default meta;
