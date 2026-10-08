// Lint fixtures that replay the four kinds of mistake student reviewers found in rounds 2-3 (LOG.md: false claims in
// lesson narration, a NaN on a one-token input, a step index that no longer existed, a stage with no badge). Each one is
// the clean 6-step lesson with one step replaced by the faulty version; lint must report exactly the expected code.
import type { StepDef, LessonDef } from '../../types/lesson.ts';
import { counter, lesson6, squash } from './lesson6.ts';

export interface Finding {
	id: string;
	/** what the reviewer saw on the page */
	finding: string;
	expect: { code: string; level: 'error' | 'warn'; step: string; fixture?: string; lang?: 'en' | 'zh' };
	lesson: LessonDef;
}

function replace(id: string, step: StepDef): LessonDef {
	return { ...lesson6, steps: lesson6.steps.map((s) => ('id' in s && s.id === id ? step : s)) };
}

/** 1. A metric typed into the narration that disagreed with the screen. */
const wrongMetric: StepDef = {
	...squash,
	narration: {
		en: 'tanh squashes the sum {x} to {y}. This model is right 97% of the time on test data.',
		zh: 'tanh 把和 {x} 压成 {y}。这个模型在测试数据上 97% 的时候是对的。',
		vars: (_rt, at) => ({ x: at.unit!.values.x, y: at.unit!.values.y })
	}
};

/** 2. A quantity that is 0/0 for a one-letter input (the average change between neighbouring letters). */
const nanOnOne: StepDef = {
	...squash,
	narration: {
		en: 'tanh squashes the sum {x} to {y}. Between neighbouring letters the sum changes by {step} on average.',
		zh: 'tanh 把和 {x} 压成 {y}。相邻字母之间，和平均变化 {step}。',
		vars: (rt, at) => {
			const t = rt.sources.fwd.trace!.t['B0.lin'];
			const [T, D] = t.shape;
			let s = 0;
			for (let i = 1; i < T; i++) s += Math.abs(t.data[i * D] - t.data[(i - 1) * D]);
			return { x: at.unit!.values.x, y: at.unit!.values.y, step: s / (T - 1) };
		}
	}
};

/** 3. A fixed frame index written for the preset text ("the 8th letter"), stale on shorter inputs. */
const staleIndex: StepDef = { ...counter, at: 7, predict: counter.predict };

/** 4. A step that forgot its provenance, so its stage would carry no badge. */
const noBadge = { ...squash } as Partial<StepDef>;
delete noBadge.provenance;

export const FINDINGS: Finding[] = [
	{
		id: 'wrong-metric',
		finding: 'The narration quoted an accuracy as typed text; the measured value on the page was different.',
		expect: { code: 'AK402', level: 'warn', step: 'squash', lang: 'en' },
		lesson: replace('squash', wrongMetric)
	},
	{
		id: 'nan-one-token',
		finding: 'A one-token input made the narration show NaN.',
		expect: { code: 'AK401', level: 'error', step: 'squash', fixture: 'edge:one' },
		lesson: replace('squash', nanOnOne)
	},
	{
		id: 'stale-step-index',
		finding: 'A step pointed at a fixed position that did not exist for a shorter input.',
		expect: { code: 'AK404', level: 'error', step: 'count', fixture: 'edge:one' },
		lesson: replace('count', staleIndex)
	},
	{
		id: 'missing-badge',
		finding: 'A stage showed numbers with no provenance badge.',
		expect: { code: 'AK502', level: 'error', step: 'squash' },
		lesson: replace('squash', noBadge as StepDef)
	}
];
