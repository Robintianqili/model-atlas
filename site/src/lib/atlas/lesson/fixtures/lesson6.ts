// The 6-step fixture lesson (W1-C acceptance 5) on the 'fixture-seq' package: two auto steps (one with an override),
// three authored steps over a trace and a process source, and the auto output step marked as the limits step.
import type { Tensor } from '../../types/core.ts';
import type { StepDef } from '../../types/lesson.ts';
import { auto, defineLesson } from '../define.ts';
import { primaryTrace } from '../facts.ts';
import { pickUnit } from '../../runtime/selection.ts';
import { ALPHABET, type CountState } from './seq.ts';

export const squash: StepDef = {
	id: 'squash',
	title: { en: 'Squash the sums', zh: '把和压扁' },
	source: 'fwd',
	select: (rt) => ({ layer: 'B0.act', unit: pickUnit(primaryTrace({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel })?.t['B0.act'], 'argmaxAbs') }),
	view: 'zoom.elementwise',
	provenance: 'live',
	narration: {
		en: 'tanh squashes the sum {x} to {y}. Large sums end up near the edges, small ones barely change.',
		zh: 'tanh 把和 {x} 压成 {y}。大的和被推到两端附近，小的几乎不变。',
		vars: (_rt, at) => ({ x: at.unit!.values.x, y: at.unit!.values.y })
	},
	look: { en: 'Look at where {x} lands on the curve.', zh: '看 {x} 落在曲线的哪里。', vars: (_rt, at) => ({ x: at.unit!.values.x }) }
};

function mostCommon(count: Tensor): { id: number; c: number } {
	let id = 0;
	for (let i = 1; i < count.data.length; i++) if (count.data[i] > count.data[id]) id = i;
	return { id, c: count.data[id] };
}

export const counter: StepDef = {
	id: 'count',
	title: { en: 'A counter reads the letters', zh: '计数器逐个读字母' },
	source: 'unroll',
	at: { phase: 'read', last: true },
	view: 'zoom.formula',
	provenance: 'live',
	predict: {
		id: 'seq-count',
		question: { en: 'The counter reads one letter per step. How many steps does your text take?', zh: '计数器每步读一个字母。你的文字需要多少步？' },
		options: [
			{ en: 'One per letter', zh: '每个字母一步' },
			{ en: 'One per word', zh: '每个词一步' }
		],
		answer: 0,
		explain: { en: 'Each letter is one event on the tape.', zh: '每个字母都是记录里的一个事件。' }
	},
	narration: {
		en: 'Letters read so far: {run.length|int}. The most common one is {top|tok}, with {share|pct} of them.',
		zh: '计数器读了 {run.length|int} 个字母。最常见的是 {top|tok}，占 {share|pct}。',
		vars: (_rt, at) => {
			const s = at.state as CountState;
			const m = mostCommon(s.count);
			return { top: ALPHABET[m.id], share: m.c / (s.t + 1) };
		}
	}
};

export const nudge: StepDef = {
	id: 'nudge',
	title: { en: 'Your turn: change the answer', zh: '轮到你：改变答案' },
	source: 'fwd',
	at: { layer: 'prob' },
	view: 'kit.probbars',
	provenance: 'live',
	task: {
		prompt: { en: 'Change the text until the model answers "odd".', zh: '修改文字，直到模型回答“古怪”。' },
		done: (rt) => (rt.fact('top.label') as { en?: string } | undefined)?.en === 'odd',
		hint: { en: 'Try letters you rarely use.', zh: '试试不常用的字母。' }
	},
	narration: {
		en: 'Right now the model says {top.label} with {top.p|pct}. The gap to {runnerUp.label} is {margin|pct}.',
		zh: '现在模型说 {top.label}，概率 {top.p|pct}。它比 {runnerUp.label} 高 {margin|pct}。'
	}
};

export const lesson6 = defineLesson({
	title: { en: 'From letters to an answer', zh: '从字母到答案' },
	summary: { en: 'Six steps through a tiny letter classifier and a letter counter.', zh: '用六步走过一个小小的字母分类器和一个字母计数器。' },
	fixtures: [{ id: 'mixed', label: { en: 'A short phrase', zh: '一个短语' }, raw: 'a quiet day' }],
	steps: [
		auto('input'),
		auto('op:B0.lin', { title: { en: 'A weighted sum, cell by cell', zh: '一格一格的加权和' } }),
		squash,
		counter,
		nudge,
		auto('output', { kindOf: 'limits' })
	]
});
