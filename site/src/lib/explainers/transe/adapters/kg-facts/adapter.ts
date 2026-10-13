// transe: local input adapter 'kg-facts' (InputKind 'graph'): the facts of the small knowledge graph (kg.ts), editable,
// and a question (head, relation, ?). The model's input is the facts as an [n, 3] table of ids (head, relation, tail).
// No fact: nothing to train on, so the input is empty.
import type { I18n, InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { id, inputOf, LOCATED_IN, PT_FACT, SPEAKS, START, START_FACTS, type KgRaw } from '../../kg.ts';

export const NO_FACTS: I18n = { en: 'There is no fact to learn from: add at least one fact, or pick a starting set.', zh: '没有可以学习的事实：至少加上一个事实，或者选一组起始事实。' };
const L = (en: string, zh: string) => ({ en, zh });

/** the starting sets: the start; the start with Portugal's continent; the start without any language fact */
export const PRESETS: { id: string; label: I18n; raw: KgRaw }[] = [
	{ id: 'start', label: L('The start: 33 facts', '起始：33 个事实'), raw: START },
	{ id: 'with-pt', label: L('+ Portugal is in Europe', '+ 葡萄牙位于欧洲'), raw: { facts: [...START_FACTS, PT_FACT], query: [PT_FACT[0], LOCATED_IN] } },
	{ id: 'no-lang', label: L('Without the languages', '去掉语言'), raw: { facts: START_FACTS.filter((f) => f[1] !== SPEAKS), query: [id('Canada'), LOCATED_IN] } }
];

const adapter: InputAdapter<KgRaw> = {
	id: 'kg-facts',
	kind: 'graph',
	preprocess(raw): Preprocessed {
		const g = inputOf(raw), n = g.facts.length;
		const input = { shape: [n, 3], data: Int32Array.from(g.facts.flat()), dtype: 'i32' as const };
		return {
			tensors: { input },
			steps: [{ id: 'facts', label: { en: `The facts: ${n} rows of (head, relation, tail) ids`, zh: `事实：${n} 行（头实体，关系，尾实体）编号` } }],
			empty: !n,
			notes: n ? [] : [NO_FACTS]
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		{ id: 'none', label: L('no fact at all', '一个事实都没有'), raw: { facts: [], query: START.query } },
		{ id: 'one', label: L('a single fact: (Paris, capital of, France)', '只有一个事实：（巴黎，首都所属，法国）'), raw: { facts: [START_FACTS[0]], query: START.query } }
	],
	layer0: { view: 'transe/kg', narration: { what: { en: 'The facts the model learns from.', zh: '模型学习用的事实。' } } }
};

export default adapter;
