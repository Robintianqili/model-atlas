// transe view 'kg': the knowledge graph itself. Three columns (capitals, countries, then continents and languages),
// one arrow per fact from head to tail, coloured by relation (its name next to the colour). Facts the learner added
// are dashed; props.mark rings entities and draws their facts thicker (the lesson's example fact, Portugal).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { added, ENT, inputOf, REL } from '../../kg.ts';

type P = Omit<StageProps, 'ctx'>;
export function describeKg(p: P): I18n {
	const g = inputOf(p.rt.input), n = REL.map((_, k) => g.facts.filter((f) => f[1] === k).length), a = added(p.rt.input).length;
	return {
		en: `A knowledge graph of ${ENT.length} entities (11 countries, 11 capitals, 4 continents, 8 languages) and ${g.facts.length} facts: ${REL.map((r, k) => `${n[k]} “${r.en}”`).join(', ')}${a ? `; ${a} added by you` : ''}.`,
		zh: `一个知识图谱：${ENT.length} 个实体（11 个国家、11 个首都、4 个大洲、8 种语言）和 ${g.facts.length} 个事实：${REL.map((r, k) => `“${r.zh}” ${n[k]} 个`).join('，')}${a ? `；其中 ${a} 个是你加的` : ''}。`
	};
}

const meta = {
	key: 'transe/kg',
	title: { en: 'The knowledge graph', zh: '知识图谱' },
	kinds: ['train', 'process'],
	describe: describeKg,
	async demo() {
		const [{ fakeRuntime }, { START }] = await Promise.all([import('../../../../atlas/runtime/fake.ts'), import('../../kg.ts')]);
		const rt = fakeRuntime({ slug: 'transe', input: START, tapes: {} });
		return { rt, source: 'run', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mark: [0, 11] } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
