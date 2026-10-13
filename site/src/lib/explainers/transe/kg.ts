// transe: the small knowledge graph (generated here, no download) and the learner's input. 34 entities of four types
// (11 countries, their 11 capitals, 4 continents, 8 languages) and 3 relations; a fact is a triple (head, relation,
// tail) of ids. The start facts leave out where Portugal is, so the lesson can ask the model and the learner can add it.
// Pure and shared by the adapter, the trainer, the process, the views and the lesson; pipeline/transe reads the same
// ids from the train fixture and the process input, never from a copy of its own.
import type { I18n } from '../../atlas/types/index.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
/** entity types, in id order: countries 0-10, capitals 11-21, continents 22-25, languages 26-33 */
export const TYPES = [L('country', '国家'), L('capital', '首都'), L('continent', '大洲'), L('language', '语言')];
/** [country, its Chinese, capital, its Chinese] */
const C: [string, string, string, string][] = [
	['France', '法国', 'Paris', '巴黎'], ['Germany', '德国', 'Berlin', '柏林'], ['Italy', '意大利', 'Rome', '罗马'], ['Spain', '西班牙', 'Madrid', '马德里'],
	['Portugal', '葡萄牙', 'Lisbon', '里斯本'], ['Japan', '日本', 'Tokyo', '东京'], ['China', '中国', 'Beijing', '北京'], ['Brazil', '巴西', 'Brasília', '巴西利亚'],
	['Argentina', '阿根廷', 'Buenos Aires', '布宜诺斯艾利斯'], ['Mexico', '墨西哥', 'Mexico City', '墨西哥城'], ['Canada', '加拿大', 'Ottawa', '渥太华']
];
const CONT = [L('Europe', '欧洲'), L('Asia', '亚洲'), L('South America', '南美洲'), L('North America', '北美洲')];
const LANG = [L('French', '法语'), L('German', '德语'), L('Italian', '意大利语'), L('Spanish', '西班牙语'), L('Portuguese', '葡萄牙语'), L('Japanese', '日语'), L('Chinese', '汉语'), L('English', '英语')];
export interface Entity { name: I18n; type: number }
export const ENT: Entity[] = [
	...C.map(([en, zh]) => ({ name: L(en, zh), type: 0 })),
	...C.map(([, , en, zh]) => ({ name: L(en, zh), type: 1 })),
	...CONT.map((name) => ({ name, type: 2 })),
	...LANG.map((name) => ({ name, type: 3 }))
];
export const NE = ENT.length;
/** relations: capital → country (one-to-one), country → continent (many-to-one), country → language */
export const REL = [L('capital of', '首都所属'), L('located in', '位于'), L('speaks', '讲')];
export const NR = REL.length;
export const [CAPITAL_OF, LOCATED_IN, SPEAKS] = [0, 1, 2];
/** the type of each relation's tails in the start facts (a country, a continent, a language) */
export const TAIL_TYPE = [0, 2, 3];
/** an entity's id by its English name */
export const id = (en: string) => ENT.findIndex((e) => e.name.en === en);
/** the ids of one entity type */
export const ofType = (k: number) => ENT.flatMap((e, i) => (e.type === k ? [i] : []));

export type Fact = [number, number, number];
const where = ['Europe', 'Europe', 'Europe', 'Europe', 'Europe', 'Asia', 'Asia', 'South America', 'South America', 'North America', 'North America'];
const speaks = [['French'], ['German'], ['Italian'], ['Spanish'], ['Portuguese'], ['Japanese'], ['Chinese'], ['Portuguese'], ['Spanish'], ['Spanish'], ['English', 'French']];
/** each country's continent (world knowledge, also Portugal's: the facts may leave one out) */
export const CONTINENT = where.map(id);
/** the fact the start leaves out (Portugal is located in Europe): the lesson's question and the task's answer */
export const PT_FACT: Fact = [id('Portugal'), LOCATED_IN, id('Europe')];
export const START_FACTS: Fact[] = [
	...C.map((_, i): Fact => [11 + i, CAPITAL_OF, i]),
	...where.flatMap((w, i): Fact[] => (i === PT_FACT[0] ? [] : [[i, LOCATED_IN, id(w)]])),
	...speaks.flatMap((ls, i) => ls.map((l): Fact => [i, SPEAKS, id(l)]))
];

/** the learner's input: the facts, and a question (head, relation) for link prediction */
export interface KgRaw { facts: Fact[]; query: [number, number] }
export const START: KgRaw = { facts: START_FACTS, query: [id('Paris'), CAPITAL_OF] };
const ok = (x: unknown, n: number) => Number.isInteger(x) && (x as number) >= 0 && (x as number) < n;
/** a valid input: facts with valid ids, without duplicates (the first kept), and a valid question (default: Paris,
 *  capital of) */
export function inputOf(raw: unknown): KgRaw {
	const r = (raw ?? {}) as Partial<KgRaw>, seen = new Set<string>(), facts: Fact[] = [];
	for (const f of Array.isArray(r.facts) ? r.facts : START_FACTS)
		if (Array.isArray(f) && ok(f[0], NE) && ok(f[1], NR) && ok(f[2], NE) && f[0] !== f[2] && !seen.has(f.join())) seen.add(f.join()), facts.push([f[0], f[1], f[2]]);
	const q = Array.isArray(r.query) && ok(r.query[0], NE) && ok(r.query[1], NR) ? ([r.query[0], r.query[1]] as [number, number]) : START.query;
	return { facts, query: q };
}
export const factsOf = (raw: unknown) => inputOf(raw).facts;
export const hasFact = (fs: Fact[], f: Fact) => fs.some((x) => x[0] === f[0] && x[1] === f[1] && x[2] === f[2]);
/** the facts that are not in the start set */
export const added = (raw: unknown) => factsOf(raw).filter((f) => !hasFact(START_FACTS, f));
export const nameOf = (i: number): I18n => ENT[i]?.name ?? L('?', '?');
/** "(Paris, capital of, France)" in both languages */
export const factText = (f: Fact): I18n => ({ en: `(${ENT[f[0]].name.en}, ${REL[f[1]].en}, ${ENT[f[2]].name.en})`, zh: `（${ENT[f[0]].name.zh}，${REL[f[1]].zh}，${ENT[f[2]].name.zh}）` });
