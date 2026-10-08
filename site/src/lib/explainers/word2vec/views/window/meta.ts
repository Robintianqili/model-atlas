// word2vec view 'window': the learner's sentences as words with their ids. At a 'center' frame of the pairs process the
// center word is filled, its window outlined and the (center, context) pairs it gives are listed; at the 'vocab' frame
// (or props.mode 'vocab') the vocabulary is listed with ids and counts.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { corpusOf, type Corpus } from '../../corpus.ts';

/** the pairs a center frame (metrics s, i) gives, as word pairs */
export function pairsAt(c: Corpus, s: number, i: number): [string, string][] {
	const ids = c.sentences[s] ?? [];
	const out: [string, string][] = [];
	for (let j = Math.max(0, i - c.window); j <= Math.min(ids.length - 1, i + c.window); j++) if (j !== i) out.push([c.words[ids[i]], c.words[ids[j]]]);
	return out;
}

export function describeWindow(p: Omit<StageProps, 'ctx'>): I18n {
	const c = corpusOf(p.rt.input as never), m = p.at.frame?.metrics;
	if (!c.pairs.length) return { en: 'Your sentences: no pairs yet.', zh: '你的句子：还没有训练对。' };
	if (p.at.frame?.phase !== 'center' || !m) {
		const v = c.words.map((w, i) => `${i} ${w} (${c.counts[i]})`).join(', ');
		return { en: `${c.sentences.length} sentences, ${c.words.length} different words. Ids and counts: ${v}.`, zh: `${c.sentences.length} 个句子，${c.words.length} 个不同的词。编号和次数：${v}。` };
	}
	const ps = pairsAt(c, m.s, m.i).map(([a, b]) => `(${a}, ${b})`).join(' ');
	return { en: `Sentence ${m.s + 1}, word ${m.i + 1}: center “${c.words[m.c]}”, window ${c.window}, pairs ${ps}; ${m.n} of ${c.pairs.length} pairs so far.`, zh: `第 ${m.s + 1} 句第 ${m.i + 1} 个词：中心词“${c.words[m.c]}”，窗口 ${c.window}，训练对 ${ps}；到目前为止 ${m.n} / ${c.pairs.length} 对。` };
}

const meta = {
	key: 'word2vec/window',
	title: { en: 'Your sentences and the sliding window', zh: '你的句子和滑动窗口' },
	kinds: ['process'],
	describe: describeWindow,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { pairs }, { PETS }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts'), import('../../corpus.ts')]);
		const tape = record(pairs, PETS, {}, {});
		const rt = fakeRuntime({ slug: 'word2vec', input: PETS, tapes: { pairs: tape } });
		return { rt, source: 'pairs', at: atFor(rt, 'pairs', 3), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
