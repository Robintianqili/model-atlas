// word2vec: the processes (kind C).
//   pairs  the sliding window: a 'vocab' frame (the vocabulary), then one 'center' frame per word of every sentence, left
//          to right, with the number of context words it gets. No Python reference (notes, kit issue 4): its rules are
//          covered by claims.test.ts.
//   fit    trainer 'sg' as a process source: the frames of trainingProcess(sg), nothing computed another way (a claim
//          test checks they equal the 'train' source's). It makes no frames for a corpus without pairs, so the lesson
//          shows the input's empty note there (a train source always records frames; see notes, kit issues).
// Neither draws a random number of its own: fit's only randomness is the trainer's (init and the pair of each step).
import type { ProcessDef, ProcessRun, TrainState } from '../../atlas/types/index.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { corpusOf, type CorpusRaw } from './corpus.ts';
import { sg, type SgOpt } from './trainer.ts';

const inner = trainingProcess(sg);
const L = (en: string, zh: string) => ({ en, zh });

export const fit: ProcessDef<CorpusRaw, SgOpt, TrainState> = {
	...inner,
	id: 'fit',
	*run(input, ctx): ProcessRun<TrainState, SgOpt> {
		if (!corpusOf(input).pairs.length) return;
		yield* inner.run(input, ctx);
	}
};

type NoOpt = Record<string, never>;
export const pairs: ProcessDef<CorpusRaw, NoOpt, Record<string, never>> = {
	id: 'pairs',
	kind: 'algorithm',
	title: L('Slide a window over your sentences', '让窗口滑过你的句子'),
	phases: {
		vocab: { label: L('Words and ids', '词和编号'), kind: 'input', narration: { what: L('{frame.V|int} different words, numbered from 0, most frequent first.', '{frame.V|int} 个不同的词，从 0 开始编号，出现次数多的排在前面。') } },
		center: { label: L('One center word', '一个中心词'), kind: 'other', narration: { what: L('This center word gives {frame.k|int} pairs; {frame.n|int} so far.', '这个中心词给出 {frame.k|int} 对；到目前为止共 {frame.n|int} 对。') } }
	},
	schema: {},
	defaults: {},
	*run(input): ProcessRun<Record<string, never>, NoOpt> {
		const c = corpusOf(input);
		if (!c.pairs.length) return;
		const tokens = c.sentences.reduce((a, s) => a + s.length, 0);
		yield { phase: 'vocab', state: {}, metrics: { V: c.words.length, S: c.sentences.length, N: tokens, P: c.pairs.length, dropped: c.dropped.length, window: c.window } };
		const kOf = (len: number, i: number) => Math.min(len - 1, i + c.window) - Math.max(0, i - c.window);
		// ex = 1 on the lesson's example (step 'pairs'): the first center word with the most context words, a full window
		// when the text has one; the scrubber opens there
		const kmax = Math.max(...c.sentences.flatMap((ids) => ids.map((_, i) => kOf(ids.length, i))));
		let n = 0, t = 0, ex = 1;
		for (let s = 0; s < c.sentences.length; s++) {
			const ids = c.sentences[s];
			for (let i = 0; i < ids.length; i++) {
				const k = kOf(ids.length, i);
				n += k;
				yield { phase: 'center', step: t++, state: {}, metrics: { s, i, c: ids[i], k, n, ex: ex && k === kmax ? 1 : 0 } };
				if (k === kmax) ex = 0;
			}
		}
	}
};
