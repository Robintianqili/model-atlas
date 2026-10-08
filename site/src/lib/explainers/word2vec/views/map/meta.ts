// word2vec view 'map': the word vectors (rows of W_in) of a training tape, flattened to 2-D by PCA. The axes are the
// PCA of the tape's last frame, so the dots move on fixed axes while the frames play. props.focus: a word id to ring,
// with a line to its nearest word (cosine similarity of the full vectors); props.list: every word's nearest word;
// props.vs: two sources whose last frames are compared (a word whose nearest word differs is marked).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { corpusOf } from '../../corpus.ts';
import { finalPca, neighbours, project, wordVectors } from '../../vectors.ts';

export function mapPic(p: Omit<StageProps, 'ctx'>) {
	const raw = p.rt.input as never, tape = p.at.tape;
	const vs = wordVectors(raw, p.at.state), pca = finalPca(raw, tape);
	if (vs.length < 2 || !pca) return null;
	const words = corpusOf(raw).words, nn = vs.map((_, i) => neighbours(vs, i)[0]);
	const last = (s: string) => { const t = p.rt.sources[s]?.tape; return t?.length ? wordVectors(raw, t.state(t.length - 1)) : []; };
	const vsNames = (p.props.vs as string[] | undefined) ?? [];
	const pair = vsNames.map(last);
	const cmp = pair.length === 2 && pair.every((x) => x.length === vs.length);
	const moved = new Set(cmp ? vs.map((_, i) => i).filter((i) => neighbours(pair[0], i)[0].j !== neighbours(pair[1], i)[0].j) : []);
	const focus = typeof p.props.focus === 'number' && p.props.focus < vs.length ? (p.props.focus as number) : -1;
	const f = p.at.frame;
	return { words, xy: project(vs, pca), final: project(wordVectors(raw, tape!.state(tape!.length - 1)), pca), share: pca.share, nn, moved, cmp, focus, step: f?.step ?? 0, loss: f?.metrics?.avgLoss ?? NaN, list: !!p.props.list };
}

export function describeMap(p: Omit<StageProps, 'ctx'>): I18n {
	const m = mapPic(p);
	if (!m) return { en: 'The word map: nothing to show yet.', zh: '词向量图：还没有可显示的内容。' };
	const pairs = m.words.map((w, i) => `${w} → ${m.words[m.nn[i].j]}`).join(', ');
	return {
		en: `${m.words.length} word vectors after ${m.step} training steps, drawn by PCA (the map keeps ${fmtPct(m.share)} of their spread). Nearest word of each, by cosine similarity: ${pairs}.${Number.isFinite(m.loss) ? ` Average loss ${fmt3(m.loss)}.` : ''}`,
		zh: `训练 ${m.step} 步后的 ${m.words.length} 个词向量，用 PCA 画成平面图（保留了 ${fmtPct(m.share)} 的分布）。按余弦相似度，每个词最近的词：${pairs}。${Number.isFinite(m.loss) ? `平均损失 ${fmt3(m.loss)}。` : ''}`
	};
}

const meta = {
	key: 'word2vec/map',
	title: { en: 'The word vectors on a map', zh: '词向量图' },
	kinds: ['train', 'process'],
	describe: describeMap,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { fit }, { PETS }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts'), import('../../corpus.ts')]);
		const tape = record(fit, PETS, { steps: 600 }, { seed: 11 });
		const rt = fakeRuntime({ slug: 'word2vec', input: PETS, tapes: { run: tape } });
		return { rt, source: 'run', at: atFor(rt, 'run', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { focus: 4, list: true } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
