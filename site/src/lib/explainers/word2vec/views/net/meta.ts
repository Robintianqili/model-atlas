// word2vec view 'net': one training step on one (center, context) pair, as a table with one row per word of the
// vocabulary: its one-hot input x, its row of W_in, its output vector (a column of W_out), its score u and its share p.
// props.show picks the columns of the lesson step: onehot | row | scores | softmax | loss | update (the change of both
// tables). Reads a fine frame of source `one` (vectors.ts worked()).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { worked, workedWeight } from '../../vectors.ts';

export const SHOWS = ['onehot', 'row', 'scores', 'softmax', 'loss', 'update'] as const;
const vec = (v: number[]) => `(${v.map((x) => fmt3(x)).join(', ')})`;

export function describeNet(p: Omit<StageProps, 'ctx'>): I18n {
	const w = worked(p.rt.input as never, p.at);
	if (!w) return { en: 'One training step: nothing to show yet.', zh: '一步训练：还没有可显示的内容。' };
	const [c, o] = [w.words[w.c], w.words[w.o]], ww = workedWeight(w);
	const head = { en: `Training pair: center “${c}” (id ${w.c}), context “${o}” (id ${w.o}), ${w.V} words. `, zh: `训练对：中心词“${c}”（编号 ${w.c}），上下文词“${o}”（编号 ${w.o}），共 ${w.V} 个词。` };
	const show = String(p.props.show ?? 'row');
	const body: Record<string, I18n> = {
		onehot: { en: `The input x is ${w.V} numbers, 1 at id ${w.c} and 0 elsewhere.`, zh: `输入 x 是 ${w.V} 个数，编号 ${w.c} 处为 1，其余为 0。` },
		row: { en: `h is row ${w.c} of W_in: ${vec(w.h)}.`, zh: `h 是 W_in 的第 ${w.c} 行：${vec(w.h)}。` },
		scores: { en: `Each word's score is h · its output vector; “${o}” scores ${fmt3(w.u[w.o])}.`, zh: `每个词的分数是 h 与它的输出向量的点积；“${o}”的分数是 ${fmt3(w.u[w.o])}。` },
		softmax: { en: `Softmax shares: “${o}” gets ${fmtPct(w.p[w.o])}; the shares range from ${fmtPct(Math.min(...w.p))} to ${fmtPct(Math.max(...w.p))}.`, zh: `softmax 份额：“${o}”得到 ${fmtPct(w.p[w.o])}；各份额在 ${fmtPct(Math.min(...w.p))} 到 ${fmtPct(Math.max(...w.p))} 之间。` },
		loss: { en: `Loss = −ln ${fmt3(w.p[w.o], 4)} = ${fmt3(w.loss, 4)}.`, zh: `损失 = −ln ${fmt3(w.p[w.o], 4)} = ${fmt3(w.loss, 4)}。` },
		update: {
			en: `After the update (learning rate ${fmt3(w.lr)}), number ${ww?.k} of the row of “${c}” is ${fmt3(ww?.new ?? NaN)} (was ${fmt3(ww?.old ?? NaN)}) and the score of “${o}” is ${fmt3(w.u1 ?? NaN)} (was ${fmt3(w.u[w.o])}).`,
			zh: `更新之后（学习率 ${fmt3(w.lr)}），“${c}”那一行第 ${ww?.k} 个数是 ${fmt3(ww?.new ?? NaN)}（原来是 ${fmt3(ww?.old ?? NaN)}），“${o}”的分数是 ${fmt3(w.u1 ?? NaN)}（原来是 ${fmt3(w.u[w.o])}）。`
		}
	};
	const b = body[show] ?? body.row;
	return { en: head.en + b.en, zh: head.zh + b.zh };
}

const meta = {
	key: 'word2vec/net',
	title: { en: 'One training step, word by word', zh: '一步训练，逐词展开' },
	kinds: ['train', 'process'],
	describe: describeNet,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { fit }, { PETS }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts'), import('../../corpus.ts')]);
		const tape = record(fit, PETS, { steps: 1 }, { seed: 11, detail: 'fine' });
		const rt = fakeRuntime({ slug: 'word2vec', input: PETS, tapes: { one: tape } });
		return { rt, source: 'one', at: atFor(rt, 'one', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'update' } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
