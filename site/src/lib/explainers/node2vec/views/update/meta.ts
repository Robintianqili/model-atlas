// node2vec view 'update': one skip-gram training step on one (center, context) pair of members, read at the 'update'
// frame of a fine train source (vectors.ts worked()). A table with one row per number of the vectors: the center
// member's vector h, the context member's output vector, their products (which add up to the score), the gradient of h
// and h after the step; above it the score, the share, the loss, and the one number worked out as old − lr × g = new.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { worked } from '../../vectors.ts';

export function describeUpdate(p: Omit<StageProps, 'ctx'>): I18n {
	const w = worked(p.at);
	if (!w) return { en: 'One training step: nothing to show yet.', zh: '一步训练：还没有可显示的内容。' };
	return {
		en: `Training pair: center member ${w.c}, context member ${w.o}. Score ${fmt3(w.u[w.o])}, share ${fmtPct(w.p[w.o])}, loss ${fmt3(w.loss, 4)}. Row k = ${w.k} of h (counting from 0): ${fmt3(w.old)} − ${w.lr} × (${fmt3(w.g)}) = ${fmt3(w.new)}. After the step: score ${fmt3(w.u1)}, loss ${fmt3(w.loss1, 4)}.`,
		zh: `训练对：中心成员 ${w.c}，上下文成员 ${w.o}。分数 ${fmt3(w.u[w.o])}，份额 ${fmtPct(w.p[w.o])}，损失 ${fmt3(w.loss, 4)}。h 中 k = ${w.k} 的那个数（从 0 数起）：${fmt3(w.old)} − ${w.lr} × (${fmt3(w.g)}) = ${fmt3(w.new)}。这一步之后：分数 ${fmt3(w.u1)}，损失 ${fmt3(w.loss1, 4)}。`
	};
}

const meta = {
	key: 'node2vec/update',
	title: { en: 'One skip-gram step, number by number', zh: '一步 skip-gram，逐个数展开' },
	kinds: ['train'],
	describe: describeUpdate,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { sg }, { START }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts'), import('../../walks.ts')]);
		const tape = record(trainingProcess(sg), START, { steps: 1 }, { seed: 1, detail: 'fine' });
		const rt = fakeRuntime({ slug: 'node2vec', input: START, tapes: { one: tape } });
		return { rt, source: 'one', at: atFor(rt, 'one', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
