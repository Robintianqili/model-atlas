// contrastive-learning view 'matrix' (props.mode; data.ts reads the numbers):
//   cos | P | G   the nce frame: cosines, row softmax (a toggle shows the column softmax) or the gradient, rows = drawings
//                 (thumbnails), columns = captions (numbered, listed under the table), the true pairs outlined
//   pair          drawing 0's and caption 0's vectors, their products and the sum (the cosine)
//   tau           the loss of the batch over the slider's range of τ (log axis), the current τ marked
//   progress      the progress frame: the cosines at one checkpoint of the training run, on one fixed colour domain
//   simclr        the simclr frame: the 2N distorted views and their cosines, each view's partner outlined
//   shift         drawing 0, a distorted view of it and the drawing moved across, with their cosines
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { nce } from '../../process.ts';
import { START } from '../../adapters/batch/adapter.ts';
import { viewData } from './data.ts';

export function describeMatrix(p: Omit<StageProps, 'ctx'>): I18n {
	const d = viewData(p);
	if (d.kind === 'curve') return { en: `The batch’s InfoNCE loss for τ from 0.01 to 2 (log scale): ${fmt3(d.now)} at τ = ${fmt3(d.tau)}, lowest ${fmt3(d.best.loss)} at τ = ${fmt3(d.best.tau)}; a blind guess scores ${fmt3(d.chance)}.`, zh: `这一批的 InfoNCE 损失随 τ 从 0.01 到 2 的变化（对数刻度）：τ = ${fmt3(d.tau)} 时为 ${fmt3(d.now)}，最低 ${fmt3(d.best.loss)}（τ = ${fmt3(d.best.tau)}）；瞎猜的损失是 ${fmt3(d.chance)}。` };
	if (d.kind === 'pair') return { en: `Drawing 0’s vector and caption 0’s vector, ${d.a.length} numbers each; their products add up to the cosine ${fmt3(d.cos)}.`, zh: `画 0 的向量和描述 0 的向量，各 ${d.a.length} 个数；它们的乘积加起来是余弦 ${fmt3(d.cos)}。` };
	if (d.kind === 'shift') return { en: `Drawing 0 against a distorted view of it: cosine ${fmt3(d.jitter)}; against itself moved across: ${fmt3(d.shift)}.`, zh: `画 0 与它的一个变形视图：余弦 ${fmt3(d.jitter)}；与移到另一边的它自己：${fmt3(d.shift)}。` };
	if (d.kind === 'table' && d.M) {
		const n = Math.round(Math.sqrt(d.M.length));
		const diag = d.marks.map(([i, j]) => fmt3(d.M![i * n + j]));
		return { en: `A ${n}×${n} table of ${d.what.en}; the outlined cells (the true pairs): ${diag.join(', ')}.`, zh: `一张 ${n}×${n} 的表：${d.what.zh}；带框的格子（正确的对）：${diag.join('，')}。` };
	}
	return { en: 'The batch table: nothing to show yet.', zh: '这一批的表：还没有可显示的内容。' };
}

const meta = {
	key: 'contrastive-learning/matrix',
	title: { en: 'A batch’s table of pairs', zh: '一批图文对的表' },
	kinds: ['process'],
	describe: describeMatrix,
	async demo() {
		// made-up unit vectors (illustrative): three pairs in two dimensions
		const tape = record(nce, { N: 3, D: 2, tau: 0.1, I: [1, 0, 0, 1, 0.6, 0.8], T: [1, 0, 0.6, 0.8, 0, 1] }, {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'contrastive-learning', input: START, tapes: { nce: tape } });
		return { rt, source: 'nce', at: atFor(rt, 'nce', 0), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'cos' } };
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
