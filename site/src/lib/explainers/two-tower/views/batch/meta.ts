// two-tower view 'batch' (kind process): the inbatch run as a table (View.svelte); props.show 'S' | 'P' | 'G' picks the
// reading shown first. fieldAt() reads a field from the frame of its phase, if the step's cursor has reached it.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { demoStage } from '../../demo.ts';

export const SHOW = { S: { en: 'Scores', zh: '分数' }, P: { en: 'Shares (softmax of each row)', zh: '份额（每行的 softmax）' }, G: { en: 'Gradient of the loss', zh: '损失的梯度' } } satisfies Record<string, I18n>;
const PHASE = { S: 'scores', P: 'softmax', G: 'grad' } as const;
export function fieldAt(p: Omit<StageProps, 'ctx'>, k: keyof typeof SHOW): Tensor | null {
	const tp = p.at.tape, i = tp ? tp.find({ phase: PHASE[k] }) : -1;
	return tp && i >= 0 && i <= p.at.index ? ((tp.state(i) as Record<string, Tensor>)[k] ?? null) : null;
}

const meta = {
	key: 'two-tower/batch',
	title: { en: 'A batch scored against itself', zh: '一批数据自己给自己打分' },
	kinds: ['process'],
	describe: (p: StageProps): I18n => {
		const k = ((p.props.show as keyof typeof SHOW) ?? 'P'), M = fieldAt(p, k);
		if (!M) return { en: 'The batch table: nothing to show yet.', zh: '这一批的表：还没有可显示的内容。' };
		const B = M.shape[0], d = Array.from({ length: B }, (_, i) => fmt3(M.data[i * B + i])).join(', ');
		return { en: `A ${B}×${B} table of ${SHOW[k].en.toLowerCase()}: row i is viewer i, column j the movie viewer j liked; the diagonal (true pairs): ${d}.`, zh: `一张 ${B}×${B} 的${SHOW[k].zh}表：第 i 行是观众 i，第 j 列是观众 j 喜欢的电影；对角线（正确的配对）：${d}。` };
	},
	demo: () => demoStage({ show: 'G' }, 'inbatch'),
	minHeight: 320
} satisfies ViewMeta;

export default meta;
