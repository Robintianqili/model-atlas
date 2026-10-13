// View 'positional-encodings/order': the text as typed and shuffled (process 'pe', phase 'order'), each token coloured by
// the attention weight the last token gives it in TinyGPT's layer 0 (the mean of the 4 heads). props.mode 'nopos' shows
// the run with the position table set to zero, 'pos' the model as trained; props.final also shows how much the last token's
// final hidden state (after the last layer) changed.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { tokText } from '../../vocab.ts';
import { arr, demoStage, stateOf } from '../demo.ts';

export function orderOf(p: Pick<StageProps, 'at' | 'props'>) {
	const s = stateOf(p);
	const pos = p.props?.mode === 'pos';
	const m = p.at?.frame?.metrics ?? {};
	const rows = [
		{ key: 'typed', label: { en: 'As typed', zh: '原顺序' }, ids: arr(s.typed), w: arr(pos ? s.w1 : s.w0) },
		{ key: 'shuffled', label: { en: 'Shuffled', zh: '打乱后' }, ids: arr(s.shuffled), w: arr(pos ? s.w1s : s.w0s) }
	];
	return { pos, rows, d: pos ? m.d1 : m.d0, final: !pos && !!p.props?.final, dF: m.dF0, max: Math.max(1e-9, ...rows.flatMap((r) => r.w)) };
}

export function describeOrder(o: ReturnType<typeof orderOf>): I18n {
	const [a, b] = o.rows;
	if (!a.ids.length) return { en: 'The text has not been read yet.', zh: '还没有读入文字。' };
	const list = (r: (typeof o.rows)[number]) => r.ids.map((id, i) => `${tokText(id).trim() || '␣'} ${fmtPct(r.w[i])}`).join(', ');
	const how: I18n = o.pos ? { en: 'with TinyGPT’s position table', zh: '使用 TinyGPT 的位置表' } : { en: 'with the position table set to zero', zh: '位置表设为零' };
	return {
		en: `The last token’s layer-0 attention weights (mean of 4 heads), ${how.en}. As typed: ${list(a)}. Shuffled: ${list(b)}. Its attention output changes by at most ${fmt3(o.d)}.${o.final ? ` Its final hidden state changes by at most ${fmt3(o.dF)}.` : ''}`,
		zh: `最后一个词元在第 0 层的注意力权重（4 个头的平均），${how.zh}。原顺序：${list(a)}。打乱后：${list(b)}。它的注意力输出最多变化 ${fmt3(o.d)}。${o.final ? `它的最终隐藏状态最多变化 ${fmt3(o.dF)}。` : ''}`
	};
}

const meta = {
	key: 'positional-encodings/order',
	title: { en: 'The same tokens in two orders', zh: '同样的词元，两种顺序' },
	kinds: ['process'],
	describe: (p: StageProps) => describeOrder(orderOf(p)),
	demo: () => demoStage('order', { mode: 'nopos' }),
	minHeight: 220
} satisfies ViewMeta;

export default meta;
