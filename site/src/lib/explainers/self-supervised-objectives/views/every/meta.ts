// self-supervised-objectives view 'every': the causal objective at every position of the learner's text at once. For
// each token from the second one on, TinyGPT's probability for the token that really comes next, read from its 'probs'
// row of the token before (one bar per token, in text order, bars to scale from 0 to 1), and the mean −log p: the
// causal loss on this text.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct, visibleToken } from '../../../../atlas/core/format.ts';
import { vocab } from '../../adapters/masked-text/adapter.ts';
import { nextProbs, say } from '../../text.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;

export function everyOf(p: P) {
	const N = nextProbs(p.rt);
	const items = N.ps.map((q) => ({ label: `${q.i}: ${visibleToken(vocab(q.id))}`, p: q.p, id: q.i, color: q.i === N.worst.i ? 'var(--accent)' : undefined }));
	const caption: I18n = N.targets
		? { en: `${N.targets} answers from one text. Mean −log p, the causal loss: ${fmt3(N.loss)} nats. Least expected (accent colour): “${say(N.worst.id)}”, token ${N.worst.i}, ${fmtPct(N.worst.p)}.`, zh: `一段文字给出 ${N.targets} 个答案。−log p 的平均值，也就是因果损失：${fmt3(N.loss)} 奈特。最出乎意料的（强调色）：第 ${N.worst.i} 个词元“${say(N.worst.id)}”，${fmtPct(N.worst.p)}。` }
		: { en: 'TinyGPT needs at least two tokens: the first one has nothing before it.', zh: 'TinyGPT 至少需要两个词元：第一个词元前面什么都没有。' };
	return { items, caption, n: N.targets };
}

const meta = {
	key: 'self-supervised-objectives/every',
	title: { en: 'Every token is an answer', zh: '每个词元都是一个答案' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const E = everyOf(p);
		const list = E.items.map((x) => `${x.label} ${fmtPct(x.p)}`).join('; ');
		return { en: `TinyGPT's probability for each real next token: ${list}. ${E.caption.en}`, zh: `TinyGPT 给每个真实下一个词元的概率：${list}。${E.caption.zh}` };
	},
	demo: async () => demoProps(),
	minHeight: 260
} satisfies ViewMeta;

export default meta;
