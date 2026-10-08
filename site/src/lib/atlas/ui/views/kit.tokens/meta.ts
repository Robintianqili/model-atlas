// kit.tokens (W1-D2): the kit's TokenChips on the learner's tokens.
import type { ViewMeta } from '../../../types/view.ts';
import type { Props } from '../kit.parts/stage.ts';
import { demoStage } from '../kit.parts/demo.ts';
import { tokensOf } from './tokens.ts';

export function describeTokens(p: Props) {
	const k = tokensOf(p);
	if (!k) return { en: 'Tokens: nothing to show yet.', zh: '词元：还没有可显示的内容。' };
	const list = k.items.map((x) => `“${x.text}”${x.id !== undefined ? ` (${x.id})` : ''}`).join(' ');
	return { en: `The text as ${k.items.length} tokens with their ids: ${list}.`, zh: `文本被切成 ${k.items.length} 个词元及其编号：${list}。` };
}

const meta = {
	key: 'kit.tokens',
	title: { en: 'The text as tokens', zh: '文本切成的词元' },
	kinds: ['trace'],
	accepts: () => false,
	describe: describeTokens,
	demo: async () => demoStage('gpt', 'tok_emb', [3, 0]),
	minHeight: 90
} satisfies ViewMeta;

export default meta;
