// kit.attention (W1-D2): the kit's AttentionMatrix on an [H, Tq, Tk] layer of attention weights, one head at a time.
import type { ViewMeta } from '../../../types/view.ts';
import { keyTexts, stageModel, stageTrace, tokenIds, tokenTexts, type Props } from '../kit.parts/stage.ts';
import { demoStage } from '../kit.parts/demo.ts';
import { fmtPct } from '../../../core/format.ts';
import { attnOf } from './attn.ts';

export function describeAttention(p: Props) {
	const a = attnOf(p);
	if (!a) return { en: 'Attention weights: nothing to show yet.', zh: '注意力权重：还没有可显示的内容。' };
	const ids = tokenIds(stageTrace(p), stageModel(p));
	const toks = tokenTexts(p, a.tq, ids);
	const keys = keyTexts(p, a.tk, ids);
	const r = a.w.subarray(a.query * a.tk, (a.query + 1) * a.tk);
	const top = Array.from(r, (x, j) => ({ x, j })).sort((u, v) => v.x - u.x).slice(0, 3);
	const list = top.map((x) => `“${keys[x.j]}” ${fmtPct(x.x)}`);
	return {
		en: `${a.label.en}, head ${a.head} of ${a.heads} (counting from 0): ${a.tq} queries × ${a.tk} keys${a.causal ? ', future keys masked' : ''}. Query “${toks[a.query]}” looks most at ${list.join(', ')}.`,
		zh: `${a.label.zh ?? a.label.en}，头 ${a.head}（共 ${a.heads} 个，从 0 数起）：${a.tq} 个查询 × ${a.tk} 个键${a.causal ? '，未来的键被遮住' : ''}。查询“${toks[a.query]}”最关注 ${list.join('，')}。`
	};
}

const meta = {
	key: 'kit.attention',
	title: { en: 'Attention weights, one head at a time', zh: '注意力权重，一次一个头' },
	kinds: ['trace'],
	accepts: () => false,
	describe: describeAttention,
	demo: async () => demoStage('gpt', 'weights', [1, 5, 0]),
	minHeight: 320
} satisfies ViewMeta;

export default meta;
