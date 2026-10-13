<!--
  PITFALLS: two misconception checks with live evidence, then a short list of other traps.
-->
<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import Term from '$lib/kit/Term.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import type { BPETokenizer } from '$lib/models/bpe';
	import Misconception from './Misconception.svelte';
	import Tok from './Tok.svelte';
	import type { RealResult } from './realtok';
	import type { LangSummary } from './types';
	import { PRESETS } from './presets';
	import { fmtBytes } from '$lib/kit/format';
	import { REAL, REAL_BY_ID, REAL_TOTAL_BYTES, ROW_COLOR, ROW_NAME } from './tokenizers';
	import { real, conn, loadAll, loadReal } from './state.svelte';

	interface Props {
		tiny: BPETokenizer;
		summary: LangSummary[];
	}
	let { tiny, summary }: Props = $props();

	const EXAMPLES = ['unbelievable', ' unbelievable', 'the', ' the', 'The', 'HELLO', ' strawberry', '我', '12345678'];
	let gpt2 = $state.raw<RealResult[] | null>(null);

	$effect(() => {
		const c = conn.client;
		if (!c || real.status.gpt2 !== 'ready') return;
		c.encodeMany(EXAMPLES, ['gpt2'])
			.then((r) => (gpt2 = r.map((x) => x.gpt2)))
			.catch(() => {});
	});
	const langName = (li: number) => t(PRESETS.languages[li].name);
	// a reader who jumps here from the table of contents never scrolled past Part 3, which loads the files
	const allReal = $derived(REAL.every((r) => real.status[r.id] === 'ready'));
</script>

<div class="pf">
	<Misconception
		statement={{ en: 'A token is a word.', zh: '一个词元就是一个词。' }}
		truth={false}
		verdict={{ en: 'A token is whatever piece the tokenizer learned: a word with its space, part of a word, one letter, or even one byte of a character.', zh: '词元是分词器学到的任何片段：带空格的词、词的一部分、一个字母，甚至一个字符的一个字节。' }}
	>
		{#snippet evidence()}
			<div class="ev">
				<div class="tscroll" use:keyScroll><table>
					<thead>
						<tr>
							<th scope="col">{t({ en: 'Text', zh: '文本' })}</th>
							<th scope="col"><span class="sw" style:background="var(--cat-{ROW_COLOR.tinygpt})"></span>TinyGPT</th>
							<th scope="col"><span class="sw" style:background="var(--cat-{ROW_COLOR.gpt2})"></span>GPT-2</th>
						</tr>
					</thead>
					<tbody>
						{#each EXAMPLES as ex, i (i)}
							<tr>
								<th scope="row"><code>“{ex}”</code></th>
								<td>
									{#each tiny.encode(ex, { allowSpecial: false }) as id, j (j)}<Tok text={tiny.tokenToDisplay(id)} small /> {/each}
									<span class="ids num">{tiny.encode(ex, { allowSpecial: false }).join(' ')}</span>
								</td>
								<td>
									{#if gpt2?.[i]}
										{#each gpt2[i].tokens as tk, j (j)}<Tok text={tk.text} small /> {/each}
										<span class="ids num">{gpt2[i].tokens.map((x) => x.id).join(' ')}</span>
									{:else}
										<span class="muted">{t({ en: 'loads with Part 3', zh: '随第 3 部分加载' })}</span>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table></div>
				{#if real.status.gpt2 === 'idle' || real.status.gpt2 === 'error'}
					<button type="button" class="btn ghost load" onclick={() => loadReal('gpt2')}>{t({ en: 'Load GPT-2', zh: '加载 GPT-2' })} ({fmtBytes(REAL_BY_ID.gpt2.bytes)})</button>
				{/if}
				<ProvenanceBadge mode="live" compact />
				<Lang>
					{#snippet en()}
						<p>
							In GPT-2, “the”, “ the” and “The” are three different tokens with three different ids (TinyGPT even cuts “the” into
							t + he), so the model has to learn separately that they are the same word. And the model cannot look inside a token: in GPT-2, “ strawberry” arrives as a single id
							(41236), not as ten letters, which is one reason language models are bad at counting letters.
						</p>
					{/snippet}
					{#snippet zh()}
						<p>在 GPT-2 里，“the”“ the”“The”是三个不同的词元，ID 也各不相同（TinyGPT 甚至把“the”切成 t + he），所以模型得分别学会它们是同一个词。而且模型看不到词元内部：在 GPT-2 里，“ strawberry”到达模型时只是一个 ID（41236），而不是十个字母，这也是语言模型数不清字母的原因之一。</p>
					{/snippet}
				</Lang>
			</div>
		{/snippet}
	</Misconception>

	<Misconception
		statement={{ en: 'The same meaning costs the same number of tokens in every language.', zh: '同样的意思，在每种语言里需要的词元数都一样。' }}
		truth={false}
		verdict={{ en: 'The cost depends on how much of that language the tokenizer’s training text contained.', zh: '代价取决于分词器的训练文本里有多少这种语言。' }}
	>
		{#snippet evidence()}
			<div class="ev">
				{#if summary.length}
					<ul class="sum">
						{#each summary as s (s.col)}
							<li>
								<span class="sw" style:background="var(--cat-{ROW_COLOR[s.col]})"></span><b>{t(ROW_NAME[s.col])}</b>:
								{#if i18n.lang === 'zh'}英语 <b class="num">{s.en}</b>，最便宜的是{langName(s.min.li)} <b class="num">{s.min.v}</b>，最贵的是{langName(s.max.li)} <b class="num">{s.max.v}</b>（英语的 {(s.max.v / s.en).toFixed(1)} 倍）{:else}English <b class="num">{s.en}</b>, cheapest {langName(s.min.li)} <b class="num">{s.min.v}</b>, most expensive {langName(s.max.li)} <b class="num">{s.max.v}</b> <span class="muted">({(s.max.v / s.en).toFixed(1)}× English)</span>{/if}

							</li>
						{/each}
					</ul>
					<ProvenanceBadge mode="live" compact />
				{/if}
				{#if !allReal}
					<button type="button" class="btn ghost load" onclick={loadAll}>{t({ en: 'Load the real tokenizers', zh: '加载真实分词器' })} ({fmtBytes(REAL_TOTAL_BYTES)})</button>
				{/if}
				<Lang>
					{#snippet en()}
						<p>
							Same sentence, same meaning, very different bills. The cheapest language is not always English: it is whichever
							language the tokenizer saw most. Qwen2.5, trained with a lot of Chinese, writes the Chinese sentence in fewer tokens
							than the English one.
						</p>
					{/snippet}
					{#snippet zh()}
						<p>同一句话、同样的意思，账单却差别很大。最便宜的语言也不一定是英语，而是分词器见得最多的那种语言。用大量中文训练的 Qwen2.5，写中文那句话用的词元比写英文那句还少。</p>
					{/snippet}
				</Lang>
			</div>
		{/snippet}
	</Misconception>

	<div class="prose more">
		<Lang>
			{#snippet en()}
				<h3>Other traps</h3>
				<ul>
					<li>
						<b>Numbers are cut in odd ways.</b> GPT-2 turns 12345678 into 123 | 45 | 678, GPT-4 into 123 | 456 | 78, and Qwen2.5 into
						eight single digits. The same digit can belong to different tokens in different numbers, which is one reason arithmetic
						is hard for language models.
					</li>
					<li><b>Characters and words do not predict the token count.</b> To know the cost, run the model’s own tokenizer.</li>
					<li>
						<b>A tokenizer belongs to its model.</b> The model’s <Term id="embedding" /> table has one row per token id. Give it ids
						from another tokenizer and every id points at the wrong row.
					</li>
					<li>
						<b>Byte fallback is not understanding.</b> TinyGPT can encode Chinese because every text is bytes, but it never saw those
						byte ids during training, so it cannot use them.
					</li>
					<li>
						<b>A bigger vocabulary is not free.</b> It shortens sequences, but every token adds a row to the embedding table and a
						score to the output layer: 1,024 × 76 = 77,824 embedding numbers in TinyGPT, 50,257 × 768 ≈ 38.6 million in GPT-2.
					</li>
				</ul>
			{/snippet}
			{#snippet zh()}
				<h3>其他陷阱</h3>
				<ul>
					<li><b>数字被切得很怪。</b>GPT-2 把 12345678 切成 123 | 45 | 678，GPT-4 切成 123 | 456 | 78，Qwen2.5 则切成 8 个单独的数字。同一个数字在不同的数里可能属于不同的词元，这是语言模型做算术困难的原因之一。</li>
					<li><b>字符数和词数推算不出词元数。</b>想知道代价，就用模型自己的分词器跑一遍。</li>
					<li><b>分词器是和模型绑定的。</b>模型的<Term id="embedding" />表里每个词元 ID 对应一行。换用另一个分词器的 ID，每个 ID 都会指向错误的那一行。</li>
					<li><b>能按字节编码不等于能理解。</b>因为任何文本都是字节，TinyGPT 可以给中文编码，但它在训练中从没见过这些字节 ID，所以用不上它们。</li>
					<li><b>更大的词表是有代价的。</b>它能缩短序列，但每个词元都要在嵌入表里加一行、在输出层里加一个分数：TinyGPT 是 1,024 × 76 = 77,824 个嵌入数值，GPT-2 是 50,257 × 768 ≈ 3,860 万个。</li>
				</ul>
			{/snippet}
		</Lang>
	</div>
</div>

<style>
	.pf {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
		max-width: none;
	}
	.ev {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.ev p {
		margin: 0;
	}
	.tscroll {
		overflow-x: auto;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		width: 100%;
	}
	th,
	td {
		padding: 4px 6px;
		border-bottom: 1px solid var(--rule);
		text-align: left;
		vertical-align: top;
	}
	thead th {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	th[scope='row'] {
		font-weight: 400;
		white-space: pre;
	}
	td {
		line-height: 1.9;
	}
	.ids {
		display: block;
		font-size: var(--t-xs);
		color: var(--muted);
		line-height: 1.3;
	}
	.sw {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 2px;
		margin-right: 4px;
	}
	.sum {
		margin: 0;
		padding: 0;
		list-style: none;
		display: grid;
		gap: 3px;
		font-size: var(--fs-sm);
	}
	.more {
		max-width: var(--measure);
	}
	.load {
		justify-self: start;
	}
</style>
