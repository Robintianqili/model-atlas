<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
	import Term from '$lib/kit/Term.svelte';

	let { part }: { part: 'intro' | 'after' } = $props();
</script>

<div class="prose">
	{#if part === 'intro'}
		<Lang>
			{#snippet en()}
				<p>
					Tokens are the unit you pay in. Model APIs charge per token, the context window (how much text a model can read at
					once) is counted in tokens, and every token costs the model one more step of computation. A tokenizer that learned its
					merges mostly from English gives English the longest, cheapest pieces.
				</p>
				<p>
					The table counts one sentence, “The weather is nice today, so let’s go to the park.”, written in ten languages. Every
					number is computed live by the tokenizer in its column. Select a cell to see the cut.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>词元就是你付费的单位。模型 API 按词元收费，上下文窗口（模型一次能读多少文本）按词元计算，模型每多处理一个词元就要多做一步计算。一个主要从英文学到合并规则的分词器，会把最长、最划算的片段留给英语。</p>
				<p>下表统计同一句话“今天天气很好，我们去公园吧。”用十种语言写出来时的词元数。每个数字都由所在列的分词器实时计算。选中一个格子可以看到它是怎么切的。</p>
			{/snippet}
		</Lang>
	{:else}
		<Lang>
			{#snippet en()}
				<p>
					In GPT-2, the same meaning in Hindi or Russian costs several times as many tokens as in English. Newer tokenizers
					(cl100k, Qwen2.5) shrink the gap because they learned merges from more languages, but the gap does not disappear. BERT’s
					numbers can look small for some languages only because whole words became [UNK]: short, but the meaning is gone.
				</p>
				<p>
					Your own BPE shows the cause directly. Train it on the Chinese story in Part 1: the Chinese row gets cheaper, and the
					English row falls back towards bytes. The training text decides which languages get the cheap tokens. A bigger
					vocabulary helps every language, but it is not free: every token needs its own row in the model’s
					<Term id="embedding" /> table and its own output score.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>在 GPT-2 里，同样的意思用印地语或俄语写，词元数是英语的好几倍。较新的分词器（cl100k、Qwen2.5）从更多语言里学习合并规则，差距变小了，但并没有消失。BERT 在某些语言上的数字看起来很小，只是因为整个词都变成了 [UNK]：短是短了，意思却没了。</p>
				<p>你自己的 BPE 能直接说明原因。在第 1 部分用中文故事训练它：中文那一行变便宜了，英文那一行则退回接近字节。训练文本决定了哪些语言能得到便宜的词元。更大的词表对每种语言都有帮助，但并不是免费的：每个词元都要在模型的<Term id="embedding" />表里占一行，在输出层里占一个分数。</p>
			{/snippet}
		</Lang>
	{/if}
</div>
