<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
</script>

<div class="prose">
	<Lang>
		{#snippet en()}
			<p>
				TinyGPT’s tokenizer was trained exactly like your toy, but on 100 MB of English children’s stories (TinyStories), and it
				stopped after 767 merges: 256 bytes + 767 merges + 1 special token (&lt;|endoftext|&gt;, which marks the end of a story) =
				1,024 tokens.
			</p>
			<p>
				To cut new text, it does not count anything. It replays its merge list: inside each pre-token it repeatedly merges the
				neighbouring pair that was learned earliest (the lowest <em>rank</em>), until no learned pair is left. The steps below
				use the same sentence as above.
			</p>
		{/snippet}
		{#snippet zh()}
			<p>TinyGPT 的分词器和你的玩具用的是完全相同的训练方法，只是训练文本换成了 100 MB 的英文儿童故事（TinyStories），并在 767 次合并后停止：256 个字节 + 767 次合并 + 1 个特殊词元（&lt;|endoftext|&gt;，标记一个故事的结束）= 1,024 个词元。</p>
			<p>切分新文本时，它不再计数，而是重放合并列表：在每个预分词片段内部，反复合并最早学到（<em>排名</em>最小）的那一对相邻词元，直到没有学过的对为止。下面的步骤用的就是上面那句话。</p>
		{/snippet}
	</Lang>
</div>
