<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
	import Term from '$lib/kit/Term.svelte';
</script>

<div class="prose">
	<Lang>
		{#snippet en()}
			<p>
				Large models use the same ideas with far bigger vocabularies, learned from far more and more varied text. Here are four
				real tokenizers, running in your browser, next to yours and TinyGPT’s. Hover over or tap a token to light up the same
				characters in every row.
			</p>
			<ul>
				<li><b>GPT-2</b> (byte-level BPE, 50,257 tokens) learned its merges mostly from English web pages.</li>
				<li>
					<b>BERT</b> uses <Term id="wordpiece" />, an older relative of BPE. It lowercases the text, splits it on spaces and
					punctuation, and then cuts each word into the longest pieces it knows; a piece that continues a word starts with ##.
					If a word cannot be spelled with its pieces, the whole word becomes [UNK] (“unknown”) and its content is lost.
				</li>
				<li><b>GPT-4’s cl100k_base</b> (about 100,000 tokens) covers more languages and a lot of code.</li>
				<li>
					<b>Qwen2.5</b> (about 151,000 tokens) was built to handle Chinese well: common words such as 自然, 语言 and 处理 are single
					tokens.
				</li>
			</ul>
		{/snippet}
		{#snippet zh()}
			<p>大模型用的是同样的思路，只是词表大得多，学习用的文本也多得多、杂得多。下面是四个真实的分词器，在你的浏览器里运行，和你的分词器以及 TinyGPT 的分词器放在一起。把鼠标移到（或点一下）某个词元，每一行里对应的字符都会亮起来。</p>
			<ul>
				<li><b>GPT-2</b>（字节级 BPE，50,257 个词元）的合并规则主要学自英文网页。</li>
				<li><b>BERT</b> 用的是 <Term id="wordpiece" />，它是 BPE 的一个更早的近亲。它先把文本转成小写，按空格和标点切开，再把每个词切成它认识的最长片段；接在词中间的片段以 ## 开头。如果一个词无法用它的片段拼出来，整个词就变成 [UNK]（“未知”），内容也就丢了。</li>
				<li><b>GPT-4 的 cl100k_base</b>（约 100,000 个词元）覆盖了更多语言和大量代码。</li>
				<li><b>Qwen2.5</b>（约 151,000 个词元）专门照顾中文：像“自然”“语言”“处理”这样的常用词都是单个词元。</li>
			</ul>
		{/snippet}
	</Lang>
</div>
