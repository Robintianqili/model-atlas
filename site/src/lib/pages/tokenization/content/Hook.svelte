<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
	import Term from '$lib/kit/Term.svelte';
</script>

<div class="prose">
	<Lang>
		{#snippet en()}
			<p>
				You type a sentence. The model never sees your letters, and it does not see words either. It sees a short list of
				whole numbers. The fixed program that turns text into that list is the <Term id="tokenizer" />: it cuts the text into
				pieces called <Term id="token">tokens</Term> and replaces each piece with its number (its id) in a
				<Term id="vocabulary" />. The tokenizer is built before the model is trained, and it never changes afterwards.
			</p>
			<p>Type anything below. TinyGPT’s tokenizer cuts it up as you type.</p>
		{/snippet}
		{#snippet zh()}
			<p>
				你输入一句话。模型看不到你的字母，也看不到词，它看到的只是一小串整数。把文本变成这串整数的固定程序叫做<Term id="tokenizer" />：它把文本切成叫做<Term id="token">词元</Term>（token）的片段，再把每个片段换成它在<Term id="vocabulary" />里的编号（ID）。分词器在模型训练之前就定好了，之后再也不会改变。
			</p>
			<p>在下面随便输入点什么。TinyGPT 的分词器会边输入边切分。</p>
		{/snippet}
	</Lang>
</div>
