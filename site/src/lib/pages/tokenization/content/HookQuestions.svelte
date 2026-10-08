<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
</script>

<div class="prose">
	<Lang>
		{#snippet en()}
			<p>Three questions run through this page:</p>
			<ol>
				<li><b>Who decides where the cuts go?</b> In Part 1 you build a tokenizer yourself, merge by merge.</li>
				<li><b>What does a real tokenizer do with your text, byte by byte?</b> Part 2 opens TinyGPT’s.</li>
				<li>
					<b>Why does 我爱自然语言处理 (“I love natural language processing”) cost 5 tokens in one model and 24 in another?</b> Part 3 runs four real tokenizers
					side by side, and the comparison after it measures ten languages.
				</li>
			</ol>
		{/snippet}
		{#snippet zh()}
			<p>这一页围绕三个问题展开：</p>
			<ol>
				<li><b>切在哪里，由谁决定？</b>第 1 部分里，你会一次合并一次合并地亲手训练一个分词器。</li>
				<li><b>真实的分词器是怎样一个字节一个字节地处理你的文本的？</b>第 2 部分打开 TinyGPT 的分词器。</li>
				<li><b>为什么“我爱自然语言处理”在一个模型里只要 5 个词元，在另一个模型里却要 24 个？</b>第 3 部分把四个真实分词器并排运行，之后的对比部分再量一量十种语言。</li>
			</ol>
		{/snippet}
	</Lang>
</div>
