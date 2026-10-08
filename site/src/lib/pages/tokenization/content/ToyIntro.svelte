<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
	import Term from '$lib/kit/Term.svelte';
</script>

<div class="prose">
	<Lang>
		{#snippet en()}
			<p>
				<Term id="bpe">Byte-pair encoding (BPE)</Term> is the recipe behind the tokenizers of GPT-2, GPT-4, Llama, Qwen and TinyGPT.
				It learns its vocabulary from text with four rules:
			</p>
			<ol>
				<li>
					<b>Start from bytes.</b> Any text is stored as a list of <Term id="utf-8" /> bytes, and a byte has only 256 possible
					values. Those 256 values are the first 256 tokens, so any text at all can be written down.
				</li>
				<li>
					<b>Split into pre-tokens.</b> A regular expression cuts the text into word-like pieces. A word keeps the space in front
					of it (“ cat”). Merges happen only inside a piece, never across two.
				</li>
				<li>
					<b>Count and merge.</b> Count every pair of neighbouring tokens in the whole text. The most frequent pair becomes one new
					token, and every place where it occurs is rewritten with the new token.
				</li>
				<li><b>Repeat</b> until the vocabulary has the size you want.</li>
			</ol>
			<p>
				The result is a list of merges in the order they were learned. That list <em>is</em> the tokenizer. Below, you train one
				on your own text, in your browser. Before you press play, predict the first merge on the English story.
			</p>
		{/snippet}
		{#snippet zh()}
			<p>
				<Term id="bpe">字节对编码（BPE）</Term>是 GPT-2、GPT-4、Llama、Qwen 和 TinyGPT 的分词器背后的方法。它用四条规则从文本中学出词表：
			</p>
			<ol>
				<li><b>从字节开始。</b>任何文本都以一串 <Term id="utf-8" /> 字节存储，而一个字节只有 256 种取值。这 256 个值就是最初的 256 个词元，所以任何文本都写得出来。</li>
				<li><b>切成预分词片段。</b>一个正则表达式先把文本切成类似单词的片段。单词带着它前面的空格（“ cat”）。合并只在片段内部进行，绝不跨越两个片段。</li>
				<li><b>计数并合并。</b>统计整段文本里每一对相邻词元出现的次数。出现最多的那一对变成一个新词元，它出现的每个地方都改写成这个新词元。</li>
				<li><b>重复</b>，直到词表达到你想要的大小。</li>
			</ol>
			<p>结果是一张按学习顺序排列的合并列表。这张列表<em>就是</em>分词器。下面你会在浏览器里用自己的文本训练一个。按下播放之前，先预测一下英文故事上的第一次合并。</p>
		{/snippet}
	</Lang>
</div>
