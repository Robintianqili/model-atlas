<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
</script>

<div class="prose">
	<Lang>
		{#snippet en()}
			<h3>What the toy shows</h3>
			<ul>
				<li>
					<b>Frequent pieces win first.</b> On the English story the first merges are “ t” (24 times), “he” (21) and then
					“ the” (16): a space plus a common letter, then the most common word. The space is part of the token because the
					pre-tokenizer keeps it with the word.
				</li>
				<li>
					<b>Chinese starts below the character.</b> On the Chinese story the first merge is the byte pair EF BC, which is not a
					character at all: it is the shared start of the full-width punctuation marks “，”, “：” and “！”. BPE only sees bytes;
					characters appear later, when their bytes happen to be merged.
				</li>
				<li>
					<b>Each merge saves as many tokens as the pair occurred.</b> That is why the curve drops steeply at first and then
					flattens: later pairs are rarer.
				</li>
				<li>
					<b>Too many merges memorize.</b> Keep going and every pre-token becomes a single token. From the first merge with count 1
					on, the tokenizer is just memorizing words that appear once. Real tokenizers learn from gigabytes of text and stop long
					before that, so their tokens are pieces that are common in general, not in one paragraph.
				</li>
			</ul>
		{/snippet}
		{#snippet zh()}
			<h3>这个玩具说明了什么</h3>
			<ul>
				<li><b>常见的片段先被合并。</b>在英文故事上，最先合并的是“ t”（24 次）、“he”（21 次），然后是“ the”（16 次）：先是空格加一个常见字母，然后就是最常见的词。空格之所以在词元里，是因为预分词把空格和后面的词放在了一起。</li>
				<li><b>中文从比字更小的单位开始。</b>在中文故事上，第一次合并的是字节对 EF BC，它根本不是一个字：它是全角标点“，”“：”“！”共同的开头。BPE 只看得到字节；只有当一个字的几个字节恰好被合并时，这个字才会出现在词表里。</li>
				<li><b>每次合并省下的词元数，等于这一对出现的次数。</b>所以曲线一开始下降得很快，后来变平：越往后，剩下的对越少见。</li>
				<li><b>合并太多就是在死记。</b>一直合并下去，每个预分词片段都会变成一个词元。从第一次计数为 1 的合并开始，分词器只是在背下只出现一次的词。真实的分词器用动辄数 GB 的文本训练，并且远在这之前就停下，所以它们的词元是在各种文本里都常见的片段，而不是只在某一段话里常见。</li>
			</ul>
		{/snippet}
	</Lang>
</div>
