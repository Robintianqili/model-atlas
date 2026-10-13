<!--
  The lesson's cover stage (redesign spec §5.6, answer first): the learner's sentence as TinyGPT reads it, live:
  characters → UTF-8 bytes → token ids, the tokens as chips with their ids, and the list of numbers the model gets.
  Explore's "What does a model read?" section shows the same with the sentence editor (ReadsStrip).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import ReadsStrip from './ReadsStrip.svelte';
	import type { TokPage } from './page.svelte';

	let { P }: { P: TokPage } = $props();
</script>

<figure class="cover">
	<blockquote class="sent" lang={/[一-鿿]/.test(P.sentence) ? 'zh-CN' : undefined}>{P.sentence || t({ en: '(empty)', zh: '（空）' })}</blockquote>
	<ReadsStrip {P} />
	<figcaption>
		{t({
			en: 'Your sentence as TinyGPT reads it, computed in your browser. In seven steps the lesson builds a tokenizer like this one from bytes, on a text of your own.',
			zh: '这就是 TinyGPT 读到的你的句子，在你的浏览器里现场算出。课程用七步，在你自己的文本上，从字节开始搭出一个这样的分词器。'
		})}
	</figcaption>
</figure>

<style>
	.cover {
		display: grid;
		gap: var(--s4);
		margin: 0;
		min-width: 0;
	}
	.sent {
		margin: 0;
		padding: var(--s3) var(--s4);
		border-left: 3px solid var(--ink);
		background: var(--panel);
		border-radius: 0 var(--r3) var(--r3) 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		overflow-wrap: anywhere;
	}
	figcaption {
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		max-width: var(--measure);
	}
</style>
