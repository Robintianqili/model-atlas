<!--
  The input sheet (redesign spec §4.4 InputSheet; the stage header's chip and #input open it): the lesson's two texts,
  the one the current step uses first. The training text trains BPE (lesson steps 1–5, Explore Part 1); the sentence
  is the new text (the cover, lesson steps 6–7, Explore Parts 2–3). Every step updates as you type.
-->
<script lang="ts">
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import { getLessonApp } from '$lib/kit/lesson/app';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { PRESETS } from './presets';
	import { MAX_SENTENCE, type TokPage } from './page.svelte';

	let { P }: { P: TokPage } = $props();
	const app = getLessonApp();
	const textFirst = $derived(!!app && !app.pos.cover && app.pos.at < 5);
	const K = $derived(P.trace && P.trained === P.trainText ? P.trace.merges.length : null);
</script>

{#snippet trainBox()}
	<section class="box" aria-labelledby="tok-in-text">
		<h3 id="tok-in-text">{t({ en: 'Your text', zh: '你的文本' })} <span class="sub">{t({ en: 'trains your tokenizer (steps 1–5, Explore Part 1)', zh: '用来训练你的分词器（第 1–5 步、探索第 1 部分）' })}</span></h3>
		<TextInputBar
			bind:value={P.trainText}
			multiline
			rows={5}
			maxLength={2000}
			label={{ en: 'Training text (type or paste your own)', zh: '训练文本（可输入或粘贴你自己的）' }}
			presets={PRESETS.train.map((p) => ({ text: p.text, label: p.label }))}
		/>
		<p class="meta num" aria-live="polite">
			{fmtInt(new TextEncoder().encode(P.trainText).length)} {t({ en: 'bytes', zh: '字节' })}{#if P.training}{' · '}{t({ en: 'training…', zh: '训练中…' })}{:else if K !== null}{' · '}{fmtInt(K)} {t({ en: 'merges learned', zh: '次合并' })}{/if}
		</p>
	</section>
{/snippet}
{#snippet sentenceBox()}
	<section class="box" aria-labelledby="tok-in-sent">
		<h3 id="tok-in-sent">{t({ en: 'Your sentence', zh: '你的句子' })} <span class="sub">{t({ en: 'new text for the tokenizers (the cover, steps 6–7, Explore Parts 2–3)', zh: '交给分词器的新文本（封面、第 6–7 步、探索第 2–3 部分）' })}</span></h3>
		<TextInputBar bind:value={P.sentence} maxLength={MAX_SENTENCE} label={{ en: 'Your sentence', zh: '你的句子' }} presets={PRESETS.sentence} />
	</section>
{/snippet}

<div class="sheet">
	{#if textFirst}{@render trainBox()}{@render sentenceBox()}{:else}{@render sentenceBox()}{@render trainBox()}{/if}
	<p class="hint">{t({ en: 'The step you are on updates as you type. Your text never leaves your device.', zh: '你一边输入，当前这一步就一边更新。你的文本不会离开你的设备。' })}</p>
</div>

<style>
	.sheet {
		display: grid;
		gap: var(--s5);
	}
	.box {
		display: grid;
		gap: var(--s2);
		min-width: 0;
	}
	h3 {
		margin: 0;
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.sub {
		display: block;
		color: var(--muted);
		font-size: var(--t-sm);
		font-weight: var(--w-body);
	}
	.meta {
		margin: 0;
		color: var(--muted);
		font-size: var(--t-xs);
	}
	.hint {
		margin: 0;
		color: var(--muted);
		font-size: var(--t-sm);
	}
</style>
