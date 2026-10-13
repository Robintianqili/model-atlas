<!--
  Explore (redesign spec §5.9; a lazy panel of the next-token-prediction page, #explore): every part of the lesson at full
  size, with more to try.
    #explore-hook     your sentence (the editor and its live summary)
    #explore-toy      one sentence is many lessons (the sliding window) and the loss −log p toy (the old #toy)
    #explore-real     your text through TinyGPT: loss and perplexity, colored by surprise (#real)
    #explore-compare  watch it learn (the training replay) and decoding (#compare)
  It reads and writes the page's NtpState (vp.S), so the sentence and the loaded checkpoints are the lesson's.
-->
<script lang="ts">
	import Section from '$lib/kit/Section.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import YourSentence from '../YourSentence.svelte';
	import Part1Windows from '../Part1Windows.svelte';
	import LossToy from '../LossToy.svelte';
	import Surprisal from '../Surprisal.svelte';
	import Learn from '../Learn.svelte';
	import Decoding from '../Decoding.svelte';
	import type { NtpState } from '../state.svelte';

	let { vp }: { vp: { S: NtpState }; section?: string } = $props();
	const S = $derived(vp.S);
</script>

<div class="explore">
	<Section id="explore-hook" title={{ en: 'Your sentence', zh: '你的句子' }}>
		<YourSentence {S} />
	</Section>
	<Section id="explore-toy" title={{ en: 'One sentence is many lessons', zh: '一句话，许多道练习题' }}>
		<Part1Windows {S} />
		<div class="sub">
			<h3>{t({ en: 'The loss: −log p', zh: '损失：−log p' })}</h3>
			<LossToy {S} />
		</div>
	</Section>
	<Section id="explore-real" title={{ en: 'Your text through TinyGPT: loss and perplexity', zh: '用 TinyGPT 读你的文本：损失与困惑度' }} provenance="live">
		<Surprisal {S} />
	</Section>
	<Section id="explore-compare" title={{ en: 'Watch it learn', zh: '看它学习' }}>
		<Learn {S} />
		<Decoding {S} />
	</Section>
</div>

<style>
	.explore {
		display: grid;
		gap: var(--s7);
	}
	.sub {
		display: grid;
		gap: var(--s3);
		margin-top: var(--s5);
		padding-top: var(--s5);
		border-top: 1px dashed var(--rule);
	}
	.sub h3 {
		margin: 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
</style>
