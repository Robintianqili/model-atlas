<!-- word2vec "Train it yourself": the train source (trainer sg, the twin-checked one) run to the end on the current input,
     then its word map with every word's nearest word, and the loss of each step's pair with its trailing mean. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../atlas/types/index.ts';
	import { atFor } from '../../../atlas/runtime/sources.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import WordMap from '../views/map/View.svelte';
	import LossCurve from '../../../atlas/ui/views/loss-curve/View.svelte';

	let { store }: { store: AtlasStore } = $props();
	let busy = $state(false);
	async function train() {
		busy = true;
		try {
			await store.rt.ensure('train', 'last');
		} finally {
			busy = false;
			store.touch();
		}
	}
	const stage = (props: Record<string, unknown>): StageProps | null => {
		const rt = store.r, tape = rt.sources.train?.tape;
		if (!tape?.length) return null;
		return { rt, source: 'train', at: atFor(rt, 'train', tape.length - 1), focus: [], sel: {}, provenance: rt.sources.train.provenance, autoplay: false, onend: () => {}, props };
	};
	const map = $derived(stage({ list: true }));
	const curve = $derived(stage({ smooth: 100, labels: { loss: { en: 'loss of each step’s pair', zh: '每一步那一对的损失' } } }));
</script>

<p class="muted">{t({ en: 'Train on your sentences in your browser (plain SGD, one random pair per step), then read every word’s nearest word. Change your input above and train again.', zh: '在你的浏览器里用你的句子训练（普通 SGD，每步一个随机训练对），然后看每个词最近的词。改一改上面的输入，再训练一次。' })}</p>
<p><button type="button" class="btn primary" onclick={train} disabled={busy}>{busy ? t({ en: 'Training…', zh: '训练中…' }) : t({ en: 'Train', zh: '训练' })}</button></p>
{#if map && curve}
	<div class="two"><WordMap {...map} /><LossCurve {...curve} /></div>
{/if}

<style>
	.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 16px; }
</style>
