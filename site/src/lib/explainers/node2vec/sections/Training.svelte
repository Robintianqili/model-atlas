<!-- node2vec "Train it yourself" (the word2vec package's section, adapted): the `run` source (trainer sg, the twin-checked
     one) run to the end on the current input, then its member map with the nearest members of the member in focus, and
     the loss of each step's pair with its trailing mean (the raw per-pair loss is too noisy to read). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../atlas/types/index.ts';
	import { atFor } from '../../../atlas/runtime/sources.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import MemberMap from '../views/map/View.svelte';
	import LossCurve from '../../../atlas/ui/views/loss-curve/View.svelte';
	import { inputOf } from '../walks.ts';

	let { store }: { store: AtlasStore } = $props();
	let busy = $state(false);
	async function train() {
		busy = true;
		try {
			await store.rt.ensure('run', 'last');
		} finally {
			busy = false;
			store.touch();
		}
	}
	const stage = (props: Record<string, unknown>): StageProps | null => {
		const rt = store.r, tape = rt.sources.run?.tape;
		if (!tape?.length) return null;
		return { rt, source: 'run', at: atFor(rt, 'run', tape.length - 1), focus: [], sel: {}, provenance: rt.sources.run.provenance, autoplay: false, onend: () => {}, props };
	};
	const map = $derived(stage({ focus: inputOf(store.r.input).focus, list: true }));
	const curve = $derived(stage({ smooth: 100, labels: { loss: { en: 'loss of each step’s pair', zh: '每一步那一对的损失' } } }));
</script>

<p class="muted">{t({ en: 'Train the member vectors on your graph, p and q in your browser (plain SGD, one random pair of the walks per step), then read the map and the nearest members of the member in focus. Change your input above and train again.', zh: '在你的浏览器里用你的图、p 和 q 训练成员向量（普通 SGD，每步一个从游走中随机抽出的训练对），然后看成员向量图和关注成员最近的成员。改一改上面的输入，再训练一次。' })}</p>
<p><button type="button" class="btn primary" onclick={train} disabled={busy}>{busy ? t({ en: 'Training…', zh: '训练中…' }) : t({ en: 'Train', zh: '训练' })}</button></p>
{#if map && curve}
	<div class="two"><MemberMap {...map} /><LossCurve {...curve} /></div>
{/if}

<style>
	.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 16px; }
</style>
