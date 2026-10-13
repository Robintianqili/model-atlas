<!-- /train/: the atlas filtered to Kind = Training methods (redesign spec §6.5; the URL is kept). -->
<script lang="ts">
	import AtlasView from '$lib/kit/catalog/AtlasView.svelte';
	import {  } from '$lib/i18n/i18n.svelte';
	import { ct } from '$lib/kit/catalog/text';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const c = $derived(data.index.counts);
	const planned = $derived(data.index.planned.filter((p) => p.kind === 'concept').length);
	const sections = $derived(data.index.groups.filter((g) => g.kind === 'concept' && g.live).length);
</script>

<svelte:head><title>{ct('methods')} · Model Atlas</title></svelte:head>

<AtlasView
	index={data.index}
	kind="concept"
	title={{ en: 'Training methods', zh: '训练方法' }}
	lede={{
		en: `How models are trained, fine-tuned, compressed and served: ${c.methods} lessons you can run now in ${sections} sections, each on a small model you can watch. ${planned} more are planned.`,
		zh: `模型如何训练、微调、压缩与部署：现在可以学的 ${c.methods} 节课，分在 ${sections} 个部分，每节都在一个看得见的小模型上。另有 ${planned} 节在计划中。`
	}}
/>
