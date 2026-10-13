<!-- /atlas/: every lesson, filterable (redesign spec §6.4; kit/catalog/AtlasView.svelte). -->
<script lang="ts">
	import AtlasView from '$lib/kit/catalog/AtlasView.svelte';
	import {  } from '$lib/i18n/i18n.svelte';
	import { ct } from '$lib/kit/catalog/text';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<svelte:head><title>{ct('allLessons')} · Model Atlas</title></svelte:head>

<AtlasView index={data.index} title={{ en: 'All lessons', zh: '全部课程' }} />
