<!--
  Lesson Lab (/dev/atlas/lab/<slug>/, design §5.1 and §8.2 W1-D1). The route node only loads LabView.svelte (the lab UI,
  the runtime and AutoModelPage are lazy chunks), so adding this dev route does not reshape the chunks other pages share.
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	let View = $state<Component<any> | null>(null);
	let failed = $state('');
	onMount(() => {
		import('./LabView.svelte').then(
			(m) => (View = m.default),
			(e) => (failed = e instanceof Error ? e.message : String(e))
		);
	});
</script>

{#if View}
	<View {data} />
{:else}
	<div class="container lab-wait">
		<!-- no i18n here: the route node imports nothing the lab chunk does not (see the comment at the top) -->
		<p class="muted">{failed || 'Loading the Lesson Lab… · 正在加载课程实验室…'}</p>
	</div>
{/if}

<style>
	.lab-wait {
		padding-top: var(--s5);
	}
</style>
