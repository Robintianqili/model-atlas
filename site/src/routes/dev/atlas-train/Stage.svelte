<!--
  Renders one kind-B view with StageProps passed one by one (no spread props, which would grow the Svelte runtime chunk
  that every page preloads).
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import type { StageProps } from '$lib/atlas/types/view.ts';

	let { view, p }: { view: Component<StageProps>; p: StageProps } = $props();
	const View = $derived(view);
</script>

<View rt={p.rt} source={p.source} at={p.at} focus={p.focus} sel={p.sel} ctx={p.ctx} provenance={p.provenance} autoplay={p.autoplay} onend={p.onend} props={p.props} />
