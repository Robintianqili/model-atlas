<!--
  TinyCNN's Code view (redesign spec §5.9), loaded only when the view opens: the forward pass in code, with the numbers
  of the lesson's input, and how it was checked against PyTorch.
-->
<script lang="ts">
	import CodeSection from '../CodeSection.svelte';
	import type { CnnStore } from '../store.svelte';

	let { vp }: { vp: { S: CnnStore }; section?: string } = $props();
</script>

<CodeSection S={vp.S} />
