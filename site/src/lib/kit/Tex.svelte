<!--
  Inline (or display) math in prose: <Tex tex="\\mathrm{softmax}(z)_i" />.
  Use sym() from $lib/kit/tex to color symbols like their tensors.
-->
<script lang="ts">
	import 'katex/dist/katex.min.css';
	import '$lib/styles/math.css';
	import { renderTex, fillTex } from './tex';
	import { keyScroll } from './keyscroll';

	interface Props {
		tex: string;
		display?: boolean;
		values?: Record<string, number | string>;
	}
	let { tex, display = false, values }: Props = $props();
	const html = $derived(renderTex(values ? fillTex(tex, values) : tex, display));
</script>

{#if display}<div class="tex-block" use:keyScroll={'.katex-display'}>{@html html}</div>{:else}<span class="tex-inline">{@html html}</span>{/if}
