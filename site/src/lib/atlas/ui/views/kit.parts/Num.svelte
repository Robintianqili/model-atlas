<!--
  The kit's <Num> (monospace, 3 significant digits, the full value on hover, focus or tap) with the atlas number format:
  every minus sign is U+2212, also in exponents ("−7.70e−4"; kit/format.ts changes only the first '-'). Atlas views
  import this one, so a view never mixes two minus signs. <Num value={x} digits={4} unit="%" signed />
-->
<script lang="ts">
	import Tooltip from '$lib/kit/Tooltip.svelte';
	import { fmt3, fmtFull } from '../../../core/format.ts';

	interface Props {
		value: number | null | undefined;
		digits?: number;
		unit?: string;
		plain?: boolean;
		/** color the text by sign (blue negative, orange positive) */
		signed?: boolean;
	}
	let { value, digits = 3, unit = '', plain = false, signed = false }: Props = $props();

	const short = $derived(fmt3(value, digits));
	const full = $derived(fmtFull(value));
	const sign = $derived(signed && value ? (value < 0 ? 'neg' : 'pos') : '');
</script>

{#if plain || short === full}
	<span class="num n {sign}">{short}{unit}</span>
{:else}
	<Tooltip tip={full + unit} width={240}>
		<span class="num n {sign}" aria-label={full + unit}>{short}{unit}</span>
	</Tooltip>
{/if}

<style>
	.n {
		white-space: nowrap;
	}
	.neg {
		color: var(--div-neg-text);
	}
	.pos {
		color: var(--div-pos-text);
	}
</style>
