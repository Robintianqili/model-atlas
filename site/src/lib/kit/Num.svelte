<!--
  A number in the visual grammar: monospace, 3 significant digits, full precision on hover,
  focus or tap. <Num value={0.123456789} />  <Num value={x} digits={4} unit="%" />
  Set `plain` to render only the text (for dense tables where tooltips would be noise).
-->
<script lang="ts">
	import Tooltip from './Tooltip.svelte';
	import { fmt3, fmtFull } from './format';

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
