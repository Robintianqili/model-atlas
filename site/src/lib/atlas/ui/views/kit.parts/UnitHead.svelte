<!--
  The first line of every zoom view: which layer, which op, which output number.
  "Conv 1-2 · Convolution · channel 4, row 10, column 12 = 1.23"
-->
<script lang="ts">
	import Num from './Num.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		layer: Text;
		op?: Text;
		where?: Text;
		value?: number | null;
		shape?: readonly (number | string)[];
	}
	let { layer, op, where, value = null, shape }: Props = $props();
</script>

<p class="uh">
	<strong class="ly">{t(layer)}</strong>
	{#if op}<span class="op">{t(op)}</span>{/if}
	{#if shape}<span class="num muted shape">[{shape.join('×')}]</span>{/if}
	{#if where}<span class="wh">{t(where)}{#if value !== null}<span class="eq">&nbsp;=&nbsp;<Num {value} signed /></span>{/if}</span>{/if}
</p>

<style>
	.uh {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 2px 10px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		min-width: 0;
	}
	.ly {
		color: var(--ink);
		font-weight: 650;
	}
	.op {
		font-size: var(--fs-xs);
		padding: 0 6px;
		border-radius: var(--r-pill);
		background: var(--surface-2);
		color: var(--ink-2);
	}
	.shape {
		font-size: var(--fs-xs);
	}
	.wh {
		min-width: 0;
		overflow-wrap: anywhere;
	}
	.eq {
		white-space: nowrap;
	}
</style>
