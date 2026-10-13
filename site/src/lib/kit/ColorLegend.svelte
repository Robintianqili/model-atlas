<!--
  Legend for the value scales. Diverging shows −m · 0 · +m; sequential shows lo → hi.
  <ColorLegend kind="diverging" domain={[-2.1, 2.1]} />
  <ColorLegend kind="sequential" domain={[0, 1]} label="attention weight" />
-->
<script lang="ts">
	import { gradientCss, type Domain, type ScaleKind } from './color';
	import { themeState } from './theme.svelte';
	import { fmt3 } from './format';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		kind?: ScaleKind;
		domain: Domain;
		label?: Text;
		width?: number;
	}
	let { kind = 'diverging', domain, label, width = 180 }: Props = $props();

	const bg = $derived(gradientCss(kind, themeState.resolved, 13));
	const m = $derived(Math.max(Math.abs(domain[0]), Math.abs(domain[1])));
</script>

<div class="legend" style="--w: {width}px">
	{#if label}<span class="title">{t(label)}</span>{/if}
	<span class="bar" style="background: {bg}" aria-hidden="true"></span>
	<span class="ticks num">
		{#if kind === 'diverging'}
			<span>{fmt3(-m)}</span><span>0</span><span>{fmt3(m)}</span>
		{:else}
			<span>{fmt3(domain[0])}</span><span>{fmt3(domain[1])}</span>
		{/if}
	</span>
	<span class="visually-hidden">
		{#if kind === 'diverging'}
			{t({ en: `${ui('negative')}: blue, ${ui('zero')}: gray, ${ui('positive')}: orange`, zh: `${ui('negative')}：蓝色，${ui('zero')}：灰色，${ui('positive')}：橙色` })}
		{:else}
			{t({ en: `${ui('low')} → ${ui('high')}: light → dark purple`, zh: `${ui('low')} → ${ui('high')}：浅紫 → 深紫` })}
		{/if}
	</span>
</div>

<style>
	.legend {
		display: inline-grid;
		gap: 2px;
		width: min(var(--w), 100%);
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.title {
		color: var(--ink-2);
	}
	.bar {
		height: 10px;
		border-radius: 3px;
		border: 1px solid var(--rule);
	}
	.ticks {
		display: flex;
		justify-content: space-between;
		font-size: var(--t-xs);
	}
</style>
