<!--
  TensorView: picks the right view for a traced tensor from its shape.
    [N]            -> TensorStrip
    [R, C]         -> TensorGrid
    [C, H, W]      -> TileStack + the selected channel enlarged (TensorGrid)
    [1, C, H, W] / [1, …]  leading 1s are dropped
  <TensorView tensor={trace.t[layerId]} label={layerLabel} scope={scope} scopeDomains={doms} />
-->
<script lang="ts">
	import TensorGrid from './TensorGrid.svelte';
	import TensorStrip from './TensorStrip.svelte';
	import TileStack from './TileStack.svelte';
	import ColorLegend from './ColorLegend.svelte';
	import { domainOf, type ColorScope, type Domain, type ScaleKind } from './color';
	import { ui, t, colon, type Text } from '$lib/i18n/i18n.svelte';
	import type { CellValue, TensorLike } from './types';

	interface Props {
		tensor: TensorLike;
		label: Text;
		kind?: ScaleKind;
		scope?: ColorScope;
		scopeDomains?: { layer?: Domain; global?: Domain };
		/** selected channel for [C,H,W] tensors (bindable) */
		channel?: number;
		legend?: boolean;
		onhover?: (c: CellValue | null) => void;
	}
	let {
		tensor,
		label,
		kind = 'diverging',
		scope = 'layer',
		scopeDomains = {},
		channel = $bindable(0),
		legend = true,
		onhover
	}: Props = $props();

	const dims = $derived.by(() => {
		const d = [...tensor.shape];
		while (d.length > 1 && d[0] === 1) d.shift();
		return d;
	});
	const data = $derived(tensor.data);
	const chData = $derived.by(() => {
		if (dims.length !== 3) return data;
		const hw = dims[1] * dims[2];
		const c = Math.min(channel, dims[0] - 1);
		return data instanceof Float32Array ? data.subarray(c * hw, (c + 1) * hw) : Array.prototype.slice.call(data, c * hw, (c + 1) * hw);
	});
	// 'unit' = the shown map (one channel for [C,H,W]); 'layer' = the whole tensor unless the page
	// passes scopeDomains.layer; 'global' = scopeDomains.global (falls back to the layer).
	const viewDomain = $derived<Domain>(
		scope === 'unit'
			? domainOf(kind, dims.length === 3 ? chData : data)
			: scope === 'global'
				? (scopeDomains.global ?? scopeDomains.layer ?? domainOf(kind, data))
				: (scopeDomains.layer ?? domainOf(kind, data))
	);
</script>

<div class="tv">
	{#if dims.length === 1}
		<TensorStrip {data} {kind} domain={viewDomain} name={label} shape={tensor.shape} />
	{:else if dims.length === 2}
		<TensorGrid {data} rows={dims[0]} cols={dims[1]} {kind} domain={viewDomain} {label} {onhover} />
	{:else if dims.length === 3}
		<TileStack
			{data}
			channels={dims[0]}
			height={dims[1]}
			width={dims[2]}
			{kind}
			{scope}
			{scopeDomains}
			bind:selected={channel}
			{label}
			tile={dims[0] > 16 ? 44 : 60}
		/>
		<div class="big">
			<span class="cap">{t(label)} · {ui('channel')} {channel}</span>
			<TensorGrid data={chData} rows={dims[1]} cols={dims[2]} {kind} domain={viewDomain} maxCell={22} label="{t(label)} {ui('channel')} {channel}" {onhover} />
		</div>
	{:else}
		<p class="muted">{ui('shape')}{colon()} [{tensor.shape.join('×')}] — {ui('noDataYet')}</p>
	{/if}
	{#if legend && dims.length <= 3}<ColorLegend {kind} domain={viewDomain} />{/if}
</div>

<style>
	.tv {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		min-width: 0;
	}
	.big {
		display: grid;
		gap: 4px;
		max-width: 520px;
	}
	.cap {
		font-size: var(--fs-xs);
		color: var(--muted);
		font-family: var(--font-mono);
	}
</style>
