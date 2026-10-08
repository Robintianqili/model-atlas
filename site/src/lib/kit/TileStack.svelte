<!--
  TileStack: a [C, H, W] tensor as small multiples, one tile per channel, with channel labels
  and a selected channel (bindable).

  <TileStack data={trace.t.conv1.data} channels={8} height={26} width={26}
             scope="layer" bind:selected={ch} label="conv1 output" />

  scope: 'unit' = each tile uses its own range; 'layer' = one range for all tiles (default);
  'global' = `scopeDomains.global` (computed by the page across layers).
  Keyboard: tiles form one tab stop; arrow keys move the selection.
-->
<script lang="ts">
	import TensorGrid from './TensorGrid.svelte';
	import { domainOf, type ColorScope, type Domain, type ScaleKind } from './color';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		data: Float32Array | ArrayLike<number>;
		channels: number;
		height: number;
		width: number;
		kind?: ScaleKind;
		scope?: ColorScope;
		scopeDomains?: { layer?: Domain; global?: Domain };
		selected?: number;
		channelLabels?: string[];
		/** approximate tile size in CSS px */
		tile?: number;
		label: Text;
		/** bump to redraw after in-place mutation */
		version?: number;
		onselect?: (channel: number) => void;
	}
	let {
		data,
		channels,
		height,
		width,
		kind = 'diverging',
		scope = 'layer',
		scopeDomains = {},
		selected = $bindable(0),
		channelLabels,
		tile = 64,
		label,
		version = 0,
		onselect
	}: Props = $props();

	const hw = $derived(height * width);
	const slices = $derived(
		Array.from({ length: channels }, (_, c) =>
			data instanceof Float32Array ? data.subarray(c * hw, (c + 1) * hw) : Array.prototype.slice.call(data, c * hw, (c + 1) * hw)
		)
	);
	const layerDomain = $derived(scopeDomains.layer ?? domainOf(kind, data));
	const tileDomain = $derived(scope === 'unit' ? undefined : scope === 'global' ? (scopeDomains.global ?? layerDomain) : layerDomain);
	const cell = $derived(Math.max(1, Math.floor(tile / Math.max(height, width))));
	let buttons: HTMLButtonElement[] = $state([]);
	const gid = $props.id();

	function choose(c: number) {
		selected = c;
		onselect?.(c);
	}
	function onkeydown(e: KeyboardEvent) {
		let c = selected;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') c = Math.min(channels - 1, c + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') c = Math.max(0, c - 1);
		else if (e.key === 'Home') c = 0;
		else if (e.key === 'End') c = channels - 1;
		else return;
		e.preventDefault();
		choose(c);
		buttons[c]?.focus();
	}
</script>

<div class="stack" role="listbox" aria-labelledby={gid} aria-orientation="horizontal" tabindex="-1" {onkeydown}>
	<span id={gid} class="visually-hidden">{t(label)}: {channels} {ui('channels')}</span>
	{#each slices as slice, c (c)}
		<button
			bind:this={buttons[c]}
			type="button"
			class="tile"
			class:sel={selected === c}
			role="option"
			aria-selected={selected === c}
			tabindex={selected === c ? 0 : -1}
			onclick={() => choose(c)}
		>
			<TensorGrid
				data={slice}
				rows={height}
				cols={width}
				{kind}
				domain={tileDomain}
				{cell}
				gridLines={false}
				labels={false}
				interactive={false}
				readout={false}
				{version}
				label="{t(label)} {ui('channel')} {c}"
			/>
			<span class="cl num">{channelLabels?.[c] ?? c}</span>
		</button>
	{/each}
</div>

<style>
	.stack {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.tile {
		display: grid;
		justify-items: center;
		gap: 2px;
		padding: 4px;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		background: var(--surface);
		cursor: pointer;
	}
	.tile:hover {
		border-color: var(--rule-strong);
	}
	.tile.sel {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.cl {
		font-size: 11px;
		color: var(--muted);
		line-height: 1.2;
	}
	.tile.sel .cl {
		color: var(--ink);
		font-weight: 700;
	}
</style>
