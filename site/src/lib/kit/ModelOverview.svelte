<!--
  ModelOverview: the layer pipeline generated from spec.layers. Left to right on wide
  screens, top to bottom on narrow ones (container query at 560 px). Each node is a button
  with a thumbnail slot, the op, and the output shape. Edges show flowing dashes for the
  forward pass (static when reduced motion is on).

  <ModelOverview layers={spec.layers} bind:current>
    {#snippet thumb(layer)}<TensorGrid … cell={2} interactive={false} />{/snippet}
  </ModelOverview>

  Keyboard: one tab stop; arrow keys move, Enter/Space select (selection follows focus).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { keyScroll } from './keyscroll';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmtShape } from './format';
	import RoleChip from './RoleChip.svelte';
	import type { LayerNode } from './types';

	interface Props {
		layers: LayerNode[];
		current?: string;
		thumb?: Snippet<[LayerNode]>;
		label?: Text;
		/** animate edges (forward-pass dashes); off when reduced motion is on */
		flowing?: boolean;
		onselect?: (id: string) => void;
	}
	let { layers, current = $bindable(''), thumb, label, flowing = true, onselect }: Props = $props();

	let nodes: HTMLButtonElement[] = $state([]);
	const idx = $derived(layers.findIndex((l) => l.id === current));

	function choose(i: number) {
		current = layers[i].id;
		onselect?.(current);
	}
	function onkeydown(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(layers.length - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = layers.length - 1;
		else return;
		e.preventDefault();
		choose(j);
		nodes[j]?.focus();
	}
</script>

<nav class="ov" aria-label={label ? t(label) : ui('overview')}>
	<ol class="pipe" class:flowing use:keyScroll>
		{#each layers as l, i (l.id)}
			<li class="item">
				<button
					bind:this={nodes[i]}
					type="button"
					class="node"
					class:cur={l.id === current}
					class:frozen={l.status === 'frozen'}
					class:trainable={l.status === 'trainable'}
					aria-current={l.id === current ? 'step' : undefined}
					tabindex={i === (idx < 0 ? 0 : idx) ? 0 : -1}
					onclick={() => choose(i)}
					onkeydown={(e) => onkeydown(e, i)}
				>
					{#if thumb}<span class="thumb">{@render thumb(l)}</span>{/if}
					<span class="name">{t(l.label)}</span>
					{#if l.op}<span class="op num">{l.op}</span>{/if}
					{#if l.outShape}<span class="shape num">{fmtShape(l.outShape)}</span>{/if}
					{#if l.status}<span class="st"><RoleChip role={l.status} iconOnly /></span>{/if}
				</button>
				{#if i < layers.length - 1}<span class="edge" aria-hidden="true"></span>{/if}
			</li>
		{/each}
	</ol>
</nav>

<style>
	.ov {
		container-type: inline-size;
		min-width: 0;
	}
	.pipe {
		list-style: none;
		margin: 0;
		padding: 6px 2px 10px;
		display: flex;
		align-items: stretch;
		overflow-x: auto;
		scroll-snap-type: x proximity;
	}
	.item {
		display: flex;
		align-items: center;
		flex: 0 0 auto;
		scroll-snap-align: start;
	}
	.node {
		position: relative;
		display: grid;
		justify-items: center;
		align-content: start;
		gap: 3px;
		min-width: 84px;
		max-width: 132px;
		padding: 8px 8px 6px;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		color: var(--ink);
		text-align: center;
		transition:
			border-color var(--dur-1) var(--ease),
			transform var(--dur-1) var(--ease);
	}
	.node:hover {
		border-color: var(--ink-2);
	}
	.node.cur {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
		background: color-mix(in srgb, var(--accent-soft) 60%, var(--surface));
	}
	.node.frozen {
		background: var(--surface-2);
		color: var(--muted);
	}
	.node.frozen .thumb {
		filter: grayscale(0.9) opacity(0.75);
	}
	.node.trainable {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.thumb {
		display: grid;
		place-items: center;
		min-height: 36px;
		max-width: 112px;
		overflow: hidden;
	}
	.name {
		font-size: var(--fs-sm);
		font-weight: 650;
		line-height: 1.2;
	}
	.op {
		font-size: 11px;
		color: var(--muted);
	}
	.shape {
		font-size: 11px;
		color: var(--ink-2);
	}
	.st {
		position: absolute;
		top: -9px;
		right: -6px;
	}
	.edge {
		width: 26px;
		height: 2px;
		margin: 0 2px;
		flex: 0 0 auto;
		background: repeating-linear-gradient(90deg, var(--ink-2) 0 5px, transparent 5px 9px);
		background-size: 9px 2px;
	}
	.flowing .edge {
		animation: flow-x 0.9s linear infinite;
	}
	@keyframes flow-x {
		from {
			background-position: 0 0;
		}
		to {
			background-position: 9px 0;
		}
	}
	@keyframes flow-y {
		from {
			background-position: 0 0;
		}
		to {
			background-position: 0 9px;
		}
	}
	@container (max-width: 560px) {
		.pipe {
			flex-direction: column;
			align-items: stretch;
			overflow-x: visible;
			padding: 2px 0;
		}
		.item {
			flex-direction: column;
			align-items: stretch;
		}
		.node {
			grid-template-columns: auto 1fr auto;
			grid-template-areas: 'thumb name shape' 'thumb op shape';
			justify-items: start;
			align-items: center;
			max-width: none;
			text-align: left;
			column-gap: 10px;
		}
		.thumb {
			grid-area: thumb;
			width: 72px;
		}
		.name {
			grid-area: name;
		}
		.op {
			grid-area: op;
		}
		.shape {
			grid-area: shape;
			justify-self: end;
		}
		.edge {
			width: 2px;
			height: 16px;
			margin: 2px auto;
			background: repeating-linear-gradient(180deg, var(--ink-2) 0 5px, transparent 5px 9px);
			background-size: 2px 9px;
		}
		.flowing .edge {
			animation-name: flow-y;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.flowing .edge {
			animation: none;
		}
	}
</style>
