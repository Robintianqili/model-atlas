<!--
  TensorStrip: a vector as a strip of colored cells. Long vectors are elided in the middle
  ("…") and the exact shape label is always visible.

  <TensorStrip data={embedding} name="token 3 embedding" shape={['T', 64]} />
  <TensorStrip data={probs} kind="sequential" domain={[0, 1]} maxCells={10} vertical />

  Keyboard: Tab focuses the strip, arrow keys move between visible cells, Enter selects.
-->
<script lang="ts">
	import { makeScale, scopeDomain, type ColorScope, type Domain, type ScaleKind } from './color';
	import { themeState } from './theme.svelte';
	import { fmt3, fmtFull, fmtShape } from './format';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		data: ArrayLike<number>;
		kind?: ScaleKind;
		domain?: Domain;
		scope?: ColorScope;
		scopeDomains?: { layer?: Domain; global?: Domain };
		/** maximum cells drawn before eliding (default 24) */
		maxCells?: number;
		/** shape label; defaults to [length] */
		shape?: (number | string)[];
		/** accessible name and caption */
		name?: Text;
		cellSize?: number;
		vertical?: boolean;
		/** show 3-digit values inside cells (needs cellSize ≥ 34) */
		values?: boolean;
		selected?: number | null;
		onhover?: (i: number | null, value?: number) => void;
		onselect?: (i: number, value: number) => void;
	}
	let {
		data,
		kind = 'diverging',
		domain,
		scope = 'unit',
		scopeDomains = {},
		maxCells = 24,
		shape,
		name,
		cellSize = 16,
		vertical = false,
		values = false,
		selected = $bindable(null),
		onhover,
		onselect
	}: Props = $props();

	const n = $derived(data.length);
	const scale = $derived(makeScale(kind, domain ?? scopeDomain(kind, scope, data, scopeDomains), themeState.resolved));
	const layout = $derived.by(() => {
		if (n <= maxCells) return { head: Array.from({ length: n }, (_, i) => i), tail: [] as number[], hidden: 0 };
		const head = Math.ceil((maxCells - 1) * 0.75);
		const tail = maxCells - 1 - head;
		return {
			head: Array.from({ length: head }, (_, i) => i),
			tail: Array.from({ length: tail }, (_, i) => n - tail + i),
			hidden: n - head - tail
		};
	});
	const visible = $derived([...layout.head, ...layout.tail]);
	let hover = $state<number | null>(null);
	let cursor = $state<number | null>(null);
	let focused = $state(false);
	const active = $derived(hover ?? (focused ? cursor : null));

	function enter(i: number | null) {
		hover = i;
		onhover?.(i, i === null ? undefined : data[i]);
	}
	function pick(i: number) {
		selected = i;
		cursor = i;
		onselect?.(i, data[i]);
	}
	function onkeydown(e: KeyboardEvent) {
		const pos = cursor === null ? -1 : visible.indexOf(cursor);
		const next = vertical ? 'ArrowDown' : 'ArrowRight';
		const prev = vertical ? 'ArrowUp' : 'ArrowLeft';
		let p = pos;
		if (e.key === next) p = Math.min(visible.length - 1, pos + 1);
		else if (e.key === prev) p = Math.max(0, pos - 1);
		else if (e.key === 'Home') p = 0;
		else if (e.key === 'End') p = visible.length - 1;
		else if ((e.key === 'Enter' || e.key === ' ') && cursor !== null) {
			e.preventDefault();
			pick(cursor);
			return;
		} else return;
		e.preventDefault();
		cursor = visible[Math.max(0, p)];
		hover = null;
		onhover?.(cursor, data[cursor]);
	}
</script>

{#snippet cellEl(i: number)}
	{@const v = data[i]}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<span
		class="c"
		class:on={active === i}
		class:wide={values}
		class:sel={selected === i}
		style="background: {scale.css(v)}; color: {scale.ink(v)}"
		onpointerenter={() => enter(i)}
		onclick={() => pick(i)}
		role="presentation"
		>{#if values}<span class="v">{fmt3(v)}</span>{/if}</span
	>
{/snippet}

<div class="strip-wrap" class:vertical>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (composite widget: arrow keys move between cells) -->
	<div
		class="strip"
		style="--cs: {cellSize}px"
		tabindex="0"
		role="application"
		aria-roledescription="vector"
		aria-label="{name ? t(name) + ' ' : ''}{fmtShape(shape ?? [n])}"
		onpointerleave={() => enter(null)}
		onfocus={() => (focused = true)}
		onblur={() => (focused = false)}
		{onkeydown}
	>
		{#each layout.head as i (i)}{@render cellEl(i)}{/each}
		{#if layout.hidden > 0}
			<span class="gap" title={ui('elided', { n: layout.hidden })} aria-hidden="true">…</span>
		{/if}
		{#each layout.tail as i (i)}{@render cellEl(i)}{/each}
	</div>
	<span class="shape num">{fmtShape(shape ?? [n])}</span>
</div>
<div class="readout num" aria-hidden="true">
	{#if active !== null}[{active}] = {fmt3(data[active])}{:else}&nbsp;{/if}
</div>
<div class="visually-hidden" aria-live="polite">
	{#if focused && cursor !== null}{ui('itemAt', { i: cursor, v: fmtFull(data[cursor]) })}{/if}
</div>

<style>
	.strip-wrap {
		display: flex;
		align-items: center;
		gap: 8px;
		max-width: 100%;
		flex-wrap: wrap;
	}
	.strip {
		display: inline-flex;
		gap: 1px;
		padding: 1px;
		border-radius: 3px;
		background: var(--surface);
		border: 1px solid var(--rule);
		max-width: 100%;
		overflow-x: auto;
		overflow-y: hidden;
	}
	.vertical {
		flex-direction: column;
		align-items: flex-start;
	}
	.vertical .strip {
		flex-direction: column;
	}
	.c {
		width: var(--cs);
		height: var(--cs);
		flex: 0 0 auto;
		display: grid;
		place-items: center;
		cursor: crosshair;
	}
	.c.wide {
		width: auto;
		min-width: var(--cs);
		padding: 0 4px;
	}
	.v {
		font-family: var(--font-mono);
		font-size: 10px;
		white-space: nowrap;
	}
	.c.on {
		outline: 2px solid var(--highlight);
		outline-offset: -1px;
		position: relative;
		z-index: 1;
	}
	.c.sel {
		box-shadow: inset 0 0 0 2px var(--accent);
	}
	.gap {
		display: grid;
		place-items: center;
		min-width: calc(var(--cs) * 1.4);
		height: var(--cs);
		font-size: 12px;
		color: var(--muted);
	}
	.vertical .gap {
		width: var(--cs);
		min-width: 0;
		height: calc(var(--cs) * 1.2);
	}
	.shape {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.readout {
		min-height: 1.4em;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.strip:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
</style>
