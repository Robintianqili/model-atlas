<!--
  Token chips for the tokenization page (a richer cousin of the kit's TokenChips):
  - raw-byte pieces (<0xE7>) are drawn as hex in a dashed box, so "part of a character" is visible;
  - per-chip marks: 'new' (just merged), 'pair' (an occurrence of a hovered pair), 'link' (same text
    hovered in another view);
  - gaps between pre-tokens; ids under the chips; a tooltip with id and bytes;
  - one tab stop: arrows move, Enter/Space select, Escape hides the tooltip.
  Built for thousands of chips: events are handled once on the container, each chip's content is a
  cached, escaped HTML string, and `limit` caps how many are drawn until "Show all" is pressed.
-->
<script lang="ts" module>
	import { visibleToken } from '$lib/kit/format';

	const cache = new Map<string, string>();
	const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

	/** "处<0xE7><0x90>" -> '处<span class="hex">E7 90</span>' (escaped; whitespace made visible). */
	export function chipHtml(s: string): string {
		let h = cache.get(s);
		if (h !== undefined) return h;
		h = '';
		const re = /((?:<0x[0-9A-F]{2}>)+)/g;
		let last = 0;
		for (const m of s.matchAll(re)) {
			const i = m.index ?? 0;
			if (i > last) h += esc(visibleToken(s.slice(last, i)));
			h += `<span class="hex">${m[1].replace(/<0x([0-9A-F]{2})>/g, '$1 ').trim()}</span>`;
			last = i + m[1].length;
		}
		if (last < s.length || s === '') h += esc(visibleToken(s.slice(last)));
		if (cache.size > 20000) cache.clear();
		cache.set(s, h);
		return h;
	}
</script>

<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { ChipItem } from './types';

	interface Props {
		items: ChipItem[];
		marks?: (string | null | undefined)[];
		selected?: number | null;
		showIds?: boolean;
		label: Text;
		/** cap the box height and scroll inside it */
		maxHeight?: number;
		dense?: boolean;
		/** draw at most this many chips until "Show all" is pressed */
		limit?: number;
		onhover?: (i: number | null) => void;
		onselect?: (i: number) => void;
	}
	let {
		items,
		marks = [],
		selected = $bindable(null),
		showIds = false,
		label,
		maxHeight,
		dense = false,
		limit = Infinity,
		onhover,
		onselect
	}: Props = $props();

	let focusIdx = $state(0);
	let tip = $state<number | null>(null);
	let box: HTMLDivElement | undefined = $state();
	let tipEl: HTMLSpanElement | undefined = $state();
	let shift = $state(0);
	let all = $state(false);
	const gid = $props.id();

	const shown = $derived(all || items.length <= limit ? items : items.slice(0, limit));
	const hidden = $derived(items.length - shown.length);

	// keep the tooltip inside the viewport (and inside a scrolling box)
	$effect(() => {
		if (tip === null || !tipEl) return;
		// measure where the unshifted tooltip would be (centred on its chip), not the tooltip itself:
		// a tooltip that is re-created (tap, then focus) may still carry the previous shift
		const cell = tipEl.parentElement?.getBoundingClientRect();
		if (!cell) return;
		const w = tipEl.offsetWidth;
		const left = cell.left + cell.width / 2 - w / 2;
		const right = left + w;
		const b = box?.getBoundingClientRect();
		const lo = Math.max(8, b && maxHeight !== undefined ? b.left + 4 : 8);
		const hi = Math.min(innerWidth - 8, b && maxHeight !== undefined ? b.right - 4 : innerWidth - 8);
		shift = left < lo ? lo - left : right > hi ? hi - right : 0;
	});

	function idxOf(e: Event): number | null {
		const el = (e.target as HTMLElement | null)?.closest?.('[data-i]') as HTMLElement | null;
		if (!el || !box?.contains(el)) return null;
		return Number(el.dataset.i);
	}
	function chipAt(i: number): HTMLElement | null {
		return box?.querySelector<HTMLElement>(`[data-i="${i}"]`) ?? null;
	}
	function show(i: number | null) {
		if (tip === i) return;
		tip = i;
		onhover?.(i);
	}
	function select(i: number) {
		selected = i;
		onselect?.(i);
	}
	function onkeydown(e: KeyboardEvent) {
		const i = idxOf(e);
		if (i === null) return;
		const n = shown.length;
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(n - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = n - 1;
		else if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			select(i);
			return;
		} else if (e.key === 'Escape') {
			tip = null;
			return;
		} else return;
		e.preventDefault();
		focusIdx = j;
		chipAt(j)?.focus();
	}
	const tabIdx = $derived(Math.max(0, Math.min(shown.length - 1, selected ?? focusIdx)));
</script>

<div class="wrap">
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions (a listbox; the options inside are focusable) -->
	<div
		class="chips"
		class:dense
		class:scroll={maxHeight !== undefined}
		style:max-height={maxHeight !== undefined ? `${maxHeight}px` : undefined}
		bind:this={box}
		role="listbox"
		aria-orientation="horizontal"
		aria-label={t(label)}
		tabindex={items.length ? undefined : 0}
		onclick={(e) => {
			const i = idxOf(e);
			if (i !== null) select(i);
		}}
		{onkeydown}
		onpointerover={(e) => {
			const i = idxOf(e);
			if (i !== null) show(i);
		}}
		onpointerleave={() => show(null)}
		onfocusin={(e) => {
			const i = idxOf(e);
			if (i !== null) {
				focusIdx = i;
				show(i);
			}
		}}
		onfocusout={() => show(null)}
	>
		{#each shown as tok, i (i)}
			<span class="cell" class:gap={tok.gap && i > 0}>
				<span
					data-i={i}
					class="chip c{i % 4} {tok.kind ?? ''} {marks[i] ?? ''}"
					class:sel={selected === i}
					role="option"
					aria-selected={selected === i}
					tabindex={i === tabIdx ? 0 : -1}
					aria-describedby={tip === i ? `${gid}-tip` : undefined}
				>{@html chipHtml(tok.text)}</span>
				{#if showIds && tok.id !== undefined}<span class="id num">{tok.id}</span>{/if}
				{#if tip === i}
					<span class="tip" role="tooltip" id="{gid}-tip" bind:this={tipEl} style:transform="translateX(calc(-50% + {shift}px))">
						{#if tok.id !== undefined}<span>ID <b class="num">{tok.id}</b></span>{/if}
						{#if tok.hex}<span class="num">{tok.hex}</span>{/if}
						{#if tok.note}<span>{tok.note}</span>{/if}
					</span>
				{/if}
			</span>
		{/each}
	</div>
	{#if hidden > 0}
		<button type="button" class="btn ghost more" onclick={() => (all = true)}>
			{t({ en: `Show all ${items.length} tokens (${hidden} more; slow for long texts)`, zh: `显示全部 ${items.length} 个词元（还有 ${hidden} 个；长文本会变慢）` })}
		</button>
	{/if}
</div>

<style>
	.wrap {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 4px;
		min-width: 0;
	}
	.more {
		justify-self: start;
		min-height: 32px;
		font-size: var(--fs-xs);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 2px;
		align-items: flex-start;
		min-width: 0;
		padding: 2px;
	}
	.chips.scroll {
		contain: content; /* re-layout after a merge stays inside the box */
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: 26px 4px 4px; /* room for the tooltip of the first row */
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.cell {
		position: relative;
		display: inline-flex;
		flex-direction: column;
		align-items: center;
		min-width: 0;
	}
	.cell.gap {
		margin-left: 5px;
	}
	.chip {
		display: inline-block;
		min-width: 1.3em;
		max-width: 100%;
		padding: 1px 5px;
		border-radius: 4px;
		border: 1px solid transparent;
		font-family: var(--font-mono);
		font-size: 0.92rem;
		line-height: 1.5;
		white-space: pre;
		overflow: hidden;
		text-overflow: ellipsis;
		text-align: center;
		cursor: pointer;
		color: var(--ink);
		touch-action: manipulation;
	}
	.dense .chip {
		font-size: 0.82rem;
		padding: 0 4px;
	}
	.c0 {
		background: color-mix(in srgb, var(--cat-0) 16%, var(--surface));
	}
	.c1 {
		background: color-mix(in srgb, var(--cat-3) 22%, var(--surface));
	}
	.c2 {
		background: color-mix(in srgb, var(--cat-2) 16%, var(--surface));
	}
	.c3 {
		background: color-mix(in srgb, var(--cat-4) 18%, var(--surface));
	}
	.chip :global(.hex) {
		font-size: 0.72em;
		letter-spacing: 0.02em;
		color: var(--ink-2);
	}
	.chip.partial {
		border: 1px dashed var(--ink-2);
		background: var(--surface-2);
	}
	.chip.special {
		border: 1px dashed var(--ink-2);
		background: var(--surface-2);
		font-size: 0.78rem;
	}
	.chip.unk {
		border: 1.5px solid var(--k);
		background: color-mix(in srgb, var(--k) 14%, var(--surface));
		font-size: 0.78rem;
		font-weight: 700;
	}
	.chip.cont {
		border-left: 2px solid var(--ink-2);
	}
	.chip.new {
		box-shadow: 0 0 0 2px var(--accent);
	}
	.chip.pair {
		box-shadow: 0 0 0 2px var(--seq-mid);
		background: color-mix(in srgb, var(--seq-mid) 30%, var(--surface));
	}
	.chip.link {
		box-shadow: 0 0 0 2px var(--highlight);
		background: var(--accent-soft);
	}
	.chip.sel {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.chip:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.id {
		font-size: 10px;
		color: var(--muted);
		line-height: 1.3;
	}
	.tip {
		position: absolute;
		bottom: calc(100% + 6px);
		left: 50%;
		z-index: 50;
		display: grid;
		gap: 1px;
		width: max-content;
		max-width: 240px;
		padding: 5px 8px;
		border-radius: var(--r2);
		background: var(--ink);
		color: var(--bg);
		font-size: var(--fs-xs);
		line-height: 1.4;
		pointer-events: none;
		box-shadow: var(--shadow-2);
		white-space: normal;
	}
</style>
