<!--
  TokenChips: tokens as alternating-color chips. Hover, focus or tap shows the id and bytes.
  Selectable with mouse, touch or keyboard (arrow keys move, Enter/Space select).

  <TokenChips tokens={[{ text: 'Hel', id: 72 }, { text: 'lo', id: 9 }]} bind:selected={i} showIds />

  Whitespace is made visible (␣ for space, ↵ for newline). Special tokens get a dashed border.
-->
<script lang="ts">
	import { visibleToken, utf8Hex } from './format';
	import { ui, t, colon, type Text } from '$lib/i18n/i18n.svelte';
	import type { TokenItem } from './types';

	interface Props {
		tokens: TokenItem[];
		selected?: number | null;
		/** show ids under the chips */
		showIds?: boolean;
		/** allow selection (otherwise chips are display-only but still focusable for the tooltip) */
		selectable?: boolean;
		label?: Text;
		onselect?: (index: number, token: TokenItem) => void;
		onhover?: (index: number | null) => void;
		/** highlight these indices (e.g. the tokens a query attends to) */
		marked?: number[];
	}
	let {
		tokens,
		selected = $bindable(null),
		showIds = false,
		selectable = true,
		label,
		onselect,
		onhover,
		marked = []
	}: Props = $props();

	let focusIdx = $state(0);
	let tip = $state<number | null>(null);
	let chips: HTMLElement[] = $state([]);
	const gid = $props.id();

	function select(i: number) {
		if (!selectable) return;
		selected = i;
		onselect?.(i, tokens[i]);
	}
	function onkeydown(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(tokens.length - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = tokens.length - 1;
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
		chips[j]?.focus();
	}
	function bytesOf(tok: TokenItem): string {
		return tok.bytes ? tok.bytes.map((b) => b.toString(16).padStart(2, '0')).join(' ') : utf8Hex(tok.text);
	}
	function show(i: number | null) {
		tip = i;
		onhover?.(i);
	}
</script>

<div class="chips" role="listbox" aria-orientation="horizontal" aria-labelledby={label ? gid : undefined} aria-label={label ? undefined : ui('tokens')}>
	{#if label}<span id={gid} class="visually-hidden">{t(label)}</span>{/if}
	{#each tokens as tok, i (i)}
		<span class="cell">
			<span
				bind:this={chips[i]}
				class="chip c{i % 4}"
				class:special={tok.special}
				class:gen={tok.tone === 'generated'}
				class:muted={tok.tone === 'muted'}
				class:sel={selected === i}
				class:marked={marked.includes(i)}
				role="option"
				aria-selected={selected === i}
				tabindex={i === (selected ?? focusIdx) ? 0 : -1}
				onclick={() => select(i)}
				onkeydown={(e) => onkeydown(e, i)}
				onpointerenter={() => show(i)}
				onpointerleave={() => show(null)}
				onfocus={() => {
					focusIdx = i;
					show(i);
				}}
				onblur={() => show(null)}
				aria-describedby={tip === i ? `${gid}-tip` : undefined}
			>{visibleToken(tok.text)}</span>
			{#if showIds && tok.id !== undefined}<span class="id num">{tok.id}</span>{/if}
			{#if tip === i}
				<span class="tip" role="tooltip" id="{gid}-tip">
					<span><b>{ui('token')} {i}</b>{#if tok.special}{' · '}{ui('special')}{/if}</span>
					{#if tok.id !== undefined}<span>{ui('tokenId')}{colon()} <span class="num">{tok.id}</span></span>{/if}
					<span>{ui('bytes')}{colon()} <span class="num">{bytesOf(tok)}</span></span>
				</span>
			{/if}
		</span>
	{/each}
</div>

<style>
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 2px;
		align-items: flex-start;
	}
	.cell {
		position: relative;
		display: inline-flex;
		flex-direction: column;
		align-items: center;
	}
	.chip {
		display: inline-block;
		min-width: 1.4em;
		padding: 2px 5px;
		border-radius: 4px;
		border: 1px solid transparent;
		font-family: var(--font-mono);
		font-size: 0.95rem;
		line-height: 1.5;
		white-space: pre;
		text-align: center;
		cursor: pointer;
		color: var(--ink);
		touch-action: manipulation;
	}
	/* alternating tints, mixed from fixed hues so they never look like data colors */
	.c0 {
		background: color-mix(in srgb, var(--cat-0) 18%, var(--surface));
	}
	.c1 {
		background: color-mix(in srgb, var(--cat-3) 22%, var(--surface));
	}
	.c2 {
		background: color-mix(in srgb, var(--cat-2) 18%, var(--surface));
	}
	.c3 {
		background: color-mix(in srgb, var(--cat-4) 20%, var(--surface));
	}
	.chip.special {
		border: 1px dashed var(--ink-2);
		background: var(--surface-2);
		font-size: 0.8rem;
	}
	.chip.gen {
		background: color-mix(in srgb, var(--generated) 16%, var(--surface));
		border-color: var(--generated);
	}
	.chip.muted {
		opacity: 0.45;
		text-decoration: line-through;
		text-decoration-color: var(--muted);
	}
	.chip.marked {
		box-shadow: inset 0 -3px 0 var(--k);
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
		transform: translateX(-50%);
		z-index: 50;
		display: grid;
		gap: 1px;
		width: max-content;
		max-width: 260px;
		padding: 6px 9px;
		border-radius: var(--r2);
		background: var(--ink);
		color: var(--bg);
		font-size: var(--fs-xs);
		line-height: 1.4;
		pointer-events: none;
		box-shadow: var(--shadow-2);
	}
</style>
