<!--
  Tooltip: shows `tip` (text) or the `content` snippet on hover, keyboard focus or tap.
  Escape closes it. The wrapper is focusable unless `focusable={false}` (use that when the
  child is already a button or link; focus inside still opens the tip).

  <Tooltip tip="Full value: 0.123456">0.123</Tooltip>
  <Tooltip>{#snippet content()}<b>Rich</b> tip{/snippet}<span>hover me</span></Tooltip>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		tip?: string;
		content?: Snippet;
		children: Snippet;
		focusable?: boolean;
		/** preferred side; flips automatically near the viewport edge */
		placement?: 'top' | 'bottom';
		/** extra class on the wrapper */
		class?: string;
		/** max bubble width in px */
		width?: number;
	}

	let {
		tip = '',
		content,
		children,
		focusable = true,
		placement = 'top',
		class: klass = '',
		width = 280
	}: Props = $props();

	const id = $props.id();
	let open = $state(false);
	let pinned = $state(false); // opened by tap/click: stays until tapped again or Escape
	let wrap: HTMLSpanElement | undefined = $state();
	let bubble: HTMLSpanElement | undefined = $state();
	let side = $state<'top' | 'bottom'>('top');
	let shift = $state(0);

	$effect(() => {
		side = placement;
	});

	$effect(() => {
		if (!open || !bubble || !wrap) return;
		// keep the bubble inside the viewport
		shift = 0;
		const r = bubble.getBoundingClientRect();
		const margin = 8;
		if (r.left < margin) shift = margin - r.left;
		else if (r.right > innerWidth - margin) shift = innerWidth - margin - r.right;
		const wr = wrap.getBoundingClientRect();
		if (placement === 'top' && wr.top - r.height < margin + 56) side = 'bottom';
		else side = placement;
	});

	function show() {
		open = true;
	}
	function hide() {
		if (!pinned) open = false;
	}
	function toggle(e: MouseEvent) {
		// a tap on touch devices: pin open / closed. Mouse users get hover anyway.
		if ((e as PointerEvent).pointerType === 'mouse') return;
		pinned = !pinned;
		open = pinned;
	}
	function onkey(e: KeyboardEvent) {
		if (e.key === 'Escape' && open) {
			open = false;
			pinned = false;
			e.stopPropagation();
		}
	}
	function onfocusout(e: FocusEvent) {
		if (wrap && e.relatedTarget instanceof Node && wrap.contains(e.relatedTarget)) return;
		pinned = false;
		open = false;
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<span
	bind:this={wrap}
	class="tt {klass}"
	tabindex={focusable ? 0 : undefined}
	aria-describedby={open ? id : undefined}
	role={focusable ? 'button' : undefined}
	onmouseenter={show}
	onmouseleave={hide}
	onfocusin={show}
	{onfocusout}
	onclick={toggle}
	onkeydown={onkey}
>
	{@render children()}
	{#if open && (tip || content)}
		<span
			bind:this={bubble}
			{id}
			role="tooltip"
			class="bubble {side}"
			style="--shift: {shift}px; max-width: min({width}px, calc(100vw - 16px));"
		>
			{#if content}{@render content()}{:else}{tip}{/if}
		</span>
	{/if}
</span>

<style>
	.tt {
		position: relative;
		display: inline-block;
		cursor: help;
	}
	.tt[role='button']:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
		border-radius: 3px;
	}
	.bubble {
		position: absolute;
		left: 50%;
		z-index: 60;
		width: max-content;
		transform: translateX(calc(-50% + var(--shift)));
		padding: 8px 10px;
		border-radius: var(--r2);
		background: var(--ink);
		color: var(--bg);
		font-family: var(--font-body);
		font-size: var(--fs-sm);
		font-weight: 400;
		line-height: 1.45;
		letter-spacing: normal;
		text-transform: none;
		text-align: left;
		white-space: normal;
		box-shadow: var(--shadow-2);
		cursor: auto;
	}
	/* invisible bridge so the pointer can move from the trigger into the bubble */
	.bubble::after {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		height: 10px;
	}
	.bubble.top {
		bottom: calc(100% + 8px);
	}
	.bubble.top::after {
		top: 100%;
	}
	.bubble.bottom {
		top: calc(100% + 8px);
	}
	.bubble.bottom::after {
		bottom: 100%;
	}
	.bubble :global(a) {
		color: inherit;
	}
</style>
