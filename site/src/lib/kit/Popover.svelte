<!--
  Popover: a small panel anchored to the control that opened it (redesign spec §4.4: the step list above the rail, the
  provenance detail, About). It is a native modal <dialog>: focus stays inside, Esc and a click outside close it, and
  focus goes back to the opener. On phones (below 760 px) it becomes a bottom sheet, so it never runs off screen.

    <button bind:this={btn} aria-expanded={open} onclick={() => (open = !open)}>Steps</button>
    <Popover bind:open anchor={btn} label="Steps" placement="top">…</Popover>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		open: boolean;
		/** the control it hangs from */
		anchor?: HTMLElement | null;
		/** its accessible name */
		label: string;
		/** above or below the anchor (flips when there is no room) */
		placement?: 'top' | 'bottom';
		/** which edges line up with the anchor */
		align?: 'start' | 'center' | 'end';
		/** width in px (capped by the window) */
		width?: number;
		onclose?: () => void;
		children: Snippet;
		/** a sticky footer (actions, key hints) */
		footer?: Snippet;
	}
	let { open = $bindable(false), anchor = null, label, placement = 'bottom', align = 'start', width = 360, onclose, children, footer }: Props =
		$props();

	let dlg: HTMLDialogElement | undefined = $state();
	let pos = $state({ left: 0, top: 0, maxH: 400, side: 'bottom' as 'top' | 'bottom' });
	let opener: Element | null = null;

	const GAP = 8;
	const MARGIN = 8;

	function place() {
		if (!anchor || !dlg) return;
		const r = anchor.getBoundingClientRect();
		const w = Math.min(width, innerWidth - 2 * MARGIN);
		let left = align === 'start' ? r.left : align === 'end' ? r.right - w : r.left + r.width / 2 - w / 2;
		left = Math.max(MARGIN, Math.min(left, innerWidth - MARGIN - w));
		const above = r.top - GAP - MARGIN;
		const below = innerHeight - r.bottom - GAP - MARGIN;
		const side = placement === 'top' ? (above >= 200 || above >= below ? 'top' : 'bottom') : below >= 200 || below >= above ? 'bottom' : 'top';
		const maxH = Math.max(120, side === 'top' ? above : below);
		const top = side === 'top' ? r.top - GAP : r.bottom + GAP;
		pos = { left, top, maxH, side };
	}

	$effect(() => {
		const d = dlg;
		if (!d) return;
		if (open && !d.open) {
			opener = document.activeElement;
			place();
			d.showModal();
		} else if (!open && d.open) d.close();
	});

	$effect(() => {
		if (!open) return;
		const on = () => place();
		addEventListener('resize', on);
		addEventListener('scroll', on, true);
		return () => {
			removeEventListener('resize', on);
			removeEventListener('scroll', on, true);
		};
	});

	function closed() {
		open = false;
		onclose?.();
		if (opener instanceof HTMLElement && opener.isConnected && document.activeElement !== opener) opener.focus();
		opener = null;
	}

	function backdrop(e: MouseEvent) {
		if (e.target === dlg) open = false;
	}
</script>

<dialog
	bind:this={dlg}
	class="pop {pos.side}"
	aria-label={label}
	style:--pop-left="{pos.left}px"
	style:--pop-top="{pos.top}px"
	style:--pop-w="{width}px"
	style:--pop-max-h="{pos.maxH}px"
	onclose={closed}
	onclick={backdrop}
>
	<div class="box">
		<div class="body">{@render children()}</div>
		{#if footer}<div class="foot">{@render footer()}</div>{/if}
	</div>
</dialog>

<style>
	.pop {
		position: fixed;
		inset: auto;
		margin: 0;
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--ink);
		max-width: none;
		max-height: none;
		overflow: visible;
		left: var(--pop-left);
		width: min(var(--pop-w), calc(100vw - 16px));
	}
	.pop.bottom {
		top: var(--pop-top);
	}
	.pop.top {
		top: var(--pop-top);
		transform: translateY(-100%);
	}
	.pop::backdrop {
		background: transparent;
	}
	.box {
		display: flex;
		flex-direction: column;
		max-height: var(--pop-max-h);
		border: 1px solid var(--rule-strong);
		border-radius: var(--r4);
		background: var(--panel);
		box-shadow: var(--shadow-2);
		overflow: hidden;
	}
	.body {
		overflow: auto;
		overscroll-behavior: contain;
		padding: var(--s3);
	}
	.foot {
		border-top: 1px solid var(--rule);
		padding: var(--s2) var(--s3);
	}
	.pop[open] .box {
		animation: pop var(--dur-2) var(--ease);
	}
	@keyframes pop {
		from {
			opacity: 0;
			transform: translateY(4px);
		}
	}
	/* phones: a bottom sheet */
	@media (max-width: 759px) {
		.pop,
		.pop.top,
		.pop.bottom {
			inset: auto 0 0 0;
			top: auto;
			left: 0;
			width: 100%;
			transform: none;
		}
		.pop::backdrop {
			background: rgb(10 12 14 / 0.42);
		}
		.box {
			max-height: 85dvh;
			border-radius: var(--r5) var(--r5) 0 0;
			border-width: 1px 0 0;
			padding-bottom: env(safe-area-inset-bottom, 0px);
		}
		.body {
			padding: var(--s4);
		}
	}
</style>
