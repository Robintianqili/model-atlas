<!--
  Sheet: a modal panel (redesign spec §4.4-§4.5): a bottom sheet on phones, a centred dialog or an end-side panel on wider
  screens. It is a native <dialog> opened with showModal(), so focus stays inside, the page behind is inert and Esc
  closes it; focus goes back to the element that opened it. A click on the backdrop closes it too.

    <Sheet bind:open title={ui('menu')}>…</Sheet>
    <Sheet bind:open label="Steps" placement="bottom">…</Sheet>

  placement: 'auto' (bottom sheet below 760 px, centred dialog above; the default), 'bottom', 'center' or 'end'.
  Without a title, give a label (its accessible name); the close button is always there.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from './Icon.svelte';
	import { x } from './icons';
	import { ui } from '$lib/i18n/i18n.svelte';

	interface Props {
		open: boolean;
		/** a visible heading (also the accessible name) */
		title?: string;
		/** the accessible name when there is no title */
		label?: string;
		placement?: 'auto' | 'bottom' | 'center' | 'end';
		/** max width of a centred or end panel, in px */
		width?: number;
		onclose?: () => void;
		children: Snippet;
		/** a sticky footer (actions, key hints) */
		footer?: Snippet;
	}
	let { open = $bindable(false), title, label, placement = 'auto', width = 520, onclose, children, footer }: Props = $props();

	const tid = $props.id();
	let dlg: HTMLDialogElement | undefined = $state();
	let opener: Element | null = null;

	$effect(() => {
		const d = dlg;
		if (!d) return;
		if (open && !d.open) {
			opener = document.activeElement;
			d.showModal();
		} else if (!open && d.open) d.close();
	});

	function closed() {
		open = false;
		onclose?.();
		// focus goes back to the opener (browsers that already did it keep it)
		if (opener instanceof HTMLElement && opener.isConnected && document.activeElement !== opener) opener.focus();
		opener = null;
	}

	function backdrop(e: MouseEvent) {
		// a click on the dialog box itself (outside its content) is a click on the backdrop
		if (e.target === dlg) open = false;
	}
</script>

<dialog
	bind:this={dlg}
	class="sheet {placement}"
	style:--sheet-w="{width}px"
	aria-labelledby={title ? tid : undefined}
	aria-label={title ? undefined : label}
	onclose={closed}
	onclick={backdrop}
>
	<div class="box">
		<div class="head">
			{#if title}<h2 id={tid}>{title}</h2>{:else}<span></span>{/if}
			<button type="button" class="icon-btn close" aria-label={ui('close')} onclick={() => (open = false)}><Icon d={x} /></button>
		</div>
		<div class="body">
			{@render children()}
		</div>
		{#if footer}<div class="foot">{@render footer()}</div>{/if}
	</div>
</dialog>

<style>
	.sheet {
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--ink);
		max-width: none;
		max-height: none;
		overflow: visible;
	}
	.sheet::backdrop {
		background: rgb(10 12 14 / 0.42);
	}
	.box {
		display: flex;
		flex-direction: column;
		background: var(--panel);
		color: var(--ink);
		box-shadow: var(--shadow-2);
		overflow: hidden;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--s3);
		padding: var(--s2) var(--s2) 0 var(--s5);
		min-height: 52px;
	}
	.head h2 {
		margin: 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.body {
		overflow: auto;
		overscroll-behavior: contain;
		padding: var(--s2) var(--s5) var(--s5);
	}
	.foot {
		border-top: 1px solid var(--rule);
		padding: var(--s3) var(--s5);
		background: var(--panel);
	}

	/* centred dialog */
	.sheet.center .box {
		width: min(var(--sheet-w), calc(100vw - 32px));
		max-height: min(80vh, 760px);
		border-radius: var(--r5);
	}
	/* end-side panel (full height) */
	.sheet.end {
		margin: 0 0 0 auto;
		height: 100%;
	}
	.sheet.end .box {
		width: min(var(--sheet-w), 100vw);
		height: 100dvh;
		border-radius: var(--r5) 0 0 var(--r5);
	}
	/* bottom sheet: full width, rounded top, above the safe area */
	.sheet.bottom {
		margin: auto 0 0;
		width: 100%;
	}
	.sheet.bottom .box {
		width: 100%;
		max-height: 85dvh;
		border-radius: var(--r5) var(--r5) 0 0;
		padding-bottom: env(safe-area-inset-bottom, 0px);
	}
	.sheet.auto .box {
		width: min(var(--sheet-w), calc(100vw - 32px));
		max-height: min(80vh, 760px);
		border-radius: var(--r5);
	}
	@media (max-width: 759px) {
		.sheet.auto {
			margin: auto 0 0;
			width: 100%;
		}
		.sheet.auto .box {
			width: 100%;
			max-height: 85dvh;
			border-radius: var(--r5) var(--r5) 0 0;
			padding-bottom: env(safe-area-inset-bottom, 0px);
		}
		.head {
			padding-left: var(--s4);
		}
		.body {
			padding: var(--s2) var(--s4) var(--s4);
		}
		.foot {
			padding: var(--s3) var(--s4);
		}
	}

	/* motion: the panel slides and the backdrop fades (300 ms; instant under reduced motion, base.css) */
	.sheet[open] .box {
		animation: rise var(--dur-3) var(--ease);
	}
	.sheet.end[open] .box {
		animation-name: slide;
	}
	.sheet[open]::backdrop {
		animation: fade var(--dur-2) var(--ease);
	}
	@keyframes rise {
		from {
			transform: translateY(16px);
			opacity: 0;
		}
	}
	@keyframes slide {
		from {
			transform: translateX(24px);
			opacity: 0;
		}
	}
	@keyframes fade {
		from {
			opacity: 0;
		}
	}
</style>
