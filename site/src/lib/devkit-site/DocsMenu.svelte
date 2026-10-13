<!--
  The DevKit's menu below 960 px (redesign spec §6.8: "Phones: the nav becomes a sheet"): a native modal <dialog> that
  holds the docs tree, as a bottom sheet on phones and a start-side panel on tablets. showModal() traps focus and makes
  the page inert, Esc closes it, a click on the backdrop closes it, and focus goes back to the button that opened it.

  It follows the look and behaviour of $lib/kit/Sheet.svelte but is kept local on purpose: kit Sheet is otherwise
  imported only by the site header, and importing it from a DevKit page moves Sheet's icon dependencies out of the shared
  shell chunk into a chunk of their own that every page preloads (measured: +155 B gzip on the planned model pages,
  whose budget had about 0.4 KB left).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { ui } from '$lib/i18n/i18n.svelte';
	import Glyph from './Glyph.svelte';
	import { x } from './icons.ts';

	let { open = $bindable(false), title, children }: { open: boolean; title: string; children: Snippet } = $props();
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
		if (opener instanceof HTMLElement && opener.isConnected && document.activeElement !== opener) opener.focus();
		opener = null;
	}
</script>

<dialog bind:this={dlg} class="menu-sheet" aria-labelledby={tid} onclose={closed} onclick={(e) => e.target === dlg && (open = false)}>
	<div class="box">
		<div class="head">
			<h2 id={tid}>{title}</h2>
			<button type="button" class="icon-btn" aria-label={ui('close')} onclick={() => (open = false)}><Glyph d={x} /></button>
		</div>
		<div class="body">{@render children()}</div>
	</div>
</dialog>

<style>
	.menu-sheet {
		margin: 0 auto 0 0;
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--ink);
		max-width: none;
		max-height: none;
		height: 100%;
		overflow: visible;
	}
	.menu-sheet::backdrop {
		background: rgb(10 12 14 / 0.42);
	}
	.box {
		display: flex;
		flex-direction: column;
		width: min(400px, 100vw);
		height: 100dvh;
		background: var(--panel);
		box-shadow: var(--shadow-2);
		border-radius: 0 var(--r5) var(--r5) 0;
		overflow: hidden;
	}
	.head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--s3);
		min-height: 56px;
		padding: var(--s2) var(--s2) 0 var(--s5);
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
	/* phones: a bottom sheet above the safe area */
	@media (max-width: 759px) {
		.menu-sheet {
			margin: auto 0 0;
			width: 100%;
			height: fit-content;
		}
		.box {
			width: 100%;
			height: auto;
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
	}
	/* motion: the panel rises or slides in (300 ms; instant under reduced motion, base.css) */
	.menu-sheet[open] .box {
		animation: slide var(--dur-3) var(--ease);
	}
	.menu-sheet[open]::backdrop {
		animation: fade var(--dur-2) var(--ease);
	}
	@media (max-width: 759px) {
		.menu-sheet[open] .box {
			animation-name: rise;
		}
	}
	@keyframes slide {
		from {
			transform: translateX(-24px);
			opacity: 0;
		}
	}
	@keyframes rise {
		from {
			transform: translateY(16px);
			opacity: 0;
		}
	}
	@keyframes fade {
		from {
			opacity: 0;
		}
	}
</style>
