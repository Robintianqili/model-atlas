<!--
  Rendered Markdown of the DevKit pages (made at build time by md.ts from the repository's own files). Swaps the site-path
  token for the page's relative way to the site root (base.ts relBase: the same string on the server and in the browser),
  labels the code blocks' copy buttons in the current language and copies on click (one listener for the whole block).
  The styles below are the reading styles of every generated text on /devkit/.
-->
<script lang="ts">
	import { page } from '$app/state';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import { relBase, withBase } from './base.ts';
	import { keyScroll } from '$lib/kit/keyscroll';

	interface Props {
		html: string;
		/** the language of the text (the generated texts are English) */
		lang?: string;
		class?: string;
	}
	let { html, lang = 'en', class: cls = '' }: Props = $props();
	const out = $derived(withBase(html, relBase(page.url.pathname)));
	let el: HTMLDivElement | undefined = $state();

	// copy buttons: label them in the current language, copy the block's text on click
	$effect(() => {
		if (!el) return;
		void out;
		const label = i18n.lang === 'zh' ? '复制' : 'Copy';
		for (const b of el.querySelectorAll<HTMLButtonElement>('button[data-copy]')) {
			b.textContent = label;
			b.setAttribute('aria-label', t({ en: 'Copy this code', zh: '复制这段代码' }));
		}
	});
	$effect(() => {
		const node = el;
		if (!node) return;
		const onClick = async (e: MouseEvent) => {
			const b = (e.target as HTMLElement | null)?.closest<HTMLButtonElement>('button[data-copy]');
			if (!b) return;
			const code = b.closest('.code')?.querySelector('pre')?.textContent ?? '';
			try {
				await navigator.clipboard.writeText(code);
				b.textContent = '✓';
				setTimeout(() => (b.textContent = i18n.lang === 'zh' ? '复制' : 'Copy'), 1400);
			} catch {
				/* clipboard blocked */
			}
		};
		node.addEventListener('click', onClick);
		return () => node.removeEventListener('click', onClick);
	});
</script>

<!-- keyScroll: wide code blocks, tables and formulas get a tab stop while they scroll (keyboard access) -->
<div class="dk-md {cls}" {lang} bind:this={el} use:keyScroll={'pre, .table-wrap, .katex-display'}>{@html out}</div>

<style>
	.dk-md {
		min-width: 0;
		max-width: 100%;
		overflow-wrap: break-word;
	}
	.dk-md :global(> :first-child) {
		margin-top: 0;
	}
	.dk-md :global(p),
	.dk-md :global(li) {
		max-width: var(--measure);
	}
	.dk-md :global(ul),
	.dk-md :global(ol) {
		padding-left: 1.4em;
		margin: 0 0 var(--s4);
	}
	.dk-md :global(li + li) {
		margin-top: 6px;
	}
	.dk-md :global(li > ul),
	.dk-md :global(li > ol) {
		margin: 6px 0 0;
	}
	.dk-md :global(li > p) {
		margin: 0 0 8px;
	}
	.dk-md :global(h2),
	.dk-md :global(h3),
	.dk-md :global(h4) {
		position: relative;
		margin: var(--s6) 0 var(--s3);
		scroll-margin-top: calc(var(--header-h) + 16px);
	}
	.dk-md :global(h2) {
		font-size: var(--fs-xl);
	}
	.dk-md :global(h3) {
		font-size: var(--fs-lg);
	}
	.dk-md :global(h4) {
		font-size: var(--fs-md);
	}
	.dk-md :global(.anchor) {
		margin-left: 0.35em;
		color: var(--muted);
		text-decoration: none;
		font-weight: 400;
		opacity: 0;
		transition: opacity var(--dur-1) var(--ease);
	}
	.dk-md :global(:is(h2, h3, h4):hover .anchor),
	.dk-md :global(.anchor:focus-visible) {
		opacity: 1;
	}
	.dk-md :global(:target) {
		background: linear-gradient(transparent 60%, var(--accent-soft) 60%);
	}
	.dk-md :global(code) {
		padding: 0.05em 0.3em;
		border-radius: 4px;
		background: var(--surface-2);
		font-size: 0.86em;
		overflow-wrap: anywhere;
	}
	.dk-md :global(pre code) {
		padding: 0;
		background: none;
		font-size: inherit;
		overflow-wrap: normal;
	}
	.dk-md :global(.code) {
		margin: 0 0 var(--s4);
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface-2);
		overflow: hidden;
		min-width: 0;
	}
	.dk-md :global(.code-head) {
		display: flex;
		align-items: center;
		justify-content: space-between;
		min-height: 30px;
		padding: 0 4px 0 12px;
		border-bottom: 1px solid var(--rule);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.dk-md :global(.code-lang) {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.dk-md :global(.code-copy) {
		flex: none;
		min-height: 28px;
		padding: 0 10px;
		border: 0;
		border-radius: var(--r1);
		background: transparent;
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	.dk-md :global(.code-copy:hover) {
		background: var(--surface-3);
	}
	.dk-md :global(pre) {
		margin: 0;
		padding: 10px 12px;
		border-radius: 0;
		background: none;
		font-size: 0.8rem;
		line-height: 1.55;
		tab-size: 2;
	}
	.dk-md :global(.tc) {
		color: var(--dk-com);
		font-style: italic;
	}
	.dk-md :global(.ts) {
		color: var(--dk-str);
	}
	.dk-md :global(.tk) {
		color: var(--dk-kw);
		font-weight: 600;
	}
	.dk-md :global(.tn) {
		color: var(--dk-num);
	}
	.dk-md :global(.tp) {
		color: var(--dk-key);
	}
	.dk-md :global(.table-wrap) {
		margin: 0 0 var(--s4);
		overflow-x: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.dk-md :global(table) {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--fs-sm);
		line-height: 1.45;
	}
	.dk-md :global(th),
	.dk-md :global(td) {
		padding: 7px 10px;
		text-align: left;
		vertical-align: top;
		border-bottom: 1px solid var(--rule);
	}
	.dk-md :global(th) {
		background: var(--surface-2);
		font-weight: 600;
		white-space: nowrap;
	}
	.dk-md :global(tr:last-child td) {
		border-bottom: 0;
	}
	.dk-md :global(td code) {
		overflow-wrap: break-word;
	}
	.dk-md :global(td:first-child code) {
		white-space: nowrap;
	}
	.dk-md :global(blockquote) {
		margin: 0 0 var(--s4);
		padding: 8px 14px;
		border-left: 3px solid var(--accent);
		background: var(--surface-2);
		border-radius: 0 var(--r2) var(--r2) 0;
		color: var(--ink-2);
	}
	.dk-md :global(blockquote p:last-child) {
		margin-bottom: 0;
	}
	.dk-md :global(hr) {
		margin: var(--s5) 0;
	}
</style>
