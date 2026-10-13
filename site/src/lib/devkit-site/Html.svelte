<!--
  Rendered Markdown of the DevKit pages (made at build time by md.ts from the repository's own files). Swaps the site-path
  token for the page's relative way to the site root (base.ts relBase: the same string on the server and in the browser),
  labels the code blocks' copy buttons and the heading anchors in the current language and copies on click (one
  listener for the whole block). The styles below are the reading styles of every generated text on /devkit/ (redesign
  spec §6.8): text 16/26, code blocks mono 14/22 on --well with a copy button, API tables at 14 px.
  class="compact": 14/22 text, for notes and the error-code explanations.
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
	const copyLabel = () => (i18n.lang === 'zh' ? '复制' : 'Copy');

	// copy buttons and heading anchors: label them in the current language
	$effect(() => {
		if (!el) return;
		void out;
		const label = copyLabel();
		for (const b of el.querySelectorAll<HTMLButtonElement>('button[data-copy]')) {
			b.textContent = label;
			b.setAttribute('aria-label', t({ en: 'Copy this code', zh: '复制这段代码' }));
		}
		for (const a of el.querySelectorAll<HTMLAnchorElement>('a.anchor')) a.setAttribute('aria-label', t({ en: 'Link to this heading', zh: '链接到这个标题' }));
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
				b.textContent = i18n.lang === 'zh' ? '已复制' : 'Copied';
				setTimeout(() => (b.textContent = copyLabel()), 1400);
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
		font-size: var(--t-md);
		line-height: 26px;
		overflow-wrap: break-word;
	}
	.dk-md.compact {
		font-size: var(--t-sm);
		line-height: 22px;
	}
	.dk-md :global(> :first-child) {
		margin-top: 0;
	}
	.dk-md :global(> :last-child) {
		margin-bottom: 0;
	}
	.dk-md :global(p) {
		margin: 0 0 var(--s4);
	}
	.dk-md :global(p),
	.dk-md :global(li) {
		max-width: 75ch;
	}
	.dk-md :global(ul),
	.dk-md :global(ol) {
		padding-left: 1.4em;
		margin: 0 0 var(--s4);
	}
	.dk-md :global(li + li) {
		margin-top: 6px;
	}
	.dk-md :global(li::marker) {
		color: var(--muted);
	}
	.dk-md :global(li > ul),
	.dk-md :global(li > ol) {
		margin: var(--s2) 0 0;
	}
	.dk-md :global(li > p) {
		margin: 0 0 var(--s2);
	}

	/* headings: the page's h1 is the section title; ### of DEVKIT.md is h2 here */
	.dk-md :global(h2),
	.dk-md :global(h3),
	.dk-md :global(h4) {
		position: relative;
		scroll-margin-top: calc(var(--header-h) + 16px);
	}
	.dk-md :global(h2) {
		margin: var(--s5) 0 var(--s3);
		padding-top: var(--s4);
		border-top: 1px solid var(--rule);
		font-size: var(--t-xl);
		line-height: var(--lh-xl);
	}
	.dk-md :global(h3) {
		margin: var(--s5) 0 var(--s2);
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.dk-md :global(h4) {
		margin: var(--s5) 0 var(--s2);
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.dk-md :global(> h2:first-child) {
		padding-top: 0;
		border-top: 0;
	}
	.dk-md :global(.anchor) {
		margin-left: 0.35em;
		color: var(--muted);
		font-weight: var(--w-body);
		text-decoration: none;
		opacity: 0;
		transition: opacity var(--dur-1) var(--ease);
	}
	.dk-md :global(:is(h2, h3, h4):hover .anchor),
	.dk-md :global(.anchor:focus-visible) {
		opacity: 1;
	}
	/* the heading a link opened: the "here" bar beside it */
	.dk-md :global(:is(h2, h3, h4):target) {
		padding-left: var(--s3);
		border-left: 3px solid var(--here);
	}

	/* inline code: the mono at the text's size (font-size-adjust matches the x-height, base.css) */
	.dk-md :global(code) {
		padding: 1px 5px;
		border-radius: var(--r1);
		background: color-mix(in srgb, var(--sunken) 75%, transparent);
		overflow-wrap: anywhere;
	}
	/* code reads as typed: no ligatures (=> stays two characters) */
	.dk-md :global(code),
	.dk-md :global(pre) {
		font-variant-ligatures: none;
	}
	.dk-md :global(pre code) {
		padding: 0;
		border: 0;
		background: none;
		overflow-wrap: normal;
	}

	/* code blocks: mono 14/22 on --well, a head with the language or file and the copy button */
	.dk-md :global(.code) {
		margin: 0 0 var(--s4);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--well);
		overflow: hidden;
		min-width: 0;
	}
	.dk-md :global(.code-head) {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--s3);
		min-height: 36px;
		padding: 0 4px 0 var(--s4);
		border-bottom: 1px solid var(--rule);
		background: color-mix(in srgb, var(--sunken) 45%, var(--well));
		font: var(--w-body) var(--t-xs) / var(--lh-xs) var(--font-mono);
		font-size-adjust: none;
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
		min-height: 30px;
		padding: 0 var(--s3);
		border: 1px solid transparent;
		border-radius: var(--r2);
		background: transparent;
		color: var(--ink-2);
		font: var(--w-ui) var(--t-xs) / var(--lh-xs) var(--font-sans);
	}
	.dk-md :global(.code-copy:hover) {
		border-color: var(--line-control);
		background: var(--panel);
		color: var(--ink);
	}
	.dk-md :global(pre) {
		margin: 0;
		padding: var(--s3) var(--s4);
		border-radius: 0;
		background: none;
		font-size: var(--t-sm);
		line-height: 22px;
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
		font-weight: var(--w-ui);
	}
	.dk-md :global(.tn) {
		color: var(--dk-num);
	}
	.dk-md :global(.tp) {
		color: var(--dk-key);
	}

	/* API tables: 14 px on the panel, a well header row */
	.dk-md :global(.table-wrap) {
		margin: 0 0 var(--s4);
		overflow-x: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
	}
	.dk-md :global(table) {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.dk-md :global(th),
	.dk-md :global(td) {
		padding: var(--s2) var(--s3);
		text-align: left;
		vertical-align: top;
		border-bottom: 1px solid var(--rule);
	}
	.dk-md :global(th) {
		background: var(--well);
		color: var(--ink);
		font-weight: var(--w-ui);
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
		padding: var(--s3) var(--s4);
		border: 1px solid var(--rule);
		border-left: 3px solid var(--line-control);
		border-radius: 0 var(--r3) var(--r3) 0;
		background: var(--panel);
		color: var(--ink-2);
	}
	.dk-md :global(blockquote p:last-child) {
		margin-bottom: 0;
	}
	.dk-md :global(hr) {
		margin: var(--s6) 0;
	}
	@media (max-width: 759px) {
		.dk-md :global(pre) {
			padding-inline: var(--s3);
		}
		.dk-md :global(th),
		.dk-md :global(td) {
			padding: var(--s2);
		}
	}
</style>
