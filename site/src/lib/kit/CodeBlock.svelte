<!--
  Code listing for the Code tab: line numbers, highlighted lines (the line that computed the
  selected tensor), an optional caption and a copy button. No syntax-highlighting dependency.
  <CodeBlock code={src} lang="python" highlight={[4, 5]} caption="pipeline/tinycnn/model.py" />
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from './keyscroll';

	interface Props {
		code: string;
		lang?: string;
		highlight?: number[];
		caption?: Text;
		/** first line number shown */
		start?: number;
	}
	let { code, lang = '', highlight = [], caption, start = 1 }: Props = $props();
	const lines = $derived(code.replace(/\n$/, '').split('\n'));
	let copied = $state(false);

	async function copy() {
		try {
			await navigator.clipboard.writeText(code);
			copied = true;
			setTimeout(() => (copied = false), 1400);
		} catch {
			/* clipboard blocked */
		}
	}
</script>

<figure class="cb">
	<figcaption>
		<span class="cap">{caption ? t(caption) : lang}</span>
		<button type="button" class="btn ghost copy" onclick={copy}>{copied ? t({ en: 'Copied', zh: '已复制' }) : t({ en: 'Copy', zh: '复制' })}</button>
	</figcaption>
	<pre use:keyScroll><code
			>{#each lines as l, i (i)}<span class="ln" class:hl={highlight.includes(i + start)}><span class="no" aria-hidden="true">{i + start}</span>{l || ' '}
</span>{/each}</code
		></pre>
</figure>

<style>
	/* redesign spec §6.8: mono 14/22 on --well, a head with the file and a copy button; the highlighted line has the
	   "here" style (an ink bar on a pale row), not the look-here yellow */
	.cb {
		margin: 0;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		overflow: hidden;
		background: var(--well);
		min-width: 0;
	}
	figcaption {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: var(--s3);
		min-height: 36px;
		padding: 0 4px 0 var(--s4);
		border-bottom: 1px solid var(--rule);
		background: color-mix(in srgb, var(--sunken) 45%, var(--well));
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	/* a file path or a sentence: the sans, wrapping on narrow screens */
	.cap {
		min-width: 0;
		padding-block: 6px;
		overflow-wrap: anywhere;
	}
	.copy {
		flex: none;
		min-height: 30px;
		padding: 0 var(--s3);
		border-radius: var(--r2);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
	}
	pre {
		margin: 0;
		font-variant-ligatures: none;
		padding: var(--s2) 0;
		border-radius: 0;
		background: none;
		font-size: var(--t-sm);
		line-height: 22px;
	}
	.ln {
		display: block;
		padding-right: var(--s4);
		white-space: pre;
	}
	.ln.hl {
		background: var(--here-bg);
		box-shadow: inset 3px 0 0 var(--here);
	}
	.no {
		display: inline-block;
		width: 3.2em;
		padding-right: 1em;
		text-align: right;
		color: var(--muted);
		user-select: none;
	}
</style>
