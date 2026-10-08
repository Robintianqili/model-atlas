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
		<span class="num">{caption ? t(caption) : lang}</span>
		<button type="button" class="btn ghost copy" onclick={copy}>{copied ? '✓' : t({ en: 'Copy', zh: '复制' })}</button>
	</figcaption>
	<pre use:keyScroll><code
			>{#each lines as l, i (i)}<span class="ln" class:hl={highlight.includes(i + start)}><span class="no" aria-hidden="true">{i + start}</span>{l || ' '}
</span>{/each}</code
		></pre>
</figure>

<style>
	.cb {
		margin: 0;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		overflow: hidden;
		background: var(--surface-2);
		min-width: 0;
	}
	figcaption {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 2px 4px 2px 12px;
		border-bottom: 1px solid var(--rule);
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.copy {
		min-height: 28px;
		font-size: var(--fs-xs);
	}
	pre {
		margin: 0;
		padding: 8px 0;
		border-radius: 0;
		font-size: 0.82rem;
	}
	.ln {
		display: block;
		padding-right: 12px;
		white-space: pre;
	}
	.ln.hl {
		background: var(--accent-soft);
		box-shadow: inset 3px 0 0 var(--accent);
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
