<!--
  UTF-8 byte view: every character of the text with its bytes (hex). Each byte is tinted by the
  token that contains it (same alternating tints as the chips), so a character split over two
  tokens shows two tints. Hover or tap a byte to highlight its token (`hover` is a token index).
-->
<script lang="ts">
	import { visibleToken } from '$lib/kit/format';
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		text: string;
		/** byte range of every token, in order */
		ranges: { start: number; end: number }[];
		hover?: number | null;
		label: Text;
		max?: number;
	}
	let { text, ranges, hover = $bindable(null), label, max = 160 }: Props = $props();

	const enc = new TextEncoder();
	const cols = $derived.by(() => {
		const out: { ch: string; bytes: { b: number; off: number; tok: number }[] }[] = [];
		let off = 0;
		let ti = 0;
		for (const ch of Array.from(text).slice(0, max)) {
			const bs = enc.encode(ch);
			const col = { ch, bytes: [] as { b: number; off: number; tok: number }[] };
			for (const b of bs) {
				while (ti < ranges.length - 1 && off >= ranges[ti].end) ti++;
				col.bytes.push({ b, off, tok: ti });
				off++;
			}
			out.push(col);
		}
		return out;
	});
	const total = $derived(enc.encode(text).length);
	const cut = $derived(Array.from(text).length > max);
</script>

<div class="bv" role="group" aria-label="{t(label)}: {total} bytes">
	{#each cols as c, i (i)}
		<div class="col" class:multi={c.bytes.length > 1}>
			<span class="ch" lang={/[一-鿿]/.test(c.ch) ? 'zh-CN' : undefined}>{visibleToken(c.ch)}</span>
			<span class="bytes">
				{#each c.bytes as b (b.off)}
					<span
						class="b num c{b.tok % 4}"
						class:link={hover === b.tok}
						role="presentation"
						onpointerenter={() => (hover = b.tok)}
						onpointerleave={() => (hover = null)}
						onclick={() => (hover = hover === b.tok ? null : b.tok)}
					>{b.b.toString(16).toUpperCase().padStart(2, '0')}</span>
				{/each}
			</span>
		</div>
	{/each}
	{#if cut}<span class="muted more">… {t({ en: 'first', zh: '只显示前' })} {max} {t({ en: 'characters shown', zh: '个字符' })}</span>{/if}
</div>

<style>
	.bv {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 3px;
		align-items: flex-start;
		min-width: 0;
	}
	.col {
		display: inline-flex;
		flex-direction: column;
		align-items: center;
		gap: 2px;
		padding: 2px;
		border-radius: 4px;
	}
	.col.multi {
		background: var(--surface-2);
	}
	.ch {
		font-family: var(--font-mono);
		font-size: 1rem;
		line-height: 1.3;
		white-space: pre;
	}
	.bytes {
		display: inline-flex;
		gap: 1px;
	}
	.b {
		font-size: var(--t-xs);
		padding: 1px 2px;
		border-radius: 3px;
		cursor: default;
		touch-action: manipulation;
		color: var(--ink);
	}
	.c0 {
		background: color-mix(in srgb, var(--cat-0) 22%, var(--surface));
	}
	.c1 {
		background: color-mix(in srgb, var(--cat-3) 30%, var(--surface));
	}
	.c2 {
		background: color-mix(in srgb, var(--cat-2) 22%, var(--surface));
	}
	.c3 {
		background: color-mix(in srgb, var(--cat-4) 24%, var(--surface));
	}
	.b.link {
		outline: 2px solid var(--highlight);
		outline-offset: 0;
		background: var(--accent-soft);
	}
	.more {
		font-size: var(--fs-xs);
		align-self: center;
	}
</style>
