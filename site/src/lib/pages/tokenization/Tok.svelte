<!--
  One static token chip (no focus, no handlers): for formulas, tables and lists.
  Raw-byte parts (<0xE7>) are drawn as hex. <Tok text=" the" /> <Tok text="我" mark="new" />
-->
<script lang="ts">
	import { visibleToken } from '$lib/kit/format';

	interface Props {
		text: string;
		mark?: 'new' | 'pair' | 'link' | null;
		kind?: 'special' | 'unk' | 'partial' | 'cont';
		small?: boolean;
	}
	let { text, mark = null, kind, small = false }: Props = $props();

	const parts = $derived.by(() => {
		const out: { hex: boolean; s: string }[] = [];
		const re = /((?:<0x[0-9A-F]{2}>)+)/g;
		let last = 0;
		for (const m of text.matchAll(re)) {
			const i = m.index ?? 0;
			if (i > last) out.push({ hex: false, s: text.slice(last, i) });
			out.push({ hex: true, s: m[1].replace(/<0x([0-9A-F]{2})>/g, '$1 ').trim() });
			last = i + m[1].length;
		}
		if (last < text.length) out.push({ hex: false, s: text.slice(last) });
		return out;
	});
	const isPartial = $derived(kind === 'partial' || (!kind && parts.some((p) => p.hex)));
</script>

<span class="tok {mark ?? ''} {kind ?? ''}" class:partial={isPartial} class:small>{#each parts as p, k (k)}{#if p.hex}<span class="hex">{p.s}</span>{:else}{visibleToken(p.s)}{/if}{/each}</span>

<style>
	.tok {
		display: inline-block;
		min-width: 1.3em;
		padding: 1px 5px;
		border-radius: 4px;
		border: 1px solid transparent;
		background: color-mix(in srgb, var(--cat-0) 16%, var(--surface));
		font-family: var(--font-mono);
		font-size: 0.92rem;
		line-height: 1.5;
		white-space: pre;
		text-align: center;
		color: var(--ink);
		vertical-align: middle;
	}
	.small {
		font-size: 0.8rem;
		padding: 0 4px;
	}
	.hex {
		font-size: max(var(--t-xs), 0.72em);
		color: var(--ink-2);
	}
	.partial {
		border: 1px dashed var(--ink-2);
		background: var(--surface-2);
	}
	.special {
		border: 1px dashed var(--ink-2);
		background: var(--surface-2);
		font-size: 0.78rem;
	}
	.unk {
		border: 1.5px solid var(--k);
		background: color-mix(in srgb, var(--k) 14%, var(--surface));
		font-weight: 700;
		font-size: 0.78rem;
	}
	.new {
		box-shadow: 0 0 0 2px var(--accent);
	}
	.pair {
		box-shadow: 0 0 0 2px var(--seq-mid);
	}
	.link {
		box-shadow: 0 0 0 2px var(--highlight);
		background: var(--accent-soft);
	}
</style>
