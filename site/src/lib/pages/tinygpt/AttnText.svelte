<!--
  The text, with each token shaded by how much the query token attends to it (one row of the
  attention matrix, sequential purple). Hover or focus a token to make it the query; click or
  press Enter to keep it. The query has a blue frame, the selected key a red one; later tokens
  are hatched (the causal mask).
-->
<script lang="ts">
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { TGState } from './state.svelte';

	let { S }: { S: TGState } = $props();

	let hoverQ = $state<number | null>(null);
	let chips: HTMLButtonElement[] = $state([]);
	const T = $derived(S.T);
	const q = $derived(hoverQ ?? S.pos);
	const w = $derived(S.run?.forward.t[`L${S.block}.attn.weights`]?.data);
	const sc = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const weightAt = (j: number) => (w && j <= q ? w[(S.head * T + q) * T + j] : 0);

	function onkeydown(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight') j = Math.min(T - 1, i + 1);
		else if (e.key === 'ArrowLeft') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = T - 1;
		else if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			S.selectPos(i);
			return;
		} else return;
		e.preventDefault();
		chips[j]?.focus();
	}
</script>

<div class="at" role="group" aria-label={t({ en: 'Attention of the query token over the text', zh: '查询词元在整段文本上的注意力' })}>
	<p class="cap">
		{t({ en: 'Where does', zh: '' })} <span class="q">{S.label(q)}</span>
		{t({ en: `look? Block ${S.block + 1}, head ${S.head + 1}:`, zh: `在看哪里？第 ${S.block + 1} 块，第 ${S.head + 1} 头：` })}
		<span class="muted">{t({ en: 'hover a token to make it the query; click to keep it', zh: '悬停在词元上把它设为查询，点击可固定' })}</span>
	</p>
	<div class="toks">
		{#each S.labels as lab, i (i)}
			{@const a = weightAt(i)}
			<button
				bind:this={chips[i]}
				type="button"
				class="tok"
				class:future={i > q}
				class:query={i === q}
				class:key={i === S.key && q === S.pos}
				style={i > q ? '' : `background: ${sc.css(a)}; color: ${sc.ink(a)}`}
				tabindex={i === S.pos ? 0 : -1}
				title={i > q ? t({ en: 'later token: masked', zh: '后面的词元：被掩码' }) : `${fmtPct(a)}`}
				aria-label="{lab}: {i > q ? t({ en: 'masked', zh: '被掩码' }) : fmtPct(a)}"
				onpointerenter={() => (hoverQ = i)}
				onpointerleave={() => (hoverQ = null)}
				onfocus={() => (hoverQ = i)}
				onblur={() => (hoverQ = null)}
				onclick={() => S.selectPos(i)}
				onkeydown={(e) => onkeydown(e, i)}>{lab}</button
			>
		{/each}
	</div>
</div>

<style>
	.at {
		display: grid;
		gap: 6px;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.q {
		font-family: var(--font-mono);
		padding: 0 4px;
		border-radius: 3px;
		box-shadow: inset 0 0 0 2px var(--q);
		white-space: pre;
	}
	.toks {
		display: flex;
		flex-wrap: wrap;
		gap: 3px 2px;
	}
	.tok {
		font-family: var(--font-mono);
		font-size: 0.92rem;
		line-height: 1.5;
		padding: 1px 4px;
		min-height: 30px;
		border: 1px solid var(--rule);
		border-radius: 4px;
		white-space: pre;
		background: var(--surface);
		color: var(--ink);
		touch-action: manipulation;
	}
	.tok.future {
		color: var(--muted);
		background: repeating-linear-gradient(-45deg, var(--surface-2) 0 4px, var(--surface) 4px 8px);
	}
	.tok.query {
		box-shadow: 0 0 0 2px var(--q);
		border-color: var(--q);
	}
	.tok.key {
		box-shadow: 0 0 0 2px var(--k);
		border-color: var(--k);
	}
	.tok:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
</style>
