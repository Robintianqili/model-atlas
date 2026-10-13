<!--
  residual-connections Capture for the local 'setup' adapter: a switch for the skip connections and a slider for the
  number of layers (3 to 20; it reports on release), with the value next to it, and the starting setups as buttons
  (short names in the compact copy beside the lesson). Reports { depth, skip }.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { MAX_DEPTH, MIN_DEPTH, setupOf, type Setup } from '../../data.ts';
	import { STARTS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Setup> = $props();

	const s = $derived(setupOf(raw));
	// the depth the slider shows while it is dragged (committed on release: each change recomputes and retrains)
	let drag = $state<number | null>(null);
	const depth = $derived(drag ?? s.depth);
</script>

<div class="setup" class:compact>
	<label class="row switch">
		<input type="checkbox" checked={s.skip === 1} onchange={(e) => onchange?.({ ...s, skip: (e.currentTarget as HTMLInputElement).checked ? 1 : 0 })} />
		<span>{t({ en: 'Skip connections', zh: '跳跃连接' })}: <b>{t(s.skip ? { en: 'on', zh: '打开' } : { en: 'off', zh: '关闭' })}</b></span>
	</label>
	<label class="row">
		<span class="name">{t(compact ? { en: 'layers', zh: '层数' } : { en: 'Number of layers', zh: '层数' })}</span>
		<input
			type="range"
			min={MIN_DEPTH}
			max={MAX_DEPTH}
			step="1"
			value={depth}
			aria-valuetext={String(depth)}
			oninput={(e) => (drag = Number((e.currentTarget as HTMLInputElement).value))}
			onchange={(e) => {
				drag = null;
				onchange?.({ ...s, depth: Number((e.currentTarget as HTMLInputElement).value) });
			}}
		/>
		<output class="val num">{depth}</output>
	</label>
	<div class="starts" role="group" aria-label={t({ en: 'Start from', zh: '从这些设置开始' })}>
		{#each STARTS as st (st.id)}
			<button type="button" class="btn" onclick={() => onchange?.({ ...st.raw })}>{t(compact ? st.short : st.label)}</button>
		{/each}
	</div>
</div>

<style>
	.setup {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 520px;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 10rem) minmax(0, 1fr) 2.6rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.compact .row {
		grid-template-columns: 3.4rem minmax(0, 1fr) 2.4rem;
	}
	.row.switch {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.switch input {
		width: 18px;
		height: 18px;
		accent-color: var(--accent);
	}
	.name {
		color: var(--ink-2);
	}
	input[type='range'] {
		width: 100%;
		min-width: 0;
		accent-color: var(--accent);
	}
	.val {
		text-align: right;
	}
	.starts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.starts .btn {
		font-size: var(--fs-xs);
	}
</style>
