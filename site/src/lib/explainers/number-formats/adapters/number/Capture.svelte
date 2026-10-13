<!--
  number-formats Capture for the local 'number' adapter: one number x (a number field and a few numbers as buttons).
  Reports { x }. The bits in the stage set x too (views/bits).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { rawOf, type Raw } from '../../fmt.ts';
	import { X_PRESETS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Raw> = $props();

	const r = $derived(rawOf(raw));
	const show = (v: number) => String(Number(v.toPrecision(9)));
	function setX(v: number) {
		if (Number.isFinite(v)) onchange?.({ x: v });
	}
</script>

<div class="number" class:compact>
	<label class="xrow">
		<span class="lab">{t({ en: 'Your number x', zh: '你的数 x' })}</span>
		<input class="num" type="number" step="any" value={show(r.x)} onchange={(e) => setX(Number((e.currentTarget as HTMLInputElement).value))} />
	</label>
	<div class="chips" role="group" aria-label={t({ en: 'Try a number', zh: '试一个数' })}>
		{#each X_PRESETS as p (p.x)}
			<button type="button" class="btn" aria-pressed={r.x === p.x} onclick={() => setX(p.x)}>{t(p.label)}</button>
		{/each}
	</div>
</div>

<style>
	.number {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 560px;
	}
	.xrow {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.lab {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	input.num {
		width: 11rem;
		min-width: 0;
		font: inherit;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		padding: 2px 4px;
		color: var(--ink);
		background: var(--surface);
		border: 1px solid var(--rule-strong);
		border-radius: 4px;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.chips .btn {
		font-size: var(--fs-xs);
	}
	.compact input.num {
		width: 100%;
	}
</style>
