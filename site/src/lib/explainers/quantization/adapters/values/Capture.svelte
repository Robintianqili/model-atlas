<!--
  quantization Capture for the local 'values' adapter: the number x of the format widgets (a number field and a few
  numbers as buttons) and the eight values of the rounding ledger (eight number fields, counting from 0, and three sets
  as buttons: weight-like, one outlier, all positive). Reports { x, vals }.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { rawOf, V_MAX, type Raw } from '../../quant.ts';
	import { PRESETS, X_PRESETS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Raw> = $props();

	const r = $derived(rawOf(raw));
	const show = (v: number) => String(Number(v.toPrecision(6)));
	function setX(v: number) {
		if (Number.isFinite(v)) onchange?.({ ...r, x: v });
	}
	function setVal(i: number, v: number) {
		if (!Number.isFinite(v)) return;
		const vals = r.vals.slice();
		vals[i] = Math.max(-V_MAX, Math.min(V_MAX, v));
		onchange?.({ ...r, vals });
	}
	const read = (e: Event) => Number((e.currentTarget as HTMLInputElement).value);
</script>

<div class="values" class:compact>
	<fieldset>
		<legend>{t({ en: 'One number x (formats)', zh: '一个数 x（格式）' })}</legend>
		<div class="xrow">
			<input class="num" type="number" step="any" value={show(r.x)} aria-label={t({ en: 'The number x', zh: '数 x' })} onchange={(e) => setX(read(e))} />
			<div class="chips" role="group" aria-label={t({ en: 'Try a number', zh: '试一个数' })}>
				{#each X_PRESETS as v (v)}
					<button type="button" class="btn" onclick={() => setX(v)}>{show(v)}</button>
				{/each}
			</div>
		</div>
	</fieldset>
	<fieldset>
		<legend>{t({ en: 'Eight values (the ledger)', zh: '八个值（账本）' })}</legend>
		<div class="grid">
			{#each r.vals as v, i (i)}
				<label class="cell">
					<span class="idx">{i}</span>
					<input class="num" type="number" step="0.01" min={-V_MAX} max={V_MAX} value={show(v)} aria-label={t({ en: `Value ${i} (counting from 0)`, zh: `编号 ${i} 的值（从 0 数起）` })} onchange={(e) => setVal(i, read(e))} />
				</label>
			{/each}
		</div>
		<div class="chips" role="group" aria-label={t({ en: 'Start from a set of values', zh: '从一组值开始' })}>
			{#each PRESETS as p (p.id)}
				<button type="button" class="btn" onclick={() => onchange?.({ ...r, vals: p.vals.slice() })}>{t(compact ? p.short : p.label)}</button>
			{/each}
		</div>
	</fieldset>
</div>

<style>
	.values {
		display: grid;
		gap: 10px;
		min-width: 0;
		max-width: 560px;
	}
	fieldset {
		border: 1px solid var(--rule);
		border-radius: 6px;
		padding: 6px 8px 8px;
		margin: 0;
		min-width: 0;
		display: grid;
		gap: 6px;
	}
	legend {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		padding: 0 4px;
	}
	.xrow {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.xrow input {
		width: 9rem;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 4px 6px;
	}
	.cell {
		display: grid;
		grid-template-columns: 1rem minmax(0, 1fr);
		align-items: center;
		gap: 3px;
		font-size: var(--fs-xs);
	}
	.idx {
		color: var(--muted);
		font-family: var(--font-mono);
		text-align: right;
	}
	input.num {
		min-width: 0;
		width: 100%;
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
	.compact .grid {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}
</style>
