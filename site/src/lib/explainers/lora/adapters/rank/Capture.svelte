<!--
  lora Capture for the local 'rank' adapter: LoRA's rank r as a slider (1 to 8) and four buttons, with what the rank
  costs on the toy layer (r·(16 + 16) trainable numbers against 16·16 = 256) and the scale α/r. Reports { r }.
-->
<script lang="ts">
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { ALPHA, D, R_MAX, R_MIN, loraCount, rankOf } from '../../data.ts';
	import { STARTS, type Rank } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Rank> = $props();

	const r = $derived(rankOf(raw));
	const n = $derived(loraCount(r, D, D));
	const set = (v: number, live = false) => onchange?.({ r: rankOf(v) }, { live });
	const scale = $derived(Number.isInteger(ALPHA / r) ? String(ALPHA / r) : (ALPHA / r).toFixed(2));
</script>

<div class="rank" class:compact>
	<label class="row">
		<span class="name">{compact ? 'r' : t({ en: 'Rank r', zh: '秩 r' })}</span>
		<input
			type="range"
			min={R_MIN}
			max={R_MAX}
			step="1"
			value={r}
			oninput={(e) => set(Number((e.currentTarget as HTMLInputElement).value), true)}
			onchange={(e) => set(Number((e.currentTarget as HTMLInputElement).value))}
		/>
		<output class="val num">{r}</output>
	</label>
	<p class="cost">
		{t({ en: 'Trainable numbers', zh: '可训练的数' })}{colon()} <strong class="num">{n}</strong>
		<span class="muted">{t({ en: `of ${D * D} (r × 32); α/r = ${scale}`, zh: `（共 ${D * D} 个，r × 32）；α/r = ${scale}` })}</span>
	</p>
	<div class="starts" role="group" aria-label={t({ en: 'Pick a rank', zh: '选一个秩' })}>
		{#each STARTS as s (s.id)}
			<button type="button" class="btn" aria-pressed={r === s.raw.r} onclick={() => onchange?.({ ...s.raw })}>{t(s.label)}</button>
		{/each}
	</div>
</div>

<style>
	.rank {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 520px;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 6rem) minmax(0, 1fr) 2.5rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.compact .row {
		grid-template-columns: 1.2rem minmax(0, 1fr) 2rem;
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
	.cost {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.muted {
		color: var(--muted);
	}
	.starts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.starts .btn {
		font-size: var(--fs-xs);
	}
	.starts .btn[aria-pressed='true'] {
		border-color: var(--accent);
		font-weight: 600;
	}
</style>
