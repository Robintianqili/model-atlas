<!--
  The stage header's input chip (redesign spec §4.4 InputChip): what the model sees now (28×28, 36 px), where it came
  from (your drawing, a preset, your photo), the model's answer, and Redraw, which opens the input sheet. Live.
-->
<script lang="ts">
	import Icon from '$lib/kit/Icon.svelte';
	import { pencil } from '$lib/kit/icons';
	import { t } from '$lib/i18n/i18n.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import type { CnnStore } from './store.svelte';

	let { S, open }: { S: CnnStore; open: () => void } = $props();
	/** a softmax share is never exactly 100% (fp32 rounds 0.99999994 up to 1) */
	const pct = (p: number) => (p >= 0.9995 ? '>99.9%' : `${(p * 100).toFixed(1)}%`);
	const kind = $derived(
		S.source.kind === 'preset'
			? t({ en: 'A preset drawing', zh: '一幅预设画' })
			: S.source.kind === 'photo'
				? t({ en: 'Your photo', zh: '你的照片' })
				: t({ en: 'Your drawing', zh: '你的画' })
	);
	const redraw = $derived(t({ en: 'Redraw', zh: '重画' }));
</script>

<div class="inchip">
	<span class="pic" aria-hidden="true"><MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={36} /></span>
	<span class="ilbl">
		<small>{kind}</small>
		{#if S.trace && S.top >= 0}
			<b>{t(S.spec.classes[S.top])} <span class="num">{pct(S.probs?.[S.top] ?? 0)}</span></b>
		{:else}
			<b>{t({ en: 'Loading…', zh: '加载中…' })}</b>
		{/if}
	</span>
	<button type="button" class="btn" onclick={open} aria-haspopup="dialog" aria-label={t({ en: 'Redraw: change the drawing', zh: '重画：更换输入的画' })}>
		<Icon d={pencil} size={16} /><span class="bt" aria-hidden="true">{redraw}</span>
	</button>
</div>

<style>
	.inchip {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		max-width: 100%;
		padding: 5px;
		border: 1px solid var(--rule);
		border-radius: 14px;
		background: var(--panel);
		box-shadow: var(--shadow-1);
	}
	.pic {
		display: grid;
		place-items: center;
		flex: none;
		width: 36px;
		height: 36px;
		overflow: hidden;
		border-radius: 8px;
		outline: 1px solid var(--rule);
	}
	.ilbl {
		display: flex;
		flex-direction: column;
		min-width: 0;
		padding-right: 2px;
		line-height: 1.2;
	}
	.ilbl small {
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
	}
	.ilbl b {
		overflow: hidden;
		font-size: var(--t-sm);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.ilbl b .num {
		color: var(--ink-2);
		font-weight: var(--w-ui);
	}
	.btn {
		min-height: 36px;
		padding: 0 12px;
		border-radius: 10px;
		font-size: var(--t-sm);
	}
	@media (max-width: 759px) {
		.inchip {
			gap: 8px;
			padding: 4px;
			border-radius: 12px;
			box-shadow: none;
		}
		.pic {
			width: 32px;
			height: 32px;
		}
		.pic :global(.mc) {
			width: 32px !important;
			height: 32px !important;
		}
		.ilbl small {
			display: none;
		}
		.btn {
			min-width: 44px;
			min-height: 44px;
			padding: 0 10px;
		}
		.bt {
			display: none;
		}
	}
</style>
