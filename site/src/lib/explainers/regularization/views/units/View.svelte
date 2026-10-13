<!--
  'regularization/units' (meta.ts): the hidden units of both layers at one update; kept units filled, switched-off units
  empty and dashed, with the count per layer and the scale of the kept ones.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { fmtSetting } from '../../../../atlas/core/format.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { unitsPic } from './meta.ts';

	let { at }: StageProps = $props();
	const pic = $derived(unitsPic({ at }));
	const L = {
		layer: { en: 'hidden layer', zh: '隐藏层' },
		kept: { en: 'kept, output × 1/(1 − p) =', zh: '保留，输出 × 1/(1 − p) =' },
		dropped: { en: 'switched off for this update (output 0)', zh: '这次更新中被关掉（输出 0）' }
	};
</script>

{#if pic}
	<div class="units">
		<p class="head">{t({ en: `Update ${fmtInt(pic.step)}`, zh: `第 ${fmtInt(pic.step)} 次更新` })} · p = <span class="num">{fmtSetting(pic.p)}</span></p>
		{#each pic.kept as row, l (l)}
			<div class="layer">
				<span class="name">{t(L.layer)} {l + 1}</span>
				<div class="row" role="img" aria-label={t({ en: `hidden layer ${l + 1}: ${fmtInt(pic.off[l])} of ${fmtInt(pic.width)} units off`, zh: `隐藏层 ${l + 1}：${fmtInt(pic.width)} 个单元中关掉 ${fmtInt(pic.off[l])} 个` })}>
					{#each row as k, i (i)}<i class:on={k}></i>{/each}
				</div>
				<span class="count num">{t({ en: `${fmtInt(pic.off[l])}/${fmtInt(pic.width)} off`, zh: `关掉 ${fmtInt(pic.off[l])}/${fmtInt(pic.width)}` })}</span>
			</div>
		{/each}
		<p class="legend">
			<span><i class="on"></i>{t(L.kept)} <span class="num">{fmtSetting(pic.scale)}</span></span>
			<span><i></i>{t(L.dropped)}</span>
		</p>
	</div>
{/if}

<style>
	.units {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 620px;
	}
	.head,
	.legend {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.layer {
		display: grid;
		grid-template-columns: 6.5rem minmax(0, 1fr) 5.5rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	@media (max-width: 480px) {
		.layer {
			grid-template-columns: minmax(0, 1fr) auto;
		}
		.layer .row {
			grid-column: 1 / -1;
			grid-row: 2;
		}
	}
	.name {
		color: var(--ink-2);
	}
	.row {
		display: grid;
		grid-template-columns: repeat(16, minmax(0, 1fr));
		gap: 3px;
	}
	i {
		display: inline-block;
		aspect-ratio: 1;
		min-width: 8px;
		border: 1.5px dashed var(--muted);
		border-radius: 3px;
		box-sizing: border-box;
	}
	i.on {
		background: var(--accent);
		border: 1.5px solid var(--accent);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
	}
	.legend span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.legend i {
		width: 12px;
		height: 12px;
	}
	.count {
		text-align: right;
	}
</style>
