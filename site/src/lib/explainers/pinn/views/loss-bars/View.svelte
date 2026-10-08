<!-- pinn/loss-bars (meta.ts): bars on a log scale and the shares of the total. Follows the frame; no animation of its own.
     Each part takes two lines (name and number, then a full-width bar), so it reads the same at 360 px and at 1280 px. -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { barsOf, describeBars } from './meta.ts';

	let { at }: StageProps = $props();
	const b = $derived(barsOf({ at }));
	const shown = $derived(b ? b.bars.filter((x) => Number.isFinite(x.value)) : []);
	const LO = -6;
	const len = (v: number) => (v <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(v) - LO) / -LO)) * 100);
	const color: Record<string, string> = { data: 'var(--k)', physics: 'var(--accent)', start: 'var(--q)' };
</script>

<figure class="lb" aria-label={t(describeBars(b))}>
	{#if !b}
		<p class="muted small">{t({ en: 'No loss has been recorded yet.', zh: '还没有记录损失。' })}</p>
	{:else}
		<p class="small head num">{t({ en: `step ${b.step} · loss = data + λ × physics + start = ${fmt3(b.total)}`, zh: `第 ${b.step} 步 · 损失 = 数据 + λ × 物理 + 起点 = ${fmt3(b.total)}` })}</p>
		<div class="share" aria-hidden="true">
			{#each shown.filter((x) => x.inLoss) as x (x.key)}<span style:flex-grow={b.total > 0 ? x.value / b.total : 0} style:background={color[x.key]}></span>{/each}
		</div>
		{#each shown as x (x.key)}
			<div class="row" class:off={!x.inLoss}>
				<p class="line small"><span class="name"><i style:background={color[x.key]}></i>{t(x.label)}</span><span class="num">{fmt3(x.value)}{#if x.inLoss && b.total > 0}<span class="muted"> · {fmtPct(x.value / b.total)}</span>{/if}</span></p>
				<span class="track"><span class="bar" style:width="{len(x.value)}%" style:background={color[x.key]}></span></span>
			</div>
		{/each}
		<div class="ticks num" aria-hidden="true">{#each ['10⁻⁶', '10⁻⁴', '10⁻²', '1'] as s (s)}<span>{s}</span>{/each}</div>
		<p class="muted small">{t({ en: 'Bars on a log scale: each tick is 100 times the one before. The top bar splits the total into its parts.', zh: '条形用对数刻度：每一格是前一格的 100 倍。最上面的条把总损失按各部分分开。' })}</p>
	{/if}
</figure>

<style>
	.lb { margin: 0; display: grid; gap: 8px; min-width: 0; grid-template-columns: minmax(0, 1fr); }
	.head { margin: 0; color: var(--ink-2); }
	.share { display: flex; height: 16px; border-radius: 3px; overflow: hidden; border: 1px solid var(--rule); background: var(--surface-2); }
	.share span { min-width: 2px; }
	.row { display: grid; gap: 3px; }
	.row.off { opacity: 0.7; }
	.row.off .bar { background-image: repeating-linear-gradient(45deg, transparent 0 4px, var(--surface) 4px 7px); }
	.line { display: flex; justify-content: space-between; gap: 8px; margin: 0; color: var(--ink-2); }
	.name { overflow-wrap: anywhere; }
	.name i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
	.track { position: relative; height: 12px; background: var(--surface-2); border-radius: 3px; overflow: hidden; }
	.bar { position: absolute; inset: 0 auto 0 0; border-radius: 3px; }
	.ticks { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }
	p { margin: 0; }
	.small { font-size: var(--fs-sm); }
</style>
