<!--
  'lora/matrices': the frame's real matrices as heat maps (meta.ts has the props). 'merge': B × A × α/r = ΔW, then
  W0 + ΔW = W. 'target': ΔW, ΔW* and ΔW* − ΔW on one scale. Every grid is the kit's TensorGrid (keyboard and hover read
  out each number).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtNum } from '../../../../atlas/core/format.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { D, WORLD } from '../../data.ts';
	import { matOf } from './meta.ts';

	let { at, props }: StageProps = $props();
	const pic = $derived(matOf({ at }));
	const mode = $derived(props.mode === 'target' ? 'target' : 'merge');
	const sym = (m: number): [number, number] => [-m, m];
	// fixed domains, so colors keep their meaning while the run plays (a value past the end clamps)
	const AB = sym(0.6);
	const dDom = $derived(sym(Math.max(0.45, ...Array.from(pic?.dW ?? [], Math.abs))));
	const wDom = sym(0.9);
	const CELL = 9;
	const L = {
		B: { en: 'B, trainable', zh: 'B，可训练' },
		A: { en: 'A, trainable', zh: 'A，可训练' },
		dW: { en: 'ΔW = (α/r)·B·A', zh: 'ΔW = (α/r)·B·A' },
		W0: { en: 'W0, frozen', zh: 'W0，冻结' },
		W: { en: 'W = W0 + ΔW', zh: 'W = W0 + ΔW' },
		learned: { en: 'ΔW, learned', zh: 'ΔW，学到的' },
		needed: { en: 'ΔW*, needed by the new task', zh: 'ΔW*，新任务需要的' },
		missing: { en: 'ΔW* − ΔW, still missing', zh: 'ΔW* − ΔW，还缺的' }
	};
</script>

{#snippet grid(data: ArrayLike<number>, rows: number, cols: number, name: { en: string; zh: string }, domain: [number, number])}
	<figure>
		<TensorGrid {data} {rows} {cols} {domain} cell={CELL} labels={false} readout={false} label={name} />
		<figcaption>{t(name)} <span class="dim num">{rows}&nbsp;×&nbsp;{cols}</span></figcaption>
	</figure>
{/snippet}

{#if pic}
	{#if mode === 'merge'}
		<div class="row">
			{@render grid(pic.B.data, D, pic.r, L.B, AB)}
			<span class="op" aria-hidden="true">×</span>
			{@render grid(pic.A.data, pic.r, D, L.A, AB)}
			<span class="op" aria-hidden="true">× {fmtNum(pic.scale)} =</span>
			{@render grid(pic.dW, D, D, L.dW, dDom)}
		</div>
		<div class="row">
			{@render grid(WORLD.W0, D, D, L.W0, wDom)}
			<span class="op" aria-hidden="true">+ ΔW =</span>
			{@render grid(pic.W, D, D, L.W, wDom)}
		</div>
	{:else}
		<div class="row">
			{@render grid(pic.dW, D, D, L.learned, dDom)}
			{@render grid(WORLD.dWstar, D, D, L.needed, dDom)}
			{@render grid(pic.missing, D, D, L.missing, dDom)}
		</div>
	{/if}
	<p class="key">
		<span class="sw neg"></span>{t({ en: 'below 0', zh: '小于 0' })}
		<span class="sw mid"></span>0
		<span class="sw pos"></span>{t({ en: 'above 0', zh: '大于 0' })}
		<span class="dim">· {t({ en: 'stronger color: further from 0', zh: '颜色越浓，离 0 越远' })}</span>
	</p>
{/if}

<style>
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px 10px;
		margin-bottom: 10px;
	}
	figure {
		margin: 0;
		min-width: 0;
		display: grid;
		justify-items: center;
		gap: 4px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		text-align: center;
		max-width: 12rem;
	}
	.op {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.dim {
		color: var(--muted);
	}
	.key {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 12px;
		border: 1px solid var(--rule);
	}
	.sw.neg {
		background: var(--div-neg);
	}
	.sw.mid {
		background: var(--div-mid);
	}
	.sw.pos {
		background: var(--div-pos);
	}
</style>
