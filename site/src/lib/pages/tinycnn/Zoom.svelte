<!--
  Zoom: the arithmetic of the unit selected in the overview, with its context kept (where it is in
  the network, and which part of the drawing it can see).
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Term from '$lib/kit/Term.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import ConvDetail from './ConvDetail.svelte';
	import ReluDetail from './ReluDetail.svelte';
	import PoolDetail from './PoolDetail.svelte';
	import FlattenDetail from './FlattenDetail.svelte';
	import DenseDetail from './DenseDetail.svelte';
	import SoftmaxDetail from './SoftmaxDetail.svelte';
	import InputDetail from './InputDetail.svelte';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import { isSpatial } from './data';
	import type { CnnStore } from './store.svelte';

	let { S, el = $bindable() }: { S: CnnStore; el?: HTMLElement } = $props();

	const L = $derived(S.layer(S.sel.layer));
	const spatial = $derived(isSpatial(L));
	const rf = $derived(spatial ? S.receptiveField(L.id, S.sel.y, S.sel.x) : { y0: 0, x0: 0, y1: 27, x1: 27 });
	const rfSize = $derived(rf ? rf.y1 - rf.y0 + 1 : 28);
	const unit = $derived(
		L.op === 'input'
			? ''
			: spatial
				? ` · ${t({ en: 'channel', zh: '通道' })} ${S.sel.ch} · ${t({ en: 'cell', zh: '格子' })} (${S.sel.y}, ${S.sel.x})`
				: L.op === 'dense' || L.op === 'softmax'
					? ` · ${t(S.spec.classes[S.sel.ch])}`
					: ''
	);
	const kindText = $derived(
		L.op === 'conv2d'
			? t({ en: 'convolution', zh: '卷积' })
			: L.op === 'relu'
				? 'ReLU'
				: L.op === 'maxpool2d'
					? t({ en: 'max pooling', zh: '最大池化' })
					: L.op === 'flatten'
						? t({ en: 'flatten', zh: '展平' })
						: L.op === 'dense'
							? t({ en: 'dense layer', zh: '全连接层' })
							: L.op === 'softmax'
								? 'softmax'
								: t({ en: 'input', zh: '输入' })
	);
</script>

<section class="zoom" bind:this={el} tabindex="-1" aria-labelledby="zoom-title">
	<header class="zh">
		<div class="ttl">
			<span class="eyebrow">{t({ en: 'Zoom in', zh: '放大' })} · {kindText}</span>
			<h3 id="zoom-title">{t(L.label)}<span class="unit">{unit}</span></h3>
			{#if L.desc}<p class="desc">{t(L.desc)}</p>{/if}
		</div>
		<div class="ctx">
			{#if S.trace && rf}
				<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={72} boxes={[{ ...rf, color: 'var(--accent)' }]} />
			{/if}
			<p class="rf">
				{#if L.op === 'input'}
					{t({ en: 'One pixel of the input.', zh: '输入中的一个像素。' })}
				{:else if spatial}
					{t({ en: `This cell sees a ${rfSize}×${rfSize} patch of the 28×28 drawing`, zh: `这个格子能看到 28×28 画面中 ${rfSize}×${rfSize} 的区域` })}{i18n.lang === 'zh' ? '（' : ' ('}<Term id="receptive-field" />{i18n.lang === 'zh' ? '）。' : ').'}
				{:else}
					{t({ en: 'Depends on the whole drawing.', zh: '取决于整幅画。' })}
				{/if}
			</p>
			<ProvenanceBadge mode="live" compact precision={S.spec.precision} />
		</div>
	</header>
	{#if !S.trace}
		<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
	{:else if L.op === 'conv2d'}
		<ConvDetail {S} />
	{:else if L.op === 'relu'}
		<ReluDetail {S} />
	{:else if L.op === 'maxpool2d'}
		<PoolDetail {S} />
	{:else if L.op === 'flatten'}
		<FlattenDetail {S} />
	{:else if L.op === 'dense'}
		<DenseDetail {S} />
	{:else if L.op === 'softmax'}
		<SoftmaxDetail {S} />
	{:else}
		<InputDetail {S} />
	{/if}
</section>

<style>
	.zoom {
		margin-top: 16px;
		padding: 14px 16px 16px;
		border: 1px solid var(--rule);
		border-left: 4px solid var(--accent);
		border-radius: var(--r3);
		background: var(--surface);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
		outline: none;
	}
	.zh {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 10px 24px;
	}
	.ttl {
		display: grid;
		gap: 2px;
		min-width: 0;
		flex: 1 1 320px;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-xl);
	}
	.unit {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		font-weight: 500;
		color: var(--ink-2);
	}
	.desc {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.ctx {
		display: flex;
		align-items: center;
		gap: 10px;
		flex: 0 1 380px;
	}
	.rf {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	/* phones: slightly smaller formulas so the multi-line ones fit without sideways scrolling */
	@media (max-width: 480px) {
		.zoom :global(.fc .katex) {
			font-size: 0.95em;
		}
	}
	@media (max-width: 380px) {
		.zoom :global(.fc .katex) {
			font-size: 0.86em;
		}
	}
</style>
