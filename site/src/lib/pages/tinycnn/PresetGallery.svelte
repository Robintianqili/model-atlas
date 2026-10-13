<!--
  Preset drawings: 6 Quick, Draw! test-split drawings for each of the 10 classes, plus the two
  held-out classes (ladder, envelope) the model never learned.
-->
<script lang="ts">
	import MapCanvas from './MapCanvas.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { b64ToFloats, type PresetGroup } from './data';
	import type { CnnStore } from './store.svelte';

	let { S }: { S: CnnStore } = $props();

	const src = $derived(S.source);
	const activeCls = $derived(src.kind === 'preset' ? src.cls : null);
	const activeIdx = $derived(src.kind === 'preset' ? src.idx : -1);
	const groups = $derived(S.presets ? [...S.presets.classes, ...S.presets.heldOut] : []);
	const active = $derived(groups.find((g) => g.id === activeCls) ?? null);
	// decoded thumbnails, computed once per file
	const thumbs = $derived.by(() => {
		const m = new Map<string, Float32Array[]>();
		for (const g of groups) m.set(g.id, g.items.map((it) => b64ToFloats(it.b64)));
		return m;
	});

	function pick(g: PresetGroup) {
		// first click loads drawing 1; clicking the active class again moves to its next drawing
		const i = activeCls === g.id ? (activeIdx + 1) % g.items.length : 0;
		S.loadPreset(g.id, i);
	}
	const classIndex = (id: string) => S.spec.classes.findIndex((c) => c.id === id);
</script>

<div class="pg">
	<div class="pg-head">
		<h3>{t({ en: 'Or start from a real drawing', zh: '或者从一幅真实的画开始' })}</h3>
		<ProvenanceBadge
			mode="precomputed"
			compact
			note={{ en: 'Drawings from the Quick, Draw! test split (CC BY 4.0). The model never trained on them.', zh: '来自 Quick, Draw! 测试集的画（CC BY 4.0），模型训练时从未见过。' }}
		/>
	</div>
	<p class="muted small">
		{t({
			en: 'Test-set drawings from Quick, Draw! (the model never trained on them). Click a class again for its next drawing.',
			zh: '来自 Quick, Draw! 测试集的画（模型训练时没见过）。再点一次同一类别会换下一幅。'
		})}
	</p>
	{#if !S.presets}
		<p class="muted">{t({ en: 'Loading drawings…', zh: '正在加载画…' })}</p>
	{:else}
		<div class="grid" role="group" aria-label={t({ en: 'Preset drawings, 10 classes', zh: '预设画，10 个类别' })}>
			{#each S.presets.classes as g (g.id)}
				{@const k = classIndex(g.id)}
				<button type="button" class="pc" class:on={activeCls === g.id} aria-pressed={activeCls === g.id} onclick={() => pick(g)}>
					<MapCanvas data={thumbs.get(g.id)![activeCls === g.id ? activeIdx : 0]} rows={28} cols={28} domain={[0, 1]} paper size={44} />
					<span class="nm"><span class="sw" style="background: {categorical(k, themeState.resolved)}"></span>{t(g)}</span>
				</button>
			{/each}
		</div>
		<h4 class="ood-h">{t({ en: 'Not one of the 10 classes the model learned: see what it does', zh: '不属于模型学过的 10 个类别：看看它会怎么做' })}</h4>
		<div class="grid ood" role="group" aria-label={t({ en: 'Drawings of classes the model never learned', zh: '模型没学过的类别的画' })}>
			{#each S.presets.heldOut as g (g.id)}
				<button type="button" class="pc" class:on={activeCls === g.id} aria-pressed={activeCls === g.id} onclick={() => pick(g)}>
					<MapCanvas data={thumbs.get(g.id)![activeCls === g.id ? activeIdx : 0]} rows={28} cols={28} domain={[0, 1]} paper size={44} />
					<span class="nm">{t(g)} <span class="tag">{t({ en: 'unknown', zh: '未学过' })}</span></span>
				</button>
			{/each}
		</div>
		{#if active}
			<div class="variants" role="group" aria-label={t({ en: `Drawings of ${active.en}`, zh: `${active.zh}的画` })}>
				<span class="muted small">{t(active)}:</span>
				{#each active.items as it, i (it.fileIndex)}
					<button type="button" class="vb" class:on={activeIdx === i} aria-pressed={activeIdx === i} aria-label="{t(active)} #{i + 1}" onclick={() => S.loadPreset(active.id, i)}>
						<MapCanvas data={thumbs.get(active.id)![i]} rows={28} cols={28} domain={[0, 1]} paper size={32} />
						<span class="num">#{i + 1}</span>
					</button>
				{/each}
			</div>
		{/if}
	{/if}
</div>

<style>
	.pg {
		margin-top: var(--s5);
		display: grid;
		gap: 8px;
	}
	.pg-head {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-lg);
	}
	.small {
		font-size: var(--fs-sm);
		margin: 0;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(104px, 1fr));
		gap: 6px;
	}
	.ood {
		grid-template-columns: repeat(auto-fill, minmax(120px, 160px));
	}
	.pc {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 4px 8px 4px 4px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		font-size: var(--fs-sm);
		text-align: left;
		min-height: 52px;
	}
	.pc:hover {
		border-color: var(--rule-strong);
	}
	.pc.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.nm {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		flex-wrap: wrap;
		line-height: 1.2;
		min-width: 0;
	}
	.sw {
		flex: 0 0 auto;
	}
	@media (max-width: 480px) {
		.pc .sw {
			display: none;
		}
		.pc {
			gap: 6px;
		}
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
	}
	.tag {
		font-size: var(--t-xs);
		padding: 0 5px;
		border-radius: var(--r-pill);
		background: var(--illus-soft);
		color: var(--illus);
		font-weight: 650;
	}
	.ood-h {
		margin: 6px 0 0;
		font-size: var(--fs-md);
	}
	.variants {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
		padding-top: 4px;
	}
	.vb {
		display: grid;
		justify-items: center;
		gap: 1px;
		padding: 3px;
		border: 1px solid var(--rule);
		border-radius: var(--r1);
		background: var(--surface);
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.vb.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
		color: var(--ink);
	}
</style>
