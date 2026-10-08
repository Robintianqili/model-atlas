<script lang="ts">
	import { base } from '$app/paths';
	import ModelCard from '$lib/kit/ModelCard.svelte';
	import InputIcon from '$lib/kit/InputIcon.svelte';
	import { AREAS } from '$lib/catalog/areas';
	import { MODELS, sortModels } from '$lib/catalog/models';
	import type { InputKind, Tier } from '$lib/catalog/types';
	import { t, ui, colon } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';

	let status = $state<'all' | 'live' | 'planned'>('all');
	let input = $state<InputKind | 'all'>('all');
	let tier = $state<Tier | 'all'>('all');

	const kinds = [...new Set(MODELS.flatMap((m) => m.inputs))] as InputKind[];
	const tiers: Tier[] = ['P0', 'V1', 'P1', 'P2'];
	const visible = $derived(
		MODELS.filter(
			(m) =>
				(status === 'all' || (status === 'live' ? m.status === 'live' : m.status !== 'live')) &&
				(input === 'all' || m.inputs.includes(input)) &&
				(tier === 'all' || m.tier === tier)
		)
	);
	const byArea = $derived(AREAS.map((a) => ({ area: a, models: sortModels(visible.filter((m) => m.area === a.slug)) })));
	const tierKey: Record<Tier, UIKey> = { P0: 'tierP0', V1: 'tierV1', P1: 'tierP1', P2: 'tierP2' };
</script>

<svelte:head><title>{ui('navAtlas')} · Model Atlas</title></svelte:head>

<div class="container page">
	<header class="head">
		<p class="eyebrow">/atlas/</p>
		<h1>{t({ en: 'The atlas', zh: '图鉴' })}</h1>
		<p class="lede">
			{t({
				en: `${MODELS.length} models in ${AREAS.length} application areas. Live pages run in your browser now; planned pages describe what they will show and in which order they will be built.`,
				zh: `${AREAS.length} 个应用领域中的 ${MODELS.length} 个模型。可玩页面现在就能在浏览器中运行；计划中的页面说明将展示什么以及构建顺序。`
			})}
		</p>
		<ul class="tiers">
			{#each tiers as tr (tr)}<li><span class="num">{tr}</span> {ui(tierKey[tr]).replace(/^P\d · |^V1 · /, '')}</li>{/each}
		</ul>
	</header>

	<div class="filters" role="group" aria-label={t({ en: 'Filters', zh: '筛选' })}>
		<div class="segmented" role="group" aria-label={t({ en: 'Status', zh: '状态' })}>
			<button type="button" aria-pressed={status === 'all'} onclick={() => (status = 'all')}>{t({ en: 'All', zh: '全部' })}</button>
			<button type="button" aria-pressed={status === 'live'} onclick={() => (status = 'live')}>{ui('statusLive')}</button>
			<button type="button" aria-pressed={status === 'planned'} onclick={() => (status = 'planned')}>{ui('statusPlanned')}</button>
		</div>
		<div class="segmented" role="group" aria-label={ui('tier')}>
			<button type="button" aria-pressed={tier === 'all'} onclick={() => (tier = 'all')}>{ui('tier')}{colon()} {t({ en: 'all', zh: '全部' })}</button>
			{#each tiers as tr (tr)}<button type="button" aria-pressed={tier === tr} onclick={() => (tier = tr)}>{tr}</button>{/each}
		</div>
		<label class="sel">
			<span>{ui('input')}</span>
			<select bind:value={input}>
				<option value="all">{t({ en: 'any input', zh: '任意输入' })}</option>
				{#each kinds as k (k)}<option value={k}>{ui(`in_${k}` as UIKey)}</option>{/each}
			</select>
		</label>
		<span class="muted num count" aria-live="polite">{visible.length} / {MODELS.length}</span>
	</div>

	{#each byArea as { area, models } (area.slug)}
		{#if models.length}
			<section class="area" id={area.slug} aria-labelledby="{area.slug}-h">
				<div class="area-head">
					<h2 id="{area.slug}-h">
						<span class="n num">{String(area.n).padStart(2, '0')}</span>
						<a href="{base}/atlas/{area.slug}/">{t(area.name)}</a>
					</h2>
					<p class="muted">{t(area.blurb)}</p>
				</div>
				<div class="cards">
					{#each models as m (m.slug)}<ModelCard model={m} showArea={false} />{/each}
				</div>
			</section>
		{/if}
	{/each}
	{#if visible.length === 0}
		<p class="muted">{t({ en: 'No model matches these filters.', zh: '没有符合筛选条件的模型。' })}</p>
	{/if}
	<p class="legend muted">
		<InputIcon kind="drawing" /> {ui('in_drawing')} · <InputIcon kind="text" /> {ui('in_text')} · …
		{t({ en: 'Sizes marked "design" are targets for models we will train ourselves.', zh: '标注“design”的大小是我们将自行训练的模型的设计目标。' })}
	</p>
</div>

<style>
	.page {
		padding-top: var(--s6);
	}
	.head {
		display: grid;
		gap: 8px;
	}
	.head h1,
	.head p {
		margin: 0;
	}
	.tiers {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.tiers .num {
		font-weight: 700;
		color: var(--ink);
	}
	.filters {
		position: sticky;
		top: var(--header-h);
		z-index: 5;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px 14px;
		margin: var(--s5) 0 var(--s3);
		padding: 10px 0;
		background: var(--bg);
		border-bottom: 1px solid var(--rule);
	}
	.sel {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-sm);
	}
	select {
		min-height: 34px;
		padding: 0 8px;
		border-radius: var(--r2);
		border: 1px solid var(--rule-strong);
		background: var(--surface);
	}
	.count {
		margin-left: auto;
		font-size: var(--fs-sm);
	}
	.area {
		padding-top: var(--s5);
		scroll-margin-top: calc(var(--header-h) + 70px);
	}
	.area-head h2 {
		display: flex;
		align-items: baseline;
		gap: 10px;
		margin: 0 0 4px;
		font-size: var(--fs-xl);
	}
	.area-head h2 a {
		color: var(--ink);
		text-decoration: none;
	}
	.area-head h2 a:hover {
		color: var(--accent);
	}
	.n {
		font-size: var(--fs-sm);
		color: var(--accent);
	}
	.area-head p {
		margin: 0 0 var(--s3);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr));
		gap: 12px;
	}
	.legend {
		margin-top: var(--s6);
		font-size: var(--fs-xs);
	}
</style>
