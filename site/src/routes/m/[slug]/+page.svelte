<script lang="ts">
	import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
	import ModelCard from '$lib/kit/ModelCard.svelte';
	import InputIcon from '$lib/kit/InputIcon.svelte';
	import { MODEL_BY_SLUG, MODELS, sortModels } from '$lib/catalog/models';
	import { AREA_BY_SLUG } from '$lib/catalog/areas';
	import { techText } from '$lib/catalog/tech-zh';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const model = $derived(MODEL_BY_SLUG[data.slug]);
	const area = $derived(AREA_BY_SLUG[model.area]);
	const siblings = $derived(sortModels(MODELS.filter((m) => m.area === model.area && m.slug !== model.slug)).slice(0, 6));
	const tierKey: Record<string, UIKey> = { P0: 'tierP0', V1: 'tierV1', P1: 'tierP1', P2: 'tierP2' };
	const modeText = $derived(
		model.mode === 'live'
			? { en: 'Live: computed in your browser', zh: '实时：在你的浏览器中计算' }
			: model.mode === 'precomputed'
				? { en: 'Precomputed: replayed from saved model outputs', zh: '预计算：回放保存的模型输出' }
				: { en: 'Both: live where possible, precomputed for the heavy parts', zh: '两者兼有：能实时则实时，较重部分预计算' }
	);
</script>

<!-- <title> and <meta name="description"> are set by ModelPageShell -->
{#if data.component && data.auto}
	<!-- an included explainer package: its auto page (a lazy chunk returned by +page.ts) -->
	{@const Auto = data.component}
	<Auto slug={data.auto.slug} spec={data.auto.spec} autoBytes={data.auto.autoBytes} />
{:else}
<ModelPageShell slug={model.slug} titles={{ overview: ui('whatItWillShow'), input: ui('howYouPlay'), links: ui('secLinks') }}>
	{#snippet facts()}
		<div class="planned-note" role="note">
			<strong>{ui('plannedPageTitle')}.</strong>
			{ui('plannedPageBody')}
		</div>
		<dl class="facts">
			<div><dt>{ui('area')}</dt><dd>{t(area.name)}</dd></div>
			<div><dt>{ui('family')}</dt><dd>{t(model.family)}</dd></div>
			<div><dt>{ui('tier')}</dt><dd>{ui(tierKey[model.tier])}{#if model.tierNote}<br /><span class="muted small">{t(model.tierNote)}</span>{/if}</dd></div>
			{#if model.params}<div><dt>{ui('params')}</dt><dd class="num">{t(techText(model.params))}</dd></div>{/if}
			{#if model.size}<div><dt>{ui('download')}</dt><dd class="num">{t(techText(model.size))}</dd></div>{/if}
			<div><dt>{ui('runtime')}</dt><dd>{t(techText(model.runtime))}</dd></div>
			<div><dt>{t({ en: 'Numbers', zh: '数字来源' })}</dt><dd>{t(modeText)}</dd></div>
			<div><dt>{t({ en: 'Model', zh: '模型来源' })}</dt><dd>{model.own ? t({ en: 'A tiny model we train ourselves', zh: '我们自己训练的小模型' }) : t({ en: 'An existing pretrained model', zh: '已有的预训练模型' })}</dd></div>
		</dl>
	{/snippet}
	{#snippet overview()}
		<p class="big">{t(model.shows)}</p>
		{#if model.notes}<p class="note-warn">{t(model.notes)}</p>{/if}
	{/snippet}
	{#snippet input()}
		<ul class="inputs">
			{#each model.inputs as k (k)}<li><InputIcon kind={k} size={18} /> {ui(`in_${k}` as UIKey)}</li>{/each}
		</ul>
		<p class="muted">{t(model.hook)}</p>
	{/snippet}
	{#snippet moreLinks()}
		{#if siblings.length}
			<h3 class="sib">{t({ en: `More in ${area.name.en}`, zh: `${area.name.zh ?? area.name.en}领域的更多模型` })}</h3>
			<div class="cards">{#each siblings as m (m.slug)}<ModelCard model={m} compact showArea={false} />{/each}</div>
		{/if}
	{/snippet}
</ModelPageShell>
{/if}

<style>
	.planned-note {
		padding: 12px 16px;
		border: 1.5px dashed var(--planned);
		border-radius: var(--r3);
		background: var(--planned-soft);
		color: var(--ink-2);
		font-size: var(--fs-sm);
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 150px), 1fr));
		gap: 12px 20px;
		margin: 0;
		padding: 14px 16px;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
	}
	dt {
		font-size: var(--fs-xs);
		color: var(--muted);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-family: var(--font-mono);
	}
	dd {
		margin: 0;
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.small {
		font-size: var(--fs-xs);
		font-weight: 400;
	}
	.big {
		font-size: var(--fs-lg);
		max-width: var(--measure);
	}
	.note-warn {
		font-size: var(--fs-sm);
		color: var(--illus);
		border-left: 3px solid var(--illus);
		padding-left: 10px;
	}
	.inputs {
		list-style: none;
		margin: 0 0 8px;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 8px 16px;
	}
	.inputs li {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-weight: 600;
	}
	.sib {
		font-size: var(--fs-md);
		margin-top: var(--s5);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
</style>
