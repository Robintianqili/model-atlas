<!--
  Catalog card (models and concepts), in the redesign's card language (spec §6.2): the area or section and family as a
  small kicker, the title, the hook (never clamped), and a meta line with the inputs and the size. Live is the rule, so a
  live card carries no status pill; a planned card says "Planned" in text and has a dashed outline. The whole card is one
  link. Same API as before.
  <Card href="/m/tinycnn/" title="TinyCNN" status="live" tier="P0" area="Vision"
        inputs={['drawing']} size="27 KB" hook="Draw a doodle and …" kind="model" />
  Prefer ModelCard / ConceptCard, which fill this from catalog entries; the catalog pages use kit/catalog/LessonCard.
-->
<script lang="ts">
	import StatusBadge from './StatusBadge.svelte';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { InputKind, Status, Tier } from '$lib/catalog/types';

	interface Props {
		href: string;
		title: Text;
		status: Status;
		tier?: Tier;
		kind?: 'model' | 'concept';
		/** area (models) or section (concepts) */
		area?: Text;
		family?: Text;
		inputs?: InputKind[];
		size?: Text;
		hook: Text;
		/** small corner note, e.g. "v1 page" */
		note?: Text;
		compact?: boolean;
	}
	let { href, title, status, tier, kind = 'model', area, family, inputs = [], size, hook, note, compact = false }: Props =
		$props();

	const inputName = (k: InputKind) => ui(`in_${k}` as Parameters<typeof ui>[0]);
	const kick = $derived([area ? t(area) : '', family ? t(family) : ''].filter(Boolean).join(' · '));
</script>

<a class="card {status} {kind}" class:compact {href}>
	{#if kick}<span class="kick">{kick}</span>{/if}
	<span class="title">{t(title)}</span>
	{#if !compact && t(hook)}<span class="hook">{t(hook)}</span>{/if}
	{#if status !== 'live' || inputs.length || size || note}
		<span class="meta">
			{#if status !== 'live'}<StatusBadge {status} {tier} />{/if}
			{#if inputs.length}<span>{inputs.map(inputName).join(' · ')}</span>{/if}
			{#if size}<span class="num" title={ui('download')}>{t(size)}</span>{/if}
			{#if note}<span>{t(note)}</span>{/if}
		</span>
	{/if}
</a>

<style>
	.card {
		display: flex;
		flex-direction: column;
		gap: 6px;
		height: 100%;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		color: var(--ink);
		text-decoration: none;
		transition:
			border-color var(--dur-1) var(--ease),
			box-shadow var(--dur-1) var(--ease);
	}
	.card:hover {
		color: var(--ink);
		border-color: var(--rule-strong);
		box-shadow: var(--shadow-1);
	}
	.card:hover .title {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.card.planned {
		background: transparent;
		border-style: dashed;
		border-color: var(--line-control);
	}
	.card.planned .title {
		color: var(--ink-2);
	}
	.kick {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-ui);
		color: var(--muted);
	}
	.title {
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-weight: var(--w-title);
	}
	.hook {
		flex: 1;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 12px;
		margin-top: auto;
		padding-top: 2px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.compact {
		padding: 10px 12px;
		gap: 4px;
	}
</style>
