<!--
  Catalog card (models and concepts). The whole card is one link.
  <Card href="/m/tinycnn/" title="TinyCNN" status="live" tier="P0" area="Vision"
        inputs={['drawing']} size="27 KB" hook="Draw a doodle and …" kind="model" />
  Prefer ModelCard / ConceptCard, which fill this from catalog entries.
-->
<script lang="ts">
	import StatusBadge from './StatusBadge.svelte';
	import InputIcon from './InputIcon.svelte';
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
</script>

<a class="card {status} {kind}" class:compact {href}>
	<span class="top">
		<StatusBadge {status} {tier} />
		{#if note}<span class="note">{t(note)}</span>{/if}
	</span>
	<span class="title">{t(title)}</span>
	{#if area || family}
		<span class="meta">{[area ? t(area) : '', family ? t(family) : ''].filter(Boolean).join(' · ')}</span>
	{/if}
	{#if !compact}<span class="hook">{t(hook)}</span>{/if}
	{#if inputs.length || size}
		<span class="foot">
			{#each inputs as k (k)}
				<span class="in"><InputIcon kind={k} /> {inputName(k)}</span>
			{/each}
			{#if size}<span class="size num" title={ui('download')}>↓ {t(size)}</span>{/if}
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
		border-radius: var(--r3);
		background: var(--surface);
		color: var(--ink);
		text-decoration: none;
		transition:
			border-color var(--dur-1) var(--ease),
			transform var(--dur-1) var(--ease),
			box-shadow var(--dur-1) var(--ease);
	}
	.card:hover {
		color: var(--ink);
		border-color: var(--rule-strong);
		box-shadow: var(--shadow-1);
		transform: translateY(-1px);
	}
	.card.live {
		border-color: color-mix(in srgb, var(--live) 45%, var(--rule));
		box-shadow: inset 3px 0 0 var(--live);
	}
	.card.planned {
		background: color-mix(in srgb, var(--surface) 55%, transparent);
		border-style: dashed;
	}
	.card.planned .title {
		color: var(--ink-2);
	}
	.top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
	}
	.note {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.title {
		font-family: var(--font-display);
		font-size: 1.16rem;
		font-weight: 600;
		line-height: 1.25;
	}
	.meta {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.hook {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		line-height: 1.5;
		flex: 1;
		display: -webkit-box;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 4;
		line-clamp: 4;
		overflow: hidden;
	}
	.foot {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin-top: 2px;
	}
	.in {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.size {
		margin-left: auto;
		color: var(--muted);
	}
	.compact {
		padding: 10px 12px;
		gap: 4px;
	}
	.compact .title {
		font-size: 1rem;
	}
	@media (prefers-reduced-motion: reduce) {
		.card:hover {
			transform: none;
		}
	}
</style>
