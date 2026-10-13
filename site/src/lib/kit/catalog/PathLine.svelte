<!--
  A learning path as a route line (redesign spec §4.5, B's graft): a 3 px ink line through numbered stations. Live
  stations are ink-filled and link to the lesson; planned ones are a dashed circle marked "coming". On phones the
  stations are dots only and the names follow in one line joined by arrows.
    <PathLine path={p} />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { PathItem } from './types';

	let { path: p, as = 'h3' }: { path: PathItem; as?: 'h2' | 'h3' } = $props();
	const numbers = $derived.by(() => {
		let k = 0;
		return p.stations.map((s) => (s.live ? ++k : 0));
	});
</script>

<div class="path" id="path-{p.id}">
	<div class="head">
		<svelte:element this={as} class="title">{t(p.title)}</svelte:element>
		<p class="blurb">{t(p.blurb)}</p>
		<p class="meta">
			{ct('pathMeta', { n: p.live, m: p.minutes })}{#if p.coming}{' · '}{ct('comingN', { n: p.coming })}{/if}
		</p>
	</div>
	<ol class="stations" style:--n={p.stations.length}>
		{#each p.stations as s, i (s.slug)}
			<li class:live={s.live}>
				{#if s.live}
					<a href="{base}{s.href}">
						<span class="dot" aria-hidden="true">{numbers[i]}</span>
						<span class="name">{t(s.name)}</span>
						<span class="smeta">{ui('lessonSteps', { n: s.steps ?? 0 })}</span>
					</a>
				{:else}
					<span class="st">
						<span class="dot" aria-hidden="true"></span>
						<span class="name">{t(s.name)}</span>
						<span class="smeta">{ct('coming')}</span>
					</span>
				{/if}
			</li>
		{/each}
	</ol>
	<p class="line" aria-hidden="true">
		{#each p.stations as s, i (s.slug)}{#if i}{' → '}{/if}<span class:soon={!s.live}>{t(s.name)}{#if !s.live}{' '}({ct('coming')}){/if}</span>{/each}
	</p>
</div>

<style>
	.path {
		display: grid;
		grid-template-columns: minmax(0, 300px) minmax(0, 1fr);
		gap: 32px;
		padding: 18px 0;
		border-bottom: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 16px);
	}
	.title {
		margin: 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		font-weight: var(--w-title);
	}
	.blurb {
		margin: 4px 0 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.meta {
		margin: 4px 0 0;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.stations {
		position: relative;
		align-self: center;
		display: grid;
		grid-template-columns: repeat(var(--n), minmax(0, 1fr));
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.stations::before {
		content: '';
		position: absolute;
		left: calc(50% / var(--n));
		right: calc(50% / var(--n));
		top: 15px;
		height: 3px;
		border-radius: 2px;
		background: var(--ink);
	}
	.stations li {
		position: relative;
		min-width: 0;
	}
	.stations a,
	.st {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
		padding: 0 4px;
		text-align: center;
		color: inherit;
		text-decoration: none;
	}
	.dot {
		position: relative;
		z-index: 1;
		display: grid;
		place-items: center;
		width: 32px;
		height: 32px;
		border-radius: 50%;
		background: var(--act);
		color: var(--act-ink);
		font-size: var(--t-sm);
		font-weight: var(--w-title);
		box-shadow: 0 0 0 4px var(--paper);
	}
	li:not(.live) .dot {
		background: var(--paper);
		border: 2px dashed var(--line-control);
	}
	.name {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		overflow-wrap: anywhere;
	}
	li:not(.live) .name {
		color: var(--muted);
		font-weight: var(--w-body);
	}
	.smeta {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	a:hover .name {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.line {
		display: none;
	}
	@media (max-width: 1099px) {
		.path {
			grid-template-columns: minmax(0, 1fr);
			gap: 16px;
		}
	}
	@media (max-width: 759px) {
		.path {
			gap: 10px;
			padding: 16px 0;
		}
		.stations::before {
			top: 21px;
		}
		.stations a,
		.st {
			padding: 8px 0;
		}
		.dot {
			width: 28px;
			height: 28px;
			font-size: var(--t-xs);
		}
		.name,
		.smeta {
			position: absolute;
			width: 1px;
			height: 1px;
			overflow: hidden;
			clip: rect(0 0 0 0);
			white-space: nowrap;
		}
		.line {
			display: block;
			margin: 0;
			font-size: var(--t-sm);
			line-height: var(--lh-sm);
			color: var(--ink-2);
		}
		.line .soon {
			color: var(--muted);
		}
	}
</style>
