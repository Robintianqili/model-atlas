<!--
  The end card (redesign spec §5.8), in the narration column after Finish: what you saw (the lesson's recap, else its
  chapters), your predictions (k of n right) with the cover question revealed, Up next, and Go deeper (the page's views,
  each with one line). A check draws in over 300 ms (instant under reduced motion); no confetti.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import Icon from '../Icon.svelte';
	import { arrow, check } from '../icons';
	import PredictReveal from '../PredictReveal.svelte';
	import { lui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { PredictQuestion } from '../types';
	import type { ViewTab } from './app';

	interface Props {
		recap?: Snippet;
		/** recap lines when there is no recap snippet (the chapter titles) */
		lines?: Text[];
		right: number;
		asked: number;
		prequestion?: PredictQuestion;
		upNext?: { href: string; title: Text; note?: Text };
		tabs?: ViewTab[];
		/** the lesson's own end snippet (the Walkthrough's `end`) */
		end?: Snippet;
	}
	let { recap, lines = [], right, asked, prequestion, upNext, tabs = [], end }: Props = $props();
	const views = $derived(tabs.filter((v) => v.id !== 'lesson'));
</script>

<div class="endc">
	<p class="done"><span class="ck" aria-hidden="true"><Icon d={check} size={20} /></span>{t({ en: 'You finished the lesson.', zh: '你学完了这节课。' })}</p>
	{#if recap || lines.length}
		<section aria-label={lui('recap')}>
			<p class="h">{lui('recap')}</p>
			{#if recap}{@render recap()}{:else}<ul class="recap">{#each lines as l, i (i)}<li>{t(l)}</li>{/each}</ul>{/if}
		</section>
	{/if}
	{#if asked > 0 || prequestion}
		<section aria-label={t({ en: 'Your predictions', zh: '你的预测' })}>
			{#if asked > 0}<p class="h">{lui('predictions', { k: right, n: asked })}</p>{/if}
			{#if prequestion}<PredictReveal q={prequestion} />{/if}
		</section>
	{/if}
	{#if end}<div class="own">{@render end()}</div>{/if}
	{#if upNext}
		<a class="upnext" href={upNext.href}>
			<span class="un-k">{lui('upNext')}</span>
			<span class="un-t">{t(upNext.title)}</span>
			{#if upNext.note}<span class="un-n">{t(upNext.note)}</span>{/if}
			<Icon d={arrow} size={18} class="un-i" />
		</a>
	{/if}
	{#if views.length}
		<nav aria-label={lui('goDeeper')}>
			<p class="h">{lui('goDeeper')}</p>
			<ul class="deeper">
				{#each views as v (v.id)}
					<li><a href="#{v.id}"><Icon d={v.icon} size={18} /><span><b>{t(v.label)}</b>{#if v.note}<small>{t(v.note)}</small>{/if}</span></a></li>
				{/each}
			</ul>
		</nav>
	{/if}
</div>

<style>
	.endc {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}
	.done {
		display: flex;
		align-items: center;
		gap: 10px;
		margin: 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		font-weight: var(--w-title);
	}
	.ck {
		display: grid;
		place-items: center;
		width: 32px;
		height: 32px;
		border-radius: 50%;
		background: var(--ok);
		color: var(--ok-soft);
		animation: ck var(--dur-3) var(--ease);
	}
	@keyframes ck {
		from {
			transform: scale(0.6);
			opacity: 0;
		}
	}
	.h {
		margin: 0 0 6px;
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-title);
	}
	.recap {
		margin: 0;
		padding-left: 1.2em;
	}
	.recap li + li {
		margin-top: 4px;
	}
	.upnext {
		position: relative;
		display: grid;
		gap: 2px;
		padding: 12px 44px 12px 14px;
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		color: var(--ink);
		text-decoration: none;
	}
	.upnext:hover {
		border-color: var(--ink-2);
		background: var(--well);
	}
	.un-k {
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
	}
	.un-t {
		font-weight: var(--w-title);
	}
	.un-n {
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.upnext :global(.un-i) {
		position: absolute;
		right: 14px;
		top: 50%;
		transform: translateY(-50%);
	}
	.deeper {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.deeper a {
		display: grid;
		grid-template-columns: 18px 1fr;
		gap: 10px;
		align-items: start;
		height: 100%;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		color: var(--ink-2);
		text-decoration: none;
	}
	.deeper a:hover {
		border-color: var(--line-control);
		background: var(--well);
	}
	.deeper b {
		display: block;
		color: var(--ink);
		font-size: var(--t-sm);
	}
	.deeper small {
		display: block;
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
	}
	@media (max-width: 759px) {
		.deeper {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
