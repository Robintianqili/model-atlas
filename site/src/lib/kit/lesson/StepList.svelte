<!--
  StepList (redesign spec §4.4): every step of the lesson, grouped by chapter, in a popover above the rail (desktop,
  460 px) or a bottom sheet (phones). Done steps carry a check, the current row has the here style (a 3 px ink bar on a
  pale row; never yellow), animated steps a ▸. The footer has Restart and, on devices with a pointer, the key hints.
-->
<script lang="ts">
	import Popover from '../Popover.svelte';
	import Icon from '../Icon.svelte';
	import { check, play, restart } from '../icons';
	import { lui, t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		open: boolean;
		anchor?: HTMLElement | null;
		name: string;
		titles: Text[];
		groups: { title?: Text; from: number; to: number }[];
		at: number;
		/** the cover shows (no step is current) */
		cover?: boolean;
		animated?: boolean[];
		onpick: (i: number) => void;
		onrestart: () => void;
	}
	let { open = $bindable(false), anchor = null, name, titles, groups, at, cover = false, animated = [], onpick, onrestart }: Props = $props();
	const n = $derived(titles.length);
</script>

<Popover bind:open {anchor} label={`${name} · ${t({ en: `${n} steps`, zh: `${n} 步` })}`} placement="top" align="center" width={460}>
	<p class="sh-h">{name} · {t({ en: `${n} steps`, zh: `${n} 步` })}</p>
	<ol class="slist">
		{#each groups as g, gi (gi)}
			<li class="sgrp">
				{#if g.title}<p class="sg-h">{t(g.title)}</p>{/if}
				<ol>
					{#each { length: g.to - g.from + 1 } as _, j (j)}
						{@const k = g.from + j}
						{@const done = !cover && k < at}
						{@const now = !cover && k === at}
						<li class:done class:now>
							<button type="button" aria-current={now ? 'step' : undefined} onclick={() => onpick(k)}>
								<span class="sn">{#if done}<Icon d={check} size={14} /><span class="visually-hidden">{lui('done')}: </span>{:else}<span class="num">{k + 1}</span>{/if}</span>
								<span class="st">{t(titles[k])}</span>
								{#if animated[k]}<span class="sk"><Icon d={play} size={12} /><span class="visually-hidden">{lui('animated')}</span></span>{:else}<span></span>{/if}
							</button>
						</li>
					{/each}
				</ol>
			</li>
		{/each}
	</ol>
	{#snippet footer()}
		<div class="sh-f">
			<button type="button" class="btn btn-quiet" onclick={onrestart}><Icon d={restart} size={16} />{lui('restart')}</button>
			<span class="keys">{lui('keys')}</span>
		</div>
	{/snippet}
</Popover>

<style>
	.sh-h {
		margin: 0 0 8px 10px;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		font-weight: var(--w-title);
	}
	.slist,
	.slist ol {
		display: flex;
		flex-direction: column;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.slist {
		gap: 10px;
	}
	.sg-h {
		margin: 2px 0 2px 10px;
		color: var(--muted);
		font-size: var(--t-sm);
		font-weight: var(--w-title);
	}
	.slist button {
		position: relative;
		display: grid;
		grid-template-columns: 26px 1fr 14px;
		gap: 10px;
		align-items: center;
		width: 100%;
		min-height: 40px;
		padding: 4px 10px;
		border: 0;
		border-radius: 10px;
		background: none;
		color: var(--ink-2);
		font: inherit;
		font-size: var(--t-md);
		line-height: var(--lh-md);
		text-align: left;
		cursor: pointer;
	}
	.slist button:hover {
		background: var(--well);
	}
	.sn {
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
		border: 1px solid var(--rule-strong);
		border-radius: 50%;
		color: var(--muted);
		font-size: var(--t-xs);
	}
	.done .sn {
		border-color: var(--ink-2);
		background: var(--ink-2);
		color: var(--panel);
	}
	.now button {
		background: var(--here-bg);
		color: var(--ink);
		font-weight: var(--w-title);
	}
	.now button::before {
		content: '';
		position: absolute;
		left: 0;
		top: 8px;
		bottom: 8px;
		width: 3px;
		border-radius: 3px;
		background: var(--here);
	}
	.now .sn {
		border-color: var(--ink);
		background: var(--ink);
		color: var(--panel);
	}
	.sk {
		color: var(--faint);
	}
	.sh-f {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}
	.sh-f .btn {
		flex: none;
		white-space: nowrap;
	}
	.keys {
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		text-align: right;
	}
	@media (hover: none) {
		.keys {
			display: none;
		}
	}
</style>
