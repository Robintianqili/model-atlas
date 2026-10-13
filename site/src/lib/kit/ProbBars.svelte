<!--
  ProbBars: top-k horizontal probability bars with the chosen item highlighted and optional
  cutoff lines (top-k / top-p) between ranks.

  <ProbBars items={classes.map((c, i) => ({ label: c, p: probs[i], id: i, color: categorical(i) }))}
            k={5} chosen={argmax} />
  <ProbBars items={tokens} k={10} tokens chosen={sampledId}
            cutoffs={[{ afterRank: 3, label: 'top-k = 3' }, { afterRank: nucleus, label: 'top-p = 0.9' }]} />

  - Items are sorted by p (descending) unless sort={false}.
  - `chosen` matches an item's `id` (or its label when ids are absent).
  - `tokens` makes whitespace visible (␣ ↵) in labels.
  - Bars are one purple from the sequential ramp (length carries the value) unless an item has
    its own `color` (class colors).
-->
<script lang="ts">
	import { makeScale } from './color';
	import { themeState } from './theme.svelte';
	import { fmtPct, fmt3, visibleToken } from './format';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { ProbItem } from './types';

	interface Props {
		items: ProbItem[];
		k?: number;
		sort?: boolean;
		chosen?: string | number | null;
		cutoffs?: { afterRank: number; label: string }[];
		tokens?: boolean;
		/** scale bars to the largest p instead of to 1 */
		relative?: boolean;
		label?: Text;
		onselect?: (item: ProbItem) => void;
		/** show each token's '#id' next to its text (default true; the atlas views turn it off when items have names) */
		ids?: boolean;
	}
	let {
		items,
		k = 10,
		sort = true,
		chosen = null,
		cutoffs = [],
		tokens = false,
		relative = false,
		label,
		onselect,
		ids = true
	}: Props = $props();

	const rows = $derived((sort ? [...items].sort((a, b) => b.p - a.p) : items).slice(0, k));
	const maxP = $derived(relative ? Math.max(1e-9, ...rows.map((r) => r.p)) : 1);
	const seq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const isChosen = (it: ProbItem) => chosen !== null && (it.id !== undefined ? it.id === chosen : it.label === chosen);
	const cutAfter = (rank: number) => cutoffs.filter((c) => c.afterRank === rank);
	// QA 2026-10-13: the four columns need about 224 px; in a narrower box (a view beside a picture on a phone) the
	// label column shrinks and the "chosen" word is kept for screen readers only (the row's border still marks it),
	// so the rows no longer push the page sideways (AK602 at 390 px in adversarial-tinycnn)
	let boxW = $state(0);
	const narrow = $derived(boxW > 0 && boxW < 240);
</script>

<figure class="pb" class:narrow bind:clientWidth={boxW}>
	{#if label}<figcaption>{t(label)} · <span class="muted">{ui('topK', { k: rows.length })}</span></figcaption>{/if}
	<ol>
		{#each rows as it, r (it.id ?? it.label)}
			<li class:chosen={isChosen(it)}>
				<button type="button" class="row" onclick={() => onselect?.(it)} disabled={!onselect}>
					<span class="lab" class:tok={tokens}>
						{#if it.color}<span class="sw" style="background: {it.color}" aria-hidden="true"></span>{/if}
						<span class="txt">{tokens ? visibleToken(it.label) : it.label}</span>
						{#if it.id !== undefined && tokens && ids}<span class="id num">#{it.id}</span>{/if}
					</span>
					<span class="track" aria-hidden="true">
						<span
							class="bar"
							style="width: {Math.max(0.5, (it.p / maxP) * 100)}%; background: {it.color ?? seq.css(0.72)}"
						></span>
					</span>
					<span class="val num" title={fmt3(it.p, 6)}>{fmtPct(it.p)}</span>
					<span class="badge">{isChosen(it) ? ui('chosen') : ''}</span>
				</button>
				{#each cutAfter(r + 1) as c (c.label)}
					<div class="cut" role="separator"><span>{c.label}</span></div>
				{/each}
			</li>
		{/each}
	</ol>
</figure>

<style>
	.pb {
		margin: 0;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-sm);
		margin-bottom: 6px;
	}
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	.row {
		width: 100%;
		display: grid;
		grid-template-columns: minmax(4.5em, 9em) 1fr 4.2em 4.6em;
		align-items: center;
		gap: 8px;
		padding: 3px 6px;
		border: 1px solid transparent;
		border-radius: var(--r1);
		background: none;
		text-align: left;
		font-size: var(--fs-sm);
		color: var(--ink);
		cursor: default;
	}
	.narrow .row {
		grid-template-columns: minmax(3.5em, 9em) minmax(16px, 1fr) 4.2em;
		gap: 6px;
		padding: 3px 4px;
	}
	.narrow .badge {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}
	.row:not(:disabled) {
		cursor: pointer;
	}
	.row:not(:disabled):hover {
		background: var(--surface-2);
	}
	.row:disabled {
		opacity: 1;
	}
	.chosen .row {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.lab {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
	}
	.txt {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.tok .txt {
		font-family: var(--font-mono);
		background: var(--surface-2);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
	}
	.id {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.sw {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		flex: 0 0 auto;
	}
	.track {
		height: 12px;
		background: var(--surface-2);
		border-radius: 0 3px 3px 0;
		overflow: hidden;
	}
	.bar {
		display: block;
		height: 100%;
		border-radius: 0 3px 3px 0;
		transition: width var(--dur-2) var(--ease);
	}
	.val {
		text-align: right;
		font-size: var(--fs-xs);
	}
	.badge {
		font-size: var(--t-xs);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--accent);
		font-weight: 700;
	}
	.cut {
		position: relative;
		height: 14px;
		margin: 2px 0;
	}
	.cut::before {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		border-top: 1.5px dashed var(--ink-2);
	}
	.cut span {
		position: absolute;
		right: 4px;
		top: -2px;
		padding: 0 6px;
		background: var(--surface);
		font-family: var(--font-mono);
		font-size: var(--t-xs);
		color: var(--ink-2);
	}
</style>
