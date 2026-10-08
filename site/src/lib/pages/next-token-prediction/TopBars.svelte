<!--
  The model's top guesses for one window as labeled bars (bar length = probability, 0 … 1), with
  the token that really came next pinned and marked in green even when it is far down the list.
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { visibleToken } from '$lib/kit/format';
	import type { Dist } from './lessonMath';
	import { pct } from './lessonMath';

	interface Props {
		dist: Dist;
		disp: (id: number) => string;
		k?: number;
		label?: Text;
	}
	let { dist, disp, k = 6, label }: Props = $props();

	const rows = $derived.by(() => {
		const out = dist.top.slice(0, k).map((x, i) => ({ id: x.id, p: x.p, rank: i + 1 }));
		if (!out.some((x) => x.id === dist.target)) out.push({ id: dist.target, p: dist.p, rank: dist.rank });
		return out;
	});
</script>

<figure class="tb">
	{#if label}<figcaption>{t(label)}</figcaption>{/if}
	<ol>
		{#each rows as r (r.id)}
			{#if r.rank > k}<li class="gap" aria-hidden="true">⋯</li>{/if}
			<li class:real={r.id === dist.target}>
				<span class="rk num">{r.rank}</span>
				<span class="lab"><span class="txt">{visibleToken(disp(r.id))}</span><span class="id num">#{r.id}</span></span>
				<span class="track" aria-hidden="true"><span class="bar" style="width: {Math.max(0.4, r.p * 100)}%"></span></span>
				<span class="val num">{pct(r.p)}</span>
				{#if r.id === dist.target}<span class="tag">{t({ en: 'real next', zh: '真实的下一个' })}</span>{/if}
			</li>
		{/each}
	</ol>
</figure>

<style>
	.tb {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-sm);
		font-weight: 650;
		color: var(--ink-2);
	}
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 3px;
	}
	li {
		position: relative;
		display: grid;
		grid-template-columns: 1.6em minmax(4.5em, 8.5em) minmax(0, 1fr) 4.2em;
		gap: 8px;
		align-items: center;
		padding: 3px 6px;
		border: 1.5px solid transparent;
		border-radius: var(--r1);
		font-size: var(--fs-sm);
	}
	li.real {
		border-color: var(--real);
		background: color-mix(in srgb, var(--real) 10%, var(--surface));
		padding-bottom: 18px;
	}
	li.gap {
		display: block;
		padding: 0 8px;
		color: var(--muted);
		line-height: 1;
	}
	.rk {
		color: var(--muted);
		font-size: var(--fs-xs);
		text-align: right;
	}
	.lab {
		display: flex;
		align-items: baseline;
		gap: 6px;
		min-width: 0;
	}
	.txt {
		font-family: var(--font-mono);
		white-space: pre;
		background: var(--surface-2);
		padding: 0 4px;
		border-radius: 3px;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.id {
		font-size: 10px;
		color: var(--muted);
	}
	.track {
		height: 12px;
		background: var(--surface-2);
		border-radius: 2px;
		overflow: hidden;
	}
	.bar {
		display: block;
		height: 100%;
		background: var(--seq-mid);
		transition: width var(--dur-2) var(--ease);
	}
	li.real .bar {
		background: var(--real);
	}
	.val {
		text-align: right;
		font-weight: 650;
	}
	.tag {
		position: absolute;
		left: calc(1.6em + 14px);
		bottom: 2px;
		font-size: 10px;
		font-weight: 700;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--real-text);
	}
</style>
