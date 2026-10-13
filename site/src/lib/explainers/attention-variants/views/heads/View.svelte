<!--
  'attention-variants/heads' (meta.ts): one small SVG per variant. Query heads (cat-0) in a row; below them the key/value
  heads (cat-1), one box per KV head spanning its group, with a line from every query head to the box it reads (a band per
  group when the heads are too many to draw apart). MLA: one latent box under every head. Each row's cache is printed under it.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { kindOf, type Config } from '../../data.ts';
	import { configAt, rowsOf, rowText, type RowId } from './meta.ts';

	let P: StageProps = $props();
	const rows = $derived(rowsOf(configAt(P), Array.isArray(P.props?.rows) ? (P.props.rows as RowId[]) : ['mine']));
	$effect(() => void (P.autoplay && P.onend()));
	const [W, M, QY, KY, BH] = [360, 6, 4, 46, 14];
	const geo = (c: Config) => {
		const u = (W - 2 * M) / c.heads;
		const g = c.heads / c.kvHeads;
		const gap = u > 5 ? 1.5 : 0;
		return { u, g, gap, lines: c.heads <= 48 };
	};
	const idx = (n: number) => Array.from({ length: n }, (_, i) => i);
</script>

<div class="heads">
	{#each rows as r (r.id)}
		{@const c = r.c}
		{@const G = geo(c)}
		{@const mla = kindOf(c) === 3}
		<figure>
			<svg viewBox="0 0 {W} {KY + BH + 4}" role="img" aria-label={t(rowText(c))}>
				{#if G.lines}
					{#each idx(c.heads) as h (h)}
						<line class="ln" x1={M + (h + 0.5) * G.u} y1={QY + BH} x2={mla ? M + (h + 0.5) * G.u : M + (Math.floor(h / G.g) + 0.5) * G.g * G.u} y2={KY} />
					{/each}
				{:else if !mla}
					{#each idx(c.kvHeads) as k (k)}
						<polygon class="band" points="{M + k * G.g * G.u},{QY + BH} {M + (k + 1) * G.g * G.u},{QY + BH} {M + (k + 1) * G.g * G.u - G.gap},{KY} {M + k * G.g * G.u + G.gap},{KY}" />
					{/each}
				{/if}
				{#each idx(c.heads) as h (h)}
					<rect class="q" x={M + h * G.u + G.gap / 2} y={QY} width={Math.max(0.6, G.u - G.gap)} height={BH} />
				{/each}
				{#if mla}
					<rect class="kv" x={M} y={KY} width={W - 2 * M} height={BH} />
				{:else}
					{#each idx(c.kvHeads) as k (k)}
						<rect class="kv" x={M + k * G.g * G.u + G.gap} y={KY} width={Math.max(0.6, G.g * G.u - 2 * G.gap)} height={BH} />
					{/each}
				{/if}
			</svg>
			<figcaption>{t(rowText(c))}</figcaption>
		</figure>
	{/each}
	<p class="legend">
		<span class="sw q"></span>{t({ en: 'query head', zh: '查询头' })}
		<span class="sw kv"></span>{t({ en: 'cached key + value head (MLA: the cached latent)', zh: '缓存的键头 + 值头（MLA：缓存的潜向量）' })}
	</p>
</div>

<style>
	.heads {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		max-width: 560px;
	}
	figure {
		margin: 0;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.q,
	.sw.q {
		fill: var(--cat-0);
		background: var(--cat-0);
	}
	.kv,
	.sw.kv {
		fill: var(--cat-1);
		background: var(--cat-1);
	}
	.ln {
		stroke: var(--muted);
		stroke-width: 0.8;
	}
	.band {
		fill: var(--rule);
	}
	figcaption,
	.legend {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.legend {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 12px;
	}
</style>
