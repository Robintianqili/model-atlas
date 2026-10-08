<!--
  FormulaCard: a KaTeX formula with colored symbols and live numbers, plus a "where" legend.

  <FormulaCard
    tex={`${sym('out','y_{ij}')} = \\sum_{u,v} ${sym('param','K_{uv}')}\\,${sym('in','x_{i+u,j+v}')} + ${sym('param','b')} = {{y}}`}
    values={{ y: 0.734 }}
    where={[{ tex: sym('param', 'K'), text: { en: '3×3 kernel', zh: '3×3 卷积核' } }]}
    provenance="live" />

  The formula is rendered at build time (SSR) and re-rendered in the browser when `values` change.
-->
<script lang="ts">
	import 'katex/dist/katex.min.css';
	import '$lib/styles/math.css';
	import { renderTex, fillTex } from './tex';
	import ProvenanceBadge from './ProvenanceBadge.svelte';
	import { keyScroll } from './keyscroll';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Provenance } from './types';

	interface Props {
		tex: string;
		values?: Record<string, number | string>;
		digits?: number;
		display?: boolean;
		caption?: Text;
		where?: { tex: string; text: Text }[];
		provenance?: Provenance;
	}
	let { tex, values = {}, digits = 3, display = true, caption, where = [], provenance }: Props = $props();

	const html = $derived(renderTex(fillTex(tex, values, digits), display));
</script>

<figure class="fc">
	{#if caption || provenance}
		<figcaption>
			{#if caption}<span>{t(caption)}</span>{/if}
			{#if provenance}<ProvenanceBadge mode={provenance} compact />{/if}
		</figcaption>
	{/if}
	<div class="math" use:keyScroll={'.katex-display'}>{@html html}</div>
	{#if where.length}
		<!-- the "where" label sits outside the <dl>: a <dl> may hold only <dt>/<dd> groups (axe definition-list) -->
		<div class="where">
			<span class="w">{ui('formulaWhere')}</span>
			<dl>
				{#each where as w, i (i)}
					<div>
						<dt>{@html renderTex(w.tex)}</dt>
						<dd>{t(w.text)}</dd>
					</div>
				{/each}
			</dl>
		</div>
	{/if}
</figure>

<style>
	.fc {
		margin: 0;
		padding: 12px 16px;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		min-width: 0;
	}
	figcaption {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		margin-bottom: 4px;
	}
	.math {
		overflow-x: auto;
		overflow-y: hidden;
	}
	.where,
	.where dl {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 16px;
		margin: 6px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.where dl {
		margin: 0;
	}
	.where dl div {
		display: flex;
		gap: 6px;
		align-items: baseline;
	}
	.w {
		color: var(--muted);
		font-style: italic;
	}
	dt,
	dd {
		margin: 0;
	}
</style>
