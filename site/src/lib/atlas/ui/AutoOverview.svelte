<!--
  AutoOverview (design §5.1): the model at a glance, from the spec.
    pipeline  the kit ModelOverview (≤ 20 layers, or manifest.overview.layout = 'pipeline'), with a live thumbnail of each
              layer's output;
    stack     collapsible inferred groups (L0, L1, … : ids sharing a prefix before the first '.') for deep models, with
              a note on the residual line, and the remaining layers in order.
  `current` (bindable) is the explore selection, shared with AutoInspector; the lesson never moves it.
-->
<script lang="ts">
	import ModelOverview from '$lib/kit/ModelOverview.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { LayerNode } from '$lib/kit/types';
	import type { Tensor, Trace } from '../types/core.ts';
	import type { AtlasSpec } from '../types/spec.ts';
	import { inferGroups } from '../lesson/plan.ts';
	import { layerNodes } from './kitspec.ts';
	import { say } from './strings.ts';

	interface Props {
		spec: AtlasSpec;
		trace: Trace | null;
		current?: string;
		layout?: 'pipeline' | 'stack';
		label?: string;
	}
	let { spec, trace, current = $bindable(''), layout, label }: Props = $props();

	const dims = $derived((trace?.meta?.dims ?? {}) as Record<string, number>);
	const nodes = $derived(layerNodes(spec, dims));
	const mode = $derived(layout ?? (spec.layers.length <= 20 ? 'pipeline' : 'stack'));
	const groups = $derived(inferGroups(spec));

	function thumbOf(id: string): { data: ArrayLike<number>; rows: number; cols: number } | null {
		const x: Tensor | undefined = trace?.t[id];
		if (!x) return null;
		const d = [...x.shape];
		while (d.length > 1 && d[0] === 1) d.shift();
		if (d.length >= 3) {
			const hw = d[d.length - 2] * d[d.length - 1];
			return { data: x.data.subarray(0, hw), rows: d[d.length - 2], cols: d[d.length - 1] };
		}
		if (d.length === 2) return { data: x.data, rows: d[0], cols: d[1] };
		const n = x.data.length;
		const cols = Math.min(n, 32);
		return { data: x.data.subarray(0, cols), rows: 1, cols };
	}

	type Row = { kind: 'layer'; node: LayerNode } | { kind: 'group'; id: string; nodes: LayerNode[] };
	const rows = $derived.by((): Row[] => {
		const out: Row[] = [];
		const byIndex = new Map<number, { id: string; to: number }>();
		for (const g of groups) byIndex.set(g.from, { id: g.id, to: g.to });
		for (let i = 0; i < nodes.length; i++) {
			const g = byIndex.get(i);
			if (g) {
				out.push({ kind: 'group', id: g.id, nodes: nodes.slice(i, g.to + 1) });
				i = g.to;
			} else out.push({ kind: 'layer', node: nodes[i] });
		}
		return out;
	});
	const openGroup = $derived(groups.find((g) => spec.layers.slice(g.from, g.to + 1).some((l) => l.id === current))?.id);
</script>

{#if mode === 'pipeline'}
	<ModelOverview layers={nodes} bind:current label={label ?? t(spec.name)}>
		{#snippet thumb(n)}
			{@const th = thumbOf(n.id)}
			{#if th}
				<TensorGrid data={th.data} rows={th.rows} cols={th.cols} label={t(n.label)} maxCell={3} minCell={1} gridLines={false} readout={false} interactive={false} />
			{/if}
		{/snippet}
	</ModelOverview>
{:else}
	<div class="stack" role="list" aria-label={label ?? t(spec.name)}>
		{#each rows as r (r.kind === 'group' ? `g:${r.id}` : r.node.id)}
			{#if r.kind === 'group'}
				<details class="grp" role="listitem" open={openGroup === r.id}>
					<summary><span class="gid">{r.id}</span> <span class="muted">{say('groupOf', { n: r.nodes.length })}</span></summary>
					<ol>
						{#each r.nodes as n (n.id)}
							<li>
								<button type="button" class="ly" aria-current={current === n.id ? 'true' : undefined} onclick={() => (current = n.id)}>
									<span>{t(n.label)}</span><span class="sh num">{n.outShape?.join('×')}</span>
								</button>
							</li>
						{/each}
					</ol>
				</details>
			{:else}
				<div role="listitem">
					<button type="button" class="ly" aria-current={current === r.node.id ? 'true' : undefined} onclick={() => (current = r.node.id)}>
						<span>{t(r.node.label)}</span><span class="sh num">{r.node.outShape?.join('×')}</span>
					</button>
				</div>
			{/if}
		{/each}
		{#if groups.length > 1}<p class="muted note">{say('residual')}</p>{/if}
	</div>
{/if}

<style>
	.stack {
		display: grid;
		gap: 4px;
		border-left: 3px solid var(--rule-strong);
		padding-left: 10px;
	}
	.grp {
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		padding: 4px 8px;
	}
	.grp summary {
		cursor: pointer;
		font-weight: 600;
	}
	.gid {
		font-family: var(--font-mono);
	}
	ol {
		list-style: none;
		margin: 4px 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	.ly {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		width: 100%;
		text-align: left;
		background: none;
		border: 1px solid transparent;
		border-radius: var(--r1);
		padding: 3px 6px;
		font: inherit;
		font-size: var(--fs-sm);
		color: var(--ink);
		cursor: pointer;
	}
	.ly:hover {
		background: var(--surface-2);
	}
	.ly[aria-current='true'] {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.sh {
		color: var(--muted);
		font-size: var(--fs-xs);
	}
	.note {
		font-size: var(--fs-xs);
		margin: 4px 0 0;
	}
</style>
