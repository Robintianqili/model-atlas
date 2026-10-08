<!--
  AutoInspector (design §5.1): the kit Inspector on the live trace. Values and Stats come from the trace; Math is the op's
  formula (sym() colour roles, {{name}} slots) filled with explain() at the explore unit of that layer; Code is the op's
  one-line PyTorch equivalent and, when the exporter recorded it, the file and line that define the layer.
-->
<script lang="ts">
	import Inspector from '$lib/kit/Inspector.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import type { Provenance } from '$lib/kit/types';
	import type { Trace } from '../types/core.ts';
	import type { GraphModel, UnitExplanation } from '../types/op.ts';
	import type { AtlasSpec } from '../types/spec.ts';
	import { layerNodes, parityBadge } from './kitspec.ts';
	import { say } from './strings.ts';

	interface Props {
		spec: AtlasSpec;
		model: GraphModel | null;
		trace: Trace | null;
		current?: string;
		/** the explore unit of a layer (AutoModelPage picks it with the op's UnitPick) */
		unitOf: (layer: string) => number[] | undefined;
		provenance?: Provenance;
	}
	let { spec, model, trace, current = $bindable(''), unitOf, provenance = 'live' }: Props = $props();

	const nodes = $derived(layerNodes(spec, (trace?.meta?.dims ?? {}) as Record<string, number>));
	const badge = $derived(parityBadge(spec));
	function opOf(id: string) {
		try {
			const l = model?.layer(id);
			return l && model ? { layer: l, def: model.ops.get(l.op).def } : null;
		} catch {
			return null;
		}
	}
	function explainOf(id: string): UnitExplanation | null {
		const u = unitOf(id);
		if (!model || !trace || !u) return null;
		try {
			return model.explain(trace, id, u);
		} catch {
			return null;
		}
	}
	const kindOf = (id: string) => {
		const op = opOf(id)?.def;
		return op?.op === 'softmax' || op?.color === 'sequential' ? ('sequential' as const) : ('diverging' as const);
	};
</script>

<!-- K1: maxRows: a [T, D] layer at T = 128 shows 32 rows at a time (it drew a table thousands of pixels tall) -->
<Inspector layers={nodes} bind:current tensorOf={(id) => trace?.t[id]} {kindOf} {provenance} precision={badge.precision} tolerance={badge.tolerance} maxRows={32}>
	{#snippet math(id)}
		{@const o = opOf(id)}
		{@const x = explainOf(id)}
		{#if o?.def.formula}
			<FormulaCard tex={o.def.formula} values={x?.values ?? {}} caption={x ? `${id} [${x.unit.join(', ')}]` : id} {provenance} />
		{:else}
			<p class="muted">{o ? o.def.op : id}</p>
		{/if}
	{/snippet}
	{#snippet code(id)}
		{@const o = opOf(id)}
		{#if o?.def.pytorch}
			<CodeBlock code={o.def.pytorch(o.layer)} lang="python" caption={say('codeFor')} />
		{/if}
		{#if o?.layer.code}
			<p class="muted small">{say('codeWhere', { file: o.layer.code.file, line: o.layer.code.line })}</p>
		{:else if !o?.def.pytorch}
			<p class="muted small">{say('noCode')}</p>
		{/if}
	{/snippet}
</Inspector>

<style>
	.small {
		font-size: var(--fs-xs);
	}
</style>
