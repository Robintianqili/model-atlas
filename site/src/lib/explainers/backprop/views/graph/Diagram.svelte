<!--
  backprop Diagram: a computation graph (data.ts GraphDef) drawn as HTML node boxes over an SVG of its edges, shared by
  the views 'backprop/graph' and 'backprop/net'. Wide containers lay the flow out left to right, narrow ones (a phone)
  top to bottom, so the text keeps its size. Each node shows its value and its gradient ("?" until known); each edge is
  'todo', 'done' (the backward pass crossed it) or 'active' (crossed in this frame), and may carry a label.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { I18n } from '../../../../atlas/types/core.ts';
	import type { EdgeState, GraphDef } from '../../data.ts';

	let {
		def,
		vals,
		grads,
		edges,
		labels = [],
		badge = null,
		label
	}: { def: GraphDef; vals: number[]; grads: number[]; edges: EdgeState[]; labels?: (string | null)[]; badge?: string | null; label: I18n } = $props();

	let cw = $state(640);
	const vertical = $derived(cw < 720);
	/** a vertical layout keeps to a column at most this wide, centred */
	const VW = $derived(Math.min(cw, 560));
	const NW = 100;
	const GAP = 34;
	const NH = 62;
	const PAD = 6;
	const depthMax = $derived(Math.max(...def.nodes.map((n) => n.depth)));
	const laneMax = $derived(Math.max(...def.nodes.map((n) => n.lane)));
	/** top-left corner of a node box */
	function pos(n: { depth: number; lane: number }): [number, number] {
		if (vertical) {
			const across = laneMax > 0 ? (n.lane / laneMax) * Math.max(0, VW - 2 * PAD - NW) : (VW - NW) / 2;
			return [(cw - VW) / 2 + PAD + across, PAD + n.depth * (NH + GAP)];
		}
		return [PAD + (n.depth / depthMax) * Math.max(0, cw - 2 * PAD - NW), PAD + n.lane * (NH + 12)];
	}
	const height = $derived(vertical ? 2 * PAD + depthMax * (NH + GAP) + NH : 2 * PAD + laneMax * (NH + 12) + NH);
	const at = $derived(Object.fromEntries(def.nodes.map((n) => [n.id, pos(n)])));
	/** the SVG path of edge k and its midpoint; two edges between the same nodes (L = e·e) are pulled apart */
	function edgePath(k: number): { d: string; mid: [number, number]; twin: number } {
		const e = def.edges[k];
		const [px, py] = at[e.from];
		const [cx, cy] = at[e.to];
		const twin = def.edges.filter((o) => o.from === e.from && o.to === e.to).length > 1;
		const off = twin ? (e.slot === 0 ? -11 : 11) : 0;
		const tw = twin ? e.slot + 1 : 0;
		if (vertical) {
			const [x1, y1, x2, y2] = [px + NW / 2 + off, py + NH, cx + NW / 2 + off, cy];
			const m = (y2 - y1) / 2;
			return { d: `M${x1},${y1} C${x1},${y1 + m} ${x2},${y2 - m} ${x2},${y2}`, mid: [(x1 + x2) / 2, (y1 + y2) / 2], twin: tw };
		}
		const [x1, y1, x2, y2] = [px + NW, py + NH / 2 + off, cx, cy + NH / 2 + off];
		const m = (x2 - x1) / 2;
		return { d: `M${x1},${y1} C${x1 + m},${y1} ${x2 - m},${y2} ${x2},${y2}`, mid: [(x1 + x2) / 2, (y1 + y2) / 2], twin: tw };
	}
	const paths = $derived(def.edges.map((_, k) => edgePath(k)));
	/** nodes at either end of an active edge */
	const hot = $derived(new Set(def.edges.flatMap((e, k) => (edges[k] === 'active' ? [e.from, e.to] : []))));
	const G = { en: 'grad', zh: '梯度' };
</script>

<div class="dia" bind:clientWidth={cw} style:height="{height}px" role="group" aria-label={t(label)}>
	<svg width={cw} height={height} aria-hidden="true">
		<defs>
			<marker id="bp-fwd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="mk-fwd" /></marker>
			<marker id="bp-back" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="mk-back" /></marker>
		</defs>
		{#each paths as p, k (k)}
			<path class="edge {edges[k] ?? 'todo'}" d={p.d} marker-end={edges[k] === 'active' ? undefined : 'url(#bp-fwd)'} marker-start={edges[k] === 'active' ? 'url(#bp-back)' : undefined} />
		{/each}
	</svg>
	{#each paths as p, k (k)}
		{#if labels[k]}
			<span class="elabel num tw{p.twin}" class:vert={vertical} class:active={edges[k] === 'active'} style:left="{p.mid[0]}px" style:top="{p.mid[1]}px">{labels[k]}</span>
		{/if}
	{/each}
	{#each def.nodes as n, i (n.id)}
		<div class="node {n.kind}" class:hot={hot.has(n.id)} style:left="{at[n.id][0]}px" style:top="{at[n.id][1]}px" style:width="{NW}px" style:height="{NH}px">
			<span class="nm">{n.formula ? `${n.id} = ${n.formula}` : n.id}</span>
			<span class="v num">{fmt3(vals[i])}</span>
			<span class="g num" class:unknown={!Number.isFinite(grads[i])}>{t(G)} {Number.isFinite(grads[i]) ? fmt3(grads[i]) : '?'}</span>
			{#if badge === n.id}<span class="badge">grad +=</span>{/if}
		</div>
	{/each}
</div>

<style>
	.dia {
		position: relative;
		width: 100%;
		min-width: 0;
	}
	svg {
		position: absolute;
		inset: 0;
		overflow: visible;
	}
	.edge {
		fill: none;
		stroke: var(--rule-strong);
		stroke-width: 1.5;
	}
	.edge.done {
		stroke: var(--grad);
		stroke-width: 1.8;
	}
	.edge.active {
		stroke: var(--grad);
		stroke-width: 4;
	}
	.mk-fwd {
		fill: var(--rule-strong);
	}
	.mk-back {
		fill: var(--grad);
	}
	.node {
		position: absolute;
		box-sizing: border-box;
		display: grid;
		align-content: center;
		gap: 1px;
		padding: 3px 6px;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		font-size: 11.5px;
		line-height: 1.25;
	}
	.node.param {
		border-color: var(--accent);
	}
	.node.data {
		border-style: dashed;
	}
	.node.hot {
		border-color: var(--grad);
		box-shadow: 0 0 0 2px var(--grad);
	}
	.nm {
		color: var(--ink-2);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.v {
		color: var(--ink);
		font-weight: 600;
	}
	.g {
		color: var(--grad);
	}
	.g.unknown {
		color: var(--muted);
	}
	.badge {
		position: absolute;
		top: -9px;
		right: 4px;
		padding: 0 4px;
		border-radius: var(--r-pill);
		background: var(--grad);
		color: var(--surface);
		font-size: 10px;
		font-weight: 700;
	}
	.elabel {
		position: absolute;
		transform: translate(-50%, -50%);
		padding: 0 4px;
		border-radius: var(--r1, 4px);
		background: var(--surface-2);
		border: 1px solid var(--rule);
		color: var(--ink-2);
		font-size: 10.5px;
		white-space: nowrap;
	}
	/* the two labels of a twin pair sit outside the pair: above and below it, or (vertical) to either side */
	.elabel.tw1 {
		transform: translate(-50%, calc(-100% - 2px));
	}
	.elabel.tw2 {
		transform: translate(-50%, 2px);
	}
	.elabel.vert.tw1 {
		transform: translate(calc(-100% - 4px), -50%);
	}
	.elabel.vert.tw2 {
		transform: translate(4px, -50%);
	}
	.elabel.active {
		border-color: var(--grad);
		color: var(--ink);
		font-weight: 700;
	}
</style>
