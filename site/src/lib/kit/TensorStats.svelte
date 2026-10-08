<!--
  TensorStats: shape, count, min / max / mean / std, fraction of zeros and a histogram.
  The histogram bars use the same color scale as the tensor view.
  <TensorStats tensor={trace.t.conv1} kind="diverging" />
-->
<script lang="ts">
	import Num from './Num.svelte';
	import { tensorStats, histogram } from './stats';
	import { makeScale, domainOf, type ScaleKind } from './color';
	import { themeState } from './theme.svelte';
	import { fmtShape, fmtInt, fmtPct, fmt3 } from './format';
	import { ui } from '$lib/i18n/i18n.svelte';
	import type { TensorLike } from './types';

	interface Props {
		tensor: TensorLike;
		kind?: ScaleKind;
		bins?: number;
	}
	let { tensor, kind = 'diverging', bins = 28 }: Props = $props();

	const s = $derived(tensorStats(tensor.data));
	const dom = $derived(domainOf(kind, tensor.data));
	const h = $derived(histogram(tensor.data, bins, kind === 'diverging' ? dom : [s.min, s.max]));
	const scale = $derived(makeScale(kind, dom, themeState.resolved));
	const W = 280;
	const H = 72;
</script>

<div class="ts">
	<dl class="grid">
		<div><dt>{ui('shape')}</dt><dd class="num">{fmtShape(tensor.shape)}</dd></div>
		<div><dt>{ui('count')}</dt><dd class="num">{fmtInt(s.count)}</dd></div>
		<div><dt>{ui('min')}</dt><dd><Num value={s.min} /></dd></div>
		<div><dt>{ui('max')}</dt><dd><Num value={s.max} /></dd></div>
		<div><dt>{ui('mean')}</dt><dd><Num value={s.mean} /></dd></div>
		<div><dt>{ui('std')}</dt><dd><Num value={s.std} /></dd></div>
		<div><dt>{ui('sparsity')}</dt><dd class="num">{fmtPct(s.zeros)}</dd></div>
	</dl>
	<figure class="hist">
		<figcaption>{ui('histogram')}</figcaption>
		<svg viewBox="0 0 {W} {H + 14}" width="100%" role="img" aria-label="{ui('histogram')}: {h.counts.join(', ')}">
			{#each h.counts as c, i (i)}
				{@const bw = W / h.counts.length}
				{@const bh = (c / h.max) * H}
				{@const mid = (h.edges[i] + h.edges[i + 1]) / 2}
				<rect x={i * bw + 1} y={H - bh} width={Math.max(1, bw - 2)} height={bh} rx="1.5" fill={scale.css(mid)}>
					<title>{fmt3(h.edges[i])} … {fmt3(h.edges[i + 1])}: {c}</title>
				</rect>
			{/each}
			<line x1="0" x2={W} y1={H + 0.5} y2={H + 0.5} stroke="var(--rule-strong)" />
			<text x="0" y={H + 12} class="ax">{fmt3(h.edges[0])}</text>
			<text x={W} y={H + 12} class="ax" text-anchor="end">{fmt3(h.edges[h.edges.length - 1])}</text>
		</svg>
	</figure>
</div>

<style>
	.ts {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
		gap: 8px 12px;
		margin: 0;
	}
	dt {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	dd {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.hist {
		margin: 0;
		max-width: 360px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--muted);
		margin-bottom: 2px;
	}
	.ax {
		font-family: var(--font-mono);
		font-size: 9px;
		fill: var(--muted);
	}
</style>
