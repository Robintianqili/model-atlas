<!--
  CoverStrip (redesign spec §5.6, the cover's stage for a package with a spec): answer first. The whole model on the
  learner's input as one strip, input → a few layers → output, one real channel per stage with its shape, and the
  model's answer with its top scores (real probabilities from the output layer). Live: computed from the same trace as
  the lesson. A model without an answer layer (a generator, an autoencoder) shows the strip alone.
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import Icon from '$lib/kit/Icon.svelte';
	import { right } from '$lib/kit/icons';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Tensor } from '../../types/core.ts';
	import { primaryTrace, outputId } from '../../lesson/facts.ts';
	import { fmtPct } from '../../core/format.ts';
	import type { AtlasStore } from '../runtime.svelte.ts';
	import { isAnswerLayer, topAnswers } from '../fallback.ts';
	import { shapeLabel } from '../kitspec.ts';

	let { store }: { store: AtlasStore } = $props();
	const rt = $derived(store.r);
	const trace = $derived.by(() => {
		void store.version;
		return primaryTrace({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel });
	});
	const dims = $derived((trace?.meta?.dims ?? {}) as Record<string, number>);

	function thumb(x: Tensor | undefined): { data: ArrayLike<number>; rows: number; cols: number } | null {
		if (!x) return null;
		const d = [...x.shape];
		while (d.length > 1 && d[0] === 1) d.shift();
		if (d.length >= 3) {
			const hw = d[d.length - 2] * d[d.length - 1];
			return { data: x.data.subarray(0, hw), rows: d[d.length - 2], cols: d[d.length - 1] };
		}
		if (d.length === 2) return d[0] * d[1] <= 4096 ? { data: x.data, rows: d[0], cols: d[1] } : null;
		// a vector (its first 64 values) wraps into the squarest whole grid (64 → 8×8, 10 → 2×5, 3 → 1×3), so the
		// thumbnail shows its values as cells rather than a hairline; the shape below the picture stays the real one
		const n = Math.min(x.data.length, 64);
		if (!n) return null;
		let cols = Math.ceil(Math.sqrt(n));
		while (n % cols) cols++;
		return { data: x.data.subarray(0, n), rows: n / cols, cols };
	}
	const outId = $derived(rt.spec && trace ? outputId(rt.spec, trace) : undefined);
	/** input, up to four layers between (evenly spaced), the output */
	const stages = $derived.by(() => {
		const layers = (rt.spec?.layers ?? []).filter((l) => !!trace?.t[l.id]);
		if (!layers.length) return [];
		const first = layers[0];
		const last = layers.find((l) => l.id === outId) ?? layers[layers.length - 1];
		const mid = layers.filter((l) => l !== first && l !== last && thumb(trace!.t[l.id]));
		const k = Math.min(4, mid.length);
		const pick = Array.from({ length: k }, (_, i) => mid[Math.round(((i + 1) * (mid.length + 1)) / (k + 1)) - 1]).filter((l, i, a) => l && a.indexOf(l) === i);
		return [first, ...pick, last].map((l) => ({ id: l.id, label: l.label, shape: shapeLabel(l.outShape, dims), th: thumb(trace!.t[l.id]) }));
	});
	const answers = $derived.by(() => {
		void store.version;
		const o = outId ? trace?.t[outId] : undefined;
		if (!o || !isAnswerLayer(rt, outId, trace)) return [];
		return topAnswers(rt, o, 4);
	});
	const top = $derived(answers[0]);
	/** QA 2026-10-13: thumbnails fill a square box sized so that all stages fit on one row of the strip (64-112 px;
	 * a narrow stage wraps), instead of 36 px pictures in an empty pane; cells are capped at 16 px (no value labels).
	 * A stage is box + 8 px wide (its two-line label); each arrow takes 16 px plus two 8 px gaps; the strip pads 14 px. */
	let stripW = $state(0);
	const box = $derived.by(() => {
		const k = Math.max(1, stages.length);
		const fit = stripW ? Math.floor((stripW - 28 - (k - 1) * 32) / k) - 9 : 72;
		return Math.max(64, Math.min(112, fit));
	});
	const cellOf = (th: { rows: number; cols: number }) => Math.min(16, box / Math.max(th.rows, th.cols));
</script>

<figure class="cov">
	{#if top}
		<div class="ans">
			<p class="k">{t({ en: 'It reads', zh: '它的答案' })}</p>
			<p class="a"><span class="w">{t(top.label)}</span> <span class="num p">{fmtPct(top.p)}</span></p>
			<ol class="bars" aria-label={t({ en: 'Top answers', zh: '得分最高的答案' })}>
				{#each answers as a (a.i)}
					<li>
						<span class="bl">{t(a.label)}</span>
						<span class="bt"><span class="bf" style:width="{Math.max(0.5, a.p * 100)}%"></span></span>
						<span class="num bv">{fmtPct(a.p)}</span>
					</li>
				{/each}
			</ol>
		</div>
	{/if}
	{#if stages.length > 1}
		<ol class="strip" bind:clientWidth={stripW} style:--box="{box}px">
			{#each stages as s, i (s.id)}
				{#if i}<li class="arr" aria-hidden="true"><Icon d={right} size={16} /></li>{/if}
				<li class="st">
					<div class="pic">
						{#if s.th}<TensorGrid data={s.th.data} rows={s.th.rows} cols={s.th.cols} label={t(s.label)} cell={cellOf(s.th)} gridLines={false} labels={false} readout={false} interactive={false} />{/if}
					</div>
					<span class="sl" title={t(s.label)}>{t(s.label)}</span>
					<span class="num ss">{s.shape.join('×')}</span>
				</li>
			{/each}
		</ol>
	{/if}
	<figcaption class="cap">{t({ en: 'One real channel per stage, computed from your input. The lesson opens each one.', zh: '每一站展示一个真实通道，用你的输入算出。课程会逐一打开它们。' })}</figcaption>
</figure>

<style>
	.cov {
		display: grid;
		gap: 18px;
		margin: 0;
	}
	.ans p {
		margin: 0;
	}
	.k {
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-title);
	}
	.a {
		margin: 2px 0 10px;
	}
	.w {
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
		font-weight: var(--w-display);
	}
	.p {
		color: var(--ink-2);
		font-size: var(--t-md);
	}
	.bars {
		display: grid;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
		max-width: 520px;
	}
	.bars li {
		display: grid;
		grid-template-columns: minmax(70px, 140px) 1fr 64px;
		gap: 10px;
		align-items: center;
		font-size: var(--t-sm);
	}
	.bl {
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.bt {
		height: 10px;
		border-radius: var(--r-pill);
		background: var(--sunken);
	}
	.bf {
		display: block;
		height: 100%;
		border-radius: var(--r-pill);
		background: var(--seq-hi);
	}
	.bv {
		text-align: right;
		color: var(--ink-2);
	}
	.strip {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: 8px;
		margin: 0;
		padding: 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		list-style: none;
	}
	.st {
		display: grid;
		justify-items: center;
		align-content: start;
		gap: 4px;
		min-width: 64px;
	}
	.pic {
		display: grid;
		place-items: center;
		width: var(--box, 72px);
		height: var(--box, 72px);
		overflow: hidden;
	}
	/* the stage's name on up to two lines (it was cut to one line with an ellipsis); the full name is its title */
	.sl {
		display: -webkit-box;
		max-width: calc(var(--box, 72px) + 8px);
		overflow: hidden;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-ui);
		text-align: center;
		-webkit-box-orient: vertical;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		overflow-wrap: anywhere;
	}
	.ss {
		color: var(--muted);
		font-size: var(--t-xs);
	}
	.arr {
		align-self: start;
		margin-top: calc(var(--box, 72px) / 2 - 8px);
		color: var(--faint);
	}
	.cap {
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
	}
</style>
