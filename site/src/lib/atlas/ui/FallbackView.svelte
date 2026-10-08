<!--
  FallbackView: the generic stage for a view key this build does not register (StageProps in, like any view). It shows the
  real numbers of the At: a unit's arithmetic (the window it reads, the products, the sum and the bias, and where the result
  lands in the output), a layer's output tensor (probability bars for a class output), or a process / training frame's
  scalars and tensors. Pure helpers are in fallback.ts; describe() is describeFallback().
  props.onpick(unit) (optional): clicking a cell of the output map picks that unit (the explore "unit" section uses it).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import Num from './views/kit.parts/Num.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../types/view.ts';
	import type { Tensor } from '../types/core.ts';
	import { fallbackModel, isAnswerLayer, isProbabilities, shownTerms, topAnswers } from './fallback.ts';
	import { say } from './strings.ts';

	// destructured on purpose: `let p = $props()` compiles to rest_props, a helper every page would then carry
	let { rt, source, at, autoplay, onend, props }: StageProps = $props();
	const m = $derived(fallbackModel({ rt, source, at }));
	const onpick = $derived(typeof props?.onpick === 'function' ? (props.onpick as (u: number[]) => void) : undefined);

	// a view that animates itself must report the end; this one has nothing to animate
	onMount(() => {
		if (autoplay) onend?.();
	});

	function dims(x: Tensor): number[] {
		const d = [...x.shape];
		while (d.length > 1 && d[0] === 1) d.shift();
		return d;
	}
	/** one 2-D slice of a tensor: channel `c` of [C,H,W], the matrix of [R,C], a row of a vector */
	function slice(x: Tensor, c = 0): { data: ArrayLike<number>; rows: number; cols: number } {
		const d = dims(x);
		if (d.length >= 3) {
			const hw = d[d.length - 2] * d[d.length - 1];
			const ch = Math.max(0, Math.min(d[0] - 1, c));
			return { data: x.data.subarray(ch * hw, ch * hw + hw), rows: d[d.length - 2], cols: d[d.length - 1] };
		}
		if (d.length === 2) return { data: x.data, rows: d[0], cols: d[1] };
		return { data: x.data, rows: 1, cols: d[0] ?? x.data.length };
	}
	const outSlice = $derived.by(() => {
		if (!m.out) return null;
		const d = dims(m.out);
		const u = m.unit?.unit ?? [];
		const ch = d.length >= 3 ? (u[u.length - 3] ?? 0) : 0;
		const s = slice(m.out, ch);
		let hl: { row: number; col: number }[] = [];
		if (m.unit) {
			if (d.length >= 2) hl = [{ row: u[u.length - 2] ?? 0, col: u[u.length - 1] ?? 0 }];
			else hl = [{ row: 0, col: u[u.length - 1] ?? 0 }];
		}
		return { ...s, ch, hl };
	});
	const readSlice = $derived(m.read ? { ...slice(m.read.t, m.read.channel), box: m.read.box } : null);
	const terms = $derived(m.unit ? shownTerms(m.unit) : null);
	// the model's answer (or any probability vector): top answers as bars; a [T, V] output is read at its last row
	const probs = $derived.by(() => {
		if (m.kind !== 'tensor' || !m.out) return null;
		const trace = rt.sources[source]?.trace ?? null;
		if (!isProbabilities(m.out) && !isAnswerLayer(rt, m.layer, trace)) return null;
		return topAnswers(rt as never, m.out, 10).map((x) => ({ label: t(x.label), p: x.p, id: x.i }));
	});
	const outLabel = $derived(`${m.layer ?? ''}${m.out ? ` · ${m.out.shape.join('×')}` : ''}`);
	function pick(c: { row: number; col: number }) {
		if (!onpick || !m.out) return;
		const d = dims(m.out);
		if (d.length >= 3) onpick([outSlice?.ch ?? 0, c.row, c.col]);
		else if (d.length === 2) onpick([c.row, c.col]);
		else onpick([c.col]);
	}
</script>

<div class="fb" data-atlas-fallback>
	{#if m.kind === 'unit' && m.unit}
		<div class="unit">
			{#if readSlice}
				<figure>
					<figcaption>{say('readsFrom', { t: m.read?.id ?? '' })}{m.read && m.read.t.shape.length >= 3 && m.read.t.shape[0] > 1 ? ` · ch ${m.read.channel}` : ''}</figcaption>
					{#if readSlice.rows > 1}
						<TensorGrid
							data={readSlice.data}
							rows={readSlice.rows}
							cols={readSlice.cols}
							label={say('readsFrom', { t: m.read?.id ?? '' })}
							maxCell={12}
							readout={false}
							highlights={readSlice.box ? [{ row: readSlice.box.y0, col: readSlice.box.x0, h: readSlice.box.y1 - readSlice.box.y0 + 1, w: readSlice.box.x1 - readSlice.box.x0 + 1 }] : []}
						/>
					{/if}
				</figure>
			{/if}
			{#if terms && terms.terms.length}
				<div class="terms" use:keyScroll>
					<table>
						<thead><tr><th>{say('inputCol')}</th><th aria-hidden="true"></th><th>{say('weightCol')}</th><th aria-hidden="true"></th><th>{say('productCol')}</th></tr></thead>
						<tbody>
							{#each terms.terms as tm, i (i)}
								<tr><td><Num value={tm.x} /></td><td class="op">×</td><td><Num value={tm.w} /></td><td class="op">=</td><td><Num value={tm.p} signed /></td></tr>
							{/each}
						</tbody>
					</table>
					{#if terms.more}<p class="muted small">{say('moreTerms', { n: terms.more })}</p>{/if}
					<p class="sum">
						{#if m.unit.values.s !== undefined}{say('sum')} <Num value={m.unit.values.s} />{/if}
						{#if m.unit.bias !== undefined} + {say('bias')} <Num value={m.unit.bias} />{/if}
						= <strong><Num value={m.unit.result} /></strong>
					</p>
				</div>
			{:else}
				<p class="sum">{m.unit.layer} [{m.unit.unit.join(', ')}] = <strong><Num value={m.unit.result} /></strong></p>
			{/if}
			{#if outSlice && outSlice.rows > 1}
				<figure>
					<figcaption>{say('output')} · {outLabel}</figcaption>
					<TensorGrid data={outSlice.data} rows={outSlice.rows} cols={outSlice.cols} label={outLabel} maxCell={12} readout={false} highlights={outSlice.hl} onselect={onpick ? pick : undefined} />
				</figure>
			{/if}
		</div>
	{:else if m.kind === 'tensor' && m.out}
		{#if probs}
			<ProbBars items={probs} k={probs.length} label={say('topClasses')} />
		{:else if onpick && outSlice}
			<TensorGrid data={outSlice.data} rows={outSlice.rows} cols={outSlice.cols} label={outLabel} maxCell={16} onselect={pick} />
		{:else}
			<TensorView tensor={m.out} label={outLabel} />
		{/if}
	{:else if m.kind === 'frame'}
		<p class="muted small">{say('phase')}: <strong>{m.phase}</strong>{#if m.step !== undefined}{' · '}step {m.step}{/if}</p>
		{#if m.scalars?.length}
			<dl class="scal">
				{#each m.scalars.slice(0, 12) as s (s.name)}
					<div><dt>{s.name}</dt><dd>{#if typeof s.value === 'number'}<Num value={s.value} />{:else}{s.value}{/if}</dd></div>
				{/each}
			</dl>
		{/if}
		{#each m.tensors ?? [] as x (x.name)}
			<figure class="ft">
				<figcaption>{x.name} · {x.t.shape.join('×')}</figcaption>
				<TensorView tensor={x.t} label={x.name} />
			</figure>
		{/each}
	{:else}
		<p class="muted">{say('noTensor')}</p>
	{/if}
</div>

<style>
	.fb {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.unit {
		display: flex;
		flex-wrap: wrap;
		gap: 14px;
		align-items: flex-start;
		min-width: 0;
	}
	.unit figure {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
		max-width: 100%;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--muted);
		font-family: var(--font-mono);
	}
	.terms {
		min-width: 0;
		overflow-x: auto;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
	}
	th {
		font-size: var(--fs-xs);
		color: var(--muted);
		font-weight: 500;
		text-align: right;
		padding: 0 6px;
	}
	td {
		text-align: right;
		padding: 1px 6px;
	}
	td.op {
		color: var(--muted);
		padding: 0 2px;
	}
	.sum {
		margin: 6px 0 0;
		font-size: var(--fs-sm);
	}
	.small {
		font-size: var(--fs-xs);
		margin: 2px 0 0;
	}
	.ft {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.scal {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 130px), 1fr));
		gap: 6px 14px;
		margin: 0;
	}
	.scal dt {
		font-size: var(--fs-xs);
		color: var(--muted);
		font-family: var(--font-mono);
	}
	.scal dd {
		margin: 0;
		font-weight: 600;
	}
</style>
