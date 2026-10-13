<!--
  'number-formats/gpt': TinyGPT with its weight matrices cast to a format (meta.ts has the props): the setting, four
  numbers against fp32 (perplexity, KL divergence, top-1 agreement, bits per weight), the probe's next-token bars (fp32
  and after the cast), and optionally a table of every frame and buttons for every frame.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { Frame } from '../../../../atlas/types/process.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { PROBE_TEXT } from '../../evalset.ts';
	import { configText, probeOf, tokText } from './meta.ts';

	let { rt, source, at, props }: StageProps = $props();

	let chosen = $state<number | null>(null);
	let len = $state(0);
	$effect(() => {
		const tp = at.tape;
		len = tp?.length ?? 0;
		if (!props.controls || !tp) return;
		let alive = true;
		rt.ensure(source, 'last').then(() => {
			if (alive) len = at.tape?.length ?? 0;
		});
		return () => {
			alive = false;
		};
	});
	const idx = $derived(props.controls && chosen !== null && chosen < len ? chosen : at.index);
	const frame = $derived<Frame | undefined>(idx === at.index ? at.frame : at.tape?.frame(idx));
	const q = $derived(probeOf(idx === at.index ? at : { ...at, state: at.tape?.state(idx) }));
	const m = $derived(frame?.metrics);
	const frames = $derived(Array.from({ length: len }, (_, i) => ({ i, f: at.tape!.frame(i) })));
	const rows = $derived(props.table ? frames : []);
	const pmax = $derived(Math.max(1e-9, ...q.p0, ...q.p));
	const col = $derived.by(() => {
		void themeState.resolved;
		return [categorical(0), categorical(1)];
	});

	const L = {
		ppl: { en: 'Perplexity', zh: '困惑度' },
		kl: { en: 'KL divergence', zh: 'KL 散度' },
		agree: { en: 'Top-1 same as fp32', zh: 'top-1 与 fp32 相同' },
		size: { en: 'Model size', zh: '模型大小' },
		after: { en: 'Next token after', zh: '接在后面的下一个词元' },
		fp32: { en: 'fp32', zh: 'fp32' },
		quant: { en: 'after the cast', zh: '换格式后' },
		bpw: { en: 'Bits per weight', zh: '每个权重的位数' }
	};
	const mb = (b: number) => `${fmt3(b / 1e6)} MB`;
</script>

{#if m}
	<div class="gpt">
		{#if props.controls}
			<div class="pick" role="group" aria-label={t({ en: 'Choose a format for the weights', zh: '选择权重的格式' })}>
				{#each frames as r (r.i)}
					<button type="button" class="btn" aria-pressed={r.i === idx} onclick={() => (chosen = r.i)}>{t(configText(r.f))}</button>
				{/each}
			</div>
		{/if}
		<p class="cfg"><strong>{t(configText(frame))}</strong></p>
		<div class="nums">
			<div><span class="dt">{t(L.ppl)}</span><span class="dd num">{fmt3(m.ppl)} <span class="ref">(fp32 {fmt3(m.ppl0)})</span></span></div>
			<div><span class="dt">{t(L.kl)}</span><span class="dd num">{fmt3(m.kl)}</span></div>
			<div><span class="dt">{t(L.agree)}</span><span class="dd num">{fmtPct(m.agree)}</span></div>
			<div><span class="dt">{t(L.bpw)}</span><span class="dd num">{fmt3(m.bpw)} <span class="ref">({t(L.size)} {mb(m.bytes)})</span></span></div>
		</div>
		{#if props.probe !== false}
			<figure class="probe">
				<figcaption>{t(L.after)} “{PROBE_TEXT}”</figcaption>
				<div class="legend"><span class="key"><span class="sw" style:background={col[0]}></span>{t(L.fp32)}</span><span class="key"><span class="sw" style:background={col[1]}></span>{t(L.quant)}</span></div>
				{#each q.top as id, i (i)}
					<div class="tok">
						<span class="tt">{tokText(id)}</span>
						<div class="bars">
							<div class="bar"><span style:width="{(q.p0[i] / pmax) * 100}%" style:background={col[0]}></span><span class="num v">{fmtPct(q.p0[i])}</span></div>
							<div class="bar"><span style:width="{(q.p[i] / pmax) * 100}%" style:background={col[1]}></span><span class="num v">{fmtPct(q.p[i])}</span></div>
						</div>
					</div>
				{/each}
			<p class="q1">{#if m.probeTop === m.probeTop0}{t({ en: 'Both models put the same token first.', zh: '两个模型排第一的是同一个词元。' })}{:else}{t({ en: 'After the cast the model puts first:', zh: '换格式后，模型排第一的变成了：' })} <span class="tt">{tokText(m.probeTop)}</span>{/if}</p>
			</figure>
		{/if}
		{#if rows.length}
			<div class="tbl" use:keyScroll>
				<table>
					<thead><tr><th scope="col"></th><th scope="col">{t(L.ppl)}</th><th scope="col">{t(L.kl)}</th><th scope="col">{t(L.agree)}</th><th scope="col">{t(L.bpw)}</th></tr></thead>
					<tbody>
						{#each rows as r (r.i)}
							<tr class:cur={r.i === idx}>
								<th scope="row">{t(configText(r.f))}</th>
								<td class="num">{fmt3(r.f.metrics?.ppl)}</td>
								<td class="num">{fmt3(r.f.metrics?.kl)}</td>
								<td class="num">{fmtPct(r.f.metrics?.agree ?? NaN)}</td>
								<td class="num">{fmt3(r.f.metrics?.bpw)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>
{/if}

<style>
	.gpt {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.pick {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.pick .btn {
		font-size: var(--fs-xs);
	}
	.pick .btn[aria-pressed='true'] {
		outline: 2px solid var(--accent);
		outline-offset: -2px;
		font-weight: 600;
	}
	.cfg {
		margin: 0;
	}
	.nums {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
		gap: 6px 12px;
		margin: 0;
	}
	.nums div {
		min-width: 0;
	}
	.dt {
		display: block;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.dd {
		display: block;
		margin: 0;
		font-size: var(--fs-md);
	}
	.ref {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	figure {
		margin: 0;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		margin-bottom: 4px;
	}
	.legend {
		display: flex;
		gap: 12px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin-bottom: 4px;
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.sw {
		width: 12px;
		height: 12px;
		border-radius: 2px;
		display: inline-block;
	}
	.tok {
		display: grid;
		grid-template-columns: 6rem minmax(0, 1fr) 9rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
		padding: 2px 0;
	}
	.tt {
		font-family: var(--font-mono);
		white-space: pre;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.bars {
		display: grid;
		gap: 2px;
		min-width: 0;
	}
	.bar {
		display: flex;
		align-items: center;
		gap: 4px;
		height: 12px;
		min-width: 0;
	}
	.bar span:first-child {
		display: block;
		height: 10px;
		border-radius: 2px;
		min-width: 1px;
	}
	.v {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.q1 {
		margin: 6px 0 0;
		font-size: var(--fs-sm);
	}
	.tbl {
		overflow-x: auto;
		min-width: 0;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		width: 100%;
	}
	th,
	td {
		padding: 3px 8px;
		text-align: right;
		border-bottom: 1px solid var(--rule);
		white-space: nowrap;
	}
	th[scope='row'] {
		text-align: left;
		font-weight: 500;
		white-space: normal;
		min-width: 9rem;
	}
	thead th {
		color: var(--ink-2);
		font-weight: 500;
		white-space: normal;
		vertical-align: bottom;
	}
	tr.cur td,
	tr.cur th {
		background: var(--surface-2);
		font-weight: 600;
	}
	@media (max-width: 520px) {
		.tok {
			grid-template-columns: 4.5rem minmax(0, 1fr);
		}
		}
</style>
