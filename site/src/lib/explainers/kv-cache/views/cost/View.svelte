<!--
  'kv-cache/cost' (meta.ts has the props): a line chart of the multiply-adds of every pass, without the cache and with it
  (or the running totals), the frame's pass marked; optionally the cache's size, worked out for TinyGPT and two large
  models with the same formula.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { fmtInt } from '$lib/kit/format';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtBytes } from '../../../../atlas/core/format.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { memRows } from './meta.ts';

	let { rt, source, at, props }: StageProps = $props();

	let cum = $state(false);
	let len = $state(0);
	$effect(() => {
		cum = !!props.cumulative;
	});
	$effect(() => {
		const tp = at.tape;
		len = tp?.length ?? 0;
		if (!tp) return;
		let alive = true;
		rt.ensure(source, 'last').then(() => {
			if (alive) len = at.tape?.length ?? 0;
		});
		return () => {
			alive = false;
		};
	});
	const m = $derived(at.frame?.metrics);
	const pts = $derived(Array.from({ length: len }, (_, i) => at.tape!.frame(i).metrics ?? {}));
	const ys = $derived(pts.map((p) => (cum ? [p.cumFull, p.cumCache] : [p.macsFull, p.macsCache])));
	const ymax = $derived(Math.max(1, ...ys.flat().filter((v) => Number.isFinite(v))));
	const col = $derived.by(() => {
		void themeState.resolved;
		return [categorical(1), categorical(0)];
	});
	const [W, Hh] = [600, 210];
	const pad = { l: 54, r: 14, t: 12, b: 22 };
	const xOf = (k: number) => pad.l + (len > 1 ? k / (len - 1) : 0) * (W - pad.l - pad.r);
	const yOf = (v: number) => pad.t + (1 - v / ymax) * (Hh - pad.t - pad.b);
	const path = (s: number) => ys.map((y, k) => `${k ? 'L' : 'M'}${xOf(k).toFixed(1)},${yOf(y[s]).toFixed(1)}`).join('');
	const ticks = $derived.by(() => {
		const raw = ymax / 4;
		const p = 10 ** Math.floor(Math.log10(raw));
		const step = [1, 2, 5, 10].map((s) => s * p).find((s) => s >= raw) ?? raw;
		return Array.from({ length: Math.floor(ymax / step) + 1 }, (_, i) => i * step);
	});
	const mio = (v: number) => (v === 0 ? '0' : `${Number((v / 1e6).toPrecision(3))}M`);
	const rows = $derived(props.memory ? memRows(rt, m) : []);
	const names = [
		{ en: 'without the cache', zh: '不用缓存' },
		{ en: 'with the KV cache', zh: '用 KV 缓存' }
	];
</script>

{#if m}
	<div class="cost">
		<div class="seg" role="group" aria-label={t({ en: 'What the chart counts', zh: '图里统计的是什么' })}>
			<button type="button" class="btn" aria-pressed={!cum} onclick={() => (cum = false)}>{t({ en: 'Per pass', zh: '每一遍' })}</button>
			<button type="button" class="btn" aria-pressed={cum} onclick={() => (cum = true)}>{t({ en: 'All passes so far', zh: '到目前为止的总和' })}</button>
		</div>
		<figure>
			<svg viewBox="0 0 {W} {Hh}" role="img" aria-label={t({ en: `Multiply-adds ${cum ? 'so far' : 'per pass'}, without and with the cache, over ${len} passes`, zh: `${len} 遍中${cum ? '累计' : '每一遍'}的乘加次数，不用缓存与用缓存` })}>
				{#each ticks as v (v)}
					<line class="grid" x1={pad.l} x2={W - pad.r} y1={yOf(v)} y2={yOf(v)} />
					<text class="tick" x={pad.l - 6} y={yOf(v) + 3} text-anchor="end">{mio(v)}</text>
				{/each}
				<text class="tick" x={pad.l} y={Hh - pad.b + 14} text-anchor="middle">0</text>
				<text class="tick" x={W - pad.r} y={Hh - pad.b + 14} text-anchor="end">{len - 1}</text>
				{#each [0, 1] as s (s)}<path class="line" d={path(s)} style:stroke={col[s]} />{/each}
				{#if Number.isFinite(m.k)}
					<line class="cursor" x1={xOf(m.k)} x2={xOf(m.k)} y1={pad.t} y2={Hh - pad.b} />
					{#each [0, 1] as s (s)}<circle cx={xOf(m.k)} cy={yOf(cum ? (s ? m.cumCache : m.cumFull) : s ? m.macsCache : m.macsFull)} r="4" style:fill={col[s]} />{/each}
				{/if}
			</svg>
			<figcaption>{t({ en: 'Across: the pass (one new token each), counting from 0. Up: multiply-adds (M = million).', zh: '横轴：第几遍（每遍一个新词元），从 0 数起。纵轴：乘加次数（M = 百万）。' })}</figcaption>
		</figure>
		<div class="legend">
			{#each [0, 1] as s (s)}
				<span class="key"><span class="sw" style:background={col[s]}></span>{t(names[s])}: <span class="num">{fmtInt(cum ? (s ? m.cumCache : m.cumFull) : s ? m.macsCache : m.macsFull)}</span></span>
			{/each}
			<span class="key">{t({ en: `at pass ${m.k}`, zh: `第 ${m.k} 遍` })}</span>
		</div>
		{#if rows.length}
			<div class="mem">
				<p class="f">{t({ en: 'cache = layers × heads × tokens × head size × 2 (K and V) × bytes', zh: '缓存 = 层数 × 头数 × 词元数 × 每头维度 × 2（K 和 V）× 字节数' })}</p>
				<div class="tbl" use:keyScroll>
					<table>
						<tbody>
							{#each rows as r, i (i)}
								<tr class:live={r.live}>
									<th scope="row">{t(r.label)}</th>
									<td class="num">{r.L} × {r.H} × {fmtInt(r.n)} × {r.dh} × 2 × {r.bytes}</td>
									<td class="num big">{fmtBytes(r.total)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/if}
	</div>
{/if}

<style>
	.cost {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.seg {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.btn {
		min-height: 32px;
		padding: 2px 10px;
		font-size: var(--fs-xs);
	}
	.btn[aria-pressed='true'] {
		border-color: var(--accent);
		background: var(--accent-soft);
		font-weight: 600;
	}
	figure {
		margin: 0;
		min-width: 0;
		max-width: 720px;
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.grid {
		stroke: var(--rule);
	}
	.tick {
		font-size: 10px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin-top: 4px;
	}
	.line {
		fill: none;
		stroke-width: 2.5;
	}
	.cursor {
		stroke: var(--ink-2);
		stroke-dasharray: 3 3;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
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
		flex: none;
	}
	.mem {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.f {
		margin: 0;
		font-size: var(--fs-sm);
		font-family: var(--font-mono);
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
		min-width: 8rem;
	}
	tr.live th,
	tr.live td {
		background: var(--surface-2);
	}
	.big {
		font-weight: 600;
	}
	/* a phone: the formula wraps so the size column stays in view */
	@media (max-width: 560px) {
		td.num:not(.big) {
			white-space: normal;
		}
		th[scope='row'] {
			min-width: 5.5rem;
		}
	}
</style>
