<!--
  'positional-encodings/rope' (meta.ts): a plane with the axes dim 0 and dim 1 of head 0; the query (categorical colour 0)
  and the key (colour 1) as arrows, dashed before RoPE and solid after it, with the turned angle as an arc; under it the
  dot products before and after.
-->
<script lang="ts">
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { ropeOf } from './meta.ts';

	let P: StageProps = $props();
	const o = $derived(ropeOf(P));
	const [S, C, R] = [280, 140, 110];
	const r = $derived(Math.max(1e-9, Math.hypot(o.q[0], o.q[1]), Math.hypot(o.k[0], o.k[1])));
	const X = (v: number) => C + (v / r) * R;
	const Y = (v: number) => C - (v / r) * R;
	const cq = $derived(categorical(0, themeState.resolved));
	const ck = $derived(categorical(1, themeState.resolved));
	/** an arc of radius rr from the angle of a to the angle of b, turning by `turn` radians (wraps shown as one turn mod 2π) */
	function arc(a: number[], turn: number, rr: number) {
		const a0 = Math.atan2(a[1], a[0]);
		const d = ((turn % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
		const a1 = a0 + d;
		const p = (ang: number) => `${(C + rr * Math.cos(ang)).toFixed(1)} ${(C - rr * Math.sin(ang)).toFixed(1)}`;
		return `M ${p(a0)} A ${rr} ${rr} 0 ${d > Math.PI ? 1 : 0} 0 ${p(a1)}`;
	}
	$effect(() => void (P.autoplay && P.onend()));
</script>

{#if o.ok}
	<figure class="rope">
		<svg viewBox="0 0 {S} {S}" role="img" aria-label={t({ en: 'Pair 0 of the query and the key, before and after RoPE', zh: '查询和键的第 0 对，RoPE 前后' })}>
			<defs>
				<marker id="pe-arr-q" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill={cq} /></marker>
				<marker id="pe-arr-k" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill={ck} /></marker>
			</defs>
			<circle class="ring" cx={C} cy={C} r={R} />
			<line class="axis" x1={C - R - 12} x2={C + R + 12} y1={C} y2={C} />
			<line class="axis" y1={C - R - 12} y2={C + R + 12} x1={C} x2={C} />
			<text class="ax" x={C + R + 10} y={C - 6} text-anchor="end">{t({ en: 'dim 0', zh: '维 0' })}</text>
			<text class="ax" x={C + 6} y={C - R - 2}>{t({ en: 'dim 1', zh: '维 1' })}</text>
			<path d={arc(o.q, o.m, 34)} fill="none" stroke={cq} stroke-width="1.5" />
			<path d={arc(o.k, o.n, 24)} fill="none" stroke={ck} stroke-width="1.5" />
			<line x1={C} y1={C} x2={X(o.q[0])} y2={Y(o.q[1])} stroke={cq} stroke-width="2" stroke-dasharray="4 3" />
			<line x1={C} y1={C} x2={X(o.k[0])} y2={Y(o.k[1])} stroke={ck} stroke-width="2" stroke-dasharray="4 3" />
			<line x1={C} y1={C} x2={X(o.qr[0])} y2={Y(o.qr[1])} stroke={cq} stroke-width="3" marker-end="url(#pe-arr-q)" />
			<line x1={C} y1={C} x2={X(o.kr[0])} y2={Y(o.kr[1])} stroke={ck} stroke-width="3" marker-end="url(#pe-arr-k)" />
			<text class="lbl" x={X(o.qr[0] * 1.12)} y={Y(o.qr[1] * 1.12)} text-anchor="middle">q@{o.m}</text>
			<text class="lbl" x={X(o.kr[0] * 1.12)} y={Y(o.kr[1] * 1.12)} text-anchor="middle">k@{o.n}</text>
		</svg>
		<figcaption>
			<span><i class="sw" style="background:{cq}"></i>{t({ en: `query, turned by ${o.m} rad`, zh: `查询，转了 ${o.m} 弧度` })}</span>
			<span><i class="sw" style="background:{ck}"></i>{t({ en: `key, turned by ${o.n} rad`, zh: `键，转了 ${o.n} 弧度` })}</span>
			<span class="muted">{t({ en: 'dashed: before turning', zh: '虚线：旋转之前' })}</span>
			<span>{t({ en: 'pair 0 dot product', zh: '第 0 对的点积' })}{colon()}<b class="num">{fmt3(o.pairRaw)} → {fmt3(o.pair)}</b></span>
			<span>{t({ en: 'whole head (9 pairs + 1 dim)', zh: '整个头（9 对 + 1 维）' })}{colon()}<b class="num">{fmt3(o.raw)} → {fmt3(o.score)}</b></span>
		</figcaption>
	</figure>
{/if}

<style>
	.rope { margin: 0; display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: 12px; align-items: center; min-width: 0; }
	svg { width: 100%; height: auto; display: block; }
	.ring { fill: none; stroke: var(--rule); }
	.axis { stroke: var(--rule-strong); }
	.ax, .lbl { font-size: 11px; fill: var(--ink-2); font-family: var(--font-mono); }
	.lbl { fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
	figcaption { display: grid; gap: 6px; font-size: var(--fs-sm); }
	.sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
	.muted { color: var(--muted); }
	@media (max-width: 560px) { .rope { grid-template-columns: minmax(0, 1fr); } }
</style>
