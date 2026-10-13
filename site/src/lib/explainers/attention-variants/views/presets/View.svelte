<!--
  'attention-variants/presets' (meta.ts): horizontal bars from 0 on one KiB axis, one per configuration, colored by its
  variant (cat-0 MHA, cat-1 GQA, cat-2 MQA, cat-3 MLA; legend below), the value printed after the name.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { bytesText, KIND_SHORT, kindOf, perToken } from '../../data.ts';
	import { configAt } from '../heads/meta.ts';
	import { barsOf } from './meta.ts';

	let P: StageProps = $props();
	const bars = $derived(barsOf(configAt(P)));
	$effect(() => void (P.autoplay && P.onend()));
	const [W, ML, MR, RH] = [360, 4, 8, 30];
	const KIB = 1024;
	const max = $derived.by(() => {
		const m = Math.max(...bars.map((b) => perToken(b.c))) / KIB;
		const p = 10 ** Math.floor(Math.log10(m / 4));
		const step = [1, 2, 5, 10].map((x) => x * p).find((s) => s >= m / 4) ?? 10 * p;
		return { top: Math.ceil(m / step) * step, step };
	});
	const x = (b: number) => ML + (b / KIB / max.top) * (W - ML - MR);
	const H = $derived(bars.length * RH + 30);
	const ticks = $derived(Array.from({ length: Math.round(max.top / max.step) + 1 }, (_, i) => i * max.step));
</script>

<figure class="wrap">
	<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'KV cache per token, in KiB', zh: '每个词元的 KV 缓存（KiB）' })}>
		{#each ticks as v (v)}
			<line class="grid" x1={x(v * KIB)} x2={x(v * KIB)} y1="0" y2={bars.length * RH} />
			<text class="tick" x={x(v * KIB)} y={bars.length * RH + 12} text-anchor={v === 0 ? 'start' : v === max.top ? 'end' : 'middle'}>{v.toLocaleString('en-US')}</text>
		{/each}
		{#each bars as b, i (b.id)}
			<text class="name" class:mine={b.mine} x={ML} y={i * RH + 11}>{t(b.label)} · {KIND_SHORT[kindOf(b.c)]} · <tspan class="num">{bytesText(perToken(b.c))}</tspan></text>
			<rect class="bar k{kindOf(b.c)}" class:mine={b.mine} x={x(0)} y={i * RH + 15} width={Math.max(1.5, x(perToken(b.c)) - x(0))} height="10" />
		{/each}
		<text class="axis" x={W / 2} y={H - 2} text-anchor="middle">{t({ en: 'KV cache per token (KiB = 1,024 bytes)', zh: '每个词元的 KV 缓存（KiB = 1,024 字节）' })}</text>
	</svg>
	<figcaption>
		{#each [0, 1, 2, 3] as k (k)}<span class="sw k{k}"></span>{KIND_SHORT[k]}{/each}
		<span class="note">{t({ en: 'TinyGPT in fp32 (4 bytes), the others in fp16 / bf16 (2 bytes); the outlined bar is yours.', zh: 'TinyGPT 用 fp32（4 字节），其余用 fp16 / bf16（2 字节）；带框的柱是你的配置。' })}</span>
	</figcaption>
</figure>

<style>
	.wrap {
		margin: 0;
		max-width: 560px;
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
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.axis,
	.name {
		font-size: 11px;
		fill: var(--ink-2);
	}
	.name.mine {
		font-weight: 600;
		fill: var(--ink);
	}
	.num {
		font-family: var(--font-mono);
	}
	.bar.mine {
		stroke: var(--ink);
		stroke-width: 1.5;
	}
	.k0 {
		fill: var(--cat-0);
		background: var(--cat-0);
	}
	.k1 {
		fill: var(--cat-1);
		background: var(--cat-1);
	}
	.k2 {
		fill: var(--cat-2);
		background: var(--cat-2);
	}
	.k3 {
		fill: var(--cat-3);
		background: var(--cat-3);
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 12px;
	}
	.note {
		color: var(--muted);
	}
</style>
