<!-- word2vec 'map': the word vectors flattened by PCA (see meta.ts). The ringed word is props.focus, joined to its nearest
     word; with props.list, every word's nearest word (by cosine similarity) is listed under the map. A step with
     play: 'view' animates the snapshots up to its frame (autoplay; Replay), then rests on that frame, which the
     narration describes (reduced motion: no animation). -->
<script lang="ts">
	import { untrack } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { mapPic } from './meta.ts';

	let P: StageProps = $props();
	let anim = $state(-1);
	let timer: ReturnType<typeof setInterval> | undefined;
	function replay() {
		const tp = P.at.tape;
		if (!tp) return;
		const snaps = Array.from({ length: P.at.index + 1 }, (_, i) => i).filter((i) => ['init', 'snapshot'].includes(tp.frame(i).phase));
		clearInterval(timer);
		let k = 0;
		anim = snaps[0] ?? -1;
		timer = setInterval(() => {
			if (++k < snaps.length) anim = snaps[k];
			else (clearInterval(timer), (anim = -1), P.onend());
		}, 60);
	}
	$effect(() => {
		if (P.autoplay && P.props.animate) untrack(replay);
		return () => clearInterval(timer);
	});
	const shown = $derived(anim >= 0 && P.at.tape ? { ...P, at: { index: anim, frame: P.at.tape.frame(anim), state: P.at.tape.state(anim), tape: P.at.tape } } : P);
	const m = $derived(mapPic(shown));
	let cw = $state(400);
	/** the drawing is as wide as its box (labels keep their size in a half-width compare) */
	const W = $derived(Math.max(260, cw));
	const H = $derived(Math.round(Math.min(380, Math.max(220, W * 0.62))));
	const PAD = 34;
	const box = $derived.by(() => {
		const xs = m ? m.final.map((p) => p[0]) : [0], ys = m ? m.final.map((p) => p[1]) : [0];
		const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
		return { x0, y0, sx: (W - 2 * PAD) / Math.max(1e-9, x1 - x0), sy: (H - 2 * PAD) / Math.max(1e-9, y1 - y0) };
	});
	const X = (v: number) => Math.max(4, Math.min(W - 4, PAD + (v - box.x0) * box.sx));
	const Y = (v: number) => Math.max(8, Math.min(H - 4, H - PAD - (v - box.y0) * box.sy));
	const caption = $derived.by(() => {
		if (!m) return '';
		const parts = [t({ en: `step ${fmtInt(m.step)}`, zh: `第 ${fmtInt(m.step)} 步` }), t({ en: `this flat map keeps ${fmtPct(m.share)} of the spread`, zh: `这张平面图保留了 ${fmtPct(m.share)} 的分布` })];
		if (Number.isFinite(m.loss)) parts.splice(1, 0, `${t({ en: 'average loss', zh: '平均损失' })} ${fmt3(m.loss)}`);
		if (m.focus >= 0) {
			const [a, b, c] = [m.words[m.focus], m.words[m.nn[m.focus].j], fmt3(m.nn[m.focus].cos)];
			parts.push(t({ en: `${a} → ${b} (cosine similarity ${c})`, zh: `${a} → ${b}（余弦相似度 ${c}）` }));
		}
		return parts.join(' · ');
	});
	/** label positions: each label moves up or down a line when it would overlap one placed before it */
	const labels = $derived.by(() => {
		const placed: { x: number; y: number; w: number }[] = [];
		return (m?.words ?? []).map((w, i) => {
			const wd = 6.8 * w.length, x0 = X(m!.xy[i][0]), x = x0 + 6 + wd > W ? x0 - 6 - wd : x0 + 6, y0 = Y(m!.xy[i][1]) + 4;
			const y = [0, -11, 11, -22, 22].map((d) => y0 + d).find((y) => !placed.some((b) => Math.abs(b.y - y) < 10 && x < b.x + b.w && b.x < x + wd)) ?? y0;
			placed.push({ x, y, w: wd });
			return { x, y };
		});
	});
</script>

<div class="map" style:min-height="340px" bind:clientWidth={cw}>
	{#if !m}
		<p class="muted">{t({ en: 'Training has not made any word vectors yet…', zh: '训练还没有产生词向量…' })}</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Word vectors on a map (PCA)', zh: '词向量图（PCA）' })}>
			{#if m.focus >= 0}{@const j = m.nn[m.focus].j}<line x1={X(m.xy[m.focus][0])} y1={Y(m.xy[m.focus][1])} x2={X(m.xy[j][0])} y2={Y(m.xy[j][1])} class="nn" />{/if}
			{#each m.words as w, i (i)}
				<g class:f={i === m.focus} class:moved={m.moved.has(i)}>
					<circle cx={X(m.xy[i][0])} cy={Y(m.xy[i][1])} r={i === m.focus ? 5.5 : 3.5} />
					<text x={labels[i].x} y={labels[i].y}>{w}</text>
				</g>
			{/each}
		</svg>
		<p class="small muted">{caption}{#if P.props.animate} <button type="button" class="btn ghost" onclick={replay}>{t({ en: 'Replay the training', zh: '重放训练过程' })}</button>{/if}</p>
		{#if m.list}
			<p class="list small" aria-label={t({ en: 'Nearest word of each word', zh: '每个词最近的词' })}>
				{#each m.words as w, i (i)}<span class:moved={m.moved.has(i)}>{w} → {m.words[m.nn[i].j]}</span>{/each}
			</p>
			{#if m.cmp}<p class="small muted">{t({ en: `Marked: the nearest word differs between the two runs (${m.moved.size} of ${m.words.length} words). Not marked: the same in both (${m.words.length - m.moved.size}).`, zh: `标记的词：两次运行中最近的词不一样（${m.words.length} 个词中的 ${m.moved.size} 个）。没有标记的：两次一样（${m.words.length - m.moved.size} 个）。` })}</p>{/if}
		{/if}
	{/if}
</div>

<style>
	.map { display: grid; gap: 4px; min-width: 0; align-content: start; } svg { width: 100%; height: auto; max-height: 400px; display: block; } .small { font-size: var(--fs-sm); }
	circle { fill: var(--seq-mid); } text { font-size: 12px; fill: var(--ink); } .f circle { fill: var(--accent); } .f text { font-weight: 700; }
	.nn { stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 3 2; } .moved text { fill: var(--div-neg); }
	p { margin: 0; } .btn { min-height: 28px; padding: 0 8px; font-size: var(--fs-xs); } .list { display: flex; flex-wrap: wrap; gap: 2px 10px; } .list span.moved { color: var(--div-neg-text); text-decoration: underline dotted; }
</style>
