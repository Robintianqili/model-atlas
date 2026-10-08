<!-- vae-2d/pair (see meta.ts): your drawing → 2 numbers → the redrawing; in mode 'loss' also the loss of every pixel
     and the two parts of the training loss with this drawing's numbers. Nothing animates. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmtPct } from '../../../../atlas/core/format.ts';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import Pic from '../../parts/Pic.svelte';
	import { codeOf } from '../../vae.ts';
	import { pairPic } from './meta.ts';

	let P: StageProps = $props();
	const m = $derived(pairPic(P));
	const c = $derived(codeOf(P.rt));
	const worst = $derived.by(() => {
		const px = m?.loss?.px;
		if (!px) return null;
		let b = 0;
		for (let i = 1; i < px.length; i++) if (px[i] > px[b]) b = i;
		return { i: b, v: px[b], max: px[b] };
	});
	/** 1 for a grey pixel, 0.3 for another inked one, 0 for paper (the limits mode's maps) */
	const greyMap = (a: ArrayLike<number>) => Float32Array.from(a, (v) => (v > 0.2 && v < 0.8 ? 1 : v > 0.05 ? 0.3 : 0));
	const greys = $derived(m ? [{ d: greyMap(m.x), g: m.greyIn, cap: { en: 'grey in your drawing', zh: '你的画里的灰' } }, { d: greyMap(m.y), g: m.greyOut, cap: { en: 'grey in the redrawing', zh: '重画里的灰' } }] : []);
	$effect(() => {
		if (P.autoplay && m) P.onend();
	});
</script>

{#if !m}
	<Notice text={{ en: 'Draw something first: the redrawing comes from your drawing.', zh: '先画点什么：重画来自你的画。' }} />
{:else}
	<div class="pair">
		<div class="row">
			<figure><Pic data={m.x} frame="real" label={t({ en: 'Your drawing', zh: '你的画' })} /><figcaption>{t({ en: 'your drawing (784 numbers)', zh: '你的画（784 个数）' })}</figcaption></figure>
			<p class="z"><span class="arr">→</span> <span class="num">z = (<Num value={c.z[0]} />, <Num value={c.z[1]} />)</span> <span class="arr">→</span><br /><span class="muted">{t({ en: 'only 2 numbers', zh: '只有 2 个数' })}</span></p>
			<figure><Pic data={m.y} frame="generated" label={t({ en: 'The redrawing', zh: '重画' })} /><figcaption>{t({ en: 'the redrawing (784 numbers)', zh: '重画（784 个数）' })}</figcaption></figure>
			{#if m.mode === 'loss' && m.loss && worst}
				<figure><Pic data={m.loss.px} kind="heat" max={worst.max} label={t({ en: 'Loss of each pixel', zh: '每个像素的损失' })} /><figcaption>{t({ en: 'loss of each pixel (the more it stands out, the larger)', zh: '每个像素的损失（越显眼越大）' })}</figcaption></figure>
			{/if}
		</div>
		{#if m.mode === 'limits'}
			<div class="row">
				{#each greys as q, k (k)}
					<figure><Pic data={q.d} kind="heat" label={t(q.cap)} /><figcaption>{t(q.cap)}{Number.isFinite(q.g) ? `: ${fmtPct(q.g)}` : ''}</figcaption></figure>
				{/each}
				<p class="small muted key">{t({ en: 'Full colour: the grey pixels (between 0.2 and 0.8). Faint: the other inked pixels.', zh: '颜色最显眼的：灰色像素（0.2 到 0.8 之间）。淡的：其余有墨的像素。' })}</p>
			</div>
		{/if}
		{#if m.mode === 'loss' && m.loss && worst}
			<table class="small">
				<tbody>
					<tr><th scope="row">{t({ en: 'Redrawing error: binary cross-entropy of the 784 pixels, added up', zh: '重画误差：784 个像素的二元交叉熵之和' })}</th><td><Num value={m.loss.rec} /></td></tr>
					{#each [0, 1] as k (k)}
						<tr><th scope="row">{t({ en: `KL of number ${k + 1}: ½(μ² + σ² − log σ² − 1)`, zh: `第 ${k + 1} 个数的 KL：½(μ² + σ² − log σ² − 1)` })}</th><td><Num value={m.loss.klParts[k]} /></td></tr>
					{/each}
					<tr><th scope="row">{t({ en: 'KL of both numbers together', zh: '两个数的 KL 合计' })}</th><td><Num value={m.loss.kl} /></td></tr>
					<tr class="sum"><th scope="row">{t({ en: 'Total loss: redrawing error + KL', zh: '总损失：重画误差 + KL' })}</th><td><Num value={m.loss.total} /></td></tr>
				</tbody>
			</table>
			<p class="small muted">
				{t({ en: 'The worst pixel, row', zh: '损失最大的像素：第' })} {Math.floor(worst.i / 28)}{t({ en: ', column', zh: ' 行第' })} {worst.i % 28}{t({ en: ' (counting from 0): your ink x =', zh: ' 列（从 0 数起）：你的墨 x =' })} <Num value={m.x[worst.i]} />{t({ en: ', redrawn y =', zh: '，重画的墨 y =' })} <Num value={m.y[worst.i]} />{t({ en: ', its loss', zh: '，它的损失' })} <span class="nw">−[x·log y + (1 − x)·log(1 − y)] = <Num value={worst.v} /></span>{t({ en: '.', zh: '。' })}
			</p>
		{:else}
			<p class="small muted">
				{t({ en: `Pixels more than half ink: yours ${m.inkIn}, redrawn ${m.inkOut}. Grey pixels (between 0.2 and 0.8) among the inked ones: yours ${fmtPct(m.greyIn)}, redrawn ${fmtPct(m.greyOut)}.`, zh: `墨量超过一半的像素：你的 ${m.inkIn} 个，重画的 ${m.inkOut} 个。有墨像素中灰色的（0.2 到 0.8 之间）：你的 ${fmtPct(m.greyIn)}，重画的 ${fmtPct(m.greyOut)}。` })}
			</p>
		{/if}
	</div>
{/if}

<style>
	.pair { display: grid; gap: 10px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }
	figure { margin: 0; display: grid; gap: 4px; justify-items: center; }
	figcaption, .small { font-size: var(--fs-sm); }
	figcaption { color: var(--muted); max-width: 160px; text-align: center; }
	.z { text-align: center; font-size: var(--fs-sm); }
	.arr { color: var(--muted); }
	table { border-collapse: collapse; max-width: 560px; }
	th { text-align: left; font-weight: normal; padding: 3px 12px 3px 0; }
	td { text-align: right; white-space: nowrap; }
	tr.sum th, tr.sum td { font-weight: 700; border-top: 1px solid var(--rule); }
	p { margin: 0; }
	.nw { white-space: nowrap; }
	.key { max-width: 220px; }
</style>
