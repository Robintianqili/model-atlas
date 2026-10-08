<!-- tiny-unet-seg/mask (see meta.ts): the picture with a class map over it, an opacity slider, and per mode: the shares of
     one pixel and its windows ('pred'), two class maps side by side ('compare'), or IoU per class ('iou'). Tap or use the
     arrow keys on the picture to probe another pixel. Nothing animates. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmtPct } from '../../../../atlas/core/format.ts';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import Seg from '../../parts/Seg.svelte';
	import { CLASSES, S } from '../../scenes.ts';
	import { PATHS, SURE, T, classMap, leastSure, probe, scoreOf, unsure, windowOf } from '../../seg.ts';
	import { LIMITS, modeOf } from './meta.ts';

	let P: StageProps = $props();
	const mode = $derived(modeOf(P));
	const s = $derived(scoreOf(P.rt));
	const ns = $derived(mode === 'compare' ? scoreOf(P.rt, 'noskip') : null);
	const img = $derived(T(P.rt, 'input')?.data);
	let alpha = $state(0.55);
	let picked = $state<[number, number] | null>(null);
	let seen: unknown = null;
	$effect(() => {
		if (P.rt.input !== seen) (seen = P.rt.input), (picked = null);
	});
	const sel = $derived(picked ?? (P.props.probe === 'least' ? leastSure(P.rt) : probe(P.rt)));
	const pr = $derived(T(P.rt, 'probs'));
	const shares = $derived(pr ? [0, 1, 2, 3].map((k) => pr.data[k * S * S + sel[0] * S + sel[1]]) : []);
	const boxes = $derived(P.props.rf ? PATHS.map((p) => windowOf(P.rt.sources.fwd?.model ?? P.rt.model, sel[0], sel[1], p)) : []);
	const wrong = $derived((mode === 'iou' || mode === 'limits') && s ? Uint8Array.from(s.map.k, (k, i) => +(k !== s.truth[i])) : null);
	const PATH = [{ en: 'through skip 1 (solid)', zh: '经过跳跃连接 1（实线）' }, { en: 'through skip 2 (dashed)', zh: '经过跳跃连接 2（虚线）' }, { en: 'through the bottleneck (dotted)', zh: '经过瓶颈（点线）' }];
	const counts = $derived(s ? CLASSES.map((c, k) => ({ c, n: (mode === 'truth' ? s.truth : s.map.k).filter((v) => v === k).length })).filter((q) => q.n) : []);
	const at = $derived({ en: `Pixel row ${sel[0]}, column ${sel[1]} (counting from 0)`, zh: `第 ${sel[0]} 行第 ${sel[1]} 列的像素（从 0 数起）` });
	$effect(() => {
		if (P.autoplay && s) P.onend();
	});
</script>

{#if !s || !img}
	<Notice />
{:else}
	<div class="mask" class:full={mode === 'compare'}>
		<div class="pics" class:two={mode === 'compare'}>
			{#if mode === 'compare' && ns}
				{#each [{ q: ns, src: 'noskip', c: { en: 'without skip connections', zh: '去掉跳跃连接' } }, { q: s, src: 'fwd', c: { en: 'U-Net (with skips)', zh: 'U-Net（有跳跃连接）' } }] as v (v.src)}
					{@const u = unsure(P.rt, v.src)}
					<figure>
						<Seg {img} probs={T(P.rt, 'probs', v.src)?.data} {alpha} size={300} label={t(v.c)} />
						<span class="unsure"><TensorGrid data={Float32Array.from(classMap(P.rt, v.src)?.p ?? [], (p) => 1 - p)} rows={S} cols={S} kind="sequential" domain={[0, 0.5]} interactive={false} readout={false} gridLines={false} labels={false} maxCell={9} label={t({ en: 'How unsure each pixel is: 1 − its top share', zh: '每个像素有多没把握：1 − 最大份额' })} /></span>
						<figcaption>{t(v.c)}<br /><span class="muted">{t({ en: `less than ${fmtPct(SURE, 0)} sure: ${u} pixels; border pixels right: ${v.q.edge ? Math.round(v.q.edgeAcc * v.q.edge) : 0} of ${v.q.edge}`, zh: `把握不到 ${fmtPct(SURE, 0)}：${u} 个像素；边界像素答对：${v.q.edge} 个中 ${v.q.edge ? Math.round(v.q.edgeAcc * v.q.edge) : 0} 个` })}</span></figcaption>
					</figure>
				{/each}
				<p class="small muted wide">{t({ en: 'Top: each pixel mixes the class colours by its shares. Below: how unsure each pixel is, 1 − its top share (scale 0 to 0.5); the more a pixel stands out, the less sure.', zh: '上：每个像素按份额混合各类别的颜色。下：每个像素有多没把握，即 1 − 最大份额（刻度 0 到 0.5）；越显眼越没把握。' })}</p>
			{:else}
				<figure>
					<Seg {img} cls={mode === 'truth' ? s.truth : s.map.k} {alpha} {wrong} sel={mode === 'pred' ? sel : null} {boxes} label={t(mode === 'truth' ? { en: 'Your picture with your classes', zh: '你的图和你给的类别' } : { en: 'Your picture with the model’s classes', zh: '你的图和模型给的类别' })} onpick={mode === 'pred' ? (rc) => (picked = rc) : undefined} />
					<figcaption class="muted">{t(mode === 'truth' ? { en: 'the classes you gave (the truth)', zh: '你给的类别（真实答案）' } : mode === 'pred' ? { en: 'the model’s answer: tap a pixel', zh: '模型的答案：点一个像素' } : { en: 'the model’s answer', zh: '模型的答案' })}{#if mode === 'truth' || mode === 'pred'}<br /><span class="num">{counts.map((q) => `${q.n} ${t(q.c)}`).join(' · ')}</span>{/if}</figcaption>
				</figure>
			{/if}
		</div>
		<div class="side">
			<p class="legend">{#each CLASSES as c, k (k)}<span><i class="sw k{k}"></i>{t(c)}</span>{/each}</p>
			<label class="small">{t({ en: 'Overlay opacity', zh: '着色不透明度' })} <input type="range" min="0" max="1" step="0.05" bind:value={alpha} /></label>
			{#if mode === 'pred'}
				<p class="small" aria-live="polite">{t(at)}: {t({ en: 'your class', zh: '你给的类别' })} <b>{t(CLASSES[s.truth[sel[0] * S + sel[1]]])}</b></p>
				<table class="small">
					<caption class="muted">{t({ en: 'its 4 shares after the softmax', zh: 'softmax 之后它的 4 个份额' })}</caption>
					<tbody>
						{#each shares as v, k (k)}<tr class:top={k === s.map.k[sel[0] * S + sel[1]]}><th scope="row"><i class="sw k{k}"></i>{t(CLASSES[k])}</th><td class="bar"><span style:width="{Math.max(0, Math.min(1, v)) * 100}%"></span></td><td><Num value={v} /></td></tr>{/each}
					</tbody>
				</table>
				{#if boxes.length}
					<ul class="small paths">
						{#each boxes as b, j (j)}{#if b}<li>{t(PATH[j])}: <span class="num">{b.x1 - b.x0 + 1}×{b.y1 - b.y0 + 1}</span></li>{/if}{/each}
					</ul>
				{/if}
			{:else if mode === 'iou'}
				<table class="small">
					<thead><tr><th scope="col">{t({ en: 'class', zh: '类别' })}</th><th scope="col">{t({ en: 'in both', zh: '两者都是' })}</th><th scope="col">{t({ en: 'in either', zh: '至少一方是' })}</th><th scope="col">IoU</th></tr></thead>
					<tbody>
						{#each s.per as c (c.k)}<tr><th scope="row"><i class="sw k{c.k}"></i>{t(CLASSES[c.k])}</th><td class="num">{c.inter}</td><td class="num">{c.union}</td><td class="num">{fmtPct(c.iou)}</td></tr>{/each}
						<tr class="sum"><th scope="row" colspan="3">{t({ en: 'mean IoU', zh: '平均 IoU' })}</th><td class="num">{fmtPct(s.miou)}</td></tr>
					</tbody>
				</table>
				<p class="small muted">{t(s.wrong ? { en: `A slash marks each of the ${s.wrong} pixels whose class differs from yours.`, zh: `斜线标出类别和你给的不同的 ${s.wrong} 个像素。` } : { en: 'No pixel’s class differs from yours.', zh: '没有哪个像素的类别和你给的不同。' })}</p>
			{:else if mode === 'limits'}
				<table class="small">
					<caption class="muted">{t({ en: 'mean IoU on scenes it never saw (measured)', zh: '在没见过的场景上的平均 IoU（实测）' })}</caption>
					<tbody>
						{#each LIMITS as l (l.fact)}{@const v = P.rt.num(`fact:${l.fact}`)}<tr><th scope="row">{t(l)}</th><td class="bar"><span style:width="{Math.max(0, Math.min(1, v)) * 100}%"></span></td><td class="num">{fmtPct(v)}</td></tr>{/each}
					</tbody>
				</table>
				<p class="small muted">{t({ en: `On your picture: mean IoU ${fmtPct(s.miou)}.`, zh: `在你的图上：平均 IoU ${fmtPct(s.miou)}。` })}</p>
			{/if}
		</div>
	</div>
{/if}

<style>
	.mask { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px 18px; align-items: start; }
	@media (max-width: 640px) { .mask { grid-template-columns: minmax(0, 1fr); } }
	.mask.full { grid-template-columns: minmax(0, 1fr); }
	.unsure { display: block; width: 100%; max-width: 300px; }
	.pics { display: grid; gap: 10px; }
	.pics.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
	.wide { grid-column: 1 / -1; }
	figure { margin: 0; display: grid; gap: 4px; justify-items: center; }
	figcaption, .small { font-size: var(--fs-sm); }
	figcaption { text-align: center; }
	.side { display: grid; gap: 8px; min-width: 0; }
	.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 0; font-size: var(--fs-sm); }
	.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; border: 1px solid var(--rule); vertical-align: -1px; }
	.k0 { background: var(--surface-2); } .k1 { background: var(--cat-0); } .k2 { background: var(--cat-1); } .k3 { background: var(--cat-2); }
	table { border-collapse: collapse; }
	caption { text-align: left; }
	th { text-align: left; font-weight: normal; padding: 2px 10px 2px 0; }
	td { text-align: right; padding: 2px 0 2px 8px; }
	tr.top th, tr.top td, tr.sum th, tr.sum td { font-weight: 700; }
	tr.sum th, tr.sum td { border-top: 1px solid var(--rule); }
	.bar { width: 90px; } .bar span { display: block; height: 8px; background: var(--accent); border-radius: 2px; }
	.paths { margin: 0; padding-left: 1.2em; }
	p { margin: 0; }
	input[type='range'] { width: 140px; vertical-align: middle; }
</style>
