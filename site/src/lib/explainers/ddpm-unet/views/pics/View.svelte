<!-- ddpm-unet/pics (see meta.ts): pictures of the forward process, the network's noise guess, one denoising step, a
     sampling run, the seeds' gallery and the U-Net's stages. Nothing animates by itself: the lesson's scrubber moves the frame. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps, Tensor } from '../../../../atlas/types/index.ts';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import Pic from '../../parts/Pic.svelte';
	import { PROBE, SCHED, SIDE, T, seedOf, withExtra } from '../../ddpm.ts';
	import { sample } from '../../process.ts';
	import { filmOf, galleryOf, modeOf, stepOf, traceOf } from './meta.ts';

	let P: StageProps = $props();
	const mode = $derived(modeOf(P));
	const tc = $derived(traceOf(P));
	const s = $derived(stepOf(P));
	const f = $derived(filmOf(P));
	const g = $derived(mode === 'gallery' || mode === 'pick' || mode === 'limits' ? galleryOf(P) : []);
	const frameM = $derived(P.at?.frame?.metrics ?? {});
	const seed = $derived(seedOf(P.rt.input));
	const chosen = $derived(g.find((q) => q.seed === seed));
	const set = (extra: { t?: number; seed?: number }) => (P.setInput ?? P.rt.setInput)(withExtra(P.rt.input, extra));
	// the weights of the noise schedule, √ᾱ_t (signal) and √(1 − ᾱ_t) (noise), as polylines over t = 0 … T
	const curve = (fn: (ab: number) => number) => [1, ...SCHED.abar].map((ab, i) => `${(i / T) * 200},${60 - fn(ab) * 56}`).join(' ');
	const stages = $derived.by(() => {
		const tz = (id: string) => P.rt.sources.fwd?.trace?.t[id];
		const mean = (x: Tensor | undefined) => {
			if (!x) return null;
			const [C, H, W] = x.shape, m = new Float32Array(H * W);
			for (let c = 0; c < C; c++) for (let i = 0; i < H * W; i++) m[i] += Math.abs(x.data[c * H * W + i]) / C;
			const top = Math.max(...m) || 1;
			return m.map((v) => v / top);
		};
		return ['xt', 'in.silu', 'down.silu', 'mid.silu1', 'up.silu', 'top.silu', 'out'].map((id, k) => ({ id, x: tz(id), m: k % 6 ? mean(tz(id)) : tz(id)?.data, label: P.rt.spec?.layers.find((l) => l.id === id)?.label }));
	});
	$effect(() => {
		if (mode === 'gallery' || mode === 'pick' || mode === 'limits') void P.rt.ensure('seeds', 'last');
	});
	$effect(() => {
		if (P.autoplay) P.onend();
	});
	const cap = { x0: { en: 'your drawing x₀', zh: '你的画 x₀' }, eps: { en: 'noise ε', zh: '噪声 ε' }, hat: { en: 'the network’s guess ε̂', zh: '网络的猜测 ε̂' } };
</script>

{#snippet seeds()}
	<div class="grid" role="group" aria-label={t({ en: 'Seeds: pick one', zh: '种子：选一个' })}>
		{#each g as q (q.seed)}<button type="button" class="seed" class:on={q.seed === seed} aria-pressed={q.seed === seed} onclick={() => set({ seed: q.seed })}><Pic data={q.x} size={52} frame="generated" label={t({ en: `seed ${q.seed}`, zh: `种子 ${q.seed}` })} /><span>{q.seed}</span></button>{/each}
	</div>
{/snippet}

{#snippet pn(v: number)}{#if v < 0}<span class="nw">(<Num value={v} />)</span>{:else}<Num value={v} />{/if}{/snippet}

{#snippet fig(data: ArrayLike<number> | undefined, label: { en: string; zh: string }, o: { kind?: 'ink' | 'signed'; frame?: 'real' | 'generated' | 'none'; size?: number; mark?: number; lo?: number } = {})}
	<figure><Pic {data} label={t(label)} kind={o.kind} frame={o.frame} size={o.size ?? 104} mark={o.mark} lo={o.lo} /><figcaption>{t(label)}</figcaption></figure>
{/snippet}

<div class="pics">
	{#if (mode === 'mix' || mode === 'predict' || mode === 'unet') && !tc}
		<Notice text={{ en: 'Draw something first (or pick a preset): these pictures start from your drawing.', zh: '先画点什么（或选一幅预设画）：这些图都从你的画开始。' }} />
	{:else if mode === 'mix' && tc}
		<div class="row">
			{@render fig(tc.x0, cap.x0, { frame: 'real', mark: tc.i, lo: 0 })}<span class="op">→ 2x₀ − 1, × <Num value={tc.a} /> +</span>
			{@render fig(tc.eps, cap.eps, { kind: 'signed', mark: tc.i })}<span class="op">× <Num value={tc.b} /> =</span>
			{@render fig(tc.xt, { en: `noisy drawing x_${tc.t}`, zh: `加噪的画 x_${tc.t}` }, { mark: tc.i })}
		</div>
		<p class="small">{t({ en: `The marked pixel (row ${Math.floor(tc.i / 28)}, column ${tc.i % 28}, counting from 0):`, zh: `标出的像素（第 ${Math.floor(tc.i / 28)} 行第 ${tc.i % 28} 列，从 0 数起）：` })} <Num value={tc.a} /> × (2 × <Num value={tc.x0[tc.i]} /> − 1) + <Num value={tc.b} /> × {@render pn(tc.eps[tc.i])} = <Num value={tc.xt[tc.i]} /></p>
	{:else if mode === 'noise' && P.at?.state}
		<div class="row">
			{@render fig((P.at.state as { x: Tensor }).x.data, { en: `your drawing at t = ${frameM.t}`, zh: `第 t = ${frameM.t} 步的画` }, { size: 140 })}
			<svg viewBox="-4 -4 214 78" class="sched" role="img" aria-label={t({ en: 'The two weights over the steps', zh: '两个权重随步数的变化' })}>
				<polyline points={curve((ab) => Math.sqrt(ab))} class="sig" /><polyline points={curve((ab) => Math.sqrt(1 - ab))} class="noi" />
				<line x1={(Number(frameM.t) / T) * 200} x2={(Number(frameM.t) / T) * 200} y1="0" y2="62" class="now" />
				<text x="2" y="72">t = 0</text><text x="200" y="72" text-anchor="end">t = {T}</text>
			</svg>
		</div>
		<p class="small"><span class="sigk">√ᾱ_t = <Num value={frameM.a} /></span> × {t({ en: 'drawing', zh: '画' })} + <span class="noik">√(1 − ᾱ_t) = <Num value={frameM.b} /></span> × {t({ en: 'noise', zh: '噪声' })}</p>
	{:else if mode === 'predict' && tc}
		<div class="row">
			{@render fig(tc.xt, { en: `what it sees: x_${tc.t}`, zh: `它看到的：x_${tc.t}` })}
			<div class="row answer">
				{@render fig(tc.eps, { en: 'the noise mixed in ε', zh: '混进去的噪声 ε' }, { kind: 'signed' })}
				{@render fig(tc.out, cap.hat, { kind: 'signed', frame: 'generated' })}
			</div>
		</div>
		<p class="small"><span class="answer">{t({ en: 'Mean squared difference between ε̂ and ε:', zh: 'ε̂ 和 ε 的平均平方差：' })} <Num value={tc.err} /> ·</span> {t({ en: 'try another step t:', zh: '换一个步数 t 试试：' })}
			{#each [5, 20, 50, 90] as v (v)}<button type="button" class:on={v === tc.t} onclick={() => set({ t: v })}>t = {v}</button>{/each}</p>
	{:else if mode === 'step' && s}
		<div class="row">
			{@render fig(s.before, { en: `x_${s.m.t}`, zh: `x_${s.m.t}` }, { mark: PROBE })}<span class="op">→</span>
			{@render fig(s.eps, cap.hat, { kind: 'signed', mark: PROBE })}<span class="op">→</span>
			{@render fig(s.after, { en: `x_${s.m.tNext}`, zh: `x_${s.m.tNext}` }, { frame: 'generated', mark: PROBE })}
		</div>
		<FormulaCard tex={alignTex(sample.phases.step.formula!.tex)} values={texValues(s.m)} caption={{ en: `The marked pixel (row ${Math.floor(PROBE / SIDE)}, column ${PROBE % SIDE}, counting from 0)`, zh: `标出的像素（第 ${Math.floor(PROBE / SIDE)} 行第 ${PROBE % SIDE} 列，从 0 数起）` }} />
	{:else if (mode === 'chain' || mode === 'final' || mode === 'pick') && f}
		{#if mode === 'pick'}{@render seeds()}{/if}
		<div class="row">
			{@render fig(f.x, { en: mode === 'final' ? `after ${f.K} steps` : `t = ${f.t}`, zh: mode === 'final' ? `走了 ${f.K} 步之后` : `t = ${f.t}` }, { frame: 'generated', size: mode === 'final' ? 150 : 132 })}
			{#if mode !== 'final'}<div class="film">{#each f.film as q (q.t)}{@render fig(q.x, { en: `t = ${q.t}`, zh: `t = ${q.t}` }, { size: 52 })}{/each}</div>{/if}
		</div>
	{:else if mode === 'gallery'}
		{@render seeds()}
		{#if chosen}<p class="small">{t({ en: `Seed ${seed}, on its way from noise:`, zh: `种子 ${seed} 从噪声出发的一路：` })}</p><div class="film">{#each chosen.film as q (q.t)}{@render fig(q.x, { en: `t = ${q.t}`, zh: `t = ${q.t}` }, { size: 52 })}{/each}</div>{/if}
	{:else if mode === 'limits'}
		<div class="row">
			{#if tc}{@render fig(tc.x0, { en: 'your drawing: real', zh: '你的画：真实的' }, { frame: 'real', lo: 0 })}{/if}
			<div class="grid lim">{#each g as q (q.seed)}<Pic data={q.x} size={52} frame="generated" label={t({ en: `sample, seed ${q.seed}`, zh: `样本，种子 ${q.seed}` })} />{/each}</div>
		</div>
	{:else if mode === 'unet' && tc}
		<div class="u" use:keyScroll>
			{#each stages as q, k (q.id)}
				<figure class="s{k}"><Pic data={q.m} size={k === 3 ? 40 : 56} kind={k === 6 ? 'signed' : 'ink'} lo={k ? 0 : -1} label={q.id} /><figcaption>{q.label ? t(q.label).split(' · ')[0] : q.id}<br /><span class="num">{q.x?.shape.join('×')}</span></figcaption></figure>
			{/each}
			<span class="skip a">{t({ en: 'skip connection: copied across', zh: '跳跃连接：直接复制过去' })} <span class="h">→</span><span class="v">↓</span></span><span class="skip b">{t({ en: 'skip', zh: '跳跃' })} <span class="h">→</span><span class="v">↓</span></span>
		</div>
		<p class="small muted">{t({ en: 'Inside the U, each thumbnail is the average size of a stage’s numbers over its channels: the more a pixel stands out from the background, the larger they are there.', zh: 'U 里面的每个小图是这一阶段各通道数值大小的平均：像素和背景差得越多，那里的数就越大。' })}</p>
	{:else}
		<Notice text={{ en: 'Computing the pictures…', zh: '正在计算这些图…' }} />
	{/if}
</div>

<style>
	.pics { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }
	figure { margin: 0; display: grid; gap: 3px; justify-items: center; }
	figcaption, .small { font-size: var(--fs-sm); }
	figcaption { color: var(--muted); max-width: 150px; text-align: center; }
	.op { font-size: var(--fs-sm); white-space: nowrap; }
	.nw { white-space: nowrap; }
	p { margin: 0; }
	.sched { width: 220px; max-width: 100%; }
	.sched polyline { fill: none; stroke-width: 2; }
	.sig { stroke: var(--real); } .noi { stroke: var(--muted); stroke-dasharray: 4 3; }
	.sigk { color: var(--real-text); }
	.now { stroke: var(--generated); stroke-width: 1.5; }
	.sched text { font-size: 9px; fill: var(--muted); }
	button { font: inherit; font-size: var(--fs-sm); border: 1px solid var(--rule); border-radius: 4px; background: var(--surface); color: inherit; padding: 2px 6px; cursor: pointer; }
	button.on { border-color: var(--generated); font-weight: 700; }
	.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(62px, 1fr)); gap: 6px; max-width: 640px; }
	.lim { flex: 1 1 180px; min-width: 0; }
	.seed { display: grid; justify-items: center; font-size: var(--fs-xs); gap: 2px; padding: 2px; }
	.film { display: flex; flex-wrap: wrap; gap: 4px; }
	.u { display: grid; grid-template-columns: repeat(7, auto); gap: 6px 8px; align-items: center; justify-content: start; overflow-x: auto; }
	.s0 { grid-area: 1 / 1; } .s1 { grid-area: 1 / 2; } .s2 { grid-area: 2 / 3; } .s3 { grid-area: 3 / 4; }
	.s4 { grid-area: 2 / 5; } .s5 { grid-area: 1 / 6; } .s6 { grid-area: 1 / 7; }
	.skip { font-size: var(--fs-xs); color: var(--muted); text-align: center; border-bottom: 2px dashed var(--rule); }
	.skip.a { grid-area: 1 / 3 / 2 / 6; } .skip.b { grid-area: 2 / 4; }
	:global([inert]) .answer { visibility: hidden; }
	.u figcaption { font-size: var(--fs-xs); max-width: 90px; }
	.v { display: none; }
	/* a phone: the U on its side (down the page and back), so all seven stages fit without a hidden scroll */
	@media (max-width: 560px) {
		.u { grid-template-columns: repeat(3, auto); justify-content: center; overflow-x: visible; }
		.s0 { grid-area: 1 / 1; } .s1 { grid-area: 2 / 1; } .s2 { grid-area: 3 / 2; } .s3 { grid-area: 4 / 3; }
		.s4 { grid-area: 5 / 2; } .s5 { grid-area: 6 / 1; } .s6 { grid-area: 7 / 1; }
		.skip { border-bottom: 0; border-left: 2px dashed var(--rule); align-self: stretch; display: flex; align-items: center; padding-left: 4px; text-align: left; }
		.skip.a { grid-area: 3 / 1 / 6 / 2; } .skip.b { grid-area: 4 / 2; }
		.h { display: none; } .v { display: inline; }
	}
</style>
