<!-- adversarial-tinycnn/attack (see meta.ts): the attack's frames as 28×28 pictures (hover or focus a pixel for its
     number), TinyCNN's class shares and small curves. The ε slider writes ε into the learner's input (setInput), so the
     process, the narration and the task all read the same value. Nothing animates by itself. -->
<script module lang="ts">
	/** the slider had the keyboard when ε changed (the view is drawn again once the attack has re-run) */
	let epsFocus = false;
</script>

<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import { categorical } from '$lib/kit/color';
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n, StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import Curve from '../../parts/Curve.svelte';
	import { TABLE, TABLE_EPS, cls, epsText, nearSmallest, pgdCurve } from '../../read.ts';
	import { attack } from '../../process.ts';
	import { attackOf } from './meta.ts';

	let P: StageProps = $props();
	const A = $derived(attackOf(P));
	const I = (en: string, zh: string): I18n => ({ en, zh });
	let draft = $state<number | null>(null);
	let amp = $state(true);
	const eps = $derived(draft ?? A?.fm.eps ?? 0);
	const name = (i: number) => t(cls(P.rt, i));
	const bars = (p: ArrayLike<number> | undefined) => (p ? Array.from(p, (v, i) => ({ label: name(i), p: v, id: i, color: categorical(i) })) : []);
	const minus = (a?: ArrayLike<number>, b?: ArrayLike<number>) => (a && b ? Float64Array.from(a, (v, i) => v - b[i]) : new Float64Array(784));
	const fact = (k: string) => P.rt.num(`fact:${k}`);
	/** the gradient's colour range: |g| at the 98th percentile, so the pattern shows (the top 2% saturate) */
	const gRange = $derived.by(() => {
		const g = A?.st.g?.data;
		if (!g) return 1;
		const a = Array.from(g, Math.abs).sort((p, q) => p - q);
		return a[Math.floor(0.98 * (a.length - 1))] || a[a.length - 1] || 1;
	});
	const curve = $derived(A?.mode === 'pgd' ? pgdCurve(P.at.tape, P.at.index) : null);
	const grid = $derived(A?.st.scanP ? Array.from(A.st.scanP.data, (_, i) => (i + 1) * A!.m.scanStep) : []);
	let slide = $state<HTMLInputElement>();
	// a new ε re-runs the attack, and the stage shows "Computing…" meanwhile: keep the keyboard on the slider afterwards
	$effect(() => {
		if (slide && epsFocus) (slide.focus(), (epsFocus = false));
	});
	function commit(v: number) {
		epsFocus = typeof document !== 'undefined' && document.activeElement === slide;
		const r = P.rt.input;
		const raw = Array.isArray(r) ? (typeof r[0] === 'number' ? { ink: r } : { strokes: r }) : r && typeof r === 'object' ? r : { strokes: [] };
		draft = null;
		(P.setInput ?? P.rt.setInput)({ ...raw, eps: v });
	}
	$effect(() => {
		if (P.autoplay && A) P.onend();
	});
</script>

{#snippet pic(d: ArrayLike<number> | undefined, cap: I18n, kind: 'sequential' | 'diverging' = 'sequential', m = 1, w = 224, vw = 38)}
	<figure style:width="min({w}px, {vw}vw)">
		<TensorGrid data={d ?? new Float64Array(784)} rows={28} cols={28} {kind} domain={kind === 'diverging' ? [-(m || 1), m || 1] : [0, 1]} maxCell={8} minCell={vw < 30 ? 2 : 3} label={cap} readout={kind === 'diverging'} />
		<figcaption>{t(cap)}</figcaption>
	</figure>
{/snippet}
{#snippet slider()}
	<label class="eps">
		<span>ε = <b class="num">{epsText(eps)}</b></span>
		<input bind:this={slide} type="range" min="0" max={A?.m.scanMax ?? 0.3} step="0.005" value={eps} aria-label={t(I('ε, the largest change of any pixel', 'ε：任何像素的最大改动'))} oninput={(e) => (draft = +e.currentTarget.value)} onchange={(e) => commit(+e.currentTarget.value)} />
	</label>
{/snippet}
{#snippet shares(p: ArrayLike<number> | undefined, cap: I18n)}
	<div class="bars"><p class="small">{t(cap)}</p><ProbBars items={bars(p)} k={3} ids={false} /></div>
{/snippet}

{#if !A}
	<Notice text={I('Draw something first: the attack starts from your drawing.', '先画点什么：攻击从你的画开始。')} />
{:else}
	{@const { st, m, fg, fm, y } = A}
	<div class="atk">
		{#if A.mode === 'grad'}
			<div class="row">
				{@render pic(st.x?.data, I('your drawing x', '你的画 x'))}
				{@render pic(st.g?.data, I('the gradient g = ∂L/∂x', '梯度 g = ∂L/∂x'), 'diverging', gRange)}
			</div>
			<ColorLegend kind="diverging" domain={[-gRange, gRange]} label={I('g: orange, more ink raises the loss; blue, less ink does (the top 2% of |g| show at full colour)', 'g：橙色表示多墨会让损失变大，蓝色表示少墨会（|g| 最大的 2% 以最深的颜色显示）')} width={240} />
			<p class="small muted">{t(I(`Largest |g| = ${fmt3(m.gmax)} at row ${m.row}, column ${m.col}; ${fmtPct(m.onInk)} of all |g| sits on inked pixels, which are ${fmtPct(m.inkShare)} of the grid.`, `最大的 |g| = ${fmt3(m.gmax)}，在第 ${m.row} 行第 ${m.col} 列；全部 |g| 中有 ${fmtPct(m.onInk)} 落在有墨的像素上，而它们只占网格的 ${fmtPct(m.inkShare)}。`))}</p>
		{:else if A.mode === 'sign'}
			<div class="row">
				{@render pic(st.x?.data, I('your drawing x', '你的画 x'))}
				{@render pic(st.s?.data, I('sign(g)', 'sign(g)'), 'diverging', 1)}
			</div>
			<p class="small">{t(I(`+1 (orange): ${m.nPos} pixels · −1 (blue): ${m.nNeg} · 0: ${m.nZero}`, `+1（橙）：${m.nPos} 个像素 · −1（蓝）：${m.nNeg} 个 · 0：${m.nZero} 个`))}</p>
		{:else if A.mode === 'step'}
			<!-- one row at every width, so it reads as the equation x + ε·sign(g) = x′ (pictures shrink on a phone) -->
			<div class="row eq">
				{@render pic(st.x?.data, I('x, your drawing', 'x，你的画'), 'sequential', 1, 168, 24)}<span class="op">+</span>
				{@render pic(st.s ? Float64Array.from(st.s.data, (v) => v * fm.eps) : undefined, I('ε · sign(g): full colour = ε', 'ε · sign(g)：颜色最深 = ε'), 'diverging', fm.eps, 168, 24)}<span class="op">=</span>
				{@render pic(fg.adv?.data, I('x′, clipped to [0, 1]', 'x′，裁剪到 [0, 1]'), 'sequential', 1, 168, 24)}
			</div>
			{@render slider()}
		{:else if A.mode === 'flip'}
			<div class="row">
				{@render pic(st.x?.data, I('your drawing x', '你的画 x'), 'sequential', 1, 196)}
				{@render pic(fg.adv?.data, I('x′, after one FGSM step', 'x′，走了一步 FGSM 之后'), 'sequential', 1, 196)}
				{#if amp}{@render pic(minus(fg.adv?.data, st.x?.data), I('the change x′ − x, amplified: full colour = ε', '改动 x′ − x，放大显示：颜色最深 = ε'), 'diverging', fm.eps, 196)}{/if}
			</div>
			<label class="small"><input type="checkbox" bind:checked={amp} /> {t(I('show the change, amplified', '显示放大后的改动'))}</label>
			<div class="row">
				{@render shares(st.p0?.data, I('TinyCNN on x', 'TinyCNN 看 x'))}
				{@render shares(fg.p?.data, I('TinyCNN on x′', 'TinyCNN 看 x′'))}
			</div>
		{:else if A.mode === 'pgd'}
			<div class="row">
				{@render pic(st.adv?.data, I(`x after step ${m.k} of ${m.steps}`, `第 ${m.k} 步（共 ${m.steps} 步）之后的 x`))}
				{@render pic(minus(st.adv?.data, st.x?.data), I('the change so far: never more than ε', '到目前为止的改动：从不超过 ε'), 'diverging', m.eps)}
			</div>
			{#if curve}
				<Curve xs={curve.py.map((_, i) => i)} domain={[0, m.steps]} xTicks={Array.from({ length: m.steps / 2 + 1 }, (_, i) => 2 * i)} lines={[{ y: curve.py, label: I(`p(${cls(P.rt, y).en})`, `p(“${cls(P.rt, y).zh}”)`) }, { y: curve.other, label: I('the best other class', '最强的其他类别'), dashed: true }]} xLabel={I('PGD step', 'PGD 步数')} label={I('Shares over the PGD steps', '各步 PGD 之后的份额')} />
			{/if}
			<p class="small">{t(I(`Step ${m.k}: TinyCNN says ${cls(P.rt, m.top).en} (${fmtPct(m.ptop)}); p(${cls(P.rt, y).en}) = ${fmtPct(m.py)}; loss ${fmt3(m.loss)}.`, `第 ${m.k} 步：TinyCNN 说是“${cls(P.rt, m.top).zh}”（${fmtPct(m.ptop)}）；p(“${cls(P.rt, y).zh}”) = ${fmtPct(m.py)}；损失 ${fmt3(m.loss)}。`))}</p>
		{:else if A.mode === 'budget'}
			{@render slider()}
			<div class="row">
				{@render pic(fg.adv?.data, I('x′ at this ε', '这个 ε 下的 x′'))}
				{@render shares(fg.p?.data, I(fm.flipped ? 'TinyCNN on x′: flipped' : 'TinyCNN on x′: not flipped', fm.flipped ? 'TinyCNN 看 x′：翻转了' : 'TinyCNN 看 x′：没翻转'))}
			</div>
			{#if nearSmallest(m)}
				<Curve xs={grid} domain={[0, m.scanMax]} xTicks={[0, 0.1, 0.2, 0.3]} lines={[{ y: st.scanP?.data ?? [], label: I(`p(${cls(P.rt, y).en}) after one FGSM step`, `一步 FGSM 之后的 p(“${cls(P.rt, y).zh}”)`) }]} marks={[{ x: m.minEps, label: I(`smallest ε that flips: ${epsText(m.minEps)}`, `能翻转的最小 ε：${epsText(m.minEps)}`) }]} xLabel={I('ε', 'ε')} label={I('p(y) for every ε on the slider', '滑块上每个 ε 的 p(y)')} />
			{:else if m.found === 0}
				<p class="small muted">{t(I(`One FGSM step does not flip this drawing anywhere up to ε = ${epsText(m.scanMax)}.`, `一直到 ε = ${epsText(m.scanMax)}，一步 FGSM 都翻转不了这幅画。`))}</p>
			{/if}
		{:else if A.mode === 'defend'}
			<div class="row">
				{@render pic(st.x?.data, I(`your drawing, label ${cls(P.rt, y).en}`, `你的画，标签“${cls(P.rt, y).zh}”`))}
				{@render pic(st.adv?.data, I(`x′ after PGD, label still ${cls(P.rt, y).en}`, `PGD 之后的 x′，标签仍是“${cls(P.rt, y).zh}”`))}
			</div>
			<p class="small"><a href="{base}/train/adversarial-robustness/">{t(I('Adversarial examples and robustness (planned concept page)', '对抗样本与鲁棒性（计划中的概念页）'))}</a></p>
		{:else}
			<table class="small">
				<thead><tr><th scope="col">ε</th><th scope="col">{t(I('one FGSM step', '一步 FGSM'))}</th><th scope="col">{t(I(`PGD, ${attack.defaults.steps} steps`, `PGD，${attack.defaults.steps} 步`))}</th></tr></thead>
				<tbody>
					{#each TABLE as r (r.key)}<tr><td class="num">{r.eps}</td><td class="num">{fmtPct(fact(`fgsm${r.key}`))}</td><td class="num">{fmtPct(fact(`pgd${r.key}`))}</td></tr>{/each}
				</tbody>
			</table>
			<p class="small muted">{t(I(`Share of TinyCNN’s ${fmtInt(fact('attackedN'))} test drawings it gets right that each attack flips. PGD drawings made on TinyCNN (ε = ${TABLE_EPS}) also fool TinyViT ${fmtPct(fact('transferVit'))} of the time; TinyViT’s own gradient flips ${fmtPct(fact('vitPgd10'))}.`, `TinyCNN 答对的 ${fmtInt(fact('attackedN'))} 幅测试画中，被每种攻击翻转的比例。对 TinyCNN 做出的 PGD 画（ε = ${TABLE_EPS}）同样骗过 TinyViT 的比例是 ${fmtPct(fact('transferVit'))}；用 TinyViT 自己的梯度则能翻转 ${fmtPct(fact('vitPgd10'))}。`))}</p>
		{/if}
	</div>
{/if}

<style>
	.atk { display: grid; gap: 10px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 10px 8px; align-items: start; }
	figure { margin: 0; display: grid; gap: 4px; justify-items: center; }
	figcaption { color: var(--muted); font-size: var(--fs-sm); max-width: 170px; text-align: center; }
	.op { font-size: 1.4rem; color: var(--muted); }
	.eq { flex-wrap: nowrap; gap: 6px; }
	.eq .op { padding-top: calc(min(168px, 24vw) / 2 - 0.9rem); }
	.eps { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
	.eps input { width: min(280px, 100%); }
	.bars { min-width: 200px; flex: 1 1 200px; }
	.small { font-size: var(--fs-sm); }
	p { margin: 0; }
	table { border-collapse: collapse; max-width: 460px; }
	th, td { padding: 3px 12px 3px 0; text-align: right; }
	th:first-child, td:first-child { text-align: left; }
</style>
