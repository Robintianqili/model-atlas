<!--
  "Try this": guided experiments. Each button sets the page state (the pad, a preset, the erase
  mask, the compare panel); every result shown is computed live by the same model. Each card has
  an anchor (#try-sun …) so a link can open the page with the experiment set up.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import ComparePanel from './ComparePanel.svelte';
	import Misconception from './Misconception.svelte';
	import { rasterize, type Stroke } from '$lib/kit/draw';
	import { fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { SUN, SUN_TRUNK, DISC_TRUNK, cloneStrokes, shrinkToCorner } from './sketches';
	import { simplifyStroke, type EraseMode } from './data';
	import type { CnnStore } from './store.svelte';

	let { S, baseUrl }: { S: CnnStore; baseUrl: string } = $props();

	const name = (k: number) => t(S.spec.classes[k]);
	function topOf(p: Float32Array | null | undefined): { k: number; p: number; k2: number; p2: number } | null {
		if (!p) return null;
		let k = 0;
		for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
		let k2 = k === 0 ? 1 : 0;
		for (let i = 0; i < p.length; i++) if (i !== k && p[i] > p[k2]) k2 = i;
		return { k, p: p[k], k2, p2: p[k2] };
	}
	function strokesInput(s: Stroke[]): Float32Array | null {
		if (!S.mod) return null;
		const lines = s.map((st) => simplifyStroke(st.points)).filter((p) => p.length);
		return lines.length ? S.mod.preprocessStrokes(lines).input : null;
	}
	const fmtTop = (r: ReturnType<typeof topOf>) => (r ? `${name(r.k)} ${fmtPct(r.p)} · ${name(r.k2)} ${fmtPct(r.p2)}` : '…');

	// ---------------------------------------------------------- 1. sun -> tree
	const STAGES = [
		{ s: SUN, label: { en: '1 · Draw a sun', zh: '1 · 画一个太阳' } },
		{ s: SUN_TRUNK, label: { en: '2 · Add a trunk', zh: '2 · 加上树干' } },
		{ s: DISC_TRUNK, label: { en: '3 · Remove the rays', zh: '3 · 去掉光芒' } }
	];
	let sunSeen = $state<boolean[]>([false, false, false]);
	const stageInputs = $derived(S.mod ? STAGES.map((st) => strokesInput(st.s)) : []);
	const stageTops = $derived(S.model ? stageInputs.map((x) => (x ? topOf(S.model!.predict(x).probs) : null)) : []);
	function stage(i: number) {
		S.erase = 'none';
		S.setStrokes(cloneStrokes(STAGES[i].s));
		sunSeen[i] = true;
	}

	// ---------------------------------------------------------- 2. very small
	let beforeShrink = $state.raw<Stroke[] | null>(null);
	function shrink() {
		S.erase = 'none';
		const base = S.source.kind === 'pad' && S.strokes.length ? S.strokes : SUN;
		beforeShrink = cloneStrokes(base);
		S.setStrokes(shrinkToCorner(base));
	}
	function unshrink() {
		if (beforeShrink) S.setStrokes(beforeShrink);
		beforeShrink = null;
	}
	/** the pixel path: photograph the pad (15 px brush on 280 px), then crop and resize the pixels */
	const pixelPath = $derived.by(() => {
		if (!S.mod || !S.model || S.source.kind !== 'pad' || !S.strokes.length) return null;
		const img = rasterize(S.strokes, 280);
		const r = S.mod.preprocessDrawing(img);
		return { input: r.input, top: topOf(S.model.predict(r.input).probs) };
	});
	const strokeTop = $derived(topOf(S.probs));
	const acc = $derived(
		(S.spec.training as { metrics?: { drawingPath?: { accuracy?: Record<string, number> } } } | undefined)?.metrics?.drawingPath?.accuracy ?? {}
	);

	// ---------------------------------------------------------- 3. erase half
	let beforeErase = $state<string | null>(null);
	function erase(m: EraseMode) {
		if (S.erase === 'none') {
			const tp = topOf(S.probs);
			beforeErase = tp ? `${name(tp.k)} ${fmtPct(tp.p)}` : null;
		}
		S.erase = m;
	}

	// ---------------------------------------------------------- 4. ladder
	function ladder(cls: 'ladder' | 'envelope') {
		S.erase = 'none';
		S.loadPreset(cls, cls === 'ladder' ? 2 : 4);
	}
	const isOOD = $derived(S.source.kind === 'preset' && S.source.heldOut);

	// ---------------------------------------------------------- 5. compare
	let showCompare = $state(false);
	let bCls = $state('apple');
	let bIdx = $state(4);
	function compareApples() {
		S.erase = 'none';
		S.loadPreset('apple', 1);
		bCls = 'apple';
		bIdx = 4;
		showCompare = true;
	}

	// ---------------------------------------------------------- deep links (#try-…)
	const SETUP: Record<string, () => void> = {
		'try-sun': () => stage(0),
		'try-small': shrink,
		'try-erase': () => erase('right'),
		'try-ladder': () => ladder('ladder'),
		'try-compare': compareApples
	};
	onMount(() => {
		const id = location.hash.slice(1);
		if (!SETUP[id]) return;
		let done = false;
		return $effect.root(() => {
			$effect(() => {
				if (done || !S.model) return;
				done = true;
				untrack(() => SETUP[id]());
				requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }));
			});
		});
	});
</script>

<div class="tt">
	<!-- 1 -->
	<article class="card" id="try-sun">
		<header><h3>{t({ en: 'Draw a sun, then add a trunk', zh: '先画太阳，再加树干' })}</h3><a class="anchor" href="#try-sun" aria-label={t({ en: 'Link to this experiment', zh: '本实验的链接' })}>#</a></header>
		<p class="pf"><strong>{t({ en: 'Predict first:', zh: '先预测：' })}</strong> {t({ en: 'a circle with rays on a stick. Sun or tree?', zh: '一个带光芒的圆，下面有根棍子。是太阳还是树？' })}</p>
		<div class="btns">
			{#each STAGES as st, i (i)}
				<button type="button" class="btn" onclick={() => stage(i)}>{t(st.label)}</button>
			{/each}
		</div>
		<div class="stages">
			{#each STAGES as st, i (i)}
				{#if sunSeen[i] && stageInputs[i]}
					<figure>
						<MapCanvas data={stageInputs[i]!} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={56} />
						<figcaption class="num">{fmtTop(stageTops[i])}</figcaption>
					</figure>
				{/if}
			{/each}
		</div>
		{#if sunSeen[2]}
			<p class="ex">
				{t({
					en: 'The rays are such strong evidence for "sun" that a trunk barely changes the answer. Take the rays away and the same disc on a trunk is a tree. The model adds up local evidence (rays, a vertical trunk, a round top); it has no notion that a sun cannot have a trunk.',
					zh: '光芒是“太阳”的强证据，加上树干几乎不改变答案。去掉光芒，同一个圆盘加树干就成了树。模型是在累加局部证据（光芒、竖直的树干、圆顶）；它并不知道太阳不可能有树干。'
				})}
			</p>
		{/if}
	</article>

	<!-- 2 -->
	<article class="card" id="try-small">
		<header><h3>{t({ en: 'Draw very small', zh: '画得非常小' })}</h3><a class="anchor" href="#try-small" aria-label={t({ en: 'Link to this experiment', zh: '本实验的链接' })}>#</a></header>
		<p class="pf"><strong>{t({ en: 'Predict first:', zh: '先预测：' })}</strong> {t({ en: 'a tiny drawing in a corner of the pad. Will the model still recognise it?', zh: '在画板角落画一个很小的画。模型还认得出来吗？' })}</p>
		<div class="btns">
			<button type="button" class="btn" onclick={shrink}>{t({ en: 'Shrink the drawing into a corner', zh: '把画缩小到角落' })}</button>
			{#if beforeShrink}<button type="button" class="btn ghost" onclick={unshrink}>{t({ en: 'Restore its size', zh: '恢复原来大小' })}</button>{/if}
		</div>
		{#if beforeShrink && pixelPath}
			<div class="paths">
				<figure>
					<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={84} />
					<figcaption><strong>{t({ en: 'Stroke path (this page)', zh: '笔画路径（本页使用）' })}</strong><br /><span class="num">{fmtTop(strokeTop)}</span></figcaption>
				</figure>
				<figure>
					<MapCanvas data={pixelPath.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={84} />
					<figcaption><strong>{t({ en: 'Pixel path (crop + resize the pad image)', zh: '像素路径（裁剪并缩放画板图像）' })}</strong><br /><span class="num">{fmtTop(pixelPath.top)}</span></figcaption>
				</figure>
				<ProvenanceBadge mode="live" compact />
			</div>
			<p class="ex">
				{t({
					en: 'This page frames your strokes and redraws them at the training stroke width, so size does not matter. If you instead cut the tiny drawing out of the pad image and enlarge it, a brush stroke becomes several pixels thick, much thicker than any stroke the model saw in training.',
					zh: '本页会重新框定你的笔画，并按训练时的笔画粗细重画，所以大小无关紧要。如果改为把小图从画板图像中裁出来再放大，一笔就会变成好几个像素粗，远比模型训练时见过的笔画都粗。'
				})}
				{#if acc['fixed 15 px brush, small drawing (60-100 px)']}
					{t({
						en: `Measured on 1,500 test drawings drawn on a simulated pad with this 15 px brush and sent through the pixel path: ${fmtPct(acc['fixed 15 px brush, drawing 140-260 px'] ?? 0)} when the drawing is 140–260 px wide, ${fmtPct(acc['fixed 15 px brush, small drawing (60-100 px)'])} when it is 60–100 px.`,
						zh: `在 1,500 幅测试画上实测（模拟画板、同样 15 像素笔刷、走像素路径）：画宽 140–260 像素时 ${fmtPct(acc['fixed 15 px brush, drawing 140-260 px'] ?? 0)}，只有 60–100 像素时 ${fmtPct(acc['fixed 15 px brush, small drawing (60-100 px)'])}。`
					})}
					<ProvenanceBadge mode="precomputed" compact />
				{/if}
			</p>
		{/if}
	</article>

	<!-- 3 -->
	<article class="card" id="try-erase">
		<header><h3>{t({ en: 'Erase half of the drawing', zh: '擦掉一半' })}</h3><a class="anchor" href="#try-erase" aria-label={t({ en: 'Link to this experiment', zh: '本实验的链接' })}>#</a></header>
		<p class="pf"><strong>{t({ en: 'Predict first:', zh: '先预测：' })}</strong> {t({ en: 'which half of your drawing carries the answer?', zh: '你的画的哪一半决定了答案？' })}</p>
		<div class="btns">
			<button type="button" class="btn" aria-pressed={S.erase === 'left'} onclick={() => erase('left')}>{t({ en: 'Left half', zh: '左半边' })}</button>
			<button type="button" class="btn" aria-pressed={S.erase === 'right'} onclick={() => erase('right')}>{t({ en: 'Right half', zh: '右半边' })}</button>
			<button type="button" class="btn" aria-pressed={S.erase === 'top'} onclick={() => erase('top')}>{t({ en: 'Top half', zh: '上半部' })}</button>
			<button type="button" class="btn" aria-pressed={S.erase === 'bottom'} onclick={() => erase('bottom')}>{t({ en: 'Bottom half', zh: '下半部' })}</button>
			<button type="button" class="btn ghost" disabled={S.erase === 'none'} onclick={() => (S.erase = 'none')}>{t({ en: 'Restore', zh: '恢复' })}</button>
		</div>
		{#if S.erase !== 'none'}
			<p class="res num">
				{#if beforeErase}{t({ en: 'Before:', zh: '之前：' })} {beforeErase} → {/if}{t({ en: 'now:', zh: '现在：' })} {fmtTop(strokeTop)}
				<ProvenanceBadge mode="live" compact />
			</p>
			<p class="ex">
				{t({
					en: 'The erased half becomes zeros in the 28×28 input; nothing is re-centered. Look at the overview: the maps go quiet on that side, and the cells that saw the missing part stop voting. Try the other halves too: often one half alone is enough, because the evidence is spread over the drawing.',
					zh: '被擦掉的一半在 28×28 输入里变成 0，画面不会重新居中。看看总览：那一侧的特征图安静下来，看到缺失部分的格子不再“投票”。也试试其他几半：往往一半就够了，因为证据分布在整幅画上。'
				})}
			</p>
		{/if}
	</article>

	<!-- 4 -->
	<article class="card" id="try-ladder">
		<header><h3>{t({ en: 'Show it something it never learned', zh: '给它看没学过的东西' })}</h3><a class="anchor" href="#try-ladder" aria-label={t({ en: 'Link to this experiment', zh: '本实验的链接' })}>#</a></header>
		<p class="pf"><strong>{t({ en: 'Predict first:', zh: '先预测：' })}</strong> {t({ en: 'the model knows 10 classes, and a ladder is not one of them. What will it answer?', zh: '模型只认识 10 个类别，梯子不在其中。它会怎么回答？' })}</p>
		<div class="btns">
			<button type="button" class="btn" onclick={() => ladder('ladder')}>{t({ en: 'Load a ladder', zh: '载入一幅梯子' })}</button>
			<button type="button" class="btn" onclick={() => ladder('envelope')}>{t({ en: 'Load an envelope', zh: '载入一幅信封' })}</button>
		</div>
		{#if isOOD}
			<p class="res num">{t({ en: 'Answer:', zh: '回答：' })} {fmtTop(strokeTop)} <ProvenanceBadge mode="live" compact /></p>
			<p class="ex">
				<Lang>
					{#snippet en()}It cannot say "I don't know". <Term id="softmax" /> always splits 100% among the 10 classes it was trained on, so an unfamiliar drawing still gets a class, often with a very high number. The model is a <Term id="classifier" /> for 10 things, not a detector of "anything else".{/snippet}
					{#snippet zh()}它无法说“我不知道”。<Term id="softmax" />总是把 100% 分给训练时的 10 个类别，所以陌生的画仍会被归入某一类，而且常常给出很高的数字。这个模型是针对 10 种东西的<Term id="classifier" />，不是“其他东西”的检测器。{/snippet}
				</Lang>
			</p>
		{/if}
	</article>

	<!-- 5 -->
	<article class="card wide" id="try-compare">
		<header><h3>{t({ en: 'Compare two drawings of the same class', zh: '比较同一类别的两幅画' })}</h3><a class="anchor" href="#try-compare" aria-label={t({ en: 'Link to this experiment', zh: '本实验的链接' })}>#</a></header>
		<p class="pf"><strong>{t({ en: 'Predict first:', zh: '先预测：' })}</strong> {t({ en: 'two apples drawn by two different people. In which layer do they look most alike: the pixels, the first convolutions, or the end?', zh: '两个人各画了一个苹果。它们在哪一层最相像：像素、最早的卷积，还是最后？' })}</p>
		<div class="btns">
			<button type="button" class="btn" onclick={compareApples}>{t({ en: 'Compare two apples', zh: '比较两个苹果' })}</button>
			{#if !showCompare}<button type="button" class="btn ghost" onclick={() => (showCompare = true)}>{t({ en: 'Compare my input with a preset', zh: '把我的输入和预设比较' })}</button>{/if}
		</div>
		{#if showCompare}
			<ComparePanel {S} bind:bCls bind:bIdx />
			<p class="ex">
				{t({
					en: 'Read the bars from top to bottom. Similarity does not climb steadily. In the pixels and the early maps, two drawings of the same thing can be no more alike than drawings of different things: with the two preset apples, the gray bar (an apple against a sun) is even longer. The split happens at the end. From Max pool 2 on, the same-class pair pulls ahead, and at the dense layer their 10 class scores nearly agree while the gray reference stays low. That is what training bought: throw away how a drawing looks, keep what it is.',
					zh: '从上往下看这些条。相似度并不是一路上升的。在像素和前几层特征图里，同一种东西的两幅画未必比不同东西的画更像：用两幅预设苹果时，灰色条（苹果对太阳）甚至更长。分化发生在最后：从最大池化 2 开始，同类的一对明显领先；到全连接层，它们的 10 个类别分数几乎一致，而灰色参照仍然很低。这就是训练换来的：丢掉画“长什么样”，留下它“是什么”。'
				})}
			</p>
		{/if}
	</article>

	<!-- misconception -->
	<Misconception {S} {baseUrl} />
</div>

<style>
	.tt {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
		gap: 14px;
	}
	.card {
		display: grid;
		align-content: start;
		gap: 10px;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		min-width: 0;
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.card.wide {
		grid-column: 1 / -1;
	}
	.card:target {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	header {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		align-items: baseline;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-lg);
	}
	.anchor {
		color: var(--muted);
		text-decoration: none;
		font-family: var(--font-mono);
	}
	.pf {
		margin: 0;
		font-size: var(--fs-sm);
		padding: 6px 10px;
		border-radius: var(--r1);
		background: var(--accent-soft);
	}
	.btns {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.stages,
	.paths {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 18px;
		align-items: flex-end;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		justify-items: start;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		max-width: 200px;
	}
	.res {
		margin: 0;
		font-size: var(--fs-sm);
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.ex {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
</style>
