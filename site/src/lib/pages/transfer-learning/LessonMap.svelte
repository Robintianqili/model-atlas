<!--
  Lesson step 3: where drawings land in the frozen feature space. The PCA axes and the gray /
  colored context points are precomputed (pca.json); the learner's drawings are projected live,
  and their 3 nearest neighbours are found live in all 256 features.
-->
<script lang="ts">
	import Num from '$lib/kit/Num.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import PcaMap from './PcaMap.svelte';
	import Thumb from './Thumb.svelte';
	import { clsName, type LessonState } from './lessonState.svelte';

	let { L }: { L: LessonState } = $props();
	const names = $derived<[string, string]>([t(clsName(0)), t(clsName(1))]);
	const trail = $derived(L.trail.map((d) => ({ feat: d.feat, n: d.n })));
	const padLabel = $derived(L.source.kind === 'drawn' ? { en: 'your drawing', zh: '你的画' } : { en: 'this drawing', zh: '这张画' });
	/** the next held-out test drawing of a class (the same as ‹ › on the stage header’s “Your drawing” chip) */
	function another(cls: 0 | 1) {
		const s = L.source;
		if (s.kind === 'test' && s.cls === cls) L.stepTest(1);
		else L.pickTest(cls);
	}
</script>

<div class="lm">
	{#if L.pca}
		<PcaMap pca={L.pca} items={[]} pad={L.feat} head={null} trained={false} preset={true} {names} testFeats={null} testLabels={null} {trail} showBar={false} {padLabel} badge={false} />
	{/if}
	<!-- the same picks as the “Your drawing” chip, right under the map (the chip hides ‹ › on phones) -->
	<div class="again">
		<span class="muted">{t({ en: 'Try another drawing:', zh: '换一张试试：' })}</span>
		<button type="button" class="btn" onclick={() => another(0)}><span class="dot a"></span>{t({ en: 'next ladder', zh: '下一张梯子' })}</button>
		<button type="button" class="btn" onclick={() => another(1)}><span class="dot b"></span>{t({ en: 'next envelope', zh: '下一张信封' })}</button>
	</div>
	<div class="below">
		<div class="nb">
			<h4>{t({ en: 'Closest to this drawing, in all 256 numbers', zh: '在全部 256 个数上最接近这张画的' })}</h4>
			{#if L.neighbours.length}
				<ul class="thumbs">
					{#each L.neighbours as n (n.split + n.cls + n.i)}
						<li class="cls{n.cls}">
							<Thumb data={n.input} size={48} alt={t(clsName(n.cls))} />
							<span class="nm">{t(clsName(n.cls))}</span>
							<span class="num d">{t({ en: 'distance', zh: '距离' })} <Num value={n.d} /></span>
						</li>
					{/each}
				</ul>
				<p class="muted small">{t({ en: 'Searched among the 264 shipped ladder and envelope drawings (Euclidean distance between feature vectors).', zh: '在随页面提供的 264 张梯子和信封图中查找（特征向量之间的欧氏距离）。' })}</p>
			{:else if L.presetError}
				<p class="muted small">{t({ en: 'Could not compute the features. Please reload the page.', zh: '特征计算失败，请刷新页面。' })}</p>
			{:else}
				<p class="muted small">{t({ en: 'Computing the features of 264 drawings…', zh: '正在计算 264 张画的特征…' })}</p>
			{/if}
		</div>
		{#if L.trail.length}
			<div class="tr">
				<h4>{t({ en: 'Your earlier drawings (numbered on the map)', zh: '你之前的画（地图上的编号）' })}</h4>
				<ul class="thumbs">
					{#each L.trail as d (d.n)}
						<li><Thumb data={d.input} size={36} alt={t(d.label)} /><span class="num nm">{d.n}</span></li>
					{/each}
				</ul>
			</div>
		{/if}
	</div>
</div>

<style>
	.lm {
		display: grid;
		gap: 12px;
		min-width: 0;
	}
	.again {
		display: none;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 8px;
		font-size: var(--fs-sm);
	}
	.again .btn {
		min-height: 34px;
		padding: 0 10px;
		font-size: var(--fs-sm);
		gap: 6px;
	}
	.dot {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
		background: var(--div-pos);
	}
	.dot.b {
		background: var(--div-neg);
		border-radius: 2px;
	}
	@container wt (max-width: 859.98px) {
		.again {
			display: flex;
		}
	}
	.below {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 24px;
	}
	h4 {
		margin: 0 0 6px;
		font-size: var(--fs-sm);
	}
	.thumbs {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
	}
	.thumbs li {
		display: grid;
		justify-items: center;
		gap: 2px;
		font-size: var(--fs-xs);
	}
	.thumbs li :global(canvas) {
		border: 1px solid var(--rule);
	}
	.cls0 :global(canvas) {
		border: 2px solid var(--div-pos) !important;
	}
	.cls1 :global(canvas) {
		border: 2px solid var(--div-neg) !important;
	}
	.nm {
		font-weight: 650;
	}
	.d {
		color: var(--ink-2);
	}
	.small {
		margin: 6px 0 0;
		font-size: var(--fs-xs);
		max-width: 40em;
	}
</style>
