<!--
  TinyCNN's Explore view (redesign spec §5.9), loaded only when the view opens: the whole network at once on the
  lesson's input (the page's one CnnStore): the overview with the zoom (the arithmetic of any unit), inside a layer,
  the output, try this, check yourself (the cover question revealed, the quiz) and the links. Each section has an anchor
  (#explore-overview, #explore-zoom …; the old #overview, #inspector, #output, #tryThis, #checkpoint and #try-<id> open
  them). The selection here is the page store's own; the lesson keeps its own (LessonView), so neither moves the other.
-->
<script lang="ts">
	import { tick, untrack } from 'svelte';
	import { page } from '$app/state';
	import Section from '$lib/kit/Section.svelte';
	import CatalogCard from '$lib/kit/CatalogCard.svelte';
	import PredictReveal from '$lib/kit/PredictReveal.svelte';
	import { t, ui, type Text } from '$lib/i18n/i18n.svelte';
	import type { ModelPageCatalog } from '$lib/catalog/types';
	import type { PredictQuestion } from '$lib/kit/types';
	import { motion } from '$lib/kit/theme.svelte';
	import { getLessonApp } from '$lib/kit/lesson/app';
	import InputChip from '../InputChip.svelte';
	import Overview from '../Overview.svelte';
	import Zoom from '../Zoom.svelte';
	import InspectorPanel from '../InspectorPanel.svelte';
	import OutputPanel from '../OutputPanel.svelte';
	import TryThis from '../TryThis.svelte';
	import Quiz from '../Quiz.svelte';
	import type { CnnStore } from '../store.svelte';

	let { vp, section }: { vp: { S: CnnStore; baseUrl: string; prequestion?: PredictQuestion }; section?: string } = $props();
	const S = $derived(vp.S);

	// a deep link (#output, #try-erase …): the view host scrolled there on open, but the overview and the zoom grow once
	// the model has run; scroll to the section again then (once per section)
	let scrolledFor: string | undefined;
	$effect(() => {
		const s = section;
		if (!s || !S.trace || scrolledFor === s) return;
		scrolledFor = s;
		untrack(() => tick().then(() => requestAnimationFrame(() => (document.getElementById(`explore-${s}`) ?? document.getElementById(s))?.scrollIntoView({ block: 'start' }))));
	});

	// the one input of the page: the chip opens the same input sheet as the lesson's stage header
	const app = getLessonApp();
	const openInput = () => {
		if (app) app.input.open = true;
	};

	let zoomEl = $state<HTMLElement>();
	function zoom() {
		zoomEl?.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'start' });
		zoomEl?.focus({ preventScroll: true });
	}

	const quiz: { id: string; q: Text; hint: Text; options: Text[]; answer: number; explain: Text }[] = [
		{
			id: 'tinycnn-q-products',
			q: { en: 'How many multiplications go into one cell of Conv 2-1 (before adding the bias)?', zh: '卷积 2-1 的一个格子（加偏置之前）需要多少次乘法？' },
			hint: { en: 'Click any Conv 2-1 map in the overview and look at the zoom view.', zh: '在总览中点击卷积 2-1 的任意一张图，看放大视图。' },
			options: [
				{ en: '9', zh: '9' },
				{ en: '72', zh: '72' },
				{ en: '144', zh: '144' },
				{ en: '256', zh: '256' }
			],
			answer: 1,
			explain: {
				en: 'Its input has 8 channels and each contributes a 3×3 window: 8 × 9 = 72 products, then one bias.',
				zh: '它的输入有 8 个通道，每个通道贡献一个 3×3 窗口：8 × 9 = 72 次乘法，再加一个偏置。'
			}
		},
		{
			id: 'tinycnn-q-params',
			q: { en: 'Which layer holds the most parameters?', zh: '哪一层的参数最多？' },
			hint: { en: 'See the parameter table in the Training view.', zh: '看“训练”视图里的参数表。' },
			options: [
				{ en: 'Conv 1-1, because it sees the full-size image', zh: '卷积 1-1，因为它看到的是全尺寸图像' },
				{ en: 'Conv 2-2', zh: '卷积 2-2' },
				{ en: 'Max pool 2', zh: '最大池化 2' },
				{ en: 'The dense layer', zh: '全连接层' }
			],
			answer: 3,
			explain: {
				en: 'Dense: 10 × 256 + 10 = 2,570 of the 6,722. Conv 1-1 has only 80 because its 9 weights per kernel are shared across all positions; pooling has none.',
				zh: '全连接层：10 × 256 + 10 = 2,570（共 6,722）。卷积 1-1 只有 80 个，因为每个卷积核的 9 个权重在所有位置共享；池化层一个也没有。'
			}
		}
	];

	const cat = $derived((page.data as { catalog?: ModelPageCatalog | null }).catalog ?? null);
	const linkGroups = $derived(
		[
			{ h: ui('seeAlsoConcepts'), cards: cat?.usedBy ?? [] },
			{ h: ui('prerequisites'), cards: cat?.prereqs ?? [] },
			{ h: t({ en: 'Next', zh: '下一步' }), cards: cat?.next ?? [] }
		].filter((g) => g.cards.length)
	);
	const sections = $derived(
		[
			['overview', t({ en: 'The whole network', zh: '整个网络' })],
			['zoom', t({ en: 'Zoom in', zh: '放大' })],
			['inspector', ui('secInspector')],
			['output', ui('secOutput')],
			['tryThis', ui('secTryThis')],
			['checkpoint', ui('secCheckpoint')],
			...(linkGroups.length ? [['links', ui('secLinks')]] : [])
		] as [string, string][]
	);
</script>

<div class="etop"><InputChip {S} open={openInput} /></div>

<nav class="snav" aria-label={t({ en: 'Sections', zh: '本页分节' })}>
	{#each sections as s (s[0])}<a href="#explore-{s[0]}">{s[1]}</a>{/each}
</nav>

<Section
	id="explore-overview"
	title={{ en: 'The whole network, then zoom in', zh: '整个网络，再放大细看' }}
	intro={{
		en: 'Every column is a layer, every tile a feature map, computed from your input just now. Lines show what feeds what. Click a map (or use the arrow keys) to zoom into its arithmetic.',
		zh: '每一列是一层，每个小图是一张特征图，都由你的输入刚刚算出。连线表示数据从哪里来。点击一张图（或用方向键），放大查看它的计算。'
	}}
	provenance="live"
>
	<Overview {S} onzoom={zoom} />
	<div class="zw" id="explore-zoom"><Zoom {S} bind:el={zoomEl} /></div>
</Section>
<Section
	id="explore-inspector"
	title={ui('secInspector')}
	intro={{ en: 'All channels of the selected layer, their statistics, the formula with live numbers, the code, and what each channel responds to.', zh: '选中层的全部通道、统计、带实时数字的公式、代码，以及每个通道对什么有反应。' }}
	provenance="live"
>
	<InspectorPanel {S} />
</Section>
<Section id="explore-output" title={ui('secOutput')} provenance="live">
	<OutputPanel {S} onzoom={zoom} />
</Section>
<Section id="explore-tryThis" title={ui('secTryThis')} intro={{ en: 'Each experiment sets up the page for you. Predict first, then look.', zh: '每个实验都会替你把页面设置好。先预测，再看结果。' }} provenance="live">
	<TryThis {S} baseUrl={vp.baseUrl} />
</Section>
<Section id="explore-checkpoint" title={ui('secCheckpoint')}>
	<div class="cp">
		{#if vp.prequestion}<PredictReveal q={vp.prequestion} />{/if}
		<Quiz questions={quiz} />
	</div>
</Section>
{#if linkGroups.length}
	<Section id="explore-links" title={ui('secLinks')}>
		<div class="links">
			{#each linkGroups as g (g.h)}
				<div>
					<h3>{g.h}</h3>
					<div class="cards">{#each g.cards as c (c.slug)}<CatalogCard card={c} compact />{/each}</div>
				</div>
			{/each}
		</div>
	</Section>
{/if}

<style>
	.etop {
		margin: 0 0 var(--s2);
	}
	.snav {
		position: sticky;
		top: calc(var(--ltop-h) + 8px);
		z-index: 4;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin: 0 0 var(--s2);
		padding: 8px 0;
		background: var(--paper);
	}
	.snav a {
		padding: 6px 12px;
		border: 1px solid var(--line-control);
		border-radius: var(--r-pill);
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		text-decoration: none;
	}
	.snav a:hover {
		border-color: var(--ink-2);
		color: var(--ink);
	}
	@media (max-width: 1099px) {
		.snav {
			position: static;
		}
	}
	.zw {
		scroll-margin-top: calc(var(--ltop-h) + 12px);
	}
	.cp {
		display: grid;
		gap: var(--s4);
	}
	.links {
		display: grid;
		gap: var(--s5);
	}
	.links h3 {
		font-size: var(--t-md);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
</style>
