<!--
  Home (redesign spec §6.3; mockup final/home.html): the hero with the answer-first preview of a real TinyCNN run, four
  lessons to start with, the learning paths as route lines (id="paths": the header's Paths link), the areas with their
  live/planned meters and the training methods. Everything shown comes from the build-time index (+page.server.ts).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '$lib/kit/Icon.svelte';
	import HeroPreview from '$lib/kit/catalog/HeroPreview.svelte';
	import LessonCard from '$lib/kit/catalog/LessonCard.svelte';
	import PathLine from '$lib/kit/catalog/PathLine.svelte';
	import AreaTile from '$lib/kit/catalog/AreaTile.svelte';
	import SectionChip from '$lib/kit/catalog/SectionChip.svelte';
	import { arrow, right, draw, type, points, game, graph } from '$lib/kit/catalog/glyphs';
	import { t, ui, i18n } from '$lib/i18n/i18n.svelte';
	import { ct } from '$lib/kit/catalog/text';
	import type { I18n } from '$lib/i18n/types';
	import type { InputBucket } from '$lib/kit/catalog/types';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const c = $derived(data.counts);
	const nameOf = (slug: string) => data.groups.find((g) => g.slug === slug)?.name;
	const areas = $derived(data.groups.filter((g) => g.kind === 'model'));
	const sections = $derived(data.groups.filter((g) => g.kind === 'concept'));
	const quiet = $derived(sections.filter((g) => !g.live));
	const first = $derived(data.start[0]);
	const under10 = $derived(data.start.length > 0 && data.start.every((l) => l.minutes < 10));
	const WORDS = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six'];
	const ZH = ['', '一', '两', '三', '四', '五', '六'];
	const nStart = $derived(data.start.length);

	const BUCKET: Record<InputBucket, { d: string; label: I18n }> = {
		draw: { d: draw, label: { en: 'Draw', zh: '画' } },
		type: { d: type, label: { en: 'Type', zh: '打字' } },
		points: { d: points, label: { en: 'Points and series', zh: '点与序列' } },
		play: { d: game, label: { en: 'Play', zh: '游戏' } },
		graphs: { d: graph, label: { en: 'Graphs and ratings', zh: '图与评分' } }
	};
</script>

<svelte:head>
	<title>Model Atlas · {t({ en: 'Follow every number', zh: '模型图鉴' })}</title>
</svelte:head>

<div class="container home">
	<section class="hero" aria-labelledby="hero-h">
		<div class="copy">
			<p class="kick">
				<span><b>{c.lessons}</b> {t({ en: 'step-by-step lessons', zh: '节分步课程' })}</span>
				<span><b>{c.models}</b> {t({ en: 'models', zh: '个模型' })}</span>
				<span><b>{c.methods}</b> {t({ en: 'training methods', zh: '种训练方法' })}</span>
			</p>
			<h1 id="hero-h">
				{#if i18n.lang === 'zh'}给模型你的输入，<span class="band">跟着每一个数走。</span>{:else}Give a model your input. <span class="band">Follow every number.</span>{/if}
			</h1>
			<p class="lede">
				{t({
					en: 'Short, guided lessons on how neural networks and their training work. The numbers on the screen come from real models, most of them computed in your browser from what you draw or type.',
					zh: '短小的分步课程，讲神经网络和它们的训练怎样工作。屏幕上的数都来自真实的模型，大多数就在你的浏览器里，由你画的、写的内容算出。'
				})}
			</p>
			<div class="cta">
				{#if first}
					<a class="btn btn-act" href="{base}{first.href}">
						{ct('startWith', { name: t(first.title) })}<small>· {ui('lessonSteps', { n: first.steps })}</small><Icon d={arrow} size={18} />
					</a>
				{/if}
				<a class="btn" href="{base}/atlas/">{ct('browseAll', { n: c.lessons })}</a>
			</div>
			<div class="inputs">
				<p id="by-input">{ct('byInput')}</p>
				<ul class="chips" aria-labelledby="by-input">
					{#each data.inputs as b (b.bucket)}
						{#if b.n}
							<li>
								<a class="chip" href="{base}/atlas/?input={b.bucket}"><Icon d={BUCKET[b.bucket].d} size={16} />{t(BUCKET[b.bucket].label)} <span class="c num">{b.n}</span></a>
							</li>
						{/if}
					{/each}
				</ul>
			</div>
		</div>
		<HeroPreview data={data.hero} />
	</section>

	<section class="block" aria-labelledby="start-h">
		<div class="sec-h">
			<h2 id="start-h">{ct('startHere')}</h2>
			<p>
				{t({ en: `${WORDS[nStart] ?? nStart} lessons, ${(WORDS[nStart] ?? String(nStart)).toLowerCase()} kinds of input.`, zh: `${ZH[nStart] ?? nStart}节课，${ZH[nStart] ?? nStart}种输入。` })}{#if under10}{' '}{t({ en: 'Each takes under ten minutes.', zh: '每节不到十分钟。' })}{/if}
			</p>
		</div>
		<ul class="start">
			{#each data.start as l (l.slug)}
				<li><LessonCard lesson={l} group={nameOf(l.group)} sprites={data.sprites} /></li>
			{/each}
		</ul>
	</section>

	<section class="block" id="paths" aria-labelledby="paths-h">
		<div class="sec-h">
			<h2 id="paths-h">{ct('paths')}</h2>
			<p>{t({ en: 'Lessons in a good order. Every lesson also stands on its own.', zh: '按合适的顺序排好的课程。每节课也都可以单独学。' })}</p>
		</div>
		<ol class="paths">
			{#each data.paths as p (p.id)}
				<li><PathLine path={p} /></li>
			{/each}
		</ol>
	</section>

	<section class="block" aria-labelledby="areas-h">
		<div class="sec-h">
			<h2 id="areas-h">{ct('byArea')}</h2>
			<a class="all" href="{base}/atlas/">{t({ en: `All ${c.lessons} lessons`, zh: `全部 ${c.lessons} 节课` })}<Icon d={right} size={16} /></a>
		</div>
		<ul class="areas">
			{#each areas as g (g.slug)}
				<li><AreaTile group={g} href="/atlas/{g.slug}/" /></li>
			{/each}
		</ul>
		<h3 class="sec-sub">{ct('methods')}</h3>
		<ul class="secs">
			{#each sections as g (g.slug)}
				<li class:quiet={!g.live}><SectionChip group={g} /></li>
			{/each}
		</ul>
		{#if quiet.length}
			<p class="more">{ct('planned')}{i18n.lang === 'zh' ? '：' : ': '}{quiet.map((g) => t(g.name)).join(' · ')}</p>
		{/if}
	</section>
</div>

<style>
	.home ul,
	.home ol {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.hero {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1.08fr);
		gap: 48px;
		align-items: center;
		padding: 48px 0 40px;
	}
	.kick {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		margin: 0 0 16px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		color: var(--muted);
	}
	.kick b {
		color: var(--ink);
		font-weight: var(--w-title);
		font-variant-numeric: tabular-nums;
	}
	h1 {
		margin: 0;
		font-size: var(--t-3xl);
		line-height: var(--lh-3xl);
		font-weight: var(--w-display);
		letter-spacing: -0.025em;
	}
	:global(:root:lang(zh-CN)) h1 {
		letter-spacing: 0;
		line-height: 1.3;
	}
	.band {
		padding: 0 4px;
		margin: 0 -2px;
		background: linear-gradient(transparent 12%, var(--mark-band) 12%, var(--mark-band) 92%, transparent 92%);
		color: var(--mark-band-ink);
		-webkit-box-decoration-break: clone;
		box-decoration-break: clone;
	}
	.lede {
		margin: 18px 0 0;
		max-width: 34em;
	}
	:global(:root:lang(zh-CN)) .lede {
		line-height: 1.8;
	}
	.cta {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
		align-items: center;
		margin-top: 28px;
	}
	.cta .btn {
		min-height: 52px;
		padding: 0 22px;
		border-radius: 14px;
	}
	.cta small {
		margin-left: -2px;
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		opacity: 0.78;
	}
	.inputs {
		margin-top: 28px;
	}
	.inputs p {
		margin: 0 0 10px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.chip {
		min-height: 36px;
	}
	.block {
		padding-top: 48px;
	}
	.sec-h {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 4px 16px;
		margin-bottom: 18px;
	}
	.sec-h h2 {
		margin: 0;
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
	}
	.sec-h p {
		margin: 0;
		color: var(--muted);
	}
	.all {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		color: var(--ink-2);
	}
	.start {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 16px;
	}
	.start li {
		display: flex;
		min-width: 0;
	}
	.start li > :global(*) {
		flex: 1;
	}
	.paths {
		border-top: 1px solid var(--rule);
	}
	.areas {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 12px;
	}
	.areas li {
		display: flex;
	}
	.areas li > :global(*) {
		flex: 1;
	}
	.sec-sub {
		margin: 28px 0 12px;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.secs {
		display: grid;
		grid-template-columns: repeat(5, minmax(0, 1fr));
		gap: 8px 12px;
	}
	.secs li {
		display: flex;
	}
	.secs li > :global(*) {
		flex: 1;
	}
	.more {
		display: none;
	}
	@media (max-width: 1099px) {
		.hero {
			grid-template-columns: minmax(0, 1fr);
			gap: 32px;
		}
		.start {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
		.areas {
			grid-template-columns: repeat(3, minmax(0, 1fr));
		}
		.secs {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	@media (max-width: 759px) {
		.hero {
			padding: 24px 0 8px;
			gap: 24px;
		}
		.cta .btn {
			flex: 1 1 100%;
		}
		.chips {
			gap: 6px;
		}
		.block {
			padding-top: 40px;
		}
		.sec-h h2 {
			font-size: var(--t-xl);
			line-height: var(--lh-xl);
		}
		.start {
			display: flex;
			gap: 12px;
			margin: 0 calc(-1 * var(--gutter));
			padding: 2px var(--gutter) 8px;
			overflow-x: auto;
			scroll-snap-type: x mandatory;
			scroll-padding-inline: var(--gutter);
		}
		.start li {
			flex: 0 0 78%;
			scroll-snap-align: start;
		}
		.areas {
			grid-template-columns: minmax(0, 1fr);
			gap: 0;
			border-top: 1px solid var(--rule);
		}
		.secs {
			display: flex;
			flex-wrap: wrap;
			gap: 8px;
		}
		.secs .quiet {
			display: none;
		}
		.more {
			display: block;
			margin: 10px 0 0;
			font-size: var(--t-sm);
			line-height: var(--lh-sm);
			color: var(--muted);
		}
	}
</style>
