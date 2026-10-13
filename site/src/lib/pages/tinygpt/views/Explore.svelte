<!--
  TinyGPT's Explore view (redesign spec §5.9), loaded only when the view opens: the whole model at once on the learner's
  text (the page's one TGPage): the overview and the parameters, inside a layer (the 66 traced tensors), inside a block
  (the block player and all 16 attention heads), the output (the logit lens, probabilities and sampling), try this, check
  yourself and the links. Each section has an anchor (#explore-overview …, #explore-attention, #explore-lens; the old
  #overview, #inspector, #unit, #output, #tryThis and #checkpoint open them). The selection is the page's (S), never the
  lesson's (LS).
-->
<script lang="ts">
	import { tick, untrack } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Section from '$lib/kit/Section.svelte';
	import CatalogCard from '$lib/kit/CatalogCard.svelte';
	import PredictReveal from '$lib/kit/PredictReveal.svelte';
	import { getLessonApp } from '$lib/kit/lesson/app';
	import { motion } from '$lib/kit/theme.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { ModelPageCatalog } from '$lib/catalog/types';
	import type { ModelSpec } from '$lib/kit/spec';
	import type { PredictQuestion } from '$lib/kit/types';
	import Overview from '../Overview.svelte';
	import ParamTable from '../ParamTable.svelte';
	import InspectorPanel from '../InspectorPanel.svelte';
	import BlockView from '../BlockView.svelte';
	import AttnExplorer from '../AttnExplorer.svelte';
	import LogitLens from '../LogitLens.svelte';
	import Sampling from '../Sampling.svelte';
	import TryThis from '../TryThis.svelte';
	import Checkpoint from '../Checkpoint.svelte';
	import InputChip from '../InputChip.svelte';
	import type { TGPage } from '../page.svelte';

	let { vp, section }: { vp: { P: TGPage; prequestion?: PredictQuestion }; section?: string } = $props();
	const P = $derived(vp.P);
	const S = $derived(vp.P.S);
	const app = getLessonApp();

	// a deep link (#output, #explore-lens …): the view host scrolled there on open, but the model's views above still grow
	// once it has loaded; scroll to the section again then (once per section)
	let scrolledFor: string | undefined;
	$effect(() => {
		const s = section;
		const ready = P.loadState === 'ready' && !!P.spec && S.T > 0;
		if (!s || !ready || scrolledFor === s) return;
		scrolledFor = s;
		untrack(() => tick().then(() => requestAnimationFrame(() => (document.getElementById(`explore-${s}`) ?? document.getElementById(s))?.scrollIntoView({ block: 'start' }))));
	});

	/** scroll to a part of this view (Try this, the overview's blocks); `input` opens the input sheet */
	async function goto(id: string) {
		if (id === 'input') {
			if (app) app.input.open = true;
			return;
		}
		await tick(); // let the views render the new run first, so the target does not move
		const el = document.getElementById(`explore-${id}`) ?? document.getElementById(id);
		if (!el) return;
		el.scrollIntoView({ behavior: motion.reduced ? 'auto' : 'smooth', block: 'start' });
		el.classList.remove('flash');
		void el.offsetWidth;
		el.classList.add('flash');
		setTimeout(() => el.classList.remove('flash'), 1800);
	}
	function openBlock(b: number) {
		S.block = b;
		void goto('unit');
	}

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
			['overview', ui('secOverview')],
			['inspector', ui('secInspector')],
			['unit', t({ en: 'Inside a block', zh: '块的内部' })],
			['attention', t({ en: 'All 16 heads', zh: '全部 16 个头' })],
			['output', ui('secOutput')],
			['tryThis', ui('secTryThis')],
			['checkpoint', ui('secCheckpoint')],
			['links', ui('secLinks')]
		] as [string, string][]
	);
</script>

<!-- token text: no programming ligatures ("<|" would turn into a triangle in JetBrains Mono) -->
<div class="tg">
<div class="etop"><InputChip {P} open={() => goto('input')} /></div>

<nav class="snav" aria-label={t({ en: 'Sections', zh: '本页分节' })}>
	{#each sections as s (s[0])}<a href="#explore-{s[0]}">{s[1]}</a>{/each}
</nav>

{#if P.loadState === 'error'}
	<p class="err" role="alert">{t({ en: 'The model could not be loaded:', zh: '模型加载失败：' })} {P.loadError}</p>
{/if}

<Section
	id="explore-overview"
	title={ui('secOverview')}
	intro={{ en: 'The whole model at a glance, drawn from your text. Each thumbnail is the real tensor (rows = your tokens); the outlined row is the token you are following.', zh: '一眼看完整个模型，全部由你的文本画出。每张缩略图都是真实的张量（行 = 你的词元）；带框的行是你正在跟踪的词元。' }}
	provenance="live"
>
	<Overview {S} onopen={openBlock} />
	{#if P.spec}
		<details class="params">
			<summary>{t({ en: 'Where the 364,040 parameters are', zh: '364,040 个参数分布在哪里' })}</summary>
			<ParamTable spec={P.spec} />
		</details>
	{/if}
</Section>

<Section
	id="explore-inspector"
	title={ui('secInspector')}
	intro={{ en: 'Any of the 66 tensors the model computed, with values, statistics, the arithmetic and the code. Use [ and ] to step through them.', zh: '模型算出的 66 个张量中的任意一个：数值、统计、算式和代码。用 [ 和 ] 键逐个浏览。' }}
	provenance="live"
>
	{#if P.spec}
		<InspectorPanel {S} spec={P.spec as unknown as ModelSpec} />
	{:else}
		<p class="muted">{t({ en: 'Loading…', zh: '加载中…' })}</p>
	{/if}
</Section>

<Section
	id="explore-unit"
	title={{ en: 'Inside a block', zh: '块的内部' }}
	intro={{ en: 'Zoom into one block: play the steps, then explore all 16 attention heads and redo one attention score by hand.', zh: '放大一个块：播放各个步骤，再查看全部 16 个注意力头，并亲手重算一个注意力分数。' }}
	provenance="live"
>
	<BlockView {S} bind:step={P.blockStep} />
	<div id="explore-attention" class="sub">
		<h3>{t({ en: 'All 16 attention heads', zh: '全部 16 个注意力头' })}</h3>
		<p class="muted">
			{t({
				en: 'Rows are blocks, columns are heads. Click a map to zoom in; hover the text to see one token’s attention row; click a cell of the big map to pick a query and a key for the arithmetic.',
				zh: '行是块，列是头。点击小图可放大；在文本上悬停可看一个词元的注意力行；点击大图中的格子可为算式选择查询和键。'
			})}
		</p>
		<AttnExplorer {S} bind:metric={P.metric} />
	</div>
</Section>

<Section id="explore-output" title={{ en: 'Output: logit lens and sampling', zh: '输出：logit lens 与采样' }} intro={{ en: 'When does the model decide, and how does it pick the next token?', zh: '模型在什么时候做出决定？它又是怎样挑出下一个词元的？' }} provenance="live">
	<div id="explore-lens" class="sub first">
		<h3>{t({ en: 'Logit lens: when does it decide?', zh: 'Logit lens：它在什么时候做出决定？' })}</h3>
		<LogitLens {S} />
	</div>
	<div class="sub">
		<h3>{t({ en: 'Next token: probabilities and sampling', zh: '下一个词元：概率与采样' })}</h3>
		<Sampling
			{S}
			dist={P.dist}
			bind:temperature={P.temperature}
			bind:topK={P.topK}
			bind:topP={P.topP}
			seed={P.seed}
			draws={P.draws}
			pick={P.pick}
			auto={P.auto}
			busy={P.busy}
			promptText={P.text}
			genIds={P.genIds}
			ongenerate={() => void P.genStep()}
			onauto={() => void P.autoGenerate()}
			onstop={() => (P.auto = false)}
			onreset={P.removeGenerated}
			onkeep={P.keepAsPrompt}
			onseed={P.onSeed}
		/>
	</div>
</Section>

<Section id="explore-tryThis" title={ui('secTryThis')} provenance="live">
	<TryThis {S} load={(v) => P.setPrompt(v, true)} goto={(id) => void goto(id)} setMetric={(m) => (P.metric = m)} compare={P.compare} seed={P.seed} promptText={P.text} />
</Section>

<Section id="explore-checkpoint" title={ui('secCheckpoint')}>
	<div class="cp">
		{#if vp.prequestion}<PredictReveal q={vp.prequestion} />{/if}
		<Checkpoint />
	</div>
</Section>

<Section id="explore-links" title={ui('secLinks')}>
	{#if linkGroups.length}
		<div class="links">
			{#each linkGroups as g (g.h)}
				<div>
					<h3>{g.h}</h3>
					<div class="cards">{#each g.cards as c (c.slug)}<CatalogCard card={c} compact />{/each}</div>
				</div>
			{/each}
		</div>
	{/if}
	<div class="ext">
		<h3>{t({ en: 'Elsewhere', zh: '站外资源' })}</h3>
		<ul>
			<li><a href="https://poloclub.github.io/transformer-explainer/" rel="noopener">Transformer Explainer</a> — {t({ en: 'the same steps on GPT-2 small (124M parameters), in your browser', zh: '在浏览器里演示 GPT-2 small（1.24 亿参数）的同样步骤' })}</li>
			<li><a href="https://bbycroft.net/llm" rel="noopener">LLM Visualization</a> — {t({ en: 'a 3D walk through every weight of a GPT', zh: '用 3D 走一遍 GPT 的每一个权重' })}</li>
			<li><a href="https://github.com/jessevig/bertviz" rel="noopener">BertViz</a> — {t({ en: 'the layers × heads attention grid, for big models', zh: '大模型的“层 × 头”注意力网格' })}</li>
			<li><a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens" rel="noopener">interpreting GPT: the logit lens</a> — {t({ en: 'where the logit lens comes from (nostalgebraist, 2020)', zh: 'logit lens 的出处（nostalgebraist，2020）' })}</li>
			<li><a href="https://arxiv.org/abs/1902.10186" rel="noopener">Attention is not Explanation</a> — {t({ en: 'Jain & Wallace (2019), on the misconception in Check yourself', zh: 'Jain 与 Wallace（2019），关于“检查点”里提到的误区' })}</li>
		</ul>
		<p>
			<a href="{base}/train/tokenization/">{t({ en: 'Tokenization', zh: '分词' })}</a> ·
			<a href="{base}/train/next-token-prediction/">{t({ en: 'Next-token prediction', zh: '下一个词元预测' })}</a> ·
			<a href="{base}/m/tinycnn/">TinyCNN</a>
		</p>
	</div>
</Section>

</div>

<style>
	.tg {
		font-variant-ligatures: none;
	}
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
	.err {
		color: var(--no);
		font-weight: var(--w-ui);
	}
	.params {
		margin-top: 14px;
	}
	.params summary,
	.sub h3 {
		font-size: var(--t-md);
	}
	.params summary {
		margin-bottom: 8px;
		font-weight: var(--w-ui);
		cursor: pointer;
	}
	.sub {
		margin-top: var(--s6);
		scroll-margin-top: calc(var(--ltop-h) + 12px);
	}
	.sub.first {
		margin-top: 0;
	}
	.sub > .muted {
		max-width: var(--measure);
		font-size: var(--t-sm);
	}
	.cp {
		display: grid;
		gap: var(--s4);
	}
	.links {
		display: grid;
		gap: var(--s5);
		margin-bottom: var(--s5);
	}
	.links h3,
	.ext h3 {
		font-size: var(--t-md);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
	.ext ul {
		padding-left: 1.2em;
	}
	:global(.flash) {
		animation: flash 1.8s ease-out;
		border-radius: var(--r3);
	}
	@keyframes flash {
		0%,
		40% {
			box-shadow: 0 0 0 3px var(--mark);
		}
		100% {
			box-shadow: 0 0 0 3px transparent;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		:global(.flash) {
			animation: none;
		}
	}
</style>
