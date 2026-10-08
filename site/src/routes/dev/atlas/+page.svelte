<!--
  /dev/atlas/ — the DevKit view gallery (W1-D2).
  1. Every view: each src/lib/atlas/ui/views/<key>/ (and each package view src/lib/explainers/<slug>/views/<key>/,
     shown as '<slug>/<key>') rendered from its meta.demo() and, when the meta exports variants(), its other demos.
     Demos are tiny models with made-up weights run by the real kit ops, so they are badged Illustrative; the text
     under each stage is the view's describe() (what a screen reader hears).
  2. On real models: the zoom views on TinyCNN and TinyGPT, loaded from static/models/ and run in this page by the kit
     executor; each stage shows a real UnitExplanation from explain(). Deep links: ?model=tinygpt&layer=L0.ln1&input=1
  All atlas code comes from ONE lazy module (ui/views/kit.parts/gallery.ts), imported in onMount: the prerendered page
  is only this shell, and the gallery adds as few chunk splits as possible to the pages that share kit components.
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	import { base } from '$app/paths';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import type { I18n, StageProps, ViewMeta } from '$lib/atlas/types/index.ts';

	type G = typeof import('$lib/atlas/ui/views/kit.parts/gallery.ts');
	type Stage = Omit<StageProps, 'ctx'>;

	interface Entry {
		key: string;
		meta: ViewMeta;
		comp: Component<StageProps> | null;
		stages: { label: I18n; stage: Stage }[];
		error?: string;
	}
	let g: G | null = null;
	let entries = $state.raw<Entry[]>([]);
	let loading = $state(true);
	let picked = $state<Record<string, number>>({});
	let note = $state<I18n | null>(null);
	let ViewStage = $state.raw<G['ViewStage'] | null>(null);
	let lesson = $state.raw<{ comp: G['LessonDemo']; items: ReturnType<G['lessonItems']> } | null>(null);

	async function loadGallery() {
		g = await import('$lib/atlas/ui/views/kit.parts/gallery.ts');
		note = g.DEMO_NOTE;
		ViewStage = g.ViewStage;
		const out: Entry[] = [];
		for (const v of g.allViews()) {
			try {
				const stages: { label: I18n; stage: Stage }[] = [{ label: { en: 'Demo', zh: '演示' }, stage: await v.mod.default.demo() }];
				if (v.mod.variants) for (const x of await v.mod.variants()) stages.push(x);
				out.push({ key: v.key, meta: v.mod.default, comp: v.comp, stages });
			} catch (e) {
				out.push({ key: v.key, meta: v.mod.default, comp: v.comp, stages: [], error: e instanceof Error ? e.message : String(e) });
			}
		}
		entries = out;
		try {
			lesson = { comp: g.LessonDemo, items: g.lessonItems() };
		} catch {
			lesson = null;
		}
		loading = false;
	}

	const ctx = $derived({ shown: true, reduced: motion.reduced, playing: false, done: () => {} });
	const pickOf = (key: string) => picked[key] ?? 0;
	const anchor = (key: string) => `v-${key.replace(/[^\w-]/g, '-')}`;

	// ------------------------------------------------ real models (TinyCNN, TinyGPT), run live in this page
	type RealName = 'tinycnn' | 'tinygpt';
	type RealModel = Awaited<ReturnType<G['loadReal']>>;
	type RealRun = ReturnType<RealModel['run']>;
	let realName = $state<RealName>('tinycnn');
	let realInput = $state(0);
	let realLayer = $state('conv1_2');
	let realStatus = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	let realError = $state('');
	let realModel = $state.raw<RealModel | null>(null);
	let realRun = $state.raw<RealRun | null>(null);
	let realStage = $state.raw<Stage | null>(null);
	let realComp = $state.raw<Component<StageProps> | null>(null);
	let realKey = $state('');
	/** ?view=<key> shows that view instead of the op's own zoom (e.g. zoom.formula or layer.tensor on any layer) */
	let forcedView = $state<string | null>(null);
	const realCache: Partial<Record<RealName, RealModel>> = {};

	const realLayers = $derived(realModel ? realModel.model.spec.layers : []);
	/** which view each layer of the loaded model uses: one chip per view, jumping to its first layer */
	const coverage = $derived.by(() => {
		if (!realModel || !g) return [];
		const m = new Map<string, string[]>();
		for (const l of realModel.model.spec.layers) {
			const k = g.zoomKey(realModel.model, l);
			(m.get(k) ?? m.set(k, []).get(k)!).push(l.id);
		}
		return [...m].map(([key, ids]) => ({ key, ids }));
	});
	const zoomLabel = (id: string) => {
		const l = realModel?.model.spec.layers.find((x) => x.id === id);
		return l && g && realModel ? g.zoomKey(realModel.model, l) : '';
	};

	async function runReal() {
		if (!g) return;
		realStatus = 'loading';
		try {
			const m = (realCache[realName] ??= await g.loadReal(fetch, base, realName));
			realModel = m;
			realRun = m.run(realInput);
			if (!m.model.spec.layers.some((l) => l.id === realLayer)) realLayer = m.model.spec.layers.find((l) => l.op !== 'input')?.id ?? m.model.spec.layers[0].id;
			showReal();
			realStatus = 'ready';
		} catch (e) {
			realStatus = 'error';
			realError = e instanceof Error ? e.message : String(e);
		}
	}

	function showReal() {
		if (!g || !realRun || !realModel) return;
		const layer = realModel.model.spec.layers.find((l) => l.id === realLayer);
		if (!layer) return;
		realKey = forcedView ?? g.zoomKey(realModel.model, layer);
		realComp = g.viewFor(realKey);
		realStage = g.realStage(realRun, layer.id);
	}

	onMount(() => {
		const q = new URLSearchParams(location.search);
		const m = q.get('model');
		if (m === 'tinycnn' || m === 'tinygpt') realName = m;
		const l = q.get('layer');
		if (l) realLayer = l;
		else if (realName === 'tinygpt') realLayer = 'L0.attn.scores';
		forcedView = q.get('view');
		const i = Number(q.get('input'));
		if (Number.isInteger(i) && i >= 0) realInput = i;
		void loadGallery().then(runReal);
	});
</script>

<svelte:head><title>DevKit views · Model Atlas</title></svelte:head>

<div class="container gal">
	<header class="head">
		<p class="eyebrow">/dev/atlas/</p>
		<h1>{t({ en: 'DevKit views', zh: 'DevKit 视图' })}</h1>
		<p class="lede">
			{t({
				en: 'Every view a lesson step can show, with its demo. The demos are tiny models with made-up weights, but every number in them is computed by the real kit ops. At the bottom, the zoom views run on TinyCNN and TinyGPT in your browser.',
				zh: '课程中每一步可以显示的所有视图及其演示。演示用的是权重随意设定的小模型，但其中每个数都由真实的工具包算子计算。页面底部，缩放视图在你的浏览器里运行 TinyCNN 和 TinyGPT。'
			})}
		</p>
		<p class="muted small">
			{t({ en: 'Component gallery of the site kit:', zh: '网站组件库：' })} <a href="{base}/dev/kit/">/dev/kit/</a>
		</p>
		{#if entries.length}
			<nav class="toc" aria-label={t({ en: 'Views', zh: '视图' })}>
				{#each entries as en (en.key)}<a href="#{anchor(en.key)}">{en.key}</a>{/each}
				<a href="#in-a-lesson">{t({ en: 'in a lesson', zh: '课程中' })}</a>
				<a href="#real">{t({ en: 'real models', zh: '真实模型' })}</a>
			</nav>
		{/if}
	</header>

	{#if loading}
		<p class="muted" aria-live="polite">{t({ en: 'Loading the views…', zh: '正在加载视图…' })}</p>
	{/if}

	{#each entries as en (en.key)}
		{@const vi = Math.min(pickOf(en.key), Math.max(0, en.stages.length - 1))}
		{@const st = en.stages[vi]?.stage}
		<section id={anchor(en.key)} class="demo">
			<h2>
				<span class="key">{en.key}</span>
				<span class="ttl">{t(en.meta.title)}</span>
				<ProvenanceBadge mode="illustrative" compact note={note ?? undefined} />
			</h2>
			<p class="facts num">
				{t({ en: 'kinds', zh: '类型' })}: {en.meta.kinds.join(', ') || '—'}
				· minHeight {en.meta.minHeight ?? '—'}
				{#if !en.comp}· <strong class="bad">{t({ en: 'no View.svelte', zh: '缺少 View.svelte' })}</strong>{/if}
			</p>
			{#if en.error}
				<p class="bad">{en.error}</p>
			{/if}
			{#if en.stages.length > 1}
				<div class="segmented variants" role="group" aria-label={t({ en: 'Demo', zh: '演示' })}>
					{#each en.stages as s, i (i)}
						<button type="button" aria-pressed={i === vi} onclick={() => (picked = { ...picked, [en.key]: i })}>{t(s.label)}</button>
					{/each}
				</div>
			{/if}
			{#if st && en.comp && ViewStage}
				<div class="stage" style="min-height: {en.meta.minHeight ?? 0}px">
					{#each [`${en.key}:${vi}`] as k (k)}
						<ViewStage view={en.comp} stage={st} {ctx} />
					{/each}
				</div>
				<details class="sum">
					<summary>{t({ en: 'Text summary (what a screen reader hears)', zh: '文字摘要（读屏软件读出的内容）' })}</summary>
					<p>{t(en.meta.describe({ ...st, ctx }))}</p>
				</details>
			{/if}
		</section>
	{/each}

	{#if lesson}
		{@const Lesson = lesson.comp}
		<section id="in-a-lesson" class="demo">
			<h2>
				<span class="key">{t({ en: 'In a lesson stage', zh: '在课程舞台里' })}</span>
				<ProvenanceBadge mode="illustrative" compact note={note ?? undefined} />
			</h2>
			<p class="desc">
				{t({
					en: 'Four of the views inside the kit Walkthrough, as a lesson shows them. The first step plays its slide on its own and tells the lesson when it is done; the narration is each view’s own text summary.',
					zh: '四个视图放在工具包的 Walkthrough 里，就像课程中显示的那样。第一步会自己播放滑动，结束时通知课程；旁白是每个视图自己的文字摘要。'
				})}
			</p>
			<Lesson items={lesson.items} />
		</section>
	{/if}

	<section id="real" class="demo">
		<h2>
			<span class="key">{t({ en: 'On real models', zh: '在真实模型上' })}</span>
			{#if realStatus === 'ready'}<ProvenanceBadge mode="live" compact precision="fp32" />{/if}
		</h2>
		<p class="desc">
			{t({
				en: 'TinyCNN and TinyGPT, loaded from their shipped files and run here by the kit executor (bit for bit the numbers of their own pages). Pick a layer: the stage is the zoom view its op asks for, on a real unit explanation.',
				zh: 'TinyCNN 和 TinyGPT 从它们发布的文件加载，并在这里由工具包执行器运行（与各自页面的数字逐位相同）。选一个层：舞台显示该层算子对应的缩放视图，使用真实的单元解释。'
			})}
		</p>
		<div class="controls">
			<div class="segmented" role="group" aria-label={t({ en: 'Model', zh: '模型' })}>
				<button type="button" aria-pressed={realName === 'tinycnn'} onclick={() => { realName = 'tinycnn'; realLayer = 'conv1_2'; realInput = 0; runReal(); }}>TinyCNN</button>
				<button type="button" aria-pressed={realName === 'tinygpt'} onclick={() => { realName = 'tinygpt'; realLayer = 'L0.attn.scores'; realInput = 0; runReal(); }}>TinyGPT</button>
			</div>
			{#if realModel}
				<label class="pick">
					<span>{t({ en: 'Input', zh: '输入' })}</span>
					<select bind:value={realInput} onchange={runReal}>
						{#each realModel.inputs as inp, i (inp.id)}<option value={i}>{t(inp.label)}</option>{/each}
					</select>
				</label>
				<label class="pick">
					<span>{t({ en: 'Layer', zh: '层' })}</span>
					<select bind:value={realLayer} onchange={() => { forcedView = null; showReal(); }}>
						{#each realLayers as l (l.id)}<option value={l.id}>{l.id} · {l.op} → {zoomLabel(l.id)}</option>{/each}
					</select>
				</label>
			{/if}
		</div>
		{#if coverage.length}
			<div class="cov" role="group" aria-label={t({ en: 'Views this model uses', zh: '这个模型用到的视图' })}>
				{#each coverage as c (c.key)}
					<button type="button" class="chip" aria-pressed={c.key === realKey} onclick={() => { forcedView = null; realLayer = c.ids[0]; showReal(); }}>
						<span class="num">{c.key}</span> <span class="muted">× {c.ids.length}</span>
					</button>
				{/each}
			</div>
		{/if}
		{#if realStatus === 'loading'}
			<p class="muted" aria-live="polite">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
		{:else if realStatus === 'error'}
			<p class="bad">{realError}</p>
		{/if}
		{#if realStage && realComp && ViewStage}
			<p class="facts num">
				{t({ en: 'view', zh: '视图' })}: {realKey}
				{#each ['zoom.formula', 'layer.tensor'] as alt (alt)}
					{#if alt !== realKey}· <button type="button" class="lnk" onclick={() => { forcedView = alt; showReal(); }}>{t({ en: `show as ${alt}`, zh: `用 ${alt} 显示` })}</button>{/if}
				{/each}
			</p>
			<div class="stage">
				{#each [`${realName}:${realInput}:${realLayer}`] as k (k)}
					<ViewStage view={realComp} stage={realStage} {ctx} />
				{/each}
			</div>
		{:else if realStatus === 'ready'}
			<p class="muted">{t({ en: `No view registered for ${realKey}.`, zh: `${realKey} 没有对应的视图。` })}</p>
		{/if}
	</section>
	<p class="muted small foot">{ui('navKit')}: <a href="{base}/dev/kit/">/dev/kit/</a></p>
</div>

<style>
	.gal {
		padding-top: var(--s6);
		padding-bottom: var(--s7);
	}
	.head {
		display: grid;
		gap: 8px;
		margin-bottom: var(--s5);
	}
	.head h1,
	.head p {
		margin: 0;
	}
	.small {
		font-size: var(--fs-sm);
	}
	.toc {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
		font-family: var(--font-mono);
	}
	.demo {
		padding: var(--s5) 0;
		border-top: 1px solid var(--rule);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
		scroll-margin-top: calc(var(--header-h) + 12px);
		min-width: 0;
	}
	.demo h2 {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 12px;
		font-size: var(--fs-lg);
		font-weight: 600;
	}
	.key {
		font-family: var(--font-mono);
	}
	.ttl {
		font-family: var(--font-body);
		font-weight: 400;
		color: var(--ink-2);
		font-size: var(--fs-md);
	}
	.facts {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.desc {
		margin: 0;
		color: var(--ink-2);
		max-width: var(--measure);
	}
	.segmented {
		flex-wrap: wrap;
	}
	.stage {
		min-width: 0;
		padding: var(--s4);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	@media (max-width: 560px) {
		.stage {
			padding: 10px;
		}
	}
	.sum {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.sum p {
		margin: 6px 0 0;
		max-width: var(--measure);
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px 16px;
	}
	.pick {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-sm);
		min-width: 0;
		max-width: 100%;
	}
	.pick select {
		min-width: 0;
		max-width: min(70vw, 420px);
		font: inherit;
		font-size: var(--fs-sm);
		padding: 4px 6px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		background: var(--surface);
		color: var(--ink);
	}
	.cov {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.cov .chip {
		cursor: pointer;
		font-size: var(--fs-xs);
	}
	.cov .chip[aria-pressed='true'] {
		border-color: var(--accent);
		box-shadow: 0 0 0 1.5px var(--accent);
	}
	.lnk {
		border: 0;
		background: none;
		padding: 0;
		font: inherit;
		color: var(--accent-strong);
		text-decoration: underline;
		cursor: pointer;
	}
	.bad {
		color: var(--div-neg-text);
		margin: 0;
	}
	.foot {
		margin-top: var(--s5);
	}
</style>
