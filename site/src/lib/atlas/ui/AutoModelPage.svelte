<!--
  AutoModelPage (design §5.1; redesign W2): a whole model page generated from a package (or, in the Lesson Lab, from a
  spec alone), as the lesson app (redesign spec §5). It fills the kit ModelPageShell (catalog slugs) or a plain lab
  layout (devOnly packages):
    Lesson     the guided lesson (LessonPlayer), with the cover (the answer-first strip of a package with a spec, else the
               first step's stage), the input chip in the stage header and the input sheet (the adapter's Capture and
               what the model sees)
    Explore    panels/AutoExplore (overview, inspector, the arithmetic of one unit, output, try this, check yourself, links)
    Article    panels/AutoArticle (explainers/<slug>/content/En.svelte, Zh.svelte)
    Code       panels/AutoCode (the forward pass, the parity check)
    Training   panels/AutoTraining (the run or the training record, the credits)
  The views are lazy panels: they load when their tab opens, with the section overrides they need
  (explainers/<slug>/sections/<Name>.svelte, props { store, pkg, spec }). Input and Lesson overrides load with the page.
  This component, the view registry and KaTeX live in the lazy chunk that +page.ts returns: planned pages never load them.
  The runtime starts in onMount (KIT rule 4).
  K0: without a spec (kinds B and C) the model sections are left out unless the package overrides them;
  manifest.page.sections titles, introduces or hides any section.
  K1: the KaTeX web fonts are fetched once the page is idle, so the first formula of the lesson does not swap fonts.
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	// the formula views' stylesheet, loaded with the page (see the header)
	import 'katex/dist/katex.min.css';
	import { base } from '$app/paths';
	import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
	import Section from '$lib/kit/Section.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import Icon from '$lib/kit/Icon.svelte';
	import { pencil } from '$lib/kit/icons';
	import { page as appPage } from '$app/state';
	import type { ModelPageCatalog } from '$lib/catalog/types';
	import { lui, t, ui } from '$lib/i18n/i18n.svelte';
	import type { WalkContext } from '$lib/kit/types';
	import type { AtlasSpec } from '../types/spec.ts';
	import type { PackageModules } from '../types/runtime.ts';
	import type { Tensor } from '../types/core.ts';
	import { normalizeSpec } from '../graph/legacy.ts';
	import { compileLesson, type CompiledLesson } from '../lesson/compile.ts';
	import { primaryTrace, outputId } from '../lesson/facts.ts';
	import { sourcesOf } from '../lesson/define.ts';
	import { fmt3, fmtPct } from '../core/format.ts';
	import { AtlasStore, openAtlas } from './runtime.svelte.ts';
	import { hasArticle, loadCapture, loadPackage, loadSection, sectionOverrides, type SectionName } from './packages.ts';
	import { hasView } from './registry.ts';
	import { toKitSpec } from './kitspec.ts';
	import { isAnswerLayer, topAnswers } from './fallback.ts';
	import LessonPlayer from './LessonPlayer.svelte';
	import CoverStrip from './panels/CoverStrip.svelte';
	import { STR, say } from './strings.ts';

	interface Props {
		slug: string;
		/** spec.json as embedded by the route's server load (raw; normalized here) */
		spec?: unknown;
		/** measured bytes of manifest.assets.auto (build time) */
		autoBytes?: number;
		/** an open runtime and its package (the Lesson Lab passes both) */
		store?: AtlasStore | null;
		pkg?: PackageModules | null;
		lab?: boolean;
		/** the lesson's current step (bindable) */
		lessonIndex?: number;
	}
	let { slug, spec: rawSpec, autoBytes, store: given = null, pkg: givenPkg = null, lab = false, lessonIndex = $bindable(0) }: Props = $props();

	let own = $state.raw<{ store: AtlasStore; pkg: PackageModules } | null>(null);
	let failed = $state('');
	const store = $derived(given ?? own?.store ?? null);
	const pkg = $derived(givenPkg ?? own?.pkg ?? null);
	const embedded = $derived.by((): AtlasSpec | null => {
		try {
			return rawSpec ? normalizeSpec(rawSpec) : null;
		} catch {
			return null;
		}
	});
	const rt = $derived(store?.r ?? null);
	const spec = $derived(rt?.spec ?? embedded);
	const manifest = $derived(pkg?.manifest ?? null);
	/** the catalog entry (page.data.catalog: the root layout's build-time slice), absent for a devOnly package */
	const model = $derived.by(() => {
		const c = (appPage.data as { catalog?: ModelPageCatalog | null }).catalog;
		return c && c.model.slug === slug ? c.model : undefined;
	});

	/** kit views render after mount only: the prerendered HTML has no markup whose CSS is not linked (autoentry.ts) */
	let mounted = $state(false);
	let Capture = $state<Component<any> | null>(null);
	let overrides = $state.raw<Partial<Record<SectionName, Component<any>>>>({});

	/** K1: fetch KaTeX's web fonts while the page is idle, before the lesson's first formula needs them */
	function preloadFormulaFonts() {
		const fonts = (document as Document & { fonts?: FontFaceSet }).fonts;
		if (!fonts?.load) return;
		const go = () => {
			for (const f of ['1em KaTeX_Main', 'italic 1em KaTeX_Main', 'bold 1em KaTeX_Main', 'italic 1em KaTeX_Math', '1em KaTeX_Size1', '1em KaTeX_Size2', '1em KaTeX_AMS'])
				fonts.load(f).catch(() => undefined);
		};
		const idle = (globalThis as { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
		if (idle) idle(go, { timeout: 2000 });
		else setTimeout(go, 300);
	}

	onMount(() => {
		mounted = true;
		preloadFormulaFonts();
		let alive = true;
		(async () => {
			try {
				let p = givenPkg;
				if (!given) {
					const loaded = await loadPackage(slug);
					if (!loaded) throw new Error(`no explainer package '${slug}'`);
					p = loaded.pkg;
					const s = await openAtlas(p, { base });
					if (!alive) return s.close();
					own = { store: s, pkg: p };
				}
				if (!p) return;
				const id = p.manifest.input?.adapter;
				const have = new Set(sectionOverrides(slug));
				const [cap, ...ovs] = await Promise.all([id ? loadCapture(slug, id) : null, ...(['Input', 'Lesson', 'Facts'] as const).map((n) => (have.has(n) ? loadSection(slug, n) : null))]);
				if (!alive) return;
				Capture = cap;
				overrides = { Input: ovs[0] ?? undefined, Lesson: ovs[1] ?? undefined, Facts: ovs[2] ?? undefined };
			} catch (e) {
				failed = e instanceof Error ? e.message : String(e);
			}
		})();
		return () => {
			alive = false;
			own?.store.close();
		};
	});

	// ------------------------------------------------------------------ derived page data
	const lesson = $derived.by((): CompiledLesson | null => {
		if (!store || !pkg) return null;
		return compileLesson(pkg.lesson, { spec: store.rt.spec, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes, views: hasView });
	});
	const trace = $derived.by(() => {
		void store?.version;
		return rt ? primaryTrace({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel }) : null;
	});
	// K1: a page without a spec (kinds B and C) computes in JavaScript numbers (f64): the facts say so, not "Precision —"
	const kitSpec = $derived(spec ? toKitSpec(spec, (trace?.meta?.dims ?? {}) as Record<string, number>) : manifest ? { slug, precision: 'f64' } : null);
	const pre = $derived.by(() => {
		void store?.version;
		return rt?.pre ?? null;
	});
	const presets = $derived(rt?.presets ?? []);
	/** the model's current top answer (class name, or token text for a language model) for the input chip */
	const outTop = $derived.by(() => {
		void store?.version;
		const tr = trace;
		const id = spec && tr ? outputId(spec, tr) : undefined;
		const o = id ? tr?.t[id] : undefined;
		if (!rt || !o || !isAnswerLayer(rt, id, tr)) return null;
		return topAnswers(rt, o, 1)[0] ?? null;
	});
	const input = $derived.by(() => {
		void store?.version;
		return rt?.input;
	});
	function onInput(raw: unknown, o?: { live?: boolean }) {
		if (!store) return;
		if (o?.live) store.setInputSoon(raw);
		else store.setInput(raw);
	}

	// ------------------------------------------------------------------ which views (K0)
	const secCfg = $derived(manifest?.page.sections ?? {});
	const hidden = (k: string) => !!secCfg.hide?.includes(k as never);
	const ovNames = $derived(new Set<SectionName>(sectionOverrides(slug)));
	const modelSec = (k: string, o: SectionName) => !hidden(k) && (ovNames.has(o) || !!spec);
	const trainSource = $derived(manifest ? (Object.entries(sourcesOf(manifest)).find(([, d]) => d.kind === 'train')?.[0] ?? null) : null);
	const views = $derived({
		explore:
			modelSec('overview', 'Overview') || modelSec('inspector', 'Inspector') || modelSec('unit', 'Unit') || modelSec('output', 'Output') || manifest?.page.tryThis?.length || manifest?.page.quiz?.length || manifest?.page.misconception || ovNames.has('TryThis') || ovNames.has('Checkpoint')
				? () => import('./panels/AutoExplore.svelte')
				: undefined,
		article: !hidden('article') && (hasArticle(slug) || ovNames.has('Article')) ? () => import('./panels/AutoArticle.svelte') : undefined,
		code: modelSec('code', 'Code') ? () => import('./panels/AutoCode.svelte') : undefined,
		training: (!hidden('training') && (trainSource || spec?.training || ovNames.has('Training'))) || manifest?.credits.length || ovNames.has('Credits') ? () => import('./panels/AutoTraining.svelte') : undefined
	});
	const viewProps = $derived({ store, pkg, spec, lab, slug, lessonId: lesson?.id ?? 'lesson', prequestion: manifest?.page.prequestion });

	/** a step tensor drawn as a picture: not a thin table and not a very long list (K1) */
	const thin = (x: Tensor) => x.shape.length === 2 && x.shape[1] <= 4;
	const drawable = (x: Tensor) => !thin(x) && !(x.shape.length === 1 && x.shape[0] > 4096);
	const smallTable = (x: Tensor) => thin(x) && x.shape[0] <= 12;
	const rowsOf = (x: Tensor) => Array.from({ length: x.shape[0] }, (_, r) => Array.from(x.data.subarray(r * x.shape[1], (r + 1) * x.shape[1])));
	/** the input chip's picture: the first 2-D step of what the model sees */
	const chipPic = $derived.by(() => {
		for (const s of pre?.steps ?? []) {
			const x = s.tensor;
			if (!x) continue;
			const d = x.shape.filter((v) => v > 1);
			if (d.length === 2 && d[0] <= 64 && d[1] <= 64) return { data: x.data, rows: d[0], cols: d[1] };
		}
		return null;
	});
	const editLabel = $derived(/draw/.test(manifest?.input?.adapter ?? '') ? lui('redraw') : lui('edit'));
	const kindLabel = $derived.by(() => {
		const a = manifest?.input?.adapter ?? '';
		return /draw/.test(a) ? lui('yourDrawing') : /text|char|bpe|token/.test(a) ? lui('yourText') : lui('yourInput');
	});
</script>

{#snippet pending()}
	<p class="muted">{failed ? say('modelError', { e: failed }) : say('loadingModel')}</p>
{/snippet}

{#snippet captureBox(compact: boolean)}
	{#if store && Capture}
		<Capture rt={store.rt} raw={input} {presets} options={manifest?.input?.options} onchange={onInput} {compact} />
	{:else}
		{@render pending()}
	{/if}
{/snippet}

{#snippet inputSec()}
	{#if !mounted}
		{@render pending()}
	{:else if overrides.Input}
		{@const C = overrides.Input}<C {store} {pkg} {spec} />
	{:else}
		<div class="input">
			<div>{@render captureBox(false)}</div>
			{#if pre}
				<div class="sees">
					{#each pre.steps as s (s.id)}
						{#if s.tensor && drawable(s.tensor)}
							<figure>
								<figcaption class="lbl">{t(s.label)}</figcaption>
								<TensorView tensor={s.tensor} label={t(s.label)} kind="sequential" legend={false} />
								{#if s.note}<figcaption class="muted">{t(s.note)}</figcaption>{/if}
							</figure>
						{:else if s.tensor && smallTable(s.tensor)}
							<figure>
								<figcaption class="lbl">{t(s.label)} <span class="num muted">[{s.tensor.shape.join(' × ')}]</span></figcaption>
								<table class="nums num">
									<tbody>{#each rowsOf(s.tensor) as row, r (r)}<tr>{#each row as v, c (c)}<td>{fmt3(v)}</td>{/each}</tr>{/each}</tbody>
								</table>
								{#if s.note}<figcaption class="muted">{t(s.note)}</figcaption>{/if}
							</figure>
						{:else if s.tensor}
							<p class="note">{t(s.label)}: <span class="num">[{s.tensor.shape.join(' × ')}]</span>{#if s.note} {t(s.note)}{/if}</p>
						{:else}
							<p class="note">{t(s.label)}{#if s.note} {t(s.note)}{/if}</p>
						{/if}
					{/each}
					{#each pre.notes ?? [] as n, i (i)}<p class="note">{t(n)}</p>{/each}
				</div>
			{/if}
		</div>
	{/if}
{/snippet}

<!-- the stage header's input chip (spec §4.4 InputChip): a thumbnail, what it is, the model's answer, Edit -->
{#snippet inputChip(open: () => void)}
	{#if manifest?.input || overrides.Input}
		<div class="inchip">
			{#if chipPic}
				<span class="pic" aria-hidden="true"><TensorGrid data={chipPic.data} rows={chipPic.rows} cols={chipPic.cols} label={kindLabel} kind="sequential" cell={Math.max(1, Math.floor(36 / Math.max(chipPic.rows, chipPic.cols)))} gridLines={false} readout={false} interactive={false} /></span>
			{/if}
			<span class="ilbl">
				<small>{kindLabel}</small>
				{#if outTop}<b>{t(outTop.label)} <span class="num">{fmtPct(outTop.p)}</span></b>{:else}<b>{t({ en: 'Change it any time', zh: '随时可以修改' })}</b>{/if}
			</span>
			<button type="button" class="btn" onclick={open} aria-haspopup="dialog" aria-label={editLabel}><Icon d={pencil} size={16} /><span class="bt" aria-hidden="true">{editLabel}</span></button>
		</div>
	{/if}
{/snippet}

<!-- the About sheet's credits (the Training view lists them too) -->
{#snippet credits()}
	<ul class="credits">
		{#each manifest?.credits ?? [] as c (c.name)}<li><a href={c.url} rel="noopener">{c.name}</a> · {c.license} · <span class="muted">{c.attribution}</span></li>{/each}
	</ul>
{/snippet}

{#snippet coverStage(_ctx: WalkContext)}
	{#if store}<CoverStrip {store} />{/if}
{/snippet}

{#snippet lessonSec()}
	{#if !mounted}
		{@render pending()}
	{:else if overrides.Lesson}
		{@const C = overrides.Lesson}<C {store} {pkg} {spec} />
	{:else if store && lesson && lesson.steps.length}
		<LessonPlayer {store} {lesson} label={lesson.title} bind:index={lessonIndex} {lab} cover={spec && model ? coverStage : undefined} />
	{:else}
		{@render pending()}
	{/if}
{/snippet}

{#if model}
	<ModelPageShell
		{slug}
		spec={kitSpec}
		downloadBytes={autoBytes}
		provenance="live"
		goals={manifest?.page.goals ?? []}
		prequestion={manifest?.page.prequestion}
		summary={lesson?.summary?.en ? lesson.summary : undefined}
		minutes={manifest?.catalog?.lesson?.minutes}
		lesson={lessonSec}
		{views}
		{viewProps}
		inputChip={manifest?.input || overrides.Input ? inputChip : undefined}
		inputSheet={!hidden('input') ? inputSec : undefined}
		credits={manifest?.credits.length ? credits : undefined}
	/>
{:else}
	<!-- a devOnly package (hello-xor, demo-*): not in the catalog, so the lab renders the lesson and the views as one page -->
	<article class="lab-page container">
		<header>
			<p class="eyebrow">{say('labTitle')} · {slug}</p>
			<h1>{t(spec?.name ?? { en: slug, zh: slug })}</h1>
		</header>
		{#if !hidden('input')}<Section id="input" title={ui('secInput')}>{@render inputSec()}</Section>{/if}
		<Section id="lesson" title={ui('secLesson')}>{@render lessonSec()}</Section>
		{#if store && pkg}
			{#each [views.explore, views.code, views.training] as l, i (i)}
				{#if l}
					{#await l() then m}
						{@const P = m.default}
						<Section id={['explore', 'code', 'training'][i]} title={[lui('viewExplore'), lui('viewCode'), lui('viewTraining')][i]}><P vp={viewProps} /></Section>
					{/await}
				{/if}
			{/each}
		{/if}
	</article>
{/if}

<style>
	.input {
		display: flex;
		flex-wrap: wrap;
		gap: 16px 24px;
		align-items: flex-start;
	}
	.input > div {
		min-width: 0;
		max-width: 100%;
	}
	.sees {
		display: grid;
		gap: 8px;
		max-width: min(100%, 240px);
	}
	.sees figure {
		display: grid;
		gap: 4px;
		margin: 0;
	}
	.sees figcaption,
	.note {
		margin: 0;
		font-size: var(--t-xs);
	}
	.sees .lbl {
		color: var(--ink-2);
		font-weight: var(--w-ui);
	}
	.nums {
		border-collapse: collapse;
		font-size: var(--t-xs);
	}
	.nums td {
		padding: 1px 8px 1px 0;
		text-align: right;
	}
	.note {
		padding-left: 8px;
		border-left: 3px solid var(--rule-strong);
		color: var(--ink-2);
	}
	/* the input chip (spec §4.4) */
	.inchip {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		max-width: 100%;
		padding: 5px;
		border: 1px solid var(--rule);
		border-radius: 14px;
		background: var(--panel);
		box-shadow: var(--shadow-1);
	}
	.pic {
		display: grid;
		place-items: center;
		flex: none;
		width: 36px;
		height: 36px;
		overflow: hidden;
		border-radius: 8px;
		outline: 1px solid var(--rule);
	}
	.ilbl {
		display: flex;
		flex-direction: column;
		min-width: 0;
		padding-right: 2px;
		line-height: 1.2;
	}
	.ilbl small {
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
	}
	.ilbl b {
		overflow: hidden;
		font-size: var(--t-sm);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.ilbl b .num {
		color: var(--ink-2);
		font-weight: var(--w-ui);
	}
	.inchip .btn {
		min-height: 36px;
		padding: 0 12px;
		border-radius: 10px;
		font-size: var(--t-sm);
	}
	@media (max-width: 759px) {
		.inchip {
			gap: 8px;
			padding: 4px;
			border-radius: 12px;
			box-shadow: none;
		}
		.pic {
			width: 32px;
			height: 32px;
		}
		.ilbl small,
		.bt {
			display: none;
		}
		.inchip .btn {
			width: 36px;
			padding: 0;
		}
	}
	.credits {
		margin: 0;
		padding-left: 18px;
	}
	.lab-page header h1 {
		margin: 4px 0 12px;
	}
</style>
