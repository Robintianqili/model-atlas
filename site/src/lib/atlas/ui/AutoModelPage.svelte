<!--
  AutoModelPage (design §5.1): a whole model page generated from a package (or, in the Lesson Lab, from a spec alone).
  It fills the kit ModelPageShell (catalog slugs) or a plain lab layout (devOnly packages): input (the adapter's Capture
  and what the model sees), the guided lesson (LessonPlayer), overview, inspector, the arithmetic of one unit, output,
  try-this, training, article, code, checkpoint and credits. A package replaces any section with
  explainers/<slug>/sections/<Snippet>.svelte (props: { store, pkg, spec }).
  This component, the view registry and KaTeX live in the lazy chunk that +page.ts returns: planned pages never load them.
  The runtime starts in onMount (KIT rule 4); the prerendered HTML shows the shell with the facts of the embedded spec.
  K0:
  - Without a spec (kinds B and C), the model sections (overview, inspector, unit, output, code) are left out unless the
    package overrides them; manifest.page.sections titles, introduces or hides any section.
  - The input section draws the adapter's step tensors, except long thin ones (a list of 200 points is a 200 × 2 table,
    not a picture): those are named with their shape.
  - The KaTeX stylesheet comes with this chunk, so the first formula view of the lesson does not re-lay-out the page
    (a 50-75 ms task when it arrived with that view).
  K1:
  - The KaTeX web fonts are fetched once the page is idle (document.fonts.load), so the first formula of the lesson
    does not swap fonts and re-lay-out the page while the learner steps (the AK603 long task of 52-180 ms).
  - The Training section is bilingual: readable metric names and formats (ui/training.ts; manifest.page.metrics names a
    package's own), and a train source's curve at its last training snapshot, not at a later sampling frame (K-R3).
  - The lesson's side panel shows the answer only for a one-answer output (not a per-node [n, class] output), with a
    text model's vocabulary words; the end card does not promise layers on a page without them.
  - The input section names every step and prints a short list of points as numbers; 'The arithmetic' shows the op's
    formula with its values for an op without a zoom view.
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	// the formula views' stylesheet, loaded with the page (see the header)
	import 'katex/dist/katex.min.css';
	import { base } from '$app/paths';
	import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
	import Section from '$lib/kit/Section.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import CodeBlock from '$lib/kit/CodeBlock.svelte';
	import PredictFirst from '$lib/kit/PredictFirst.svelte';
	import { MODEL_BY_SLUG } from '$lib/catalog/models';
	import { i18n, t, ui } from '$lib/i18n/i18n.svelte';
	import type { WalkContext } from '$lib/kit/types';
	import type { AtlasSpec } from '../types/spec.ts';
	import type { At } from '../types/lesson.ts';
	import type { PackageModules } from '../types/runtime.ts';
	import type { PageSection } from '../types/explainer.ts';
	import type { Tensor } from '../types/core.ts';
	import type { StageProps } from '../types/view.ts';
	import { normalizeSpec } from '../graph/legacy.ts';
	import { compileLesson, type CompiledLesson } from '../lesson/compile.ts';
	import { primaryTrace, outputId } from '../lesson/facts.ts';
	import { sourcesOf } from '../lesson/define.ts';
	import { pickUnit } from '../runtime/selection.ts';
	import { fmt3, fmtInt, fmtPct } from '../core/format.ts';
	import { atFor } from '../runtime/sources.ts';
	import { AtlasStore, openAtlas } from './runtime.svelte.ts';
	import { loadArticle, loadCapture, loadPackage, loadSection, sectionOverrides, type SectionName } from './packages.ts';
	import { hasView } from './registry.ts';
	import { toKitSpec } from './kitspec.ts';
	import { isAnswerLayer, topAnswers } from './fallback.ts';
	import { metricRows, trainingFrame } from './training.ts';
	import OpFormula from './views/kit.parts/OpFormula.svelte';
	import { focusBus } from './focus.svelte.ts';
	import LessonPlayer from './LessonPlayer.svelte';
	import AutoOverview from './AutoOverview.svelte';
	import AutoInspector from './AutoInspector.svelte';
	import ViewSlot from './ViewSlot.svelte';
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
	const model = $derived(MODEL_BY_SLUG[slug]);

	/** kit views render after mount only: the prerendered HTML has no markup whose CSS is not linked (autoentry.ts) */
	let mounted = $state(false);
	let Capture = $state<Component<any> | null>(null);
	let overrides = $state.raw<Partial<Record<SectionName, Component<any>>>>({});
	let article = $state.raw<{ en: Component<any>; zh?: Component<any> } | null>(null);

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
				const [cap, art] = await Promise.all([id ? loadCapture(slug, id) : null, loadArticle(slug)]);
				const ov: Partial<Record<SectionName, Component<any>>> = {};
				for (const n of sectionOverrides(slug)) ov[n] = await loadSection(slug, n)!;
				if (!alive) return;
				Capture = cap;
				article = art;
				overrides = ov;
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
	// K1: a page without a spec (kinds B and C) computes in JavaScript numbers (f64): the facts say so, not "Precision —"
	const kitSpec = $derived(spec ? toKitSpec(spec, (primaryTraceOf()?.meta?.dims ?? {}) as Record<string, number>) : manifest ? { slug, precision: 'f64' } : null);
	const fwd = $derived(manifest ? (Object.entries(sourcesOf(manifest)).find(([, d]) => d.kind === 'trace')?.[0] ?? null) : null);
	function primaryTraceOf() {
		if (!rt) return null;
		void store?.version;
		return primaryTrace({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel });
	}
	const trace = $derived.by(() => {
		void store?.version;
		return primaryTraceOf();
	});
	const pre = $derived.by(() => {
		void store?.version;
		return rt?.pre ?? null;
	});
	const presets = $derived(rt?.presets ?? []);
	/** the model's current top answer (class name, or token text for a language model) for the lesson's side panel */
	const outTop = $derived.by(() => {
		void store?.version;
		const tr = trace;
		const id = spec && tr ? outputId(spec, tr) : undefined;
		const o = id ? tr?.t[id] : undefined;
		if (!rt || !o || !isAnswerLayer(rt, id, tr)) return null;
		const top = topAnswers(rt, o, 1)[0];
		if (!top) return null;
		// K1 (K-R8): the line says what it is ("Next token: “␣” 83.1%", "Its answer: cat 92.0%")
		const axes = spec?.layers.find((l) => l.id === id)?.axes ?? [];
		return { ...top, what: axes[axes.length - 1] === 'vocab' ? STR.ansNext : STR.ansTop };
	});
	const input = $derived.by(() => {
		void store?.version;
		return rt?.input;
	});
	const trainSource = $derived(manifest ? (Object.entries(sourcesOf(manifest)).find(([, d]) => d.kind === 'train')?.[0] ?? null) : null);

	function onInput(raw: unknown, o?: { live?: boolean }) {
		if (!store) return;
		if (o?.live) store.setInputSoon(raw);
		else store.setInput(raw);
	}

	// ------------------------------------------------------------------ explore selection (never the lesson's)
	const firstLayer = $derived(spec?.layers.find((l) => l.op !== 'input')?.id ?? spec?.layers[0]?.id ?? '');
	let exploreLayer = $state('');
	$effect(() => {
		if (!exploreLayer && firstLayer) exploreLayer = firstLayer;
	});
	let picked = $state.raw<Record<string, number[]>>({});
	function unitOf(layer: string): number[] | undefined {
		if (picked[layer]) return picked[layer];
		const tr = trace;
		const m = rt?.model;
		if (!tr || !m) return undefined;
		try {
			const l = m.layer(layer);
			return pickUnit(tr.t[layer], m.ops.get(l.op).def.unit?.pick, { axes: l.axes });
		} catch {
			return undefined;
		}
	}
	function zoomOf(layer: string): string {
		const m = rt?.model;
		try {
			const l = m!.layer(layer);
			const z = l.ui?.zoom ?? m!.ops.get(l.op).def.zoom;
			return z && z !== 'none' ? (z.includes('.') ? z : `zoom.${z}`) : 'layer.tensor';
		} catch {
			return 'layer.tensor';
		}
	}
	const exploreFocus = $derived(store ? focusBus(store.rt, 'explore').step : []);
	/** StageProps for a layer of the primary trace, with an explicit unit (the explore views' own selection) */
	function layerStage(layer: string, unit: number[] | undefined, extra: Record<string, unknown> = {}): StageProps | null {
		void store?.version;
		if (!rt || !fwd) return null;
		const tape = rt.sources[fwd]?.tape;
		if (!tape) return null;
		const i = tape.find({ layer });
		if (i < 0) return null;
		const base: At = atFor(rt, fwd, i);
		const at: At = { ...base };
		if (unit && rt.model && trace) {
			try {
				at.unit = rt.model.explain(trace, layer, unit);
			} catch {
				/* no explanation for this op: the view shows the tensor */
			}
		}
		return {
			rt,
			source: fwd,
			at,
			focus: exploreFocus,
			sel: { layer, unit },
			provenance: rt.sources[fwd]?.provenance ?? 'live',
			autoplay: false,
			onend: () => {},
			props: extra,
			setInput: (raw: unknown) => store?.setInput(raw)
		};
	}
	const unitStage = $derived(exploreLayer ? layerStage(exploreLayer, unitOf(exploreLayer), { focusScope: 'explore', onpick: (u: number[]) => (picked = { ...picked, [exploreLayer]: u }) }) : null);
	const outLayer = $derived(spec ? (manifest?.output?.layer ?? outputId(spec, trace)) : undefined);
	const outStage = $derived(outLayer ? layerStage(outLayer, undefined, { focusScope: 'explore', ...(manifest?.output?.props ?? {}) }) : null);
	const outView = $derived(manifest?.output?.view ?? 'layer.tensor');
	// K1: the last training snapshot (a run may go on after training, e.g. sampling frames without a loss)
	const trainStage = $derived.by((): StageProps | null => {
		void store?.version;
		if (!rt || !trainSource) return null;
		const tape = rt.sources[trainSource]?.tape;
		const i = trainingFrame(tape);
		if (!tape || i < 0) return null;
		// K2: manifest.page.trainingCurve chooses what the curve draws (a policy-gradient page: the episode length, not the loss)
		return { rt, source: trainSource, at: atFor(rt, trainSource, i), focus: [], sel: {}, provenance: rt.sources[trainSource].provenance, autoplay: false, onend: () => {}, props: { ...(manifest?.page.trainingCurve ?? {}) }, setInput: (raw: unknown) => store?.setInput(raw) };
	});
	const metricNames = $derived(manifest?.page.metrics);
	/** the train source's numbers at that frame, named (K1, K-R3: raw keys before) */
	const trainMetrics = $derived(trainStage ? metricRows(trainStage.at.frame?.metrics, metricNames, ['step', 'lr']) : []);
	const trainStepAt = $derived(trainStage?.at.frame?.step);
	/** scalar training metrics only (specs may also carry matrices and lists, e.g. a confusion matrix), named (K1) */
	const trainingFacts = $derived(metricRows((spec?.training?.metrics ?? {}) as Record<string, unknown>, metricNames));
	let training = $state(false);
	async function train() {
		if (!store || !trainSource) return;
		training = true;
		try {
			await store.rt.ensure(trainSource, 'last');
		} finally {
			training = false;
			store.touch();
		}
	}

	// ------------------------------------------------------------------ try this, quiz
	let tryDone = $state<Record<string, boolean>>({});
	function setUp(id: string) {
		const tt = manifest?.page.tryThis?.find((x) => x.id === id);
		if (!tt || !store) return;
		const p = tt.setup.preset ? presets.find((x) => x.id === tt.setup.preset) : undefined;
		if (p) store.setInput(p.raw);
		else if (tt.setup.input !== undefined) store.setInput(tt.setup.input);
		tryDone = { ...tryDone, [id]: true };
	}
	function gotoStep(step: string) {
		const id = lesson?.id ?? 'lesson';
		location.hash = `${id}-${step.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '')}`;
	}
	let quizShown = $state<Record<number, boolean>>({});
	let quizPick = $state<Record<number, number>>({});
	function quizFact(key: string | undefined): number | null {
		if (!key || !rt) return null;
		void store?.version;
		const v = rt.fact(key);
		return typeof v === 'number' && Number.isFinite(v) ? v : null;
	}
	// ------------------------------------------------------------------ which sections (K0)
	const ovNames = $derived(new Set<SectionName>(sectionOverrides(slug)));
	const secCfg = $derived(manifest?.page.sections ?? {});
	const hidden = (k: PageSection) => !!secCfg.hide?.includes(k);
	/** a model section: shown when the package overrides it, or when there is a spec to fill it */
	const modelSec = (k: PageSection, o: SectionName) => !hidden(k) && (ovNames.has(o) || !!spec);
	const titles = $derived(secCfg.titles ?? {});
	const intros = $derived(secCfg.intros ?? {});
	const secTitle = (k: PageSection, fallback: string) => (titles[k] ? t(titles[k]!) : fallback);
	/** a step tensor drawn as a picture: not a thin table (points, a few features per row; K1: also short ones, which
	 *  were drawn as an unlabeled heatmap) and not a very long list */
	const thin = (x: Tensor) => x.shape.length === 2 && x.shape[1] <= 4;
	const drawable = (x: Tensor) => !thin(x) && !(x.shape.length === 1 && x.shape[0] > 4096);
	/** a short thin table printed as numbers (K1): at most 12 rows */
	const smallTable = (x: Tensor) => thin(x) && x.shape[0] <= 12;
	const rowsOf = (x: Tensor) => Array.from({ length: x.shape[0] }, (_, r) => Array.from(x.data.subarray(r * x.shape[1], (r + 1) * x.shape[1])));
	/** the op formula of a layer whose op has no zoom view ('The arithmetic' shows it with the explained values; K1) */
	function plainFormula(layer: string): string | undefined {
		const m = rt?.model;
		try {
			const l = m!.layer(layer);
			return zoomOf(layer) === 'layer.tensor' ? m!.ops.get(l.op).def.formula : undefined;
		} catch {
			return undefined;
		}
	}

	const lessonNote = $derived(manifest?.page.lessonNote ?? (lesson ? { en: `${lesson.steps.length} steps, with your input`, zh: `${lesson.steps.length} 步，用你自己的输入` } : undefined));
</script>

{#snippet pending()}
	<p class="muted">{say('loadingModel')}</p>
{/snippet}

{#snippet captureBox(compact: boolean)}
	{#if store && Capture}
		<Capture rt={store.rt} raw={input} {presets} options={manifest?.input?.options} onchange={onInput} {compact} />
	{:else if failed}
		<p class="err">{say('modelError', { e: failed })}</p>
	{:else}
		<p class="muted">{say('loadingModel')}</p>
	{/if}
{/snippet}

{#snippet inputSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Input}
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
	{/if}
{/snippet}

{#snippet lessonAside(_ctx: WalkContext)}
	<div class="you">
		<h4>{say('yourInput')}</h4>
		{#if outTop}
			<p class="ans"><span class="muted">{t(outTop.what)}:</span> {t(outTop.label)} <span class="num">{fmtPct(outTop.p)}</span></p>
		{/if}
		{@render captureBox(true)}
		<p class="hint muted">{say('inputUpdates')}</p>
	</div>
{/snippet}

{#snippet lessonEnd()}
	<div class="fin">
		<p><strong>{say('lessonEnd')}</strong> {say(spec?.layers?.length ? 'lessonEndMore' : 'lessonEndMoreFree')}</p>
		<a class="btn primary" href="#explore">{say('exploreFreely')} ↓</a>
	</div>
{/snippet}

{#snippet lessonSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Lesson}
			{@const C = overrides.Lesson}<C {store} {pkg} {spec} />
		{:else if store && lesson && lesson.steps.length}
			<LessonPlayer {store} {lesson} label={lesson.title} bind:index={lessonIndex} {lab} aside={lessonAside} end={lessonEnd} />
			<p class="foot muted">{say('lessonFoot')}</p>
		{:else}
			<p class="muted">{failed ? say('modelError', { e: failed }) : say('loadingModel')}</p>
		{/if}
	{/if}
{/snippet}

{#snippet overviewSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Overview}
			{@const C = overrides.Overview}<C {store} {pkg} {spec} />
		{:else if spec && mounted}
			<AutoOverview {spec} {trace} bind:current={exploreLayer} layout={manifest?.overview?.layout} />
		{/if}
	{/if}
{/snippet}

{#snippet inspectorSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Inspector}
			{@const C = overrides.Inspector}<C {store} {pkg} {spec} />
		{:else if spec && mounted}
			<AutoInspector {spec} model={rt?.model ?? null} {trace} bind:current={exploreLayer} {unitOf} provenance={rt?.sources[fwd ?? '']?.provenance ?? 'live'} />
		{/if}
	{/if}
{/snippet}

{#snippet unitSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Unit}
			{@const C = overrides.Unit}<C {store} {pkg} {spec} />
		{:else if unitStage}
			<p class="muted small">{say('unitPick')}</p>
			<ViewSlot view={zoomOf(exploreLayer)} stage={unitStage} {lab} />
			{@const ftex = plainFormula(exploreLayer)}
			{#if ftex && unitStage.at.unit}
				<OpFormula tex={ftex} e={unitStage.at.unit} caption={`${exploreLayer} [${unitStage.at.unit.unit.join(', ')}]`} provenance={unitStage.provenance} />
			{/if}
		{:else}
			<p class="muted">{say('loadingModel')}</p>
		{/if}
	{/if}
{/snippet}

{#snippet outputSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Output}
			{@const C = overrides.Output}<C {store} {pkg} {spec} />
		{:else if outStage}
			<ViewSlot view={outView} stage={outStage} {lab} />
		{:else}
			<p class="muted">{say('loadingModel')}</p>
		{/if}
	{/if}
{/snippet}

{#snippet tryThisSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.TryThis}
			{@const C = overrides.TryThis}<C {store} {pkg} {spec} />
		{:else}
			<div class="tries">
				{#each manifest?.page.tryThis ?? [] as tt (tt.id)}
					<article class="try" id="try-{tt.id}">
						<h3>{t(tt.title)}</h3>
						<p><span class="tag">{say('tryPredict')}</span> {t(tt.predict)}</p>
						<div class="row">
							<button type="button" class="btn" onclick={() => setUp(tt.id)}>{say('trySetUp')}</button>
							{#if tt.goto?.step}<button type="button" class="btn ghost" onclick={() => gotoStep(tt.goto!.step!)}>{say('tryGoto')}</button>{/if}
							{#if tt.goto?.layer}<button type="button" class="btn ghost" onclick={() => (exploreLayer = tt.goto!.layer!)}>{say('layer')}: {tt.goto.layer}</button>{/if}
						</div>
						{#if tryDone[tt.id]}
							<p class="small" aria-live="polite">{say('trySetDone')}</p>
							<p><span class="tag">{say('tryLook')}</span> {t(tt.look)}</p>
						{/if}
					</article>
				{/each}
			</div>
		{/if}
	{/if}
{/snippet}

{#snippet trainingSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Training}
			{@const C = overrides.Training}<C {store} {pkg} {spec} />
		{:else if trainSource}
			<div class="row">
				<button type="button" class="btn primary" onclick={train} disabled={training || !store}>{training ? say('computing') : say('runTraining')}</button>
				{#if !trainStage}<span class="muted small">{say('trNotRun')}</span>{/if}
			</div>
			{#if trainStage}
				<ViewSlot view="loss-curve" stage={trainStage} {lab} />
				{#if trainMetrics.length}
					<p class="small muted">{typeof trainStepAt === 'number' ? say('trAfter', { n: trainStepAt }) : say('trFinal')}</p>
					<dl class="facts">{#each trainMetrics as m (m.key)}<div><dt>{t(m.label)}</dt><dd class="num">{m.text}</dd></div>{/each}</dl>
				{/if}
			{/if}
		{:else if spec?.training}
			<dl class="facts">
				{#if spec.training.dataset}<div><dt>{say('trDataset')}</dt><dd>{spec.training.dataset}</dd></div>{/if}
				{#if spec.training.steps}<div><dt>{say('trSteps')}</dt><dd class="num">{fmtInt(spec.training.steps)}</dd></div>{/if}
				{#if spec.params}<div><dt>{say('trParams')}</dt><dd class="num">{fmtInt(spec.params)}</dd></div>{/if}
				{#each trainingFacts as m (m.key)}<div><dt>{t(m.label)}</dt><dd class="num">{m.text}</dd></div>{/each}
			</dl>
		{/if}
	{/if}
{/snippet}

{#snippet articleSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Article}
			{@const C = overrides.Article}<C {store} {pkg} {spec} />
		{:else if article}
			<!-- the kit Article's behaviour, inlined: importing Article here would split it out of TinyGPT's page chunk -->
			{@const En = article.en}
			{@const Zh = article.zh}
			<div class="article prose" lang={i18n.lang === 'zh' && Zh ? 'zh-CN' : 'en'}>
				{#if i18n.lang === 'zh' && Zh}
					<Zh />
				{:else}
					{#if i18n.lang === 'zh'}<p class="nt" lang="zh-CN">{ui('notTranslatedYet')}</p>{/if}
					<En />
				{/if}
			</div>
		{/if}
	{/if}
{/snippet}

{#snippet codeSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Code}
			{@const C = overrides.Code}<C {store} {pkg} {spec} />
		{:else if spec && rt?.model && mounted}
			{@const m = rt.model}
			{@const lines = spec.layers
				.map((l) => {
					try {
						const d = m.ops.get(l.op).def;
						return d.pytorch ? `${l.id.padEnd(12)} = ${d.pytorch(l)}` : null;
					} catch {
						return null;
					}
				})
				.filter((x): x is string => !!x)}
			{#if lines.length}<CodeBlock code={lines.join('\n')} lang="python" caption={say('codeFor')} />{/if}
		{/if}
	{/if}
{/snippet}

{#snippet checkpointSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Checkpoint}
			{@const C = overrides.Checkpoint}<C {store} {pkg} {spec} />
		{:else}
			{#if manifest?.page.misconception}
				{@const mc = manifest.page.misconception}
				<div class="mc">
					<p><span class="tag">{say('misconception')}</span> {t(mc.claim)}</p>
					<p><span class="tag">{say('truth')}</span> {t(mc.truth)}</p>
				</div>
			{/if}
			{#each manifest?.page.quiz ?? [] as q, i (i)}
				<div class="quiz">
					<p>{t(q.q)}</p>
					{#if q.options}
						<div class="row">
							{#each q.options as o, j (j)}
								<button type="button" class="btn" aria-pressed={quizPick[i] === j} onclick={() => (quizPick = { ...quizPick, [i]: j })}>{t(o)}</button>
							{/each}
						</div>
					{/if}
					<button type="button" class="btn ghost" onclick={() => (quizShown = { ...quizShown, [i]: true })}>{say('showAnswer')}</button>
					{#if quizShown[i]}
						<p class="small" aria-live="polite">
							{#if q.options && q.answer !== undefined}<strong>{t(q.options[q.answer])}.</strong>{/if}
							{#if q.answerFact}{@const v = quizFact(q.answerFact)}{#if v !== null}{say('measured', { v: Number.isInteger(v) ? v : fmt3(v) })}.{/if}{/if}
							{t(q.explain)}
						</p>
					{/if}
				</div>
			{/each}
		{/if}
	{/if}
{/snippet}

{#snippet creditsSec()}
	{#if !mounted}
		{@render pending()}
	{:else}
		{#if overrides.Credits}
			{@const C = overrides.Credits}<C {store} {pkg} {spec} />
		{:else if manifest?.credits.length}
			<ul class="credits">
				{#each manifest.credits as c (c.name)}
					<li><a href={c.url} rel="noopener">{c.name}</a> · {c.license} · {c.usedFor}. <span class="muted">{c.attribution}</span></li>
				{/each}
			</ul>
		{/if}
	{/if}
{/snippet}

{#snippet moreLinksSec()}
	{#if overrides.MoreLinks}{@const C = overrides.MoreLinks}<C {store} {pkg} {spec} />{/if}
{/snippet}

{#if model}
	<ModelPageShell
		{slug}
		spec={kitSpec}
		downloadBytes={autoBytes}
		provenance="live"
		goals={manifest?.page.goals ?? []}
		prequestion={manifest?.page.prequestion}
		lessonNote={lessonNote}
		input={hidden('input') ? undefined : inputSec}
		lesson={lessonSec}
		overview={modelSec('overview', 'Overview') ? overviewSec : undefined}
		inspector={modelSec('inspector', 'Inspector') ? inspectorSec : undefined}
		unit={modelSec('unit', 'Unit') ? unitSec : undefined}
		output={modelSec('output', 'Output') ? outputSec : undefined}
		tryThis={!hidden('tryThis') && (manifest?.page.tryThis?.length || overrides.TryThis) ? tryThisSec : undefined}
		training={!hidden('training') && (trainSource || spec?.training || ovNames.has('Training')) ? trainingSec : undefined}
		article={!hidden('article') && (article || overrides.Article) ? articleSec : undefined}
		code={modelSec('code', 'Code') ? codeSec : undefined}
		{titles}
		{intros}
		moreLinks={overrides.MoreLinks ? moreLinksSec : undefined}
		checkpoint={manifest?.page.quiz?.length || manifest?.page.misconception || overrides.Checkpoint ? checkpointSec : undefined}
		credits={manifest?.credits.length || overrides.Credits ? creditsSec : undefined}
	/>
{:else}
	<!-- a devOnly package (hello-xor, demo-*): not in the catalog, so the lab renders the same sections without the shell -->
	<article class="lab-page container">
		<header>
			<p class="eyebrow">{say('labTitle')} · {slug}</p>
			<h1>{t(spec?.name ?? { en: slug, zh: slug })}</h1>
		</header>
		{#if manifest?.page.goals.length}
			<Section id="goals" n={1} title={ui('secGoals')}>
				<ul>{#each manifest.page.goals as g, i (i)}<li>{t(g)}</li>{/each}</ul>
				{#if manifest.page.prequestion}<PredictFirst q={manifest.page.prequestion} />{/if}
			</Section>
		{/if}
		{#if !hidden('input')}<Section id="input" n={2} title={secTitle('input', ui('secInput'))}>{@render inputSec()}</Section>{/if}
		<Section id="lesson" n={3} title={secTitle('lesson', ui('secLesson'))}>{@render lessonSec()}</Section>
		<div id="explore"></div>
		{#if modelSec('overview', 'Overview')}<Section id="overview" n={4} title={secTitle('overview', ui('secOverview'))}>{@render overviewSec()}</Section>{/if}
		{#if modelSec('inspector', 'Inspector')}<Section id="inspector" n={5} title={secTitle('inspector', ui('secInspector'))}>{@render inspectorSec()}</Section>{/if}
		{#if modelSec('unit', 'Unit')}<Section id="unit" n={6} title={secTitle('unit', ui('secUnit'))}>{@render unitSec()}</Section>{/if}
		{#if modelSec('output', 'Output')}<Section id="output" n={7} title={secTitle('output', ui('secOutput'))}>{@render outputSec()}</Section>{/if}
		{#if trainSource && !hidden('training')}<Section id="training" n={8} title={secTitle('training', ui('secTraining'))}>{@render trainingSec()}</Section>{/if}
		{#if modelSec('code', 'Code')}<Section id="code" n={9} title={secTitle('code', ui('secCode'))}>{@render codeSec()}</Section>{/if}
		{#if manifest?.credits.length}<Section id="credits" n={10} title={STR.creditsTitle}>{@render creditsSec()}</Section>{/if}
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
		margin: 0;
		display: grid;
		gap: 4px;
	}
	.sees figcaption,
	.note {
		font-size: var(--fs-xs);
		margin: 0;
	}
	.sees .lbl {
		color: var(--ink-2);
		font-weight: 600;
	}
	.nums {
		border-collapse: collapse;
		font-size: var(--fs-xs);
	}
	.nums td {
		padding: 1px 8px 1px 0;
		text-align: right;
	}
	.note {
		color: var(--ink-2);
		border-left: 3px solid var(--rule-strong);
		padding-left: 8px;
	}
	.you {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.you h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.ans {
		margin: 0;
		font-weight: 600;
		font-size: var(--fs-sm);
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.fin {
		display: grid;
		gap: 8px;
		justify-items: start;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--live-soft);
	}
	.fin p {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.foot {
		margin: 8px 0 0;
		font-size: var(--fs-xs);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.err {
		color: var(--div-neg-text);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.tries {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr));
		gap: 12px;
	}
	.try,
	.quiz,
	.mc {
		display: grid;
		gap: 6px;
		padding: 12px 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		margin-bottom: 10px;
	}
	.try h3 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.try p,
	.quiz p,
	.mc p {
		margin: 0;
	}
	.tag {
		font-size: var(--fs-xs);
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--accent-strong);
		margin-right: 4px;
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 150px), 1fr));
		gap: 10px 18px;
		margin: 0;
	}
	.facts dt {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.facts dd {
		margin: 0;
		font-weight: 600;
	}
	.credits {
		margin: 0;
		padding-left: 18px;
		font-size: var(--fs-sm);
	}
	.nt {
		font-size: var(--fs-sm);
		color: var(--muted);
		border-left: 3px solid var(--rule-strong);
		padding-left: 10px;
	}
	.lab-page header h1 {
		margin: 4px 0 12px;
	}
</style>
