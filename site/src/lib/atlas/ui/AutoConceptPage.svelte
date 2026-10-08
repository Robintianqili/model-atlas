<!--
  AutoConceptPage (K2): a whole /train/<slug>/ concept page generated from a concept package (explainers/<slug>/ with
  type: 'concept' and a concept.ts). It fills the kit ConceptPageShell, in its order:
     goals + predict-first   manifest.page.goals, manifest.page.prequestion
     hook                    concept.ts hook: a text, or a stage (a StepDef)
     lesson                  lesson.ts through LessonPlayer (the primary path; the side panel holds the input widget)
     toy                     the input widget (the manifest's input adapter: its Capture) above concept.ts toy
     real, compare           concept.ts real and compare (StepDefs on the same runtime and sources as the lesson)
     pitfalls                manifest.page.misconception, then concept.ts pitfalls (each may link to a lesson step)
     checkpoint              back to the prediction (the shell) + manifest.page.quiz
     links                   the catalog's models for the concept (the shell), concept.ts further reading, credits
  A package replaces any of these with sections/<Hook|Lesson|Toy|Real|Compare|Pitfalls|Checkpoint|Links>.svelte
  (props { store, pkg, def }). The page's plain text comes with the page data (concept/discover.server.ts), so the
  prerendered HTML shows it; the runtime and the stages start after mount (KIT rule 4). This component and the view
  registry live in the lazy chunk behind concept/entry.ts: planned concept pages never load them.
  KaTeX: this page does not import the KaTeX stylesheet itself (a second static importer next to AutoModelPage moved it
  into a chunk of its own and added a stylesheet and about 50 B of JS to the flagship pages). Instead every view the
  lesson and the panels use is fetched while the page is idle, so a formula view's stylesheet and fonts are there before
  the learner reaches it.
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	import { base } from '$app/paths';
	import ConceptPageShell from '$lib/kit/ConceptPageShell.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { WalkContext } from '$lib/kit/types';
	import type { PackageModules } from '../types/runtime.ts';
	import { compileLesson, type AtlasCompiledStep, type CompiledLesson } from '../lesson/compile.ts';
	import { stepSlug } from '../lesson/define.ts';
	import { fmt3 } from '../core/format.ts';
	import { AtlasStore, openAtlas } from './runtime.svelte.ts';
	import { loadCapture, loadPackage } from './packages.ts';
	import { hasView, loadView } from './registry.ts';
	import LessonPlayer from './LessonPlayer.svelte';
	import ConceptPanel from './concept/ConceptPanel.svelte';
	import { conceptOverrides, loadConcept, type ConceptSection } from './concept/packages.ts';
	import { furtherHref, hookText, panelLesson, pitfallsOf, type ConceptDef, type ConceptText, type PanelKey } from './concept/panels.ts';
	import { CSTR, csay } from './concept/strings.ts';
	import { say } from './strings.ts';

	interface Props {
		slug: string;
		/** measured bytes of manifest.assets.auto (build time) */
		autoBytes?: number;
		/** the page's plain text (build time; concept/discover.server.ts) */
		text?: ConceptText | null;
		lab?: boolean;
		/** the lesson's current step (bindable) */
		lessonIndex?: number;
	}
	let { slug, text = null, lab = false, lessonIndex = $bindable(0) }: Props = $props();

	let store = $state.raw<AtlasStore | null>(null);
	let pkg = $state.raw<PackageModules | null>(null);
	let def = $state.raw<ConceptDef>({});
	let failed = $state('');
	let Capture = $state<Component<any> | null>(null);
	let overrides = $state.raw<Partial<Record<ConceptSection, Component<any>>>>({});

	/** fetch KaTeX's web fonts while the page is idle, before the first formula needs them (AutoModelPage, K1) */
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
		preloadFormulaFonts();
		let alive = true;
		let own: AtlasStore | null = null;
		(async () => {
			try {
				const loaded = await loadPackage(slug);
				if (!loaded) throw new Error(`no concept package '${slug}'`);
				const p = loaded.pkg;
				const [d, ov] = await Promise.all([loadConcept(slug), conceptOverrides(slug)]);
				const id = p.manifest.input?.adapter;
				const cap = id ? await loadCapture(slug, id) : null;
				own = await openAtlas(p, { base });
				if (!alive) return own.close();
				pkg = p;
				def = d;
				overrides = ov;
				Capture = cap;
				store = own;
			} catch (e) {
				failed = e instanceof Error ? e.message : String(e);
			}
		})();
		return () => {
			alive = false;
			own?.close();
		};
	});

	const manifest = $derived(pkg?.manifest ?? null);
	const page = $derived(manifest?.page ?? text?.page ?? null);
	const compileCtx = $derived(store && pkg ? { spec: store.rt.spec, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes, views: hasView } : null);
	const lesson = $derived.by((): CompiledLesson | null => (compileCtx && pkg ? compileLesson(pkg.lesson, compileCtx) : null));
	/** the stage panels, compiled exactly like lesson steps (the plan is not needed: panels are author steps) */
	const panels = $derived.by((): Partial<Record<PanelKey, AtlasCompiledStep>> => {
		if (!compileCtx || !manifest) return {};
		const pl = panelLesson(def, manifest);
		const c = compileLesson(pl, { ...compileCtx, plan: [] });
		const out: Partial<Record<PanelKey, AtlasCompiledStep>> = {};
		pl.keys.forEach((k, i) => (out[k] = c.steps[i]));
		return out;
	});
	// fetch every view the lesson and the panels use while the page is idle (a formula view brings KaTeX's stylesheet)
	let prefetched = false;
	$effect(() => {
		if (prefetched || !lesson) return;
		prefetched = true;
		const keys = new Set([...lesson.steps, ...Object.values(panels)].map((s) => s!.view));
		const go = () => keys.forEach((k) => void loadView(k).catch(() => null));
		const idle = (globalThis as { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number }).requestIdleCallback;
		if (idle) idle(go, { timeout: 1500 });
		else setTimeout(go, 200);
	});
	const hook = $derived(manifest ? hookText(def) : (text?.hook ?? null));
	const pitfalls = $derived(manifest ? pitfallsOf(def, manifest) : (text?.pitfalls ?? []));
	const further = $derived(manifest ? (def.further ?? []) : (text?.further ?? []));
	const credits = $derived(manifest?.credits ?? text?.credits ?? []);
	const input = $derived.by(() => {
		void store?.version;
		return store?.rt.input;
	});
	const presets = $derived(store?.rt.presets ?? []);
	function onInput(raw: unknown, o?: { live?: boolean }) {
		if (!store) return;
		if (o?.live) store.setInputSoon(raw);
		else store.setInput(raw);
	}
	const lessonNote = $derived(page?.lessonNote ?? (lesson ? { en: `${lesson.steps.length} steps, with your input`, zh: `${lesson.steps.length} 步，用你自己的输入` } : undefined));
	/** which stage panels the page has: the compiled ones once the package is loaded, the build-time list before */
	const panelKeys = $derived<PanelKey[]>(compileCtx ? (Object.keys(panels) as PanelKey[]) : (text?.panels ?? []));
	const hasInput = $derived(manifest ? !!manifest.input : !!text?.input);
	const hasHook = $derived(!!hook || panelKeys.includes('hook') || !!overrides.Hook);
	const has = (k: 'toy' | 'real' | 'compare', o: ConceptSection) => panelKeys.includes(k) || !!overrides[o];

	// ------------------------------------------------------------------ checkpoint quiz
	let quizShown = $state<Record<number, boolean>>({});
	let quizPick = $state<Record<number, number>>({});
	function quizFact(key: string | undefined): number | null {
		if (!key || !store) return null;
		void store.version;
		const v = store.rt.fact(key);
		return typeof v === 'number' && Number.isFinite(v) ? v : null;
	}
	const stepHash = (id: string) => `${lesson?.id ?? 'lesson'}-${stepSlug(id)}`;
</script>

{#snippet loading()}
	<p class="muted">{failed ? csay('error', { e: failed }) : csay('loading')}</p>
{/snippet}

{#snippet captureBox(compact: boolean)}
	{#if store && Capture}
		<Capture rt={store.rt} raw={input} {presets} options={manifest?.input?.options} onchange={onInput} {compact} />
	{:else}
		{@render loading()}
	{/if}
{/snippet}

{#snippet panel(k: PanelKey)}
	{#if store && panels[k]}
		<ConceptPanel {store} step={panels[k]!} {lab} />
	{:else if !store}
		{@render loading()}
	{/if}
{/snippet}

{#snippet hookSec()}
	{#if overrides.Hook && store && pkg}
		{@const C = overrides.Hook}<C {store} {pkg} {def} />
	{:else}
		{#if hook}<p class="hook">{t(hook)}</p>{/if}
		{#if panels.hook && store}<ConceptPanel {store} step={panels.hook} {lab} />{/if}
	{/if}
{/snippet}

{#snippet lessonAside(_ctx: WalkContext)}
	{#if manifest?.input}
		<div class="you">
			<h4>{csay('dials')}</h4>
			{@render captureBox(true)}
			<p class="hint muted">{say('inputUpdates')}</p>
		</div>
	{/if}
{/snippet}

{#snippet lessonEnd()}
	<div class="fin">
		<p><strong>{say('lessonEnd')}</strong> {csay('lessonEndConcept')}</p>
		<a class="btn primary" href="#explore">{say('exploreFreely')} ↓</a>
	</div>
{/snippet}

{#snippet lessonSec()}
	{#if overrides.Lesson && store && pkg}
		{@const C = overrides.Lesson}<C {store} {pkg} {def} />
	{:else if store && lesson && lesson.steps.length}
		<LessonPlayer {store} {lesson} label={lesson.title} bind:index={lessonIndex} {lab} aside={lessonAside} end={lessonEnd} />
		<p class="foot muted">{manifest?.input ? csay('lessonFoot') : say('lessonFoot')}</p>
	{:else}
		{@render loading()}
	{/if}
{/snippet}

{#snippet toySec()}
	{#if overrides.Toy && store && pkg}
		{@const C = overrides.Toy}<C {store} {pkg} {def} />
	{:else}
		{#if hasInput}<div class="capture">{@render captureBox(false)}</div>{/if}
		{@render panel('toy')}
	{/if}
{/snippet}

{#snippet realSec()}
	{#if overrides.Real && store && pkg}{@const C = overrides.Real}<C {store} {pkg} {def} />{:else}{@render panel('real')}{/if}
{/snippet}

{#snippet compareSec()}
	{#if overrides.Compare && store && pkg}{@const C = overrides.Compare}<C {store} {pkg} {def} />{:else}{@render panel('compare')}{/if}
{/snippet}

{#snippet pitfallsSec()}
	{#if overrides.Pitfalls && store && pkg}
		{@const C = overrides.Pitfalls}<C {store} {pkg} {def} />
	{:else}
		<ul class="pits">
			{#each pitfalls as p, i (i)}
				<li>
					<p><span class="tag">{say('misconception')}</span> {t(p.claim)}</p>
					<p><span class="tag">{say('truth')}</span> {t(p.truth)}</p>
					<!-- a button, not a link: the lesson's step ids exist only after mount (a prerendered #lesson-<id> link has no target) -->
					{#if p.step}<p><button type="button" class="btn ghost small" onclick={() => (location.hash = stepHash(p.step!))}>{csay('inLesson')} →</button></p>{/if}
				</li>
			{/each}
		</ul>
	{/if}
{/snippet}

{#snippet checkpointSec()}
	{#if overrides.Checkpoint && store && pkg}
		{@const C = overrides.Checkpoint}<C {store} {pkg} {def} />
	{:else}
		{#each page?.quiz ?? [] as q, i (i)}
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
{/snippet}

{#snippet furtherSec()}
	{#if overrides.Links && store && pkg}
		{@const C = overrides.Links}<C {store} {pkg} {def} />
	{:else}
		{#if further.length}
			<h3>{csay('further')}</h3>
			<ul>{#each further as f (f.href)}<li><a href={furtherHref(f.href, base)} rel="noopener">{t(f.label)}</a></li>{/each}</ul>
		{/if}
		{#if credits.length}
			<h3>{say('creditsTitle')}</h3>
			<ul class="credits">
				{#each credits as c (c.name)}
					<li><a href={c.url} rel="noopener">{c.name}</a> · {c.license} · {c.usedFor}. <span class="muted">{c.attribution}</span></li>
				{/each}
			</ul>
		{/if}
	{/if}
{/snippet}

<ConceptPageShell
	{slug}
	goals={page?.goals ?? []}
	prequestion={page?.prequestion}
	{lessonNote}
	exploreIntro={CSTR.exploreIntro}
	hook={hasHook ? hookSec : undefined}
	lesson={lessonSec}
	toy={hasInput || has('toy', 'Toy') ? toySec : undefined}
	real={has('real', 'Real') ? realSec : undefined}
	compare={has('compare', 'Compare') ? compareSec : undefined}
	pitfalls={pitfalls.length || overrides.Pitfalls ? pitfallsSec : undefined}
	post={page?.quiz?.length || overrides.Checkpoint ? checkpointSec : undefined}
	further={further.length || credits.length || overrides.Links ? furtherSec : undefined}
	titles={{ post: ui('secCheckpoint'), ...(page?.sections?.titles ?? {}) }}
	intros={page?.sections?.intros ?? {}}
/>

<style>
	.hook {
		font-size: var(--fs-lg);
		max-width: var(--measure);
		margin: 0 0 12px;
	}
	.capture {
		margin-bottom: 14px;
		max-width: 100%;
		min-width: 0;
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
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.pits {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 10px;
	}
	.pits li,
	.quiz {
		display: grid;
		gap: 6px;
		padding: 12px 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.quiz {
		margin-bottom: 10px;
	}
	.pits p,
	.quiz p {
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
	.credits {
		margin: 0;
		padding-left: 18px;
		font-size: var(--fs-sm);
	}
</style>
