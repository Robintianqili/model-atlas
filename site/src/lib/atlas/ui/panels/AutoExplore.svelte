<!--
  AutoExplore (redesign spec §5.9): a kit model page's Explore view, loaded only when the view opens. Every layer and
  number, freely, on the same input as the lesson (the page's one runtime): the overview, inside a layer (the
  inspector), the arithmetic of one unit, the output, try this, check yourself (the quiz and the misconception) and the
  links (concepts that use the model, prerequisites, next). Each section has an anchor (#explore-overview …; the old
  #overview, #inspector, #unit, #output, #tryThis and #checkpoint open them); a package replaces any of them with
  explainers/<slug>/sections/<Overview|Inspector|Unit|Output|TryThis|Checkpoint|MoreLinks>.svelte (props
  { store, pkg, spec }). The explore selection is its own, never the lesson's.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import Section from '$lib/kit/Section.svelte';
	import CatalogCard from '$lib/kit/CatalogCard.svelte';
	import PredictReveal from '$lib/kit/PredictReveal.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { ModelPageCatalog } from '$lib/catalog/types';
	import type { PredictQuestion } from '$lib/kit/types';
	import type { AtlasSpec } from '../../types/spec.ts';
	import type { At } from '../../types/lesson.ts';
	import type { PackageModules } from '../../types/runtime.ts';
	import type { StageProps } from '../../types/view.ts';
	import type { PageSection } from '../../types/explainer.ts';
	import { sourcesOf } from '../../lesson/define.ts';
	import { primaryTrace, outputId } from '../../lesson/facts.ts';
	import { pickUnit } from '../../runtime/selection.ts';
	import { atFor } from '../../runtime/sources.ts';
	import { fmt3 } from '../../core/format.ts';
	import type { AtlasStore } from '../runtime.svelte.ts';
	import { loadOverrides, type Overrides } from './overrides.ts';
	import { focusBus } from '../focus.svelte.ts';
	import OpFormula from '../views/kit.parts/OpFormula.svelte';
	import AutoOverview from '../AutoOverview.svelte';
	import AutoInspector from '../AutoInspector.svelte';
	import ViewSlot from '../ViewSlot.svelte';
	import { say } from '../strings.ts';

	interface Props {
		vp: {
			store: AtlasStore | null;
			pkg: PackageModules | null;
			spec: AtlasSpec | null;
			lab?: boolean;
			slug: string;
			lessonId?: string;
			prequestion?: PredictQuestion;
		};
		section?: string;
	}
	let { vp }: Props = $props();
	const store = $derived(vp.store);
	const pkg = $derived(vp.pkg);
	const spec = $derived(vp.spec);
	const lab = $derived(!!vp.lab);
	let ov = $state.raw<Overrides>({});
	let ready = $state(false);
	onMount(() => {
		loadOverrides(vp.slug, ['Overview', 'Inspector', 'Unit', 'Output', 'TryThis', 'Checkpoint', 'MoreLinks']).then((o) => {
			ov = o;
			ready = true;
		});
	});
	const rt = $derived(store?.r ?? null);
	const manifest = $derived(pkg?.manifest ?? null);
	const secCfg = $derived(manifest?.page.sections ?? {});
	const hidden = (k: PageSection) => !!secCfg.hide?.includes(k);
	const titles = $derived(secCfg.titles ?? {});
	const intros = $derived(secCfg.intros ?? {});
	const title = (k: PageSection, fallback: string) => (titles[k] ? t(titles[k]!) : fallback);

	const trace = $derived.by(() => {
		void store?.version;
		return rt ? primaryTrace({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel }) : null;
	});
	const fwd = $derived(manifest ? (Object.entries(sourcesOf(manifest)).find(([, d]) => d.kind === 'trace')?.[0] ?? null) : null);
	const presets = $derived(rt?.presets ?? []);

	// ------------------------------------------------------------------ the explore selection (never the lesson's)
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
	function layerStage(layer: string, unit: number[] | undefined, extra: Record<string, unknown> = {}): StageProps | null {
		void store?.version;
		if (!rt || !fwd) return null;
		const tape = rt.sources[fwd]?.tape;
		if (!tape) return null;
		const i = tape.find({ layer });
		if (i < 0) return null;
		const at: At = { ...atFor(rt, fwd, i) };
		if (unit && rt.model && trace) {
			try {
				at.unit = rt.model.explain(trace, layer, unit);
			} catch {
				/* no explanation for this op: the view shows the tensor */
			}
		}
		return { rt, source: fwd, at, focus: exploreFocus, sel: { layer, unit }, provenance: rt.sources[fwd]?.provenance ?? 'live', autoplay: false, onend: () => {}, props: extra, setInput: (raw: unknown) => store?.setInput(raw), density: 'regular' };
	}
	const unitStage = $derived(exploreLayer ? layerStage(exploreLayer, unitOf(exploreLayer), { focusScope: 'explore', onpick: (u: number[]) => (picked = { ...picked, [exploreLayer]: u }) }) : null);
	const outLayer = $derived(spec ? (manifest?.output?.layer ?? outputId(spec, trace)) : undefined);
	const outStage = $derived(outLayer ? layerStage(outLayer, undefined, { focusScope: 'explore', ...(manifest?.output?.props ?? {}) }) : null);
	const outView = $derived(manifest?.output?.view ?? 'layer.tensor');
	function plainFormula(layer: string): string | undefined {
		const m = rt?.model;
		try {
			const l = m!.layer(layer);
			return zoomOf(layer) === 'layer.tensor' ? m!.ops.get(l.op).def.formula : undefined;
		} catch {
			return undefined;
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
		location.hash = `${vp.lessonId ?? 'lesson'}-${step.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '')}`;
	}
	let quizShown = $state<Record<number, boolean>>({});
	let quizPick = $state<Record<number, number>>({});
	function quizFact(key: string | undefined): number | null {
		if (!key || !rt) return null;
		void store?.version;
		const v = rt.fact(key);
		return typeof v === 'number' && Number.isFinite(v) ? v : null;
	}

	// ------------------------------------------------------------------ which sections
	const has = (k: PageSection, o: keyof Overrides) => !hidden(k) && (!!ov[o] || !!spec);
	const cat = $derived((page.data as { catalog?: ModelPageCatalog | null }).catalog ?? null);
	const linkGroups = $derived(
		[
			{ h: ui('seeAlsoConcepts'), cards: cat?.usedBy ?? [] },
			{ h: ui('prerequisites'), cards: cat?.prereqs ?? [] },
			{ h: t({ en: 'Next', zh: '下一步' }), cards: cat?.next ?? [] }
		].filter((g) => g.cards.length)
	);
	const sections = $derived(
		(
			[
				['overview', has('overview', 'Overview'), title('overview', ui('secOverview'))],
				['inspector', has('inspector', 'Inspector'), title('inspector', ui('secInspector'))],
				['unit', has('unit', 'Unit'), title('unit', ui('secUnit'))],
				['output', has('output', 'Output'), title('output', ui('secOutput'))],
				['tryThis', !hidden('tryThis') && (!!manifest?.page.tryThis?.length || !!ov.TryThis), title('tryThis', ui('secTryThis'))],
				['checkpoint', !!manifest?.page.quiz?.length || !!manifest?.page.misconception || !!ov.Checkpoint || !!vp.prequestion, ui('secCheckpoint')],
				['links', linkGroups.length > 0 || !!ov.MoreLinks, ui('secLinks')]
			] as [string, boolean, string][]
		).filter(([, ok]) => ok)
	);
</script>

{#snippet loading()}<p class="muted">{say('loadingModel')}</p>{/snippet}

{#if sections.length > 2}
	<nav class="snav" aria-label={t({ en: 'Sections', zh: '本页分节' })}>
		{#each sections as s (s[0])}<a href="#explore-{s[0]}">{s[2]}</a>{/each}
	</nav>
{/if}

{#if !store || !ready}
	{@render loading()}
{:else}
	{#if has('overview', 'Overview')}
		<Section id="explore-overview" title={title('overview', ui('secOverview'))} intro={intros.overview}>
			{#if ov.Overview}{@const C = ov.Overview}<C {store} {pkg} {spec} />{:else if spec}<AutoOverview {spec} {trace} bind:current={exploreLayer} layout={manifest?.overview?.layout} />{/if}
		</Section>
	{/if}
	{#if has('inspector', 'Inspector')}
		<Section id="explore-inspector" title={title('inspector', ui('secInspector'))} intro={intros.inspector}>
			{#if ov.Inspector}{@const C = ov.Inspector}<C {store} {pkg} {spec} />{:else if spec}<AutoInspector {spec} model={rt?.model ?? null} {trace} bind:current={exploreLayer} {unitOf} provenance={rt?.sources[fwd ?? '']?.provenance ?? 'live'} />{/if}
		</Section>
	{/if}
	{#if has('unit', 'Unit')}
		<Section id="explore-unit" title={title('unit', ui('secUnit'))} intro={intros.unit}>
			{#if ov.Unit}
				{@const C = ov.Unit}<C {store} {pkg} {spec} />
			{:else if unitStage}
				<p class="muted small">{say('unitPick')}</p>
				<ViewSlot view={zoomOf(exploreLayer)} stage={unitStage} {lab} />
				{@const ftex = plainFormula(exploreLayer)}
				{#if ftex && unitStage.at.unit}
					<OpFormula tex={ftex} e={unitStage.at.unit} caption={`${exploreLayer} [${unitStage.at.unit.unit.join(', ')}]`} provenance={unitStage.provenance} />
				{/if}
			{:else}
				{@render loading()}
			{/if}
		</Section>
	{/if}
	{#if has('output', 'Output')}
		<Section id="explore-output" title={title('output', ui('secOutput'))} intro={intros.output}>
			{#if ov.Output}{@const C = ov.Output}<C {store} {pkg} {spec} />{:else if outStage}<ViewSlot view={outView} stage={outStage} {lab} />{:else}{@render loading()}{/if}
		</Section>
	{/if}
	{#if !hidden('tryThis') && (manifest?.page.tryThis?.length || ov.TryThis)}
		<Section id="explore-tryThis" title={title('tryThis', ui('secTryThis'))} intro={intros.tryThis}>
			{#if ov.TryThis}
				{@const C = ov.TryThis}<C {store} {pkg} {spec} />
			{:else}
				<div class="tries">
					{#each manifest?.page.tryThis ?? [] as tt (tt.id)}
						<article class="card" id="try-{tt.id}">
							<h3>{t(tt.title)}</h3>
							<p><span class="tag">{say('tryPredict')}</span> {t(tt.predict)}</p>
							<div class="row">
								<button type="button" class="btn" onclick={() => setUp(tt.id)}>{say('trySetUp')}</button>
								{#if tt.goto?.step}<button type="button" class="btn btn-quiet" onclick={() => gotoStep(tt.goto!.step!)}>{say('tryGoto')}</button>{/if}
								{#if tt.goto?.layer}<button type="button" class="btn btn-quiet" onclick={() => (exploreLayer = tt.goto!.layer!)}>{say('layer')}: {tt.goto.layer}</button>{/if}
							</div>
							{#if tryDone[tt.id]}
								<p class="small" aria-live="polite">{say('trySetDone')}</p>
								<p><span class="tag">{say('tryLook')}</span> {t(tt.look)}</p>
							{/if}
						</article>
					{/each}
				</div>
			{/if}
		</Section>
	{/if}
	{#if manifest?.page.quiz?.length || manifest?.page.misconception || ov.Checkpoint || vp.prequestion}
		<Section id="explore-checkpoint" title={ui('secCheckpoint')}>
			{#if vp.prequestion}<PredictReveal q={vp.prequestion} />{/if}
			{#if ov.Checkpoint}
				{@const C = ov.Checkpoint}<C {store} {pkg} {spec} />
			{:else}
				{#if manifest?.page.misconception}
					{@const mc = manifest.page.misconception}
					<div class="card">
						<p><span class="tag">{say('misconception')}</span> {t(mc.claim)}</p>
						<p><span class="tag">{say('truth')}</span> {t(mc.truth)}</p>
					</div>
				{/if}
				{#each manifest?.page.quiz ?? [] as q, i (i)}
					<div class="card">
						<p>{t(q.q)}</p>
						{#if q.options}
							<div class="row">
								{#each q.options as o, j (j)}
									<button type="button" class="btn" aria-pressed={quizPick[i] === j} onclick={() => (quizPick = { ...quizPick, [i]: j })}>{t(o)}</button>
								{/each}
							</div>
						{/if}
						<button type="button" class="btn btn-quiet" onclick={() => (quizShown = { ...quizShown, [i]: true })}>{say('showAnswer')}</button>
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
		</Section>
	{/if}
	{#if linkGroups.length || ov.MoreLinks}
		<Section id="explore-links" title={ui('secLinks')}>
			<div class="links">
				{#each linkGroups as g (g.h)}
					<div>
						<h3>{g.h}</h3>
						<div class="cards">{#each g.cards as c (c.slug)}<CatalogCard card={c} compact />{/each}</div>
					</div>
				{/each}
			</div>
			{#if ov.MoreLinks}{@const C = ov.MoreLinks}<C {store} {pkg} {spec} />{/if}
		</Section>
	{/if}
{/if}

<style>
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
	.small {
		font-size: var(--t-sm);
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
	.card {
		display: grid;
		gap: 8px;
		margin-bottom: 10px;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
	}
	.card h3,
	.card p {
		margin: 0;
	}
	.card h3 {
		font-size: var(--t-md);
	}
	.tag {
		margin-right: 4px;
		color: var(--muted);
		font-size: var(--t-sm);
		font-weight: var(--w-title);
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
