<!--
  AutoTraining (redesign spec §5.9): a kit model page's Training view, loaded only when the view opens: how the model
  learned (a train source's run, with its curve at the last training snapshot and its named metrics; or the spec's
  training record) and the credits and licences. A package replaces the training part with
  explainers/<slug>/sections/Training.svelte and the credits with Credits.svelte (props { store, pkg, spec }).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import Section from '$lib/kit/Section.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { AtlasSpec } from '../../types/spec.ts';
	import type { PackageModules } from '../../types/runtime.ts';
	import type { StageProps } from '../../types/view.ts';
	import { sourcesOf } from '../../lesson/define.ts';
	import { atFor } from '../../runtime/sources.ts';
	import { fmtInt } from '../../core/format.ts';
	import type { AtlasStore } from '../runtime.svelte.ts';
	import { loadOverrides, type Overrides } from './overrides.ts';
	import { metricRows, trainingFrame } from '../training.ts';
	import ViewSlot from '../ViewSlot.svelte';
	import { say, STR } from '../strings.ts';

	interface Props {
		vp: { store: AtlasStore | null; pkg: PackageModules | null; spec: AtlasSpec | null; lab?: boolean; slug: string };
		section?: string;
	}
	let { vp }: Props = $props();
	const store = $derived(vp.store);
	const pkg = $derived(vp.pkg);
	const spec = $derived(vp.spec);
	const manifest = $derived(pkg?.manifest ?? null);
	const rt = $derived(store?.r ?? null);
	let ov = $state.raw<Overrides>({});
	let ready = $state(false);
	onMount(() => {
		loadOverrides(vp.slug, ['Training', 'Credits']).then((o) => {
			ov = o;
			ready = true;
		});
	});
	const TrainOv = $derived(ov.Training ?? null);
	const CreditsOv = $derived(ov.Credits ?? null);
	const trainSource = $derived(manifest ? (Object.entries(sourcesOf(manifest)).find(([, d]) => d.kind === 'train')?.[0] ?? null) : null);
	const trainStage = $derived.by((): StageProps | null => {
		void store?.version;
		if (!rt || !trainSource) return null;
		const tape = rt.sources[trainSource]?.tape;
		const i = trainingFrame(tape);
		if (!tape || i < 0) return null;
		return { rt, source: trainSource, at: atFor(rt, trainSource, i), focus: [], sel: {}, provenance: rt.sources[trainSource].provenance, autoplay: false, onend: () => {}, props: { ...(manifest?.page.trainingCurve ?? {}) }, setInput: (raw: unknown) => store?.setInput(raw), density: 'regular' };
	});
	const metricNames = $derived(manifest?.page.metrics);
	const trainMetrics = $derived(trainStage ? metricRows(trainStage.at.frame?.metrics, metricNames, ['step', 'lr']) : []);
	const trainStepAt = $derived(trainStage?.at.frame?.step);
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
	const hasTraining = $derived(!!TrainOv || !!trainSource || !!spec?.training);
	const credits = $derived(manifest?.credits ?? []);
	const endDot = (s: string) => ('.!?…'.includes(s.trim().slice(-1)) ? '' : '.');
</script>

<!-- a package's Training and Credits sections take a store that is never null: wait for the runtime (a #training deep
     link opens this view before the lesson has made its store) -->
{#if !ready || ((TrainOv || CreditsOv) && !store)}
	<p class="muted">{say('loadingModel')}</p>
{:else if hasTraining}
	<Section id="training-data" title={ui('secTraining')}>
		{#if TrainOv}
			<TrainOv {store} {pkg} {spec} />
		{:else if trainSource}
			<div class="row">
				<button type="button" class="btn btn-act" onclick={train} disabled={training || !store}>{training ? say('computing') : say('runTraining')}</button>
				{#if !trainStage}<span class="muted small">{say('trNotRun')}</span>{/if}
			</div>
			{#if trainStage}
				<ViewSlot view="loss-curve" stage={trainStage} lab={!!vp.lab} />
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
	</Section>
{/if}
{#if ready && ((CreditsOv && store) || (!CreditsOv && credits.length))}
	<Section id="training-credits" title={STR.creditsTitle}>
		{#if CreditsOv}
			<CreditsOv {store} {pkg} {spec} />
		{:else}
			<ul class="credits">
				{#each credits as c (c.name)}
					<li><a href={c.url} rel="noopener">{c.name}</a> · {c.license} · {c.usedFor}{endDot(c.usedFor)} <span class="muted">{c.attribution}</span></li>
				{/each}
			</ul>
		{/if}
	</Section>
{/if}

<style>
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
		margin-bottom: var(--s3);
	}
	.small {
		font-size: var(--t-sm);
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 160px), 1fr));
		gap: 10px 18px;
		margin: 0;
	}
	.facts dt {
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
	}
	.facts dd {
		margin: 0;
		font-weight: var(--w-ui);
	}
	.credits {
		margin: 0;
		padding-left: 18px;
		font-size: var(--t-sm);
	}
</style>
