<!--
  LabView: the Lesson Lab's whole UI (loaded lazily by +page.svelte, so the route node stays tiny): the full auto page and
  lesson of one package (or of a spec folder with ?spec=models/<dir>), plus the tools a reviewer needs: fixture and
  language selectors, the lint report over every fixture in en and zh, the parity panel (kind A: the per-layer strip from
  spec.parity; kinds B and C: the per-source records of explainers/<slug>/parity.json, with the badge text each stage
  shows), and a "copy deep link" button for the current step. ?spec= is read on the client (reading url.searchParams during prerender throws).
-->
<script lang="ts">
	import { onMount, type Component } from 'svelte';
	import { base } from '$app/paths';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { i18n, setLang, t } from '$lib/i18n/i18n.svelte';
	import type { LintIssue } from '$lib/atlas/types/lesson';
	import type { SourceParity } from '$lib/atlas/types/runtime';
	import { say } from '$lib/atlas/ui/strings';
	import { sourceBadge } from '$lib/atlas/ui/kitspec';
	import type { Lab } from './lab';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	let lab = $state.raw<Lab | null>(null);
	let Auto = $state<Component<any> | null>(null);
	let error = $state('');
	let issues = $state.raw<LintIssue[] | null>(null);
	let progress = $state<{ done: number; total: number } | null>(null);
	let fixtureId = $state('');
	let lessonIndex = $state(0);
	let copied = $state<'' | 'ok' | 'fail'>('');
	let mod: typeof import('./lab') | null = null;

	onMount(() => {
		let alive = true;
		(async () => {
			try {
				const q = new URLSearchParams(location.search).get('spec');
				mod = await import('./lab');
				const [l, A] = await Promise.all([mod.openLab(data.slug, q, base, data.files), import('$lib/atlas/ui/AutoModelPage.svelte')]);
				if (!alive) return l.store.close();
				lab = l;
				Auto = A.default;
				void runLint();
			} catch (e) {
				error = e instanceof Error ? e.message : String(e);
			}
		})();
		return () => {
			alive = false;
		};
	});

	async function runLint() {
		if (!lab || !mod) return;
		issues = null;
		issues = await mod.lintAll(lab, base, (done, total) => (progress = { done, total }));
		progress = null;
	}

	function pickFixture(id: string) {
		fixtureId = id;
		const fx = lab?.fixtures.find((f) => f.id === id);
		if (fx && lab) lab.store.setInput(fx.raw);
	}

	const stepId = $derived(lab?.lesson.steps[lessonIndex]?.id ?? '');
	const deepLink = $derived.by(() => {
		if (typeof location === 'undefined' || !lab) return '';
		void lessonIndex;
		const u = new URL(location.href);
		u.search = '';
		u.searchParams.set('spec', lab.dir);
		u.hash = `${lab.lesson.id}-${stepId}`;
		return u.href;
	});
	async function copy() {
		try {
			await navigator.clipboard.writeText(deepLink);
			copied = 'ok';
		} catch {
			copied = 'fail';
		}
		setTimeout(() => (copied = ''), 2500);
	}

	const errors = $derived(issues?.filter((i) => i.level === 'error') ?? []);
	const warns = $derived(issues?.filter((i) => i.level === 'warn') ?? []);
	const spec = $derived(lab?.store.r.spec ?? null);
	const parity = $derived(spec?.parity ?? null);
	const perLayer = $derived(parity?.perLayer ? Object.entries(parity.perLayer) : []);
	const bar = $derived(parity?.bar ?? parity?.tolerance ?? 1e-4);
	// kinds B and C (K0): atlas parity --write's per-source record of twins and references (explainers/<slug>/parity.json)
	const pkgParity = $derived(lab?.pkg.parity ?? null);
	const sourceRecords = $derived(pkgParity ? Object.entries(pkgParity.sources) : []);
	/** train and process sources the lesson uses that have no record: their steps' badge says only "Live" */
	const unrecorded = $derived.by(() => {
		if (!lab) return [] as string[];
		const defs = lab.pkg.manifest.sources ?? {};
		const used = new Set(lab.lesson.steps.map((s) => s.source));
		return [...used].filter((n) => (defs[n]?.kind === 'train' || defs[n]?.kind === 'process') && !pkgParity?.sources[n]);
	});
	const fmtSourceRecord = (r: SourceParity) =>
		[
			`${r.scope} vs ${r.against}`,
			`max ${r.max === 0 ? '0' : r.max.toExponential(2)}${r.relative ? ' (relative)' : ''}${r.bar ? ` ≤ ${r.bar}` : ''}`,
			r.steps ? `${r.steps} step${r.steps === 1 ? '' : 's'}` : r.frames ? `${r.frames} frame${r.frames === 1 ? '' : 's'}` : '',
			r.fixtures ? `+ ${r.fixtures} fixtures` : '',
			r.checked
		]
			.filter(Boolean)
			.join(' · ');
	function goStep(step: string | undefined) {
		if (!step || !lab) return;
		location.hash = `${lab.lesson.id}-${step}`;
	}
</script>

<svelte:head>
	<title>{say('labTitle')} · {data.slug} · Model Atlas</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<div class="lab container">
	<header class="head">
		<p class="eyebrow">{say('labTitle')}</p>
		<h1 class="ttl">{data.slug}</h1>
		<p class="muted intro">{say('labIntro')}</p>
		{#if !data.enabled}<p class="warn">{say('labDisabled')}</p>{/if}
	</header>

	{#if error}
		<p class="warn" role="alert">{error}</p>
		{#if !data.packages.includes(data.slug)}
			<p class="muted">{say('labUnknown', { slug: data.slug })}</p>
			<ul class="links">
				{#each data.packages as s (s)}<li><a href="{base}/dev/atlas/lab/{s}/">{s}</a></li>{/each}
				{#each data.specDirs as s (s)}<li><a href="{base}/dev/atlas/lab/{s}/?spec=models/{s}">{s} (?spec=models/{s})</a></li>{/each}
			</ul>
		{/if}
	{:else if !lab}
		<p class="muted">{say('loadingModel')}</p>
	{:else}
		<section class="tools" aria-label={say('labTitle')}>
			<div class="row">
				<label class="field">
					<span>{say('labFixture')}</span>
					<select value={fixtureId} onchange={(e) => pickFixture((e.currentTarget as HTMLSelectElement).value)}>
						<option value="">{say('labCurrent')}</option>
						{#each lab.fixtures as f (f.id)}<option value={f.id}>{f.id} · {t(f.label)}</option>{/each}
					</select>
				</label>
				<div class="segmented" role="group" aria-label={say('labLang')}>
					<button type="button" class="btn" aria-pressed={i18n.lang === 'en'} onclick={() => setLang('en')}>English</button>
					<button type="button" class="btn" aria-pressed={i18n.lang === 'zh'} onclick={() => setLang('zh')}>中文</button>
				</div>
				<button type="button" class="btn" onclick={copy} data-deeplink={deepLink}>{copied === 'ok' ? say('labCopied') : say('labCopy')}</button>
				<span class="muted small">{say('labSpec')}: <code>static/{lab.dir}</code>{lab.synthetic ? ' · auto' : ''}</span>
			</div>
			{#if copied === 'fail'}<p class="small">{say('labCopyFail')} <code>{deepLink}</code></p>{/if}

			<details class="panel lint" open>
				<summary>
					<strong>{say('labLint')}</strong>
					{#if progress}
						<span class="muted">{say('labLinting', { k: progress.done, n: progress.total })}</span>
					{:else if issues}
						<span class:bad={errors.length > 0} class="sum">{say('labLintSummary', { e: errors.length, w: warns.length, f: lab.fixtures.length, s: lab.lesson.steps.length })}</span>
					{/if}
				</summary>
				{#if issues && !issues.length}<p class="muted">{say('labClean')}</p>{/if}
				{#if issues?.length}
					<ol class="issues">
						{#each [...errors, ...warns].slice(0, 200) as i, k (k)}
							<li class:error={i.level === 'error'} class:warn={i.level === 'warn'}>
								<code class="code">{i.code}</code>
								{#if i.step}<button type="button" class="linkish" onclick={() => goStep(i.step)}>{i.step}</button>{/if}
								{#if i.fixture}<span class="muted">· {i.fixture}</span>{/if}
								{#if i.lang}<span class="muted">· {i.lang}</span>{/if}
								<span>{i.message}</span>
								{#if i.fix}<span class="muted">→ {i.fix}</span>{/if}
							</li>
						{/each}
					</ol>
				{/if}
				<button type="button" class="btn small-btn" onclick={runLint} disabled={!!progress}>{say('labLintAgain')}</button>
			</details>

			<div class="panel parity">
				<!-- kinds B and C without a spec have no layers: the heading says "Parity", not "Per-layer parity" -->
				<strong>{perLayer.length || typeof parity?.max_abs_diff === 'number' || !pkgParity ? say('labParity') : t({ en: 'Parity', zh: '一致性' })}</strong>
				{#if perLayer.length}
					<div class="strip" role="list">
						{#each perLayer as [id, d] (id)}
							<span role="listitem" class="cell" class:fail={!(d <= bar)} title="{id}: {d}">{id} <span class="num">{d.toExponential(1)}</span></span>
						{/each}
					</div>
				{:else if parity && typeof parity.max_abs_diff === 'number'}
					<p class="small">
						{say('labNoPerLayer', { d: parity.max_abs_diff.toExponential(2), a: parity.against, b: bar })}
						<ProvenanceBadge mode="live" compact precision={spec?.precision} tolerance={parity.max_abs_diff} against={parity.against} />
					</p>
				{:else if !pkgParity}
					<p class="small muted">{say('labNoParity')}</p>
				{/if}
				{#if pkgParity}
					<p class="small">
						<strong>{t({ en: 'Twins and references', zh: '孪生实现与参考实现' })}</strong>
						<span class="muted">explainers/{pkgParity.slug}/parity.json · {pkgParity.checked}</span>
					</p>
					<ul class="recs">
						{#each sourceRecords as [name, r] (name)}
							{@const note = sourceBadge(r).note}
							<li class:fail={!r.pass}>
								<code>{name}</code>
								<span class="num">{fmtSourceRecord(r)}</span>
								<strong>{r.pass ? t({ en: 'pass', zh: '通过' }) : t({ en: 'FAIL', zh: '未通过' })}</strong>
								{#if note}<span class="muted">{t({ en: 'badge', zh: '徽章' })}: “{t(note)}”</span>{/if}
							</li>
						{/each}
					</ul>
				{/if}
				{#if unrecorded.length}
					<p class="small muted">
						{t({
							en: 'Lesson sources with no record, so their steps’ badge says only “Live” (a train source needs a twin, a process source pipeline/<slug>/reference.py; then atlas parity --write):',
							zh: '课程用到、但没有记录的数据源，这些步骤的徽章只显示“实时”（train 数据源需要孪生实现，process 数据源需要 pipeline/<slug>/reference.py；然后运行 atlas parity --write）：'
						})}
						{#each unrecorded as n (n)}<code>{n}</code>{/each}
					</p>
				{/if}
			</div>

			<details class="panel">
				<summary>{say('labSteps')} ({lab.lesson.steps.length})</summary>
				<ol class="steps">
					{#each lab.lesson.steps as s, k (s.id)}
						<li class:cur={k === lessonIndex}>
							<button type="button" class="linkish" onclick={() => goStep(s.id)}>{s.id}</button>
							<span class="muted">{s.origin}{s.autoId ? ` · ${s.autoId}` : ''} · {s.source} · {s.view} · {s.provenance.mode}</span>
						</li>
					{/each}
				</ol>
			</details>
			{#if lab.notes.length}
				<details class="panel">
					<summary>{say('labWarnings')} ({lab.notes.length})</summary>
					<ul class="small">{#each lab.notes as n, k (k)}<li>{n}</li>{/each}</ul>
				</details>
			{/if}
		</section>
	{/if}
</div>

{#if lab && Auto}
	<Auto slug={data.slug} pkg={lab.pkg} store={lab.store} lab bind:lessonIndex />
{/if}

<style>
	.lab {
		padding-top: var(--s5);
	}
	.head .ttl {
		margin: 2px 0 6px;
		font-family: var(--font-mono);
	}
	.intro {
		max-width: var(--measure);
		margin: 0 0 var(--s3);
	}
	.warn {
		color: var(--div-neg-text);
	}
	.tools {
		display: grid;
		gap: 10px;
		margin-bottom: var(--s5);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 14px;
		align-items: center;
	}
	.field {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-sm);
		max-width: 100%;
	}
	.field select {
		max-width: min(60vw, 320px);
		font: inherit;
		padding: 4px 6px;
		border-radius: var(--r1);
		border: 1px solid var(--rule-strong);
		background: var(--surface);
		color: var(--ink);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.panel {
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		font-size: var(--fs-sm);
		min-width: 0;
	}
	.panel summary {
		cursor: pointer;
	}
	.sum {
		margin-left: 6px;
		color: var(--ink-2);
	}
	.sum.bad {
		color: var(--div-neg-text);
		font-weight: 600;
	}
	.issues {
		margin: 8px 0;
		padding-left: 22px;
		display: grid;
		gap: 3px;
		font-size: var(--fs-xs);
		max-height: 340px;
		overflow: auto;
	}
	.issues li.error .code {
		color: var(--div-neg-text);
	}
	.issues li.warn .code {
		color: var(--illus);
	}
	.issues li {
		overflow-wrap: anywhere;
	}
	.linkish {
		background: none;
		border: none;
		padding: 0;
		font: inherit;
		color: var(--accent-strong);
		text-decoration: underline;
		cursor: pointer;
	}
	.small-btn {
		min-height: 30px;
		font-size: var(--fs-xs);
	}
	.strip {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 6px;
	}
	.cell {
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		padding: 2px 6px;
		border-radius: var(--r1);
		background: var(--live-soft);
	}
	.cell.fail {
		background: var(--surface-3);
		outline: 1.5px solid var(--div-neg);
	}
	.parity p {
		margin: 6px 0 0;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.recs {
		margin: 6px 0 0;
		padding-left: 18px;
		display: grid;
		gap: 3px;
		font-size: var(--fs-xs);
	}
	.recs li {
		overflow-wrap: anywhere;
	}
	.recs li > * + * {
		margin-left: 6px;
	}
	.recs li.fail strong {
		color: var(--div-neg-text);
	}
	.steps {
		margin: 6px 0 0;
		padding-left: 22px;
		font-size: var(--fs-xs);
		display: grid;
		gap: 2px;
	}
	.steps li.cur {
		font-weight: 700;
	}
	.links {
		font-size: var(--fs-sm);
	}
</style>
