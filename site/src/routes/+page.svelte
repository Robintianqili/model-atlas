<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import TextInputBar from '$lib/kit/TextInputBar.svelte';
	import ModelCard from '$lib/kit/ModelCard.svelte';
	import ConceptCard from '$lib/kit/ConceptCard.svelte';
	import StatusBadge from '$lib/kit/StatusBadge.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import InputIcon from '$lib/kit/InputIcon.svelte';
	import { saveHandoff } from '$lib/kit/handoff';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtShape } from '$lib/kit/format';
	import type { Stroke } from '$lib/kit/draw';
	import { AREAS } from '$lib/catalog/areas';
	import { MODELS, MODEL_BY_SLUG } from '$lib/catalog/models';
	import { CONCEPTS, CONCEPT_BY_SLUG } from '$lib/catalog/concepts';
	import { PATHS } from '$lib/catalog/paths';
	import { modelsAccepting } from '$lib/catalog';
	import type { InputKind, ModelEntry } from '$lib/catalog/types';
	import { defaultInput, demoForward } from '$lib/demo/cnn-demo';
	import { t, ui } from '$lib/i18n/i18n.svelte';

	const liveModels = MODELS.filter((m) => m.status === 'live');
	const liveConcepts = CONCEPTS.filter((c) => c.status === 'live');
	const plannedModels = MODELS.length - liveModels.length;
	const plannedConcepts = CONCEPTS.length - liveConcepts.length;
	// the first call to action: the guided lessons of the live pages (catalog `lesson`; pages without one
	// yet link to their top)
	const lessons = [
		...liveModels.map((m) => ({ slug: m.slug, name: m.name, href: `/m/${m.slug}/`, lesson: m.lesson })),
		...liveConcepts.map((c) => ({ slug: c.slug, name: c.title, href: `/train/${c.slug}/`, lesson: c.lesson }))
	];
	const liveSpecs = $derived(
		(page.data as { liveSpecs?: Record<string, { input?: { shape: number[] } }> }).liveSpecs ?? {}
	);

	// ---------------------------------------------------------------- input launcher
	let mode = $state<'drawing' | 'text'>('drawing');
	let strokes = $state<Stroke[]>([]);
	let text = $state('');
	const accepting = $derived(modelsAccepting(mode as InputKind));
	const hasInput = $derived(mode === 'drawing' ? strokes.length > 0 : text.trim().length > 0);

	function sees(m: ModelEntry): string {
		const shape = liveSpecs[m.slug]?.input?.shape;
		return shape ? fmtShape(shape) : '';
	}
	function open(m: ModelEntry) {
		if (hasInput) saveHandoff(mode === 'drawing' ? { kind: 'drawing', strokes } : { kind: 'text', text });
		goto(`${base}/m/${m.slug}/`);
	}

	// ---------------------------------------------------------------- hero art (decorative)
	const art = demoForward(defaultInput());
	const seq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const div = $derived(makeScale('diverging', [-2, 2], themeState.resolved));
	const relu = art.t.relu.data;
	const pool = art.t.pool.data;
	const probs = art.t.probs.data;
	const maxP = Math.max(...probs);

	const counts = (slug: string) => {
		const ms = MODELS.filter((m) => m.area === slug);
		return { total: ms.length, live: ms.filter((m) => m.status === 'live').length };
	};
</script>

<svelte:head>
	<title>Model Atlas · {t({ en: 'see every layer work', zh: '模型图鉴' })}</title>
</svelte:head>

<!-- ================================================================== hero -->
<section class="hero">
	<div class="container hero-inner">
		<div class="copy">
			<p class="eyebrow">Model Atlas · <span lang="zh-CN">模型图鉴</span></p>
			<h1>{t({ en: 'Give a model your own input. Watch every layer work.', zh: '给模型你自己的输入，看每一层如何工作。' })}</h1>
			<p class="lede">
				{t({
					en: 'Draw a picture or type a sentence, then follow it through a real neural network, one layer at a time, with the actual numbers. Everything runs in your browser; nothing you draw or type leaves your device.',
					zh: '画一幅画或输入一句话，然后跟着它一层一层穿过真实的神经网络，看到真实的数字。一切都在你的浏览器中运行；你画的和写的都不会离开你的设备。'
				})}
			</p>
			<p class="stats">
				<span class="big num">{liveModels.length + liveConcepts.length}</span>
				{t({ en: 'pages live now', zh: '个页面已上线' })}
				<span class="sep">·</span>
				<span class="big num">{plannedModels}</span>
				{t({ en: 'more models planned across', zh: '个模型在计划中，覆盖' })}
				<span class="big num">{AREAS.length}</span>
				{t({ en: 'areas', zh: '个领域' })}
				<span class="sep">·</span>
				<span class="big num">{plannedConcepts}</span>
				{t({ en: 'training concepts planned', zh: '个训练概念在计划中' })}
			</p>
			<nav class="lessons" aria-labelledby="lessons-h">
				<h2 id="lessons-h" class="lt">{ui('lessonHomeTitle')}</h2>
				<ul>
					{#each lessons as l, i (l.slug)}
						<li>
							<a class="btn" class:primary={i === 0} href="{base}{l.href}{l.lesson ? '#lesson' : ''}">
								{#if l.lesson}<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M4.5 2.8v10.4L13 8z" fill="currentColor" /></svg>{/if}
								{t(l.name)}{#if l.lesson}<span class="ls num">{ui('lessonSteps', { n: l.lesson.steps })}</span>{/if}
							</a>
						</li>
					{/each}
				</ul>
				<a class="more" href="{base}/atlas/">{ui('browseAtlas')} →</a>
			</nav>
		</div>
		<figure class="art">
			<!-- the picture is decoration; the caption (badge + text) stays readable (its badge is focusable) -->
			<svg viewBox="0 0 360 250" width="100%" aria-hidden="true">
				<defs>
					<pattern id="dash" width="8" height="2" patternUnits="userSpaceOnUse"><rect width="5" height="2" fill="var(--ink-2)" /></pattern>
				</defs>
				<!-- input 12×12 -->
				{#each Array.from({ length: 144 }, (_, i) => i) as i (i)}
					<rect x={10 + (i % 12) * 8} y={40 + Math.floor(i / 12) * 8} width="7" height="7" rx="1" fill={seq.css(art.t.input.data[i])} />
				{/each}
				<!-- relu tiles 4×10×10 -->
				{#each [0, 1, 2, 3] as c (c)}
					{#each Array.from({ length: 100 }, (_, i) => i) as i (i)}
						<rect x={136 + (c % 2) * 52 + (i % 10) * 5} y={22 + Math.floor(c / 2) * 62 + Math.floor(i / 10) * 5} width="4.4" height="4.4" fill={div.css(relu[c * 100 + i])} />
					{/each}
				{/each}
				<!-- pooled 4×5×5 -->
				{#each [0, 1, 2, 3] as c (c)}
					{#each Array.from({ length: 25 }, (_, i) => i) as i (i)}
						<rect x={258 + (i % 5) * 7} y={30 + c * 44 + Math.floor(i / 5) * 7} width="6" height="6" rx="1" fill={div.css(pool[c * 25 + i])} />
					{/each}
				{/each}
				<!-- output bars -->
				{#each Array.from(probs) as p, i (i)}
					<rect x="310" y={30 + i * 17} width={Math.max(2, (p / maxP) * 44)} height="11" rx="2" fill={seq.css(0.25 + (0.75 * p) / maxP)} />
				{/each}
				<rect class="flow" x="110" y="87" width="22" height="2" fill="url(#dash)" />
				<rect class="flow" x="240" y="87" width="14" height="2" fill="url(#dash)" />
				<rect class="flow" x="296" y="87" width="10" height="2" fill="url(#dash)" />
				<text x="10" y="236" class="lbl">input</text>
				<text x="136" y="236" class="lbl">conv · relu</text>
				<text x="258" y="236" class="lbl">pool</text>
				<text x="310" y="236" class="lbl">p</text>
			</svg>
			<figcaption>
				<ProvenanceBadge mode="illustrative" compact />
				<span>{t({ en: 'A hand-made filter bank, not a trained model', zh: '手工设计的滤波器组，并非训练好的模型' })}</span>
			</figcaption>
		</figure>
	</div>
</section>

<!-- ================================================================== launcher -->
<section class="container block" aria-labelledby="launcher-h">
	<h2 id="launcher-h">{ui('launcherTitle')}</h2>
	<p class="muted">{ui('launcherNothingLoaded')}</p>
	<div class="launcher">
		<div class="input-side">
			<div class="segmented" role="tablist" aria-label={ui('input')}>
				<button type="button" role="tab" aria-selected={mode === 'drawing'} onclick={() => (mode = 'drawing')}>
					<InputIcon kind="drawing" /> {ui('launcherDraw')}
				</button>
				<button type="button" role="tab" aria-selected={mode === 'text'} onclick={() => (mode = 'text')}>
					<InputIcon kind="text" /> {ui('launcherType')}
				</button>
			</div>
			<div class="pane" role="tabpanel">
				{#if mode === 'drawing'}
					<DrawPad bind:strokes size={240} showBrush={false} />
				{:else}
					<TextInputBar
						bind:value={text}
						maxLength={64}
						multiline
						rows={3}
						presets={[
							{ text: 'Once upon a time, there was a', label: { en: 'Story opening', zh: '故事开头' } },
							{ text: '我爱自然语言处理', label: { en: 'Chinese', zh: '中文' } },
							{ text: 'unbelievable!' }
						]}
					/>
				{/if}
			</div>
		</div>
		<div class="results">
			<h3>{ui('launcherAccepts')} <span class="muted num">({accepting.length})</span></h3>
			<ul class="acc">
				{#each accepting as m (m.slug)}
					<li class:planned={m.status !== 'live'}>
						<button type="button" class="acc-row" onclick={() => open(m)}>
							<span class="nm">{t(m.name)}</span>
							<StatusBadge status={m.status} tier={m.tier} />
							<span class="sees num">{sees(m) ? `${t({ en: 'sees', zh: '输入' })} ${sees(m)}` : ''}</span>
							<span class="hk">{t(m.hook)}</span>
						</button>
					</li>
				{/each}
			</ul>
		</div>
	</div>
</section>

<!-- ================================================================== live now -->
<section class="container block" aria-labelledby="live-h">
	<h2 id="live-h">{ui('livePages')}</h2>
	<div class="cards">
		{#each liveModels as m (m.slug)}<ModelCard model={m} />{/each}
		{#each liveConcepts as c (c.slug)}<ConceptCard concept={c} showSection />{/each}
	</div>
</section>

<!-- ================================================================== areas -->
<section class="container block" aria-labelledby="areas-h">
	<div class="h-row">
		<h2 id="areas-h">{t({ en: 'Eleven areas', zh: '十一个领域' })}</h2>
		<a href="{base}/atlas/">{ui('browseAtlas')} →</a>
	</div>
	<p class="muted">
		{t({
			en: 'The atlas is built to grow: every area has a planned flagship and a backlog of models. Planned pages already exist and say what they will show.',
			zh: '图鉴为持续扩展而设计：每个领域都有计划中的旗舰模型和后续模型。计划中的页面已经存在，并说明将展示什么。'
		})}
	</p>
	<ol class="areas">
		{#each AREAS as a (a.slug)}
			{@const c = counts(a.slug)}
			<li>
				<a class="area" href="{base}/atlas/{a.slug}/">
					<span class="n num">{String(a.n).padStart(2, '0')}</span>
					<span class="an">{t(a.name)}</span>
					<span class="ab">{t(a.blurb)}</span>
					<span class="ac num">
						{ui('countModels', { n: c.total })}{#if c.live}{' · '}<span class="live">{ui('countLive', { n: c.live })}</span>{/if}
					</span>
				</a>
			</li>
		{/each}
	</ol>
</section>

<!-- ================================================================== paths -->
<section class="container block" aria-labelledby="paths-h">
	<h2 id="paths-h">{t({ en: 'Learning paths', zh: '学习路径' })}</h2>
	<div class="paths">
		{#each PATHS as p (p.id)}
			<article class="path">
				<h3>{t(p.title)}</h3>
				<p class="muted">{t(p.blurb)}</p>
				<ol class="steps">
					{#each p.steps as s, i (i)}
						{@const entry = s.kind === 'model' ? MODEL_BY_SLUG[s.slug] : CONCEPT_BY_SLUG[s.slug]}
						{#if entry}
							<li class:live={entry.status === 'live'}>
								<a href="{base}/{s.kind === 'model' ? 'm' : 'train'}/{s.slug}/">
									<span class="k">{s.kind === 'model' ? ui('model') : ui('concept')}</span>
									{t('title' in entry ? entry.title : entry.name)}
								</a>
							</li>
						{/if}
					{/each}
				</ol>
			</article>
		{/each}
	</div>
</section>

<!-- ================================================================== honesty -->
<section class="container block" aria-labelledby="honest-h">
	<h2 id="honest-h">{t({ en: 'Every number says where it came from', zh: '每个数字都标明来源' })}</h2>
	<div class="prov3">
		<div><ProvenanceBadge mode="live" /><p>{ui('provLiveHelp')}</p></div>
		<div><ProvenanceBadge mode="precomputed" /><p>{ui('provPrecomputedHelp')}</p></div>
		<div><ProvenanceBadge mode="illustrative" /><p>{ui('provIllustrativeHelp')}</p></div>
	</div>
	<p><a href="{base}/about/">{ui('footerHowNumbers')} →</a></p>
</section>

<style>
	.hero {
		background:
			radial-gradient(1200px 400px at 85% -10%, color-mix(in srgb, var(--accent-soft) 70%, transparent), transparent 70%),
			var(--bg);
		border-bottom: 1px solid var(--rule);
	}
	.hero-inner {
		display: grid;
		grid-template-columns: 1.15fr 1fr;
		gap: var(--s6);
		align-items: center;
		padding-top: var(--s7);
		padding-bottom: var(--s7);
	}
	@media (max-width: 860px) {
		.hero-inner {
			grid-template-columns: 1fr;
			padding-top: var(--s6);
			padding-bottom: var(--s6);
		}
	}
	.copy {
		display: grid;
		gap: var(--s3);
	}
	.copy h1,
	.copy p {
		margin: 0;
	}
	.stats {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		line-height: 1.9;
	}
	.big {
		font-size: 1.25rem;
		font-weight: 700;
		color: var(--ink);
	}
	.sep {
		margin: 0 4px;
		color: var(--muted);
	}
	.lessons {
		display: grid;
		gap: 8px;
		margin-top: 4px;
		padding: 12px 14px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.lt {
		margin: 0;
		font-size: var(--fs-md);
		font-family: var(--font-body);
		font-weight: 700;
	}
	.lessons ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.lessons .btn {
		gap: 6px;
		min-height: 36px;
		font-size: var(--fs-sm);
	}
	.ls {
		font-size: var(--fs-xs);
		opacity: 0.8;
	}
	.ls::before {
		content: '· ';
	}
	.more {
		font-size: var(--fs-sm);
	}
	.art {
		margin: 0;
		padding: 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		box-shadow: var(--shadow-1);
	}
	.art figcaption {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-xs);
		color: var(--muted);
		margin-top: 6px;
	}
	.art .lbl {
		font-family: var(--font-mono);
		font-size: 10px;
		fill: var(--muted);
	}
	.flow {
		animation: flow 1s linear infinite;
	}
	@keyframes flow {
		from {
			transform: translateX(0);
		}
		to {
			transform: translateX(3px);
		}
	}
	.block {
		padding-top: var(--s7);
	}
	.block h2 {
		margin-bottom: var(--s2);
	}
	.h-row {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: baseline;
		gap: 8px;
	}
	.launcher {
		display: grid;
		grid-template-columns: minmax(260px, 340px) 1fr;
		gap: var(--s5);
		margin-top: var(--s4);
		align-items: start;
	}
	@media (max-width: 760px) {
		.launcher {
			grid-template-columns: 1fr;
		}
	}
	.input-side {
		display: grid;
		gap: 10px;
	}
	.segmented button {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.results h3 {
		font-size: var(--fs-md);
		font-family: var(--font-body);
		margin-bottom: 8px;
	}
	.acc {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 6px;
		max-height: 420px;
		overflow-y: auto;
		padding-right: 4px;
	}
	.acc-row {
		width: 100%;
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 2px 10px;
		padding: 10px 12px;
		text-align: left;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.acc-row:hover {
		border-color: var(--rule-strong);
		background: var(--surface-2);
	}
	.planned .acc-row {
		border-style: dashed;
		background: transparent;
	}
	.nm {
		font-weight: 700;
	}
	.sees {
		grid-column: 1 / -1;
		font-size: var(--fs-xs);
		color: var(--accent);
	}
	.sees:empty {
		display: none;
	}
	.hk {
		grid-column: 1 / -1;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr));
		gap: 12px;
		margin-top: var(--s3);
	}
	.areas {
		list-style: none;
		margin: var(--s4) 0 0;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 250px), 1fr));
		gap: 10px;
	}
	.area {
		display: grid;
		gap: 4px;
		height: 100%;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		color: var(--ink);
		text-decoration: none;
	}
	.area:hover {
		color: var(--ink);
		border-color: var(--accent);
	}
	.area .n {
		font-size: var(--fs-xs);
		color: var(--accent);
	}
	.an {
		font-family: var(--font-display);
		font-size: 1.2rem;
		font-weight: 600;
	}
	.ab {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.ac {
		font-size: var(--fs-xs);
		color: var(--muted);
		margin-top: 4px;
	}
	.ac .live {
		color: var(--live);
		font-weight: 700;
	}
	.paths {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
		gap: 12px;
		margin-top: var(--s3);
	}
	.path {
		padding: 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.path h3 {
		font-size: var(--fs-lg);
		margin-bottom: 4px;
	}
	.path p {
		font-size: var(--fs-sm);
		margin-bottom: 10px;
	}
	.steps {
		margin: 0;
		padding: 0;
		list-style: none;
		counter-reset: st;
		display: grid;
		gap: 4px;
	}
	.steps li {
		counter-increment: st;
		display: flex;
		align-items: baseline;
		gap: 8px;
	}
	.steps li::before {
		content: counter(st);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
		min-width: 1.2em;
		text-align: right;
	}
	.steps a {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		text-decoration-style: dashed;
	}
	.steps li.live a {
		color: var(--accent);
		font-weight: 700;
		text-decoration-style: solid;
	}
	.steps .k {
		font-family: var(--font-mono);
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--muted);
		margin-right: 4px;
	}
	.prov3 {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
		gap: 12px;
		margin: var(--s3) 0;
	}
	.prov3 > div {
		padding: 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.prov3 p {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
</style>
