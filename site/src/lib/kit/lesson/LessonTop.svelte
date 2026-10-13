<!--
  LessonTop (redesign spec §4.4): the lesson page's top bar, in place of the site header.
  Desktop (56 px): the mark (home) · crumbs (Atlas / Area / Name; the area crumb hides below 1100 px) · the view tabs
  (Lesson carries a k/N pill) · About · EN | 中文 · theme. Phone (52 px, static): the mark · the name over "Step k of N ·
  chapter" · 中文 · More (a sheet with the views, About, language and theme). On phones chapter-grouped progress segments
  run along its bottom edge.
  <LessonTop app={app} current="lesson" groups={groups} />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Mark from '../Mark.svelte';
	import Icon from '../Icon.svelte';
	import Sheet from '../Sheet.svelte';
	import LangToggle from '../LangToggle.svelte';
	import ThemeToggle from '../ThemeToggle.svelte';
	import { info, more } from '../icons';
	import { lui, t, ui, type Text } from '$lib/i18n/i18n.svelte';
	import type { LessonApp } from './app';
	import type { ViewId } from './views';

	interface Props {
		app: LessonApp;
		current: ViewId;
		/** the step groups (chapters) for the phone progress segments */
		groups?: { title?: Text; from: number; to: number }[];
	}
	let { app, current, groups = [] }: Props = $props();

	let aboutOpen = $state(false);
	let moreOpen = $state(false);
	const pos = $derived(app.pos);
	const count = $derived(pos.n ? (pos.cover ? String(pos.n) : `${pos.at + 1}/${pos.n}`) : '');
	const sub = $derived(
		!pos.n
			? ''
			: pos.cover
				? t({ en: `${pos.n} steps`, zh: `${pos.n} 步` })
				: `${lui('stepOf', { k: pos.at + 1, n: pos.n })}${pos.chapter ? ` · ${t(pos.chapter)}` : ''}`
	);
	const home = t({ en: 'Model Atlas home', zh: '模型图鉴首页' });
	function openAbout() {
		moreOpen = false;
		aboutOpen = true;
	}
</script>

<!-- the lesson route has no site header: its own skip link (to the lesson's main, without touching the hash) -->
{#if !app.embedded}
	<a
		class="skip"
		href="#main"
		onclick={(e) => {
			e.preventDefault();
			document.getElementById('main')?.focus();
		}}>{ui('skipToContent')}</a
	>
{/if}
<header class="ltop">
	<div class="crumb">
		<a class="home" href="{base}/" aria-label={home}><Mark size={28} /></a>
		<nav class="path" aria-label={t({ en: 'Breadcrumb', zh: '位置' })}>
			{#each app.crumbs as c, i (i)}<span class="cr" class:area={i > 0}><a href={c.href}>{t(c.label)}</a><span class="sep" aria-hidden="true">/</span></span>{/each}<span class="title" aria-current="page">{t(app.name)}</span>
		</nav>
		<div class="ttl">
			<span class="title">{t(app.name)}</span>
			{#if sub}<small>{sub}</small>{/if}
		</div>
	</div>
	<nav class="vtabs" aria-label={lui('views')}>
		{#each app.tabs as v (v.id)}
			<a href="#{v.id}" aria-current={current === v.id ? 'page' : undefined}>
				<Icon d={v.icon} size={18} class="vi" />{t(v.label)}{#if v.id === 'lesson' && count}<span class="count num">{count}</span>{/if}
			</a>
		{/each}
	</nav>
	<div class="tools">
		{#if app.about}<button type="button" class="icon-btn dt" aria-label={t(app.aboutTitle)} aria-haspopup="dialog" onclick={openAbout}><Icon d={info} /></button>{/if}
		<LangToggle />
		<span class="dt"><ThemeToggle /></span>
		<button type="button" class="icon-btn ph" aria-label={lui('more')} aria-haspopup="dialog" onclick={() => (moreOpen = true)}><Icon d={more} /></button>
	</div>
	{#if groups.length && pos.n}
		<div class="phrail" aria-hidden="true">
			{#each groups as g, gi (gi)}
				{#if gi}<i class="gap"></i>{/if}
				{#each { length: g.to - g.from + 1 } as _, j (j)}
					{@const k = g.from + j}
					<i class:done={!pos.cover && k < pos.at} class:now={!pos.cover && k === pos.at}></i>
				{/each}
			{/each}
		</div>
	{/if}
</header>

{#if app.about}
	<Sheet bind:open={aboutOpen} title={t(app.aboutTitle)} placement="end" width={420}>
		{@render app.about()}
	</Sheet>
{/if}
<Sheet bind:open={moreOpen} title={t(app.name)} placement="bottom">
	<nav class="mlist" aria-label={lui('views')}>
		{#each app.tabs as v (v.id)}
			<a href="#{v.id}" aria-current={current === v.id ? 'page' : undefined} onclick={() => (moreOpen = false)}>
				<Icon d={v.icon} size={22} />
				<span><b>{t(v.label)}</b>{#if v.note}<small>{t(v.note)}</small>{/if}</span>
			</a>
		{/each}
		{#if app.about}
			<button type="button" class="mrow-btn" onclick={openAbout}>
				<Icon d={info} size={22} />
				<span><b>{t(app.aboutTitle)}</b><small>{t({ en: 'Facts, how it was checked, credits', zh: '基本信息、如何核对、致谢' })}</small></span>
			</button>
		{/if}
	</nav>
	<div class="mrow"><span>{t({ en: 'Language', zh: '语言' })}</span><LangToggle variant="seg" /></div>
	<div class="mrow"><span>{t({ en: 'Theme', zh: '外观' })}</span><ThemeToggle /></div>
</Sheet>

<style>
	.skip {
		position: absolute;
		left: 8px;
		top: -60px;
		z-index: 100;
		padding: 8px 14px;
		border-radius: var(--r3);
		background: var(--act);
		color: var(--act-ink);
		font-weight: var(--w-ui);
	}
	.skip:focus {
		top: 8px;
	}
	.ltop {
		position: relative;
		z-index: 5;
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
		align-items: center;
		height: var(--ltop-h);
		padding: 0 12px 0 8px;
		border-bottom: 1px solid var(--rule);
		background: var(--panel);
	}
	.crumb {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
	}
	.home {
		display: grid;
		place-items: center;
		flex: none;
		width: 44px;
		height: 44px;
		border-radius: var(--r3);
	}
	.home:hover {
		background: var(--well);
	}
	.path {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
		white-space: nowrap;
	}
	.cr {
		display: flex;
		flex: 0 1 auto;
		gap: 6px;
		min-width: 0;
		overflow: hidden;
	}
	.cr a {
		min-width: 0;
		overflow: hidden;
		color: var(--muted);
		text-decoration: none;
		text-overflow: ellipsis;
	}
	/* the area crumb (with its slash) gives way first, then the name; the first crumb stays */
	.cr:not(.area) {
		flex-shrink: 0;
	}
	.cr.area {
		flex-shrink: 1000;
	}
	.cr a:hover {
		color: var(--ink);
		text-decoration: underline;
	}
	.sep {
		color: var(--faint);
	}
	.title {
		flex: 0 1 auto;
		min-width: 4em;
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
		font-size: var(--t-md);
		font-weight: var(--w-title);
		color: var(--ink);
	}
	.ttl {
		display: none;
	}
	.vtabs {
		display: flex;
		gap: 2px;
		height: var(--ltop-h);
	}
	.vtabs a {
		display: flex;
		align-items: center;
		gap: 8px;
		height: 100%;
		margin-bottom: -1px;
		padding: 0 14px;
		border-bottom: 2px solid transparent;
		color: var(--muted);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		text-decoration: none;
		white-space: nowrap;
	}
	.vtabs a:hover {
		color: var(--ink);
	}
	.vtabs a[aria-current='page'] {
		color: var(--ink);
		border-bottom-color: var(--here);
	}
	.count {
		padding: 4px 8px;
		border-radius: var(--r-pill);
		background: var(--here-bg);
		color: var(--ink-2);
		font-size: var(--t-xs);
		line-height: 1;
		font-weight: var(--w-ui);
	}
	.tools {
		justify-self: end;
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.ph,
	.phrail {
		display: none;
	}
	/* the More sheet */
	.mlist {
		display: flex;
		flex-direction: column;
	}
	.mlist a,
	.mrow-btn {
		display: grid;
		grid-template-columns: 24px 1fr;
		gap: 14px;
		align-items: center;
		min-height: 56px;
		padding: 10px 8px;
		border: 0;
		border-radius: var(--r3);
		background: none;
		color: var(--ink-2);
		text-align: left;
		text-decoration: none;
		font: inherit;
		cursor: pointer;
	}
	.mlist a[aria-current='page'] {
		background: var(--here-bg);
	}
	.mlist b {
		display: block;
		color: var(--ink);
		font-size: var(--t-md);
	}
	.mlist small {
		display: block;
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.mrow {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding: 10px 8px;
		border-top: 1px solid var(--rule);
		font-weight: var(--w-ui);
	}
	@media (max-width: 1099px) {
		.vtabs a {
			padding: 0 9px;
		}
		.vtabs :global(.vi),
		.cr.area {
			display: none;
		}
	}
	@media (max-width: 899px) {
		.vtabs a {
			padding: 0 7px;
		}
		.count {
			display: none;
		}
	}
	@media (max-width: 759px) {
		.ltop {
			grid-template-columns: minmax(0, 1fr) auto;
			padding: 0 4px 0 2px;
		}
		.vtabs,
		.dt,
		.path {
			display: none;
		}
		.ph {
			display: inline-grid;
		}
		.ttl {
			display: flex;
			flex-direction: column;
			min-width: 0;
			line-height: 1.25;
		}
		.ttl small {
			overflow: hidden;
			color: var(--muted);
			font-size: var(--t-xs);
			font-weight: var(--w-ui);
			white-space: nowrap;
			text-overflow: ellipsis;
		}
		.phrail {
			position: absolute;
			left: 0;
			right: 0;
			bottom: -1px;
			display: flex;
			gap: 3px;
			padding: 0 10px;
		}
		.phrail i {
			flex: 1;
			height: 3px;
			border-radius: var(--r-pill);
			background: var(--rule-strong);
		}
		.phrail i.done {
			background: var(--ink-2);
		}
		.phrail i.now {
			background: var(--ink);
			box-shadow: 0 0 0 2px var(--mark);
		}
		.phrail i.gap {
			flex: 0 0 4px;
			background: transparent;
		}
	}
	:global(html[lang='zh-CN']) .title {
		letter-spacing: 0;
	}
</style>
