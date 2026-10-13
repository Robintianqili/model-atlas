<!--
  ViewHost (redesign spec §5.9): a view of a lesson page other than the Lesson (Explore, How it works, Code, Training;
  Explore, Pitfalls, Sources for concepts). The top bar stays (sticky), the page scrolls normally, there is no action
  bar, and a floating pill leads back to the lesson ("Back to the lesson · step 2 of 10"). The content is a lazy panel
  (a loader, rendered only once its view opens, so the lesson shell carries none of it) or the page's own snippets.
  A section (#explore-zoom, the old #overview) is scrolled into view once the panel has rendered.
-->
<script lang="ts">
	import { tick, type Component, type Snippet } from 'svelte';
	import Icon from '../Icon.svelte';
	import { left } from '../icons';
	import { lui, t, type Text } from '$lib/i18n/i18n.svelte';
	import LessonTop from './LessonTop.svelte';
	import type { LessonApp } from './app';
	import type { ViewId } from './views';

	interface Props {
		app: LessonApp;
		view: ViewId;
		section?: string;
		title: Text;
		intro?: Text;
		/** the panel, loaded when the view opens */
		loader?: () => Promise<{ default: Component<any> }>;
		/** what the panel receives as `vp` */
		vp?: Record<string, unknown>;
		/** the page's own content for this view (the old section snippets) */
		content?: Snippet;
		/** content width: 'wide' 1240 (Explore, Code), 'narrow' 880 (Training), 'article' 680 */
		width?: 'wide' | 'narrow' | 'article';
	}
	let { app, view, section, title, intro, loader, vp = {}, content, width = 'wide' }: Props = $props();

	let Comp = $state<Component<any> | null>(null);
	let failed = $state('');
	let loadedFor: unknown = null;
	$effect(() => {
		const l = loader;
		if (!l || l === loadedFor) return;
		loadedFor = l;
		Comp = null;
		l().then(
			(m) => (Comp = m.default),
			(e) => (failed = e instanceof Error ? e.message : String(e))
		);
	});
	const ready = $derived(!loader || !!Comp);
	// the view opens at its top, or at its section once the panel has rendered
	$effect(() => {
		void view;
		const s = section;
		if (!ready) return;
		tick().then(() =>
			requestAnimationFrame(() => {
				const el = s ? (document.getElementById(`${view}-${s}`) ?? document.getElementById(s)) : null;
				if (el) el.scrollIntoView({ block: 'start' });
				else if (!app.embedded) window.scrollTo({ top: 0 });
			})
		);
	});
	const pos = $derived(app.pos);
</script>

<div class="vpage">
	<div class="vtop"><LessonTop {app} current={view} /></div>
	<svelte:element this={app.embedded ? 'div' : 'main'} id={app.embedded ? undefined : 'main'} class="vmain {width}" tabindex="-1">
		<header class="vh">
			<p class="eyebrow">{t(app.name)}</p>
			<h1>{t(title)}</h1>
			{#if intro}<p class="lede">{t(intro)}</p>{/if}
		</header>
		{#if loader}
			{#if Comp}
				<Comp {vp} {section} />
			{:else}
				<p class="muted" role="status">{failed || t({ en: 'Loading…', zh: '加载中……' })}</p>
			{/if}
		{:else if content}
			{@render content()}
		{/if}
	</svelte:element>
	{#if pos.n}
		<a class="back-pill" href="#lesson"><Icon d={left} size={16} />{pos.cover ? lui('viewLesson') : lui('backToLesson', { k: pos.at + 1, n: pos.n })}</a>
	{/if}
</div>

<style>
	.vpage {
		min-height: 100vh;
		min-height: 100dvh;
		background: var(--paper);
	}
	.vtop {
		position: sticky;
		top: 0;
		z-index: 10;
	}
	.vmain {
		display: block;
		margin: 0 auto;
		padding: var(--s6) var(--gutter) calc(var(--s8) + 24px);
		outline: none;
	}
	.vmain.wide {
		max-width: var(--page-max);
	}
	.vmain.narrow {
		max-width: calc(880px + 2 * var(--gutter));
	}
	.vmain.article {
		max-width: calc(760px + 2 * var(--gutter));
	}
	.vh {
		display: grid;
		gap: 6px;
		margin-bottom: var(--s5);
	}
	.vh p,
	.vh h1 {
		margin: 0;
	}
	.vh .eyebrow {
		color: var(--muted);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
	}
	.lede {
		max-width: var(--measure);
		color: var(--ink-2);
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.back-pill {
		position: fixed;
		right: 20px;
		bottom: calc(20px + env(safe-area-inset-bottom, 0px));
		z-index: 12;
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-height: 44px;
		padding: 0 18px 0 14px;
		border-radius: var(--r-pill);
		background: var(--act);
		color: var(--act-ink);
		box-shadow: var(--shadow-2);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		text-decoration: none;
	}
	.back-pill:hover {
		background: var(--act-hover);
		color: var(--act-ink);
	}
	@media (max-width: 759px) {
		.vtop {
			position: static;
		}
		.back-pill {
			left: 16px;
			right: 16px;
			justify-content: center;
		}
	}
</style>
