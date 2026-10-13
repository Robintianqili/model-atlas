<!--
  The glossary browser (redesign spec §6.6). At 960 px and up, two panes in one screen: on the left the search, the
  letters (pinyin initials in Chinese), the area chips and the term list (the pane scrolls); on the right the entry: the
  term in both languages, its definition, related terms and the lessons that use it, with a link to the first step that
  does. A deep link #<id> selects the entry (the URL follows the selection). Phones: the search, a sticky letter bar,
  and rows that expand in place.
    <GlossaryBrowser terms={GLOSSARY} used={data.used} lessons={data.lessons} areas={data.areas} />
-->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';
	import Icon from '../Icon.svelte';
	import { search } from '../icons';
	import { glyphFor } from './glyphs';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { I18n } from '$lib/i18n/types';
	import type { GlossaryTerm } from '$lib/catalog/types';

	interface Props {
		terms: GlossaryTerm[];
		/** term id -> [lesson index, step number, step id] */
		used: Record<string, [number, number, string][]>;
		lessons: { title: I18n; href: string; group: string }[];
		areas: { slug: string; name: I18n }[];
	}
	let { terms, used, lessons, areas }: Props = $props();

	const BY_ID = $derived(new Map(terms.map((g) => [g.id, g])));
	const SHOW_USED = 8;

	let q = $state('');
	let area = $state<string | null>(null);
	let selected = $state('');
	/** phones: the letter whose terms are listed (the letter bar works as tabs there, so the page stays short) */
	let letter = $state('');
	let wide = $state(true);
	let allUsed = $state(false);
	let listEl: HTMLElement | undefined = $state();

	// pinyin initials without a dictionary: the first character of each letter in the zh-CN (pinyin) collation
	const BOUNDS = [...'阿芭擦搭蛾发噶哈击喀垃妈拿哦啪期然撒塌挖昔压匝'];
	const PY = 'ABCDEFGHJKLMNOPQRSTWXYZ';
	const zhCol = new Intl.Collator('zh-CN');
	const enCol = new Intl.Collator('en', { sensitivity: 'base' });
	function initial(s: string, zh: boolean): string {
		const c = s.trim()[0] ?? '#';
		if (/[a-z]/i.test(c)) return c.toUpperCase();
		if (!zh || !/[一-鿿]/.test(c)) return '#';
		let k = -1;
		for (let i = 0; i < BOUNDS.length && zhCol.compare(c, BOUNDS[i]) >= 0; i++) k = i;
		return k < 0 ? '#' : PY[k];
	}
	const nameOf = (g: GlossaryTerm, zh: boolean) => (zh ? (g.term.zh ?? g.term.en) : g.term.en);

	const zh = $derived(i18n.lang === 'zh');
	// grouped by initial first (in Chinese, Latin names and pinyin initials share the letters), then in collation order
	const sorted = $derived(
		[...terms]
			.map((g) => ({ g, L: initial(nameOf(g, zh), zh).replace('#', '~') }))
			.sort((a, b) => (a.L === b.L ? (zh ? zhCol.compare(nameOf(a.g, true), nameOf(b.g, true)) : enCol.compare(a.g.term.en, b.g.term.en)) : a.L < b.L ? -1 : 1))
			.map((x) => x.g)
	);
	const groupsOf = (id: string) => new Set((used[id] ?? []).map(([li]) => lessons[li].group));
	const words = $derived(q.trim().toLowerCase().split(/\s+/).filter(Boolean));
	const visible = $derived(
		sorted.filter((g) => {
			if (area && !groupsOf(g.id).has(area)) return false;
			if (!words.length) return true;
			const hay = `${g.term.en} ${g.term.zh ?? ''} ${g.def.en} ${g.def.zh ?? ''}`.toLowerCase();
			return words.every((w) => hay.includes(w));
		})
	);
	const byLetter = $derived.by(() => {
		const out: { letter: string; items: GlossaryTerm[] }[] = [];
		for (const g of visible) {
			const L = initial(nameOf(g, zh), zh);
			if (out.at(-1)?.letter !== L) out.push({ letter: L, items: [] });
			out.at(-1)!.items.push(g);
		}
		return out;
	});
	const letters = $derived(byLetter.map((b) => b.letter));
	const tab = $derived(letters.includes(letter) ? letter : (letters[0] ?? ''));
	const listed = $derived(wide || words.length ? byLetter : byLetter.filter((b) => b.letter === tab));
	const entry = $derived(BY_ID.get(selected) ?? (wide ? visible[0] : undefined));
	const chips = $derived([...areas.map((a) => ({ slug: a.slug, name: a.name })), { slug: 'methods', name: { en: 'Training methods', zh: '训练方法' } }]);

	function select(id: string, write = true) {
		if (!BY_ID.has(id)) return;
		selected = selected === id && !wide ? '' : id;
		if (!wide) letter = initial(nameOf(BY_ID.get(id)!, zh), zh);
		allUsed = false;
		if (!wide && selected) void tick().then(() => listEl?.querySelector(`[data-id="${CSS.escape(id)}"]`)?.scrollIntoView({ block: 'start' }));
		if (write) {
			try {
				replaceState(`${page.url.pathname}${page.url.search}${selected ? `#${selected}` : ''}`, page.state);
			} catch {
				/* router not ready */
			}
		}
	}

	async function jump(L: string) {
		if (!wide) {
			letter = L;
			return;
		}
		const el = listEl?.querySelector<HTMLElement>(`[data-letter="${L}"]`);
		if (!el) return;
		el.scrollIntoView({ block: 'start' });
		await tick();
		el.nextElementSibling?.querySelector<HTMLButtonElement>('button')?.focus({ preventScroll: true });
	}

	onMount(() => {
		const mq = matchMedia('(min-width: 960px)');
		const sync = () => (wide = mq.matches);
		sync();
		mq.addEventListener('change', sync);
		const fromHash = () => {
			const id = decodeURIComponent(location.hash.slice(1));
			if (BY_ID.has(id)) {
				selected = id;
				letter = initial(nameOf(BY_ID.get(id)!, i18n.lang === 'zh'), i18n.lang === 'zh');
				void tick().then(() => listEl?.querySelector(`[data-id="${CSS.escape(id)}"]`)?.scrollIntoView({ block: wide ? 'center' : 'start' }));
			}
		};
		fromHash();
		addEventListener('hashchange', fromHash);
		return () => {
			mq.removeEventListener('change', sync);
			removeEventListener('hashchange', fromHash);
		};
	});

	const usedIn = (id: string) => used[id] ?? [];
</script>

{#snippet card(g: GlossaryTerm, inline: boolean)}
	{@const u = usedIn(g.id)}
	<div class="entry" class:inline>
		{#if !inline}
			<h2 class="term" id="term-title">{nameOf(g, zh)}</h2>
			<p class="other" lang={zh ? 'en' : 'zh-CN'}>{nameOf(g, !zh)}</p>
		{/if}
		<p class="def">{t(g.def)}</p>
		{#if g.related?.length}
			<h3 class="h">{ct('related')}</h3>
			<ul class="rel">
				{#each g.related.filter((r) => BY_ID.has(r)) as r (r)}
					<li><a class="chip" href="#{r}" onclick={(e) => (e.preventDefault(), select(r))}>{nameOf(BY_ID.get(r)!, zh)}</a></li>
				{/each}
			</ul>
		{/if}
		{#if u.length}
			<h3 class="h">{ct('usedIn')} <span class="n">{u.length}</span></h3>
			<ul class="used">
				{#each allUsed ? u : u.slice(0, SHOW_USED) as [li, k, step] (li)}
					{@const l = lessons[li]}
					<li>
						<a href="{base}{l.href}#lesson-{step}">{t(l.title)}</a>
						<span>{t({ en: `step ${k}`, zh: `第 ${k} 步` })}</span>
					</li>
				{/each}
			</ul>
			{#if u.length > SHOW_USED && !allUsed}
				<button type="button" class="linkbtn" onclick={() => (allUsed = true)}>{ct('andMore', { n: u.length - SHOW_USED })}</button>
			{/if}
		{/if}
	</div>
{/snippet}

<div class="gb">
	<div class="side">
		<div class="tools">
			<div class="find">
				<label for="gq" class="visually-hidden">{t({ en: 'Search the glossary', zh: '搜索术语表' })}</label>
				<span class="glass" aria-hidden="true"><Icon d={search} size={18} /></span>
				<input id="gq" type="search" bind:value={q} placeholder={t({ en: 'Search: softmax, 梯度, attention…', zh: '搜索：softmax、梯度、注意力…' })} autocomplete="off" />
			</div>
			<nav class="letters" aria-label={zh ? '按拼音首字母' : 'A–Z'}>
				{#each letters as L (L)}
					<button type="button" aria-pressed={wide || words.length ? undefined : tab === L} onclick={() => jump(L)}>{L}</button>
				{/each}
			</nav>
			<div class="areas" role="group" aria-label={t({ en: 'Used in lessons about', zh: '用在哪个领域的课程' })}>
				<button type="button" class="chip fill" aria-pressed={area === null} onclick={() => (area = null)}>{ct('kindAll')}</button>
				{#each chips as c (c.slug)}
					<button type="button" class="chip fill" aria-pressed={area === c.slug} onclick={() => (area = area === c.slug ? null : c.slug)}>
						<Icon d={glyphFor(c.slug === 'methods' ? 'foundations' : c.slug)} glyph size={14} />{t(c.name)}
					</button>
				{/each}
			</div>
			<p class="count" aria-live="polite">
				{visible.length === terms.length ? t({ en: `${terms.length} terms`, zh: `${terms.length} 个术语` }) : t({ en: `${visible.length} of ${terms.length} terms`, zh: `${visible.length} / ${terms.length} 个术语` })}
			</p>
		</div>
		<div class="list" bind:this={listEl}>
			{#each listed as b (b.letter)}
				<h3 class="lh" data-letter={b.letter}>{b.letter}</h3>
				<ul>
					{#each b.items as g (g.id)}
						{@const on = entry?.id === g.id}
						<li data-id={g.id}>
							<button
								type="button"
								class="row"
								aria-current={wide && on ? 'true' : undefined}
								aria-expanded={wide ? undefined : selected === g.id}
								aria-controls={wide ? 'gb-entry' : undefined}
								onclick={() => select(g.id)}
							>
								<span class="en">{nameOf(g, zh)}</span>
								<span class="zh" lang={zh ? 'en' : 'zh-CN'}>{nameOf(g, !zh)}</span>
							</button>
							{#if !wide && selected === g.id}{@render card(g, true)}{/if}
						</li>
					{/each}
				</ul>
			{/each}
			{#if !visible.length}
				<p class="none">{q.trim() ? ct('noMatch', { q: q.trim() }) : t({ en: 'No term matches.', zh: '没有匹配的术语。' })}</p>
			{/if}
		</div>
	</div>
	<section class="main" id="gb-entry" aria-live="polite" aria-labelledby={entry && wide ? 'term-title' : undefined}>
		{#if entry && wide}{@render card(entry, false)}{/if}
	</section>
</div>

<style>
	.gb {
		display: grid;
		grid-template-columns: 360px minmax(0, 1fr);
		gap: 0;
		height: clamp(480px, calc(100dvh - var(--header-h) - 132px), 820px);
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		overflow: hidden;
	}
	.side {
		display: flex;
		flex-direction: column;
		min-height: 0;
		border-right: 1px solid var(--rule);
		background: var(--paper);
	}
	.tools {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 14px 14px 8px;
		border-bottom: 1px solid var(--rule);
	}
	.find {
		position: relative;
	}
	.glass {
		position: absolute;
		left: 12px;
		top: 50%;
		transform: translateY(-50%);
		color: var(--muted);
		pointer-events: none;
	}
	input {
		width: 100%;
		height: 44px;
		padding: 0 12px 0 40px;
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		background: var(--panel);
		color: var(--ink);
		font-size: var(--t-md);
	}
	input::placeholder {
		color: var(--muted);
		opacity: 1;
	}
	.letters {
		display: flex;
		flex-wrap: wrap;
		gap: 2px;
	}
	.letters button {
		min-width: 24px;
		height: 24px;
		padding: 0 4px;
		border: 0;
		border-radius: var(--r1);
		background: none;
		color: var(--ink-2);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
		cursor: pointer;
	}
	.letters button:hover {
		background: var(--well);
		color: var(--ink);
	}
	.letters button[aria-pressed='true'] {
		background: var(--act);
		color: var(--act-ink);
	}
	.areas {
		display: flex;
		gap: 6px;
		overflow-x: auto;
		padding-bottom: 4px;
		scrollbar-width: thin;
	}
	.areas .chip {
		flex: none;
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--t-xs);
	}
	.count {
		margin: 0;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.list {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overscroll-behavior: contain;
		scrollbar-gutter: stable;
		padding: 0 6px 12px;
	}
	.list ul,
	.rel,
	.used {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.lh {
		position: sticky;
		top: 0;
		z-index: 1;
		margin: 0;
		padding: 8px 8px 2px;
		background: var(--paper);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
		scroll-margin-top: 0;
	}
	.row {
		position: relative;
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 10px;
		width: 100%;
		min-height: 40px;
		padding: 8px 10px;
		border: 0;
		border-radius: var(--r2);
		background: none;
		color: var(--ink);
		text-align: left;
		cursor: pointer;
	}
	.row:hover {
		background: var(--well);
	}
	.row .en {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
	}
	.row .zh {
		flex: none;
		max-width: 45%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.row[aria-current],
	.row[aria-expanded='true'] {
		background: var(--here-bg);
	}
	.row[aria-current]::before {
		content: '';
		position: absolute;
		left: 0;
		top: 8px;
		bottom: 8px;
		width: 3px;
		border-radius: 3px;
		background: var(--here);
	}
	.none {
		padding: 16px 8px;
		color: var(--muted);
	}
	.main {
		min-height: 0;
		overflow-y: auto;
		padding: 36px 44px;
	}
	.entry {
		max-width: 640px;
	}
	.term {
		margin: 0;
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
	}
	.other {
		margin: 4px 0 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		color: var(--muted);
	}
	.def {
		margin: 20px 0 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		color: var(--ink);
	}
	:global(:root:lang(zh-CN)) .def {
		line-height: 1.8;
	}
	.h {
		margin: 28px 0 10px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.h .n {
		font-weight: var(--w-body);
		color: var(--muted);
	}
	.rel {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.used {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 6px 24px;
	}
	.used li {
		display: flex;
		align-items: baseline;
		gap: 8px;
		min-width: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.used a {
		font-weight: var(--w-ui);
	}
	.used span {
		flex: none;
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.linkbtn {
		margin-top: 10px;
		padding: 0;
		border: 0;
		background: none;
		color: var(--ink-2);
		font: inherit;
		font-size: var(--t-sm);
		text-decoration: underline;
		text-underline-offset: 3px;
		cursor: pointer;
	}
	@media (max-width: 959px) {
		.gb {
			display: block;
			height: auto;
			border: 0;
			border-radius: 0;
			background: none;
			overflow: visible;
		}
		.side {
			display: block;
			border: 0;
			background: none;
		}
		.tools {
			padding: 0 0 8px;
		}
		.letters {
			position: sticky;
			top: var(--header-h);
			z-index: 5;
			flex-wrap: nowrap;
			margin: 0 calc(-1 * var(--gutter));
			padding: 8px var(--gutter);
			overflow-x: auto;
			background: var(--paper);
			border-bottom: 1px solid var(--rule);
			scrollbar-width: none;
		}
		.letters button {
			flex: none;
			min-width: 36px;
			height: 36px;
			font-size: var(--t-sm);
		}
		.areas {
			margin: 0 calc(-1 * var(--gutter));
			padding: 0 var(--gutter) 4px;
		}
		.areas .chip {
			min-height: 40px;
		}
		.list {
			overflow: visible;
			padding: 0;
		}
		.lh {
			position: static;
			padding: 14px 0 4px;
			background: none;
			scroll-margin-top: calc(var(--header-h) + 60px);
		}
		li[data-id] {
			scroll-margin-top: calc(var(--header-h) + 60px);
		}
		.row {
			min-height: 48px;
			padding: 10px 4px;
			border-radius: 0;
			border-bottom: 1px solid var(--rule);
		}
		.row .zh {
			font-size: var(--t-sm);
		}
		.main {
			display: none;
		}
		.entry.inline {
			padding: 12px 4px 18px;
			border-bottom: 1px solid var(--rule);
		}
		.entry.inline .def {
			margin: 0;
			font-size: var(--t-md);
			line-height: var(--lh-md);
		}
		.entry.inline .h {
			margin: 16px 0 8px;
		}
		.used {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
