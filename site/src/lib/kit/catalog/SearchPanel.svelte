<!--
  The search palette's results (redesign spec §4.5): loaded by SearchPalette on first use, so no page shell carries it.
  It fetches search-index.json once, matches as you type (lessons, areas, glossary terms, planned lessons, both
  languages), and makes the header field a combobox: arrow keys move, Enter opens, Esc closes and keeps focus in the field.
  On phones (mode 'sheet') it opens a bottom sheet with its own field.
    <SearchPanel input={el} bind:q bind:open mode="inline" />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { goto } from '$app/navigation';
	import Sheet from '../Sheet.svelte';
	import Icon from '../Icon.svelte';
	import { search as searchIcon } from '../icons';
	import { searchGroups, type SearchGroup, type SearchKind } from './search';
	import { t, ui, i18n } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { SearchEntry } from './types';

	interface Props {
		/** the header field (inline mode) */
		input?: HTMLInputElement;
		q: string;
		open: boolean;
		mode?: 'inline' | 'sheet';
	}
	let { input, q = $bindable(''), open = $bindable(false), mode = 'inline' }: Props = $props();

	let entries = $state<SearchEntry[] | null>(null);
	let failed = $state(false);
	let active = $state(-1);
	let sheetInput: HTMLInputElement | undefined = $state();
	const id = $props.id();
	/** the inline list hangs under the header field (fixed: the panel is not inside the field's box) */
	let box = $state({ top: 0, left: 0, width: 360 });
	function place() {
		if (mode !== 'inline' || !input) return;
		const r = input.getBoundingClientRect();
		const width = Math.min(Math.max(r.width, 360), innerWidth - 16);
		box = { top: r.bottom + 6, left: Math.max(8, Math.min(r.left, innerWidth - width - 8)), width };
	}

	$effect(() => {
		if (entries || failed) return;
		fetch(`${base}/search-index.json`)
			.then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
			.then((e: SearchEntry[]) => (entries = e))
			.catch(() => (failed = true));
	});

	const groups = $derived<SearchGroup[]>(entries ? searchGroups(entries, q, i18n.lang) : []);
	const flat = $derived(groups.flatMap((g) => g.hits));
	const shown = $derived(open && q.trim().length > 0);
	$effect(() => {
		if (!shown) return;
		place();
		addEventListener('resize', place);
		addEventListener('scroll', place, { passive: true });
		return () => {
			removeEventListener('resize', place);
			removeEventListener('scroll', place);
		};
	});
	$effect(() => {
		void q;
		active = -1;
	});

	const KIND: Record<SearchKind, { en: string; zh: string }> = {
		l: { en: 'Lessons', zh: '课程' },
		a: { en: 'Areas and training methods', zh: '领域与训练方法' },
		t: { en: 'Glossary', zh: '术语表' },
		p: { en: 'Planned', zh: '计划中' }
	};

	function go(e: SearchEntry) {
		open = false;
		q = '';
		void goto(`${base}${e.href}`);
	}

	function onkey(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			if (shown) {
				e.preventDefault();
				open = false;
			}
			return;
		}
		if (!shown) {
			if (e.key === 'ArrowDown' && q.trim()) open = true;
			return;
		}
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			const n = flat.length;
			if (!n) return;
			active = e.key === 'ArrowDown' ? (active + 1) % n : (active - 1 + n) % n;
			document.getElementById(`${id}-o${active}`)?.scrollIntoView({ block: 'nearest' });
		} else if (e.key === 'Enter' && active >= 0 && flat[active]) {
			e.preventDefault();
			go(flat[active]);
		}
	}

	// the header field becomes the combobox (inline mode); the attributes go away with this panel
	$effect(() => {
		const el = mode === 'inline' ? input : sheetInput;
		if (!el) return;
		el.setAttribute('role', 'combobox');
		el.setAttribute('aria-autocomplete', 'list');
		el.setAttribute('aria-controls', `${id}-list`);
		el.addEventListener('keydown', onkey);
		const reopen = () => (open = true);
		el.addEventListener('input', reopen);
		const close = (e: FocusEvent) => {
			const to = e.relatedTarget as Node | null;
			if (!to || !document.getElementById(`${id}-list`)?.contains(to)) open = mode === 'sheet' ? open : false;
		};
		el.addEventListener('focusout', close);
		return () => {
			el.removeEventListener('keydown', onkey);
			el.removeEventListener('input', reopen);
			el.removeEventListener('focusout', close);
			for (const a of ['role', 'aria-autocomplete', 'aria-controls', 'aria-expanded', 'aria-activedescendant']) el.removeAttribute(a);
		};
	});
	$effect(() => {
		const el = mode === 'inline' ? input : sheetInput;
		if (!el) return;
		el.setAttribute('aria-expanded', String(shown));
		if (shown && active >= 0) el.setAttribute('aria-activedescendant', `${id}-o${active}`);
		else el.removeAttribute('aria-activedescendant');
	});
	$effect(() => {
		if (mode === 'sheet' && open && sheetInput) sheetInput.focus();
	});

	const indexOf = (e: SearchEntry) => flat.indexOf(e);
</script>

{#snippet results()}
	<div
		class="list"
		id="{id}-list"
		role="listbox"
		aria-label={ui('searchLabel')}
		hidden={!shown}
		style:top={mode === 'inline' ? `${box.top}px` : undefined}
		style:left={mode === 'inline' ? `${box.left}px` : undefined}
		style:width={mode === 'inline' ? `${box.width}px` : undefined}
	>
		{#if shown}
			{#if failed}
				<p class="msg">{t({ en: 'The search index did not load.', zh: '搜索索引没有加载成功。' })}</p>
			{:else if !entries}
				<p class="msg">{t({ en: 'Loading…', zh: '加载中…' })}</p>
			{:else if !flat.length}
				<p class="msg">{ct('noMatch', { q: q.trim() })}</p>
			{:else}
				{#each groups as g (g.kind)}
					<div class="grp" role="group" aria-labelledby="{id}-g{g.kind}">
						<p class="gh" id="{id}-g{g.kind}" role="presentation">{t(KIND[g.kind])}</p>
						{#each g.hits as e (e.href)}
							{@const i = indexOf(e)}
							<div
								class="opt"
								id="{id}-o{i}"
								role="option"
								aria-selected={i === active}
								tabindex="-1"
								onmousedown={(ev) => ev.preventDefault()}
								onclick={() => go(e)}
								onkeydown={(ev) => ev.key === 'Enter' && go(e)}
								onmousemove={() => (active = i)}
							>
								<span class="nm" lang={i18n.lang === 'zh' ? 'zh-CN' : undefined}>{i18n.lang === 'zh' ? e.zh : e.en}</span>
								{#if e.k === 't' && e.zh !== e.en}<span class="alt" lang={i18n.lang === 'zh' ? 'en' : 'zh-CN'}>{i18n.lang === 'zh' ? e.en : e.zh}</span>{/if}
								{#if e.sub}<span class="sub">{i18n.lang === 'zh' ? e.sub[1] : e.sub[0]}</span>{/if}
							</div>
						{/each}
					</div>
				{/each}
				<a class="all" href="{base}/atlas/?q={encodeURIComponent(q.trim())}" onclick={() => (open = false)}>
					{t({ en: `Search all lessons for “${q.trim()}”`, zh: `在全部课程里搜索“${q.trim()}”` })}
				</a>
			{/if}
		{/if}
	</div>
{/snippet}

{#if mode === 'sheet'}
	<Sheet bind:open label={ui('searchOpen')} placement="bottom">
		<div class="sheet">
			<div class="field">
				<label for="{id}-sq" class="visually-hidden">{ui('searchLabel')}</label>
				<span class="glass" aria-hidden="true"><Icon d={searchIcon} size={18} /></span>
				<input bind:this={sheetInput} bind:value={q} id="{id}-sq" type="search" placeholder={ui('searchLabel')} autocomplete="off" enterkeyhint="search" />
			</div>
			{@render results()}
		</div>
	</Sheet>
{:else}
	{@render results()}
{/if}

<style>
	.list {
		position: fixed;
		z-index: 40;
		max-height: min(70vh, 520px);
		overflow-y: auto;
		padding: 6px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		box-shadow: var(--shadow-2);
	}
	.list[hidden] {
		display: none;
	}
	.sheet .list {
		position: static;
		min-width: 0;
		max-height: none;
		margin-top: 10px;
		border: 0;
		box-shadow: none;
		padding: 0;
	}
	.msg {
		margin: 0;
		padding: 10px 12px;
		font-size: var(--t-sm);
		color: var(--muted);
	}
	.grp + .grp {
		margin-top: 4px;
		padding-top: 4px;
		border-top: 1px solid var(--rule);
	}
	.gh {
		margin: 0;
		padding: 6px 10px 2px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-title);
		color: var(--muted);
	}
	.opt {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0 8px;
		padding: 7px 10px;
		border-radius: var(--r2);
		cursor: pointer;
	}
	.opt[aria-selected='true'] {
		background: var(--here-bg);
		box-shadow: inset 3px 0 0 var(--here);
	}
	.nm {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		color: var(--ink);
	}
	.alt {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	.sub {
		flex-basis: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.all {
		display: block;
		margin-top: 4px;
		padding: 8px 10px;
		border-top: 1px solid var(--rule);
		font-size: var(--t-sm);
		color: var(--ink-2);
	}
	.sheet {
		padding: 4px 0 8px;
	}
	.field {
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
	.field input {
		width: 100%;
		height: 48px;
		padding: 0 12px 0 40px;
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		background: var(--panel);
		color: var(--ink);
		font-size: var(--t-md);
	}
</style>
