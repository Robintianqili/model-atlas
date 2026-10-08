<script lang="ts">
	import { GLOSSARY, GLOSSARY_BY_ID } from '$lib/catalog/glossary';
	import { i18n, t, ui } from '$lib/i18n/i18n.svelte';

	let q = $state('');
	const sorted = [...GLOSSARY].sort((a, b) => a.term.en.localeCompare(b.term.en, 'en', { sensitivity: 'base' }));
	const visible = $derived(
		q.trim()
			? sorted.filter((g) => {
					const s = q.trim().toLowerCase();
					return (
						g.term.en.toLowerCase().includes(s) ||
						(g.term.zh ?? '').includes(q.trim()) ||
						g.def.en.toLowerCase().includes(s) ||
						(g.def.zh ?? '').includes(q.trim())
					);
				})
			: sorted
	);
	const letters = [...new Set(sorted.map((g) => g.term.en[0].toUpperCase()))];
	const first = (g: (typeof sorted)[number]) => g.term.en[0].toUpperCase();
	// one <dl> per letter (the letter heading cannot sit inside a <dl>: axe definition-list); one group while searching
	const groups = $derived.by(() => {
		if (q.trim()) return [{ letter: '', items: visible }];
		const out: { letter: string; items: typeof visible }[] = [];
		for (const g of visible) {
			const L = first(g);
			if (out.at(-1)?.letter !== L) out.push({ letter: L, items: [] });
			out.at(-1)!.items.push(g);
		}
		return out;
	});
</script>

<svelte:head><title>{ui('navGlossary')} · Model Atlas</title></svelte:head>

<div class="container page">
	<header class="head">
		<p class="eyebrow">/glossary/</p>
		<h1>{ui('navGlossary')}</h1>
		<p class="lede">
			{t({
				en: `${GLOSSARY.length} core terms, each with an English and a Chinese name and a one-sentence definition. Every page links its terms here.`,
				zh: `${GLOSSARY.length} 个核心术语，每个都有英文名、中文名和一句话定义。各页面中的术语都链接到这里。`
			})}
		</p>
		<label class="search">
			<span class="visually-hidden">{t({ en: 'Search the glossary', zh: '搜索术语表' })}</span>
			<input type="search" bind:value={q} placeholder={t({ en: 'Search: softmax, 梯度, attention…', zh: '搜索：softmax、梯度、注意力…' })} />
		</label>
		{#if !q}
			<nav class="letters" aria-label="A–Z">
				{#each letters as L (L)}<a href="#letter-{L}">{L}</a>{/each}
			</nav>
		{/if}
	</header>

	<div class="terms" aria-live="polite">
		{#each groups as gr (gr.letter)}
			{#if gr.letter}
				<div class="letter" id="letter-{gr.letter}" aria-hidden="true">{gr.letter}</div>
			{/if}
			<dl>
			{#each gr.items as g (g.id)}
			<div class="term" id={g.id}>
				<dt>
					<span class="main">{i18n.lang === 'zh' ? (g.term.zh ?? g.term.en) : g.term.en}</span>
					<span class="other" lang={i18n.lang === 'zh' ? 'en' : 'zh-CN'}>{i18n.lang === 'zh' ? g.term.en : g.term.zh}</span>
					<a class="anchor" href="#{g.id}" aria-label="Link to {g.term.en}">#</a>
				</dt>
				<dd>
					{t(g.def)}
					{#if g.related?.length}
						<span class="rel">
							{t({ en: 'See also', zh: '另见' })}:
							{#each g.related as r, j (r)}<a href="#{r}">{t(GLOSSARY_BY_ID[r].term)}</a>{j < g.related.length - 1 ? ', ' : ''}{/each}
						</span>
					{/if}
				</dd>
			</div>
			{/each}
			</dl>
		{/each}
	</div>
	{#if visible.length === 0}<p class="muted">{t({ en: 'No term matches.', zh: '没有匹配的术语。' })}</p>{/if}
</div>

<style>
	.page {
		padding-top: var(--s6);
	}
	.head {
		display: grid;
		gap: 10px;
		margin-bottom: var(--s4);
	}
	.head h1,
	.head p {
		margin: 0;
	}
	.search input {
		width: min(480px, 100%);
		min-height: 44px;
		padding: 0 14px;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.letters {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 10px;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
	}
	.terms {
		margin: 0;
		max-width: 860px;
	}
	.terms dl {
		margin: 0;
	}
	.letter {
		font-family: var(--font-display);
		font-size: var(--fs-xl);
		color: var(--accent);
		margin-top: var(--s5);
		padding-bottom: 4px;
		border-bottom: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.term {
		padding: 12px 0;
		border-bottom: 1px solid var(--rule);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.term:target {
		background: var(--accent-soft);
		border-radius: var(--r2);
		padding-inline: 10px;
	}
	dt {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
	}
	.main {
		font-family: var(--font-display);
		font-weight: 650;
		font-size: 1.15rem;
	}
	.other {
		color: var(--muted);
	}
	.anchor {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--muted);
		text-decoration: none;
		opacity: 0.6;
	}
	dd {
		margin: 4px 0 0;
		color: var(--ink-2);
	}
	.rel {
		display: block;
		margin-top: 4px;
		font-size: var(--fs-sm);
		color: var(--muted);
	}
</style>
