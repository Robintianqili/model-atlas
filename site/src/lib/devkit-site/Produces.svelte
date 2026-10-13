<!-- What a tutorial step produces: read or computed from the real hello-xor package when the site was built. -->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import Html from './Html.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { Produce, T2 } from './types.ts';

	let { p }: { p: Produce } = $props();
	const cell = (c: string | T2) => (typeof c === 'string' ? c : t(c));
	const note = { en: 'Computed in Node from the real package when this site was built (the same code runs in the browser).', zh: '在网站构建时用真实的包在 Node 里算出（浏览器里运行的是同一份代码）。' };
</script>

<div class="pr">
	<div class="bar">
		<p class="k"><span class="what">{t({ en: 'It produces', zh: '得到的结果' })}</span> <span class="cap">{t(p.caption)}</span></p>
		{#if p.provenance}<ProvenanceBadge mode={p.provenance} compact {note} />{/if}
	</div>
	<div class="body">
		{#if p.text}<p class="text" lang="en">{p.text}</p>{/if}
		{#if p.table}
			<div class="tw" use:keyScroll>
				<table>
					<thead><tr>{#each p.table.head as h, i (i)}<th scope="col">{t(h)}</th>{/each}</tr></thead>
					<tbody>
						{#each p.table.rows as r, i (i)}
							<tr>{#each r as c, j (j)}<td class:num={p.table.num?.[j]}>{cell(c)}</td>{/each}</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
		{#if p.codeHtml}<Html html={p.codeHtml} />{/if}
		{#if p.list}
			<ul class="list">
				{#each p.list as x, i (i)}<li><code>{x}</code></li>{/each}
			</ul>
		{/if}
		{#if p.note}<p class="note">{t(p.note)}</p>{/if}
	</div>
</div>

<style>
	/* one container on the panel: a head (what this step produces, its provenance), then the result */
	.pr {
		display: grid;
		min-width: 0;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		overflow: hidden;
	}
	.bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: var(--s1) var(--s3);
		padding: var(--s2) var(--s3) var(--s2) var(--s4);
		border-bottom: 1px solid var(--rule);
	}
	.k {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.what {
		color: var(--muted);
	}
	.cap {
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.body {
		display: grid;
		gap: var(--s2);
		min-width: 0;
		padding: var(--s3) var(--s4);
	}
	.text,
	.note {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.note {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	.list {
		margin: 0;
		padding-left: 1.2em;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.list li + li {
		margin-top: 2px;
	}
	.list code {
		overflow-wrap: anywhere;
	}
	.tw {
		overflow-x: auto;
		min-width: 0;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	th,
	td {
		padding: 6px var(--s2);
		text-align: left;
		vertical-align: top;
		border-bottom: 1px solid var(--rule);
	}
	tr:last-child td {
		border-bottom: 0;
	}
	th {
		font-weight: var(--w-ui);
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		white-space: nowrap;
	}
	td.num {
		font-family: var(--font-mono);
		font-variant-numeric: tabular-nums;
		font-size-adjust: var(--mono-adjust);
		white-space: nowrap;
	}
	.pr :global(.dk-md .code) {
		margin-bottom: 0;
	}
</style>
