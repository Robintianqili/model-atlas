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
		<p class="k">{t({ en: 'It produces', zh: '得到的结果' })} · <span class="cap">{t(p.caption)}</span></p>
		{#if p.provenance}<ProvenanceBadge mode={p.provenance} compact {note} />{/if}
	</div>
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

<style>
	.pr {
		display: grid;
		gap: 8px;
		min-width: 0;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-left: 3px solid var(--accent);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.bar {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 4px 10px;
	}
	.k {
		margin: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.cap {
		text-transform: none;
		letter-spacing: 0;
		font-family: var(--font-body);
		font-size: var(--fs-sm);
		color: var(--ink);
		font-weight: 600;
	}
	.text,
	.note {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.note {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.list {
		margin: 0;
		padding-left: 1.2em;
		font-size: var(--fs-sm);
	}
	.list code {
		font-size: 0.85em;
		overflow-wrap: anywhere;
	}
	.tw {
		overflow-x: auto;
		min-width: 0;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--fs-sm);
		line-height: 1.4;
	}
	th,
	td {
		padding: 5px 8px;
		text-align: left;
		vertical-align: top;
		border-bottom: 1px solid var(--rule);
	}
	th {
		font-weight: 600;
		color: var(--muted);
		font-size: var(--fs-xs);
		white-space: nowrap;
	}
	td.num {
		font-family: var(--font-mono);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	.pr :global(.dk-md .code) {
		margin-bottom: 0;
	}
</style>
