<!--
  Credits and licences on About (redesign spec §6.7): static/licenses.json grouped by kind, each group a closed
  <details> with its count, each row the name (linked to its source), the licence (linked to the full text the
  repository ships, when there is one), what it is used for and the attribution. No four-column mono table.
    <LicenceList groups={data.licences} />
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { repoFile } from '$lib/project';
	import type { I18n } from '$lib/i18n/types';

	interface Row {
		name: string;
		url: string;
		license: string;
		licenseText?: string;
		usedFor: string;
		attribution?: string;
	}
	let { groups }: { groups: { id: string; title: I18n; rows: Row[] }[] } = $props();
	const isUrl = (u: string) => /^https?:\/\//.test(u);
</script>

<div class="ll">
	{#each groups as g (g.id)}
		{#if g.rows.length}
			<details>
				<summary><span class="gt">{t(g.title)}</span><span class="n">{g.rows.length}</span></summary>
				<ul>
					{#each g.rows as r, i (i)}
						<li>
							<p class="nm">
								{#if isUrl(r.url)}<a href={r.url} rel="noopener noreferrer">{r.name}</a>{:else}{r.name}{/if}
								<span class="lic">{#if r.licenseText}<a href={repoFile(r.licenseText)} rel="noopener">{r.license}</a>{:else}{r.license}{/if}</span>
							</p>
							<p class="use">{r.usedFor}</p>
							{#if r.attribution}<p class="att">{r.attribution}</p>{/if}
						</li>
					{/each}
				</ul>
			</details>
		{/if}
	{/each}
</div>

<style>
	.ll {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	details {
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
	}
	summary {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		min-height: 48px;
		padding: 10px 16px;
		cursor: pointer;
		font-weight: var(--w-ui);
	}
	summary::marker {
		color: var(--muted);
	}
	.gt {
		flex: 1;
	}
	.n {
		font-size: var(--t-xs);
		font-weight: var(--w-body);
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}
	ul {
		margin: 0;
		padding: 0 16px 8px;
		list-style: none;
	}
	li {
		padding: 12px 0;
		border-top: 1px solid var(--rule);
	}
	li p {
		margin: 0;
		max-width: 80ch;
	}
	.nm {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 2px 12px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
	}
	.lic {
		font-size: var(--t-xs);
		font-weight: var(--w-body);
		color: var(--muted);
	}
	.lic a {
		color: var(--ink-2);
	}
	.use {
		margin-top: 4px !important;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.att {
		margin-top: 4px !important;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
</style>
