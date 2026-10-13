<!-- Previous / next at the end of a DevKit page: two link cards (rel prev / next), stacked on phones. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import Glyph from './Glyph.svelte';
	import { left, right } from './icons.ts';

	interface Link {
		href: string;
		title: string;
	}
	let { label, prev = null, next = null }: { label: string; prev?: Link | null; next?: Link | null } = $props();
</script>

<nav class="pg" aria-label={label}>
	{#if prev}
		<a class="card prev" href="{base}{prev.href}" rel="prev">
			<span class="k"><Glyph d={left} size={16} />{t({ en: 'Previous', zh: '上一页' })}</span>
			<span class="ti">{prev.title}</span>
		</a>
	{:else}<span></span>{/if}
	{#if next}
		<a class="card next" href="{base}{next.href}" rel="next">
			<span class="k">{t({ en: 'Next', zh: '下一页' })}<Glyph d={right} size={16} /></span>
			<span class="ti">{next.title}</span>
		</a>
	{/if}
</nav>

<style>
	.pg {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: var(--s3);
		margin-top: var(--s7);
		padding-top: var(--s5);
		border-top: 1px solid var(--rule);
	}
	.card {
		display: grid;
		gap: 2px;
		padding: var(--s3) var(--s4);
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		background: var(--panel);
		color: var(--ink);
		text-decoration: none;
		transition: border-color var(--dur-1) var(--ease);
	}
	.card:hover {
		border-color: var(--ink);
	}
	.next {
		text-align: right;
	}
	.k {
		display: inline-flex;
		align-items: center;
		gap: var(--s1);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.next .k {
		justify-content: flex-end;
	}
	.ti {
		font-weight: var(--w-ui);
		line-height: var(--lh-md);
	}
	@media (max-width: 559px) {
		.pg {
			grid-template-columns: minmax(0, 1fr);
		}
		.pg > span:empty {
			display: none;
		}
	}
</style>
