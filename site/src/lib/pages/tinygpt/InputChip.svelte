<!--
  The stage header's input chip (redesign spec §4.4 InputChip) on /m/tinygpt/: your text (how many tokens it became),
  the model's next-token guess with its probability, and Edit, which opens the input sheet. Live.
-->
<script lang="ts">
	import Icon from '$lib/kit/Icon.svelte';
	import { pencil, type as typeIcon } from '$lib/kit/icons';
	import { fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { argmax } from './analysis';
	import type { TGPage } from './page.svelte';

	let { P, open }: { P: TGPage; open: () => void } = $props();
	const top = $derived(P.modelProbs ? argmax(P.modelProbs) : -1);
	const n = $derived(P.promptIds.length + P.genIds.length);
</script>

<div class="inchip">
	<span class="pic" aria-hidden="true"><Icon d={typeIcon} size={20} /></span>
	<span class="ilbl">
		<small>{t({ en: `Your text · ${n} tokens`, zh: `你的文本 · ${n} 个词元` })}</small>
		{#if top >= 0 && P.modelProbs}
			<b>{t({ en: 'next:', zh: '下一个：' })} <span class="tok">{P.S.idLabel(top)}</span> <span class="num">{fmtPct(P.modelProbs[top])}</span></b>
		{:else}
			<b>{P.loadState === 'error' ? t({ en: 'The model could not be loaded', zh: '模型加载失败' }) : t({ en: 'Loading…', zh: '加载中…' })}</b>
		{/if}
	</span>
	<button type="button" class="btn" onclick={open} aria-haspopup="dialog" aria-label={t({ en: 'Edit: change the text', zh: '编辑：修改文本' })}>
		<Icon d={pencil} size={16} /><span class="bt" aria-hidden="true">{t({ en: 'Edit', zh: '编辑' })}</span>
	</button>
</div>

<style>
	.inchip {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		max-width: 100%;
		padding: 5px;
		border: 1px solid var(--rule);
		border-radius: 14px;
		background: var(--panel);
		box-shadow: var(--shadow-1);
	}
	.pic {
		display: grid;
		place-items: center;
		flex: none;
		width: 36px;
		height: 36px;
		border-radius: 8px;
		background: var(--well);
		color: var(--ink-2);
		outline: 1px solid var(--rule);
	}
	.ilbl {
		display: flex;
		flex-direction: column;
		min-width: 0;
		padding-right: 2px;
		line-height: 1.2;
	}
	.ilbl small {
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
	}
	.ilbl b {
		overflow: hidden;
		font-size: var(--t-sm);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.ilbl b .num {
		color: var(--ink-2);
		font-weight: var(--w-ui);
	}
	.tok {
		font-family: var(--font-mono);
		font-variant-ligatures: none;
		white-space: pre;
	}
	.btn {
		min-height: 36px;
		padding: 0 12px;
		border-radius: 10px;
		font-size: var(--t-sm);
	}
	@media (max-width: 759px) {
		.inchip {
			gap: 8px;
			padding: 4px;
			border-radius: 12px;
			box-shadow: none;
		}
		.pic {
			width: 32px;
			height: 32px;
		}
		.ilbl small {
			display: none;
		}
		.btn {
			min-width: 44px;
			min-height: 44px;
			padding: 0 10px;
		}
		.bt {
			display: none;
		}
	}
</style>
