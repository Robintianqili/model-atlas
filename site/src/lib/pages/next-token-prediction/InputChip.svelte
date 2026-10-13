<!--
  The stage header's input chip (redesign spec §4.4 InputChip): the learner's sentence, its token count and TinyGPT's
  loss on it (live), and Edit, which opens the input sheet (InputSheet).
-->
<script lang="ts">
	import Icon from '$lib/kit/Icon.svelte';
	import { pencil, type as typeIcon } from '$lib/kit/icons';
	import { lui, t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { NtpState } from './state.svelte';

	let { S, open }: { S: NtpState; open: () => void } = $props();
	const n = $derived(S.ids.length);
	const loss = $derived(S.fresh && S.score && S.score.scored > 0 ? S.score.meanNats : null);
	const what = $derived(t({ en: 'Your sentence', zh: '你的句子' }));
</script>

<div class="inchip">
	<span class="pic" aria-hidden="true"><Icon d={typeIcon} size={20} /></span>
	<span class="ilbl">
		<small
			>{what}{#if S.tok}{' · '}<span class="num">{n}</span> {t({ en: n === 1 ? 'token' : 'tokens', zh: '个词元' })}{#if loss !== null}{' · '}{t({ en: 'loss', zh: '损失' })} <span class="num">{fmt3(loss)}</span>{/if}{/if}</small
		>
		<b>{S.text || t({ en: '(empty)', zh: '（空）' })}</b>
	</span>
	<button type="button" class="btn" onclick={open} aria-haspopup="dialog" aria-label={`${lui('edit')}: ${what}`}>
		<Icon d={pencil} size={16} /><span class="bt" aria-hidden="true">{lui('edit')}</span>
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
	}
	.ilbl {
		display: flex;
		flex-direction: column;
		min-width: 0;
		max-width: 300px;
		line-height: 1.25;
	}
	.ilbl small {
		overflow: hidden;
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.ilbl b {
		overflow: hidden;
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.btn {
		flex: none;
		gap: 6px;
		min-height: 40px;
		padding: 0 12px;
	}
	@media (max-width: 759px) {
		.pic,
		.bt {
			display: none;
		}
		.btn {
			min-width: 44px;
			min-height: 44px;
			padding: 0;
			justify-content: center;
		}
	}
</style>
