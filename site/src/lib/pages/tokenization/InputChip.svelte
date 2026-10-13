<!--
  The stage header's input chip (redesign spec §4.4 InputChip): the text the current step uses (the training text on
  steps 1–5, the sentence on the cover and steps 6–7), its size, and Edit, which opens the input sheet (InputSheet).
-->
<script lang="ts">
	import Icon from '$lib/kit/Icon.svelte';
	import { pencil, type as typeIcon } from '$lib/kit/icons';
	import { getLessonApp } from '$lib/kit/lesson/app';
	import { lui, t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import type { TokPage } from './page.svelte';

	let { P, open }: { P: TokPage; open: () => void } = $props();
	const app = getLessonApp();
	/** steps 1–5 work on the training text; the cover and steps 6–7 on the sentence */
	const onText = $derived(!!app && !app.pos.cover && app.pos.at < 5);
	const text = $derived(onText ? P.trainText : P.sentence);
	const bytes = $derived(new TextEncoder().encode(text).length);
	const K = $derived(P.trace && P.trained === P.trainText ? P.trace.merges.length : null);
	const what = $derived(onText ? t({ en: 'Your text', zh: '你的文本' }) : t({ en: 'Your sentence', zh: '你的句子' }));
	const meta = $derived(
		onText
			? `${fmtInt(bytes)} ${t({ en: 'bytes', zh: '字节' })}${K !== null ? ` · ${fmtInt(K)} ${t({ en: 'merges learned', zh: '次合并' })}` : ''}`
			: `${fmtInt(Array.from(text).length)} ${t({ en: 'characters', zh: '个字符' })}`
	);
</script>

<div class="inchip">
	<span class="pic" aria-hidden="true"><Icon d={typeIcon} size={20} /></span>
	<span class="ilbl">
		<small>{what} · <span class="num">{meta}</span></small>
		<b lang={/[一-鿿]/.test(text) ? 'zh-CN' : undefined}>{text || t({ en: '(empty)', zh: '（空）' })}</b>
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
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
		white-space: nowrap;
		overflow: hidden;
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
		.pic {
			display: none;
		}
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
