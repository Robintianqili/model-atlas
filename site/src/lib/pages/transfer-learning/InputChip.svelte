<!--
  The stage header's input chip (redesign spec §4.4 InputChip): the drawing steps 1–6 use (as the model sees it), which
  one it is, the 10-class model's answer (from step 4 on, the new head's instead), ‹ › for the neighbouring held-out test
  drawing, and Edit, which opens the input sheet (pick a ladder or an envelope, or draw).
-->
<script lang="ts">
	import Icon from '$lib/kit/Icon.svelte';
	import { left, pencil, right } from '$lib/kit/icons';
	import { getLessonApp } from '$lib/kit/lesson/app';
	import { lui, t } from '$lib/i18n/i18n.svelte';
	import { fmtPct } from '$lib/kit/format';
	import Thumb from './Thumb.svelte';
	import { BASE_CLASSES } from './data';
	import { clsName, type LessonState } from './lessonState.svelte';

	let { L, open }: { L: LessonState; open: () => void } = $props();
	const app = getLessonApp();
	const classes = $derived(L.engine?.spec.classes ?? BASE_CLASSES);
	/** from step 4 on, the new head's answer too */
	const withHead = $derived(!!app && !app.pos.cover && app.pos.at >= 3);
	const src = $derived(L.source);
	function step(d: number) {
		L.strokes = [];
		L.stepTest(d);
	}
</script>

<div class="inchip">
	<span class="pic"><Thumb data={L.input} size={36} alt={t({ en: 'The drawing the lesson uses, as the model sees it', zh: '课程正在用的画（模型看到的样子）' })} /></span>
	<span class="ilbl">
		<small>{lui('yourDrawing')} · {t(L.label())}</small>
		<b>
			{#if withHead && L.fresh}
				{@const k = L.fresh.top as 0 | 1}
				{t({ en: `New head, step ${L.headStep}:`, zh: `新输出头，第 ${L.headStep} 步：` })} {t(clsName(k))} <span class="num">{fmtPct(L.fresh.probs[k])}</span>
			{:else if L.old}
				{t({ en: '10-class model:', zh: '10 类模型：' })} {t(classes[L.old.top])} <span class="num">{fmtPct(L.old.p)}</span>
			{:else}
				{t({ en: 'Draw something, or pick a drawing', zh: '画点什么，或者选一张画' })}
			{/if}
		</b>
	</span>
	{#if src.kind === 'test'}
		<span class="nav">
			<button type="button" class="icon-btn" onclick={() => step(-1)} aria-label={t({ en: 'Previous test drawing', zh: '上一张测试图' })}><Icon d={left} size={18} /></button>
			<button type="button" class="icon-btn" onclick={() => step(1)} aria-label={t({ en: 'Next test drawing', zh: '下一张测试图' })}><Icon d={right} size={18} /></button>
		</span>
	{/if}
	<button type="button" class="btn" onclick={open} aria-haspopup="dialog" aria-label={`${lui('edit')}: ${lui('yourDrawing')}`}>
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
		overflow: hidden;
		border-radius: 8px;
		outline: 1px solid var(--rule);
	}
	.pic :global(canvas) {
		display: block;
	}
	.ilbl {
		display: flex;
		flex-direction: column;
		min-width: 0;
		max-width: 280px;
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
	.nav {
		display: inline-flex;
		flex: none;
	}
	.nav .icon-btn {
		width: 36px;
		height: 40px;
	}
	.btn {
		flex: none;
		gap: 6px;
		min-height: 40px;
		padding: 0 12px;
	}
	@media (max-width: 759px) {
		.nav,
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
