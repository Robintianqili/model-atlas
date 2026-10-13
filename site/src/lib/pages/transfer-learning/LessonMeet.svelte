<!--
  Lesson step 1: the 10-class TinyCNN must pick one of its 10 classes for the learner's drawing
  (live), and what it picks over all 1,500 test drawings of each new class (precomputed).
-->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import Thumb from './Thumb.svelte';
	import { BASE_CLASSES } from './data';
	import type { LessonState } from './lessonState.svelte';

	let { L, tolerance }: { L: LessonState; tolerance?: number } = $props();

	const classes = $derived(L.engine?.spec.classes ?? BASE_CLASSES);
	const name = (c: { en: string; zh: string }) => (i18n.lang === 'zh' ? c.zh : c.en);
	const items = $derived(
		L.old ? classes.map((c, i) => ({ label: name(c), p: L.old!.probs[i], id: i, color: categorical(i, themeState.resolved) })) : []
	);
	const shares = (w: 'ladder' | 'envelope') => {
		const s = L.finetune?.baseModelGuesses.byHeldOutClass[w]?.top1Share ?? {};
		const rows = Object.entries(s)
			.sort((a, b) => b[1] - a[1])
			.map(([id, p]) => {
				const i = classes.findIndex((c) => c.id === id);
				return { id, label: i >= 0 ? name(classes[i]) : id, p, color: i >= 0 ? categorical(i, themeState.resolved) : 'var(--surface-3)' };
			});
		const rest = 1 - rows.reduce((a, r) => a + r.p, 0);
		if (rest > 0.0005) rows.push({ id: 'other', label: t({ en: 'other', zh: '其他' }), p: rest, color: 'var(--surface-3)' });
		return rows;
	};
</script>

<div class="meet">
	<div class="row">
		<figure class="inp">
			<Thumb data={L.input} size={120} alt={t({ en: 'The drawing, as the model sees it (28 × 28)', zh: '模型看到的画（28 × 28）' })} />
			<figcaption>{t(L.label())}<br /><span class="muted">{t({ en: '28 × 28, as the model sees it', zh: '28 × 28，模型看到的样子' })}</span></figcaption>
		</figure>
		<div class="out">
			<div class="hd">
				<h4>{t({ en: 'The 10-class model’s answer', zh: '10 类模型的回答' })}</h4>
			</div>
			{#if L.old}
				<ProbBars {items} k={10} chosen={L.old.top} label={{ en: 'Probability per known class', zh: '每个已知类别的概率' }} />
				<p class="sum num">{t({ en: 'The 10 bars add up to 100%.', zh: '这 10 个条加起来是 100%。' })}</p>
			{:else}
				<p class="muted">{t({ en: 'Draw something on the pad, or pick a ladder or an envelope.', zh: '在画板上画点什么，或者选一张梯子或信封。' })}</p>
			{/if}
		</div>
	</div>

	<div class="shares">
		<div class="hd">
			<h4>{t({ en: 'Its top guess over all 1,500 test drawings of each new class', zh: '它对每个新类别全部 1,500 张测试图的首选答案' })}</h4>
			<ProvenanceBadge mode="precomputed" compact />
		</div>
		{#each ['ladder', 'envelope'] as const as w (w)}
			{@const rows = shares(w)}
			<div class="srow">
				<span class="sl">{t(w === 'ladder' ? { en: 'Ladders', zh: '梯子' } : { en: 'Envelopes', zh: '信封' })}</span>
				<span class="shbar" aria-hidden="true">
					{#each rows as r (r.id)}<span class="seg" style:width="{r.p * 100}%" style:background={r.color}></span>{/each}
				</span>
				<span class="keys">
					{#each rows as r (r.id)}<span class="key"><span class="sw" style:background={r.color}></span>{r.label} <span class="num">{(r.p * 100).toFixed(0)}%</span></span>{/each}
				</span>
			</div>
		{/each}
	</div>
</div>

<style>
	.meet {
		container: lm / inline-size;
		display: grid;
		gap: 14px;
		min-width: 0;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px 20px;
		align-items: start;
	}
	@container lm (min-width: 440px) {
		.row {
			grid-template-columns: auto minmax(0, 1fr);
		}
	}
	.inp {
		margin: 0;
		display: flex;
		gap: 10px;
		align-items: flex-end;
	}
	.inp :global(canvas) {
		border: 1px solid var(--rule);
	}
	figcaption {
		font-size: var(--fs-sm);
		max-width: 12em;
	}
	.hd {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: center;
		gap: 4px 10px;
		margin-bottom: 6px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.sum {
		margin: 6px 0 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.shares {
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
		display: grid;
		gap: 6px;
	}
	.srow {
		display: grid;
		grid-template-columns: 5.5em minmax(0, 1fr);
		gap: 8px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.shbar {
		display: flex;
		height: 14px;
		border-radius: 4px;
		overflow: hidden;
		gap: 1px;
		background: var(--surface);
	}
	.seg {
		display: block;
		min-width: 0;
	}
	.keys {
		grid-column: 2;
		display: flex;
		flex-wrap: wrap;
		gap: 2px 12px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.key {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
	}
</style>
