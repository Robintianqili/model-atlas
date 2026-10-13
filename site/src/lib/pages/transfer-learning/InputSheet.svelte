<!--
  The input sheet (redesign spec §4.4 InputSheet; the stage header's chip and #input open it): which drawing lesson steps
  1–6 use: a held-out ladder or envelope test drawing (‹ › to step through them), or one the learner draws. Neither
  is ever added to the head's training set.
-->
<script lang="ts">
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import Icon from '$lib/kit/Icon.svelte';
	import { left, right } from '$lib/kit/icons';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtPct } from '$lib/kit/format';
	import Thumb from './Thumb.svelte';
	import { BASE_CLASSES } from './data';
	import type { LessonState } from './lessonState.svelte';

	let { L }: { L: LessonState } = $props();
	const classes = $derived(L.engine?.spec.classes ?? BASE_CLASSES);
	const src = $derived(L.source);
	function pick(cls: 0 | 1) {
		L.pickTest(cls);
		L.strokes = [];
		L.drawnInput = null;
	}
	function step(d: number) {
		L.strokes = [];
		L.stepTest(d);
	}
</script>

<div class="sheet">
	<div class="now">
		<Thumb data={L.input} size={72} alt={t({ en: 'The drawing the lesson uses, as the model sees it', zh: '课程正在用的画（模型看到的样子）' })} />
		<div>
			<p class="what">{t(L.label())}</p>
			{#if L.old}
				<p class="ans">{t({ en: '10-class model:', zh: '10 类模型：' })} <b>{t(classes[L.old.top])}</b> <span class="num">{fmtPct(L.old.p)}</span></p>
			{/if}
		</div>
	</div>

	<div class="row">
		<div class="seg" role="group" aria-label={t({ en: 'Which test drawing the lesson uses', zh: '课程用哪类测试图' })}>
			<button type="button" aria-pressed={src.kind === 'test' && src.cls === 0} onclick={() => pick(0)}>{t({ en: 'A ladder', zh: '梯子' })}</button>
			<button type="button" aria-pressed={src.kind === 'test' && src.cls === 1} onclick={() => pick(1)}>{t({ en: 'An envelope', zh: '信封' })}</button>
		</div>
		{#if src.kind === 'test'}
			<span class="nav">
				<button type="button" class="icon-btn" onclick={() => step(-1)} aria-label={t({ en: 'Previous test drawing', zh: '上一张测试图' })}><Icon d={left} size={18} /></button>
				<span class="num muted">{src.i + 1}/{L.testCount}</span>
				<button type="button" class="icon-btn" onclick={() => step(1)} aria-label={t({ en: 'Next test drawing', zh: '下一张测试图' })}><Icon d={right} size={18} /></button>
			</span>
		{/if}
	</div>

	<div class="pad">
		<p class="h">{t({ en: 'Or draw your own', zh: '或者自己画一张' })}</p>
		<DrawPad
			bind:strokes={L.strokes}
			brush={0.055}
			size={220}
			showBrush={false}
			onchange={(s) => L.drawing(s, true)}
			ondraw={(s) => L.drawing(s)}
			label={{ en: 'Lesson drawing pad', zh: '课程画板' }}
		/>
	</div>

	<p class="hint">
		{t({
			en: 'These ladders and envelopes are test drawings the lesson never trains on (held-out drawings), and your own drawings are never added to training either: the lesson’s head learns from 8 + 8 other drawings.',
			zh: '这里的梯子和信封都是课程从不拿来训练的测试图（留出的测试图），你自己画的也不会加进训练集：课程里的输出头从另外 8 + 8 张画中学习。'
		})}
	</p>
</div>

<style>
	.sheet {
		display: grid;
		gap: var(--s4);
	}
	.now {
		display: flex;
		align-items: center;
		gap: var(--s3);
	}
	.now :global(canvas) {
		border: 1px solid var(--rule);
		border-radius: var(--r2);
	}
	.now p {
		margin: 0;
	}
	.what {
		font-weight: var(--w-ui);
	}
	.ans {
		color: var(--ink-2);
		font-size: var(--t-sm);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--s2) var(--s3);
	}
	.nav {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: var(--t-sm);
	}
	.pad {
		display: grid;
		gap: var(--s2);
	}
	.h {
		margin: 0;
		font-weight: var(--w-ui);
	}
	.hint {
		margin: 0;
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
