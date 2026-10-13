<!--
  Lesson step 6: the training step repeated 200 times. A replay of the lesson's live run: loss,
  training accuracy and accuracy on the 200 held-out test drawings after every step, plus the
  learner's own drawing under the head of the current step.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import TrainCharts from './TrainCharts.svelte';
	import { LESSON_N, LESSON_STEPS, clsName, type LessonState } from './lessonState.svelte';

	interface Props {
		L: LessonState;
		autoplay?: boolean;
		ondone?: () => void;
	}
	let { L, autoplay = false, ondone }: Props = $props();
	// every visit starts the replay at step 0 (the stage is created fresh on each lesson step); with
	// reduced motion nothing plays, so it opens on the finished run (scrub or step back from there)
	untrack(() => (L.replay = motion.reduced ? LESSON_STEPS : 0));

	// the side panel shows the learner's drawing under the head of the replayed step
	$effect(() => {
		L.headStep = L.replay;
	});
	const pct = (v: number) => (v * 100).toFixed(1) + '%';
</script>

{#if L.run}
	{@const run = L.run}
	{@const e = Math.min(L.replay, run.loss.length - 1)}
	<div class="lt">
		<div class="hd">
			<p class="muted small">
				{t({ en: `${LESSON_N} ladders + ${LESSON_N} envelopes · 514 trainable parameters · ${LESSON_STEPS} steps of full-batch gradient descent`, zh: `${LESSON_N} 张梯子 + ${LESSON_N} 张信封 · 514 个可训练参数 · ${LESSON_STEPS} 步全批量梯度下降` })}
			</p>
		</div>
		<div class="tl-player">
			<StepPlayer count={run.snaps.length} bind:step={L.replay} interval={28} {autoplay} onend={ondone} label={{ en: 'Replay the training, one step at a time', zh: '逐步回放训练过程' }}>
				{#snippet children()}
					<TrainCharts loss={run.loss} trainAcc={run.trainAcc} testAcc={run.testAcc} total={LESSON_STEPS} bind:epoch={L.replay} />
				{/snippet}
			</StepPlayer>
		</div>
		<p class="steplbl num" aria-live="off">{e === 0 ? t({ en: 'Step 0: the untrained head.', zh: '第 0 步：还没训练的输出头。' }) : t({ en: `After step ${e} of ${LESSON_STEPS}`, zh: `第 ${e} / ${LESSON_STEPS} 步之后` })}</p>
		<div class="stats">
			<div class="stat">
				<span class="lbl">{t({ en: 'Training loss', zh: '训练损失' })}</span>
				<span class="val num">{run.loss[e].toFixed(3)}</span>
				<span class="sub num">{t({ en: `from ${run.loss[0].toFixed(3)}`, zh: `起点 ${run.loss[0].toFixed(3)}` })}</span>
			</div>
			<div class="stat">
				<span class="lbl">{t({ en: 'Training examples right', zh: '训练样本判对' })}</span>
				<span class="val num">{pct(run.trainAcc[e])}</span>
				<span class="sub num">{Math.round(run.trainAcc[e] * 2 * LESSON_N)}/{2 * LESSON_N}</span>
			</div>
			<div class="stat main">
				<span class="lbl">{t({ en: 'Held-out test drawings right', zh: '留出测试图判对' })}</span>
				<span class="val num">{pct(run.testAcc[e])}</span>
				<span class="sub num">{t(clsName(0))} {run.testPer[e][0]}/{run.testN[0]} · {t(clsName(1))} {run.testPer[e][1]}/{run.testN[1]}</span>
			</div>
			{#if L.fresh}
				<div class="stat">
					<span class="lbl">{t({ en: 'This drawing: p(ladder)', zh: '这张画：p（梯子）' })}</span>
					<span class="val num">{L.fresh.probs[0].toFixed(3)}</span>
					<span class="sub">{t({ en: 'not used for training', zh: '没有参与训练' })}</span>
				</div>
			{/if}
		</div>
	</div>
{:else if L.presetError}
	<p class="muted">{t({ en: 'Could not compute the features. Please reload the page.', zh: '特征计算失败，请刷新页面。' })}</p>
{:else}
	<p class="muted">{t({ en: 'Computing the frozen features of the training drawings…', zh: '正在计算训练图的冻结特征…' })}</p>
{/if}

<style>
	.lt {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.hd {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		align-items: center;
		justify-content: space-between;
	}
	.small {
		margin: 0;
		font-size: var(--fs-sm);
	}
	/* the kit player counts from 1 ("Step 201 of 201"); here step 0 is the untrained head */
	.tl-player :global(.pos) {
		display: none;
	}
	.steplbl {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.stats {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.stat {
		display: grid;
		gap: 2px;
		padding: 6px 10px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		min-width: 120px;
	}
	.stat.main {
		border-color: var(--accent);
	}
	.lbl {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.val {
		font-size: var(--fs-lg);
		font-weight: 650;
	}
	.sub {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>
