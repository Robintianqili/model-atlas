<!--
  The learner's beat in the narration (redesign spec §4.4 YourTurn / PredictCard, §5.7): a predict question with 2-4
  full-width options (keys 1-4), answered in place: the chosen option shows ✓ or ✕ in the ok / no colours and the others
  stay visible but muted. Or a task the learner does in the stage, with its status line. On phones the feedback renders
  here under the options (on wider screens the feedback bar replaces the action bar, unless `inline`).
-->
<script lang="ts">
	import Icon from '../Icon.svelte';
	import { check, help, x } from '../icons';
	import { lui, t } from '$lib/i18n/i18n.svelte';
	import type { WalkPredict, WalkTask } from '../types';

	interface Props {
		predict?: WalkPredict;
		task?: WalkTask;
		/** the chosen option (-1: revealed with Show me; null: not answered) */
		answered?: number | null;
		onanswer?: (i: number) => void;
		/** show the feedback message here on every width (no feedback bar is up) */
		inline?: boolean;
		/** the task's status, read live */
		taskDone?: boolean;
	}
	let { predict, task, answered = null, onanswer, inline = false, taskDone = false }: Props = $props();
	const gid = $props.id();
	const right = $derived(predict && answered !== null && answered >= 0 ? answered === predict.answer : null);
</script>

{#if predict}
	<div class="beat" role="group" aria-labelledby="{gid}-q">
		<p class="beat-k"><Icon d={help} size={16} />{lui('predict')}</p>
		<p class="q" id="{gid}-q">{t(predict.question)}</p>
		<div class="opts" role="radiogroup" aria-labelledby="{gid}-q">
			{#each predict.options as o, i (i)}
				{@const isAns = answered !== null && i === predict.answer}
				{@const isWrong = answered !== null && answered === i && i !== predict.answer}
				<button
					type="button"
					class="opt"
					class:ok={isAns}
					class:no={isWrong}
					class:dim={answered !== null && !isAns && !isWrong}
					role="radio"
					aria-checked={answered === i}
					onclick={() => onanswer?.(i)}
				>
					<span class="kbd hint" aria-hidden="true">{i + 1}</span>
					<span>{t(o)}</span>
					{#if isAns}<Icon d={check} size={20} class="tick" />{:else if isWrong}<Icon d={x} size={20} class="cross" />{:else}<span></span>{/if}
				</button>
			{/each}
		</div>
		{#if answered !== null}
			<div class="fbi" class:inline class:no={right === false}>
				{#if right !== null}<b>{right ? lui('right') : lui('notQuite')}</b>{/if}
				{t(predict.explain)}
			</div>
		{/if}
	</div>
{:else if task}
	<div class="beat" class:done={taskDone}>
		<p class="beat-k"><Icon d={taskDone ? check : help} size={16} />{lui('yourTurn')}</p>
		<p class="q">{t(task.prompt)}</p>
		<p class="status" aria-live="polite">{taskDone ? t(task.doneText ?? { en: 'Done.', zh: '完成。' }) : t(task.waiting ?? { en: 'Not yet: do what the task says, in the stage.', zh: '还没完成：请在演示区里按任务操作。' })}</p>
		{#if task.hint && !taskDone}
			<details class="hint"><summary>{t({ en: 'Hint', zh: '提示' })}</summary><p>{t(task.hint)}</p></details>
		{/if}
	</div>
{/if}

<style>
	.beat {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 14px 16px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
	}
	.beat.done {
		border-color: color-mix(in srgb, var(--ok) 45%, var(--rule));
	}
	.beat p {
		margin: 0;
	}
	.beat-k {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-title);
	}
	.done .beat-k {
		color: var(--ok);
	}
	.q {
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-weight: var(--w-ui);
	}
	.opts {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.opts :global(.tick) {
		color: var(--ok);
	}
	.opts :global(.cross) {
		color: var(--no);
	}
	.status {
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.hint summary {
		cursor: pointer;
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
	}
	.hint p {
		margin-top: 6px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.fbi {
		display: none;
		padding: 12px 14px;
		border: 1px solid color-mix(in srgb, var(--ok) 45%, transparent);
		border-radius: var(--r3);
		background: var(--ok-soft);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.fbi.no {
		border-color: color-mix(in srgb, var(--no) 45%, transparent);
		background: var(--no-soft);
	}
	.fbi b {
		margin-right: 4px;
		color: var(--ok);
	}
	.fbi.no b {
		color: var(--no);
	}
	.fbi.inline {
		display: block;
	}
	@media (max-width: 759px) {
		.fbi {
			display: block;
		}
	}
	:global(html[lang='zh-CN']) .fbi {
		line-height: 1.7;
	}
</style>
