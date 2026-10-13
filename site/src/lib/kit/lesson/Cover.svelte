<!--
  The cover's narration (redesign spec §5.6, step 0, answer first): "Lesson · n steps · ≈ m min · in your browser", the
  lesson's promise as the title, one sentence, "By the end you can" with the goals, and the predict-first question as
  option buttons answered in place (saved for the end card's reveal). The stage beside it is the page's cover stage.
-->
<script lang="ts">
	import Icon from '../Icon.svelte';
	import { check, help } from '../icons';
	import { predictions, restorePrediction, savePrediction } from '../predict.svelte';
	import { lui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { PredictQuestion } from '../types';

	interface Props {
		title: Text;
		summary?: Text;
		goals?: Text[];
		prequestion?: PredictQuestion;
		n: number;
		minutes?: number;
		/** a returning learner: where they were */
		resumeAt?: number;
	}
	let { title, summary, goals = [], prequestion, n, minutes, resumeAt }: Props = $props();
	const gid = $props.id();
	$effect(() => {
		if (prequestion) restorePrediction(prequestion.id);
	});
	const saved = $derived(prequestion ? predictions[prequestion.id] : undefined);
	/** spec §5.6: the cover lists 3 goals; more fold under a disclosure so the predict-first question stays near the
	 * top of the narration (QA 2026-10-13: lessons with 4-6 goals pushed it out of the first screen) */
	const shown = $derived(goals.slice(0, 3));
	const more = $derived(goals.slice(3));
</script>

<p class="eyebrow">
	{minutes ? lui('coverMeta', { n, m: minutes }) : t({ en: `Lesson · ${n} steps · in your browser`, zh: `课程 · ${n} 步 · 在你的浏览器中运行` })}
</p>
<h1 class="ctitle">{t(title)}</h1>
{#if summary}<p class="say">{t(summary)}</p>{/if}
{#if resumeAt !== undefined}<p class="resume">{lui('resumed', { k: resumeAt + 1 })}</p>{/if}
{#if goals.length}
	<div class="goals">
		<p class="h">{lui('byTheEnd')}</p>
		<ol>
			{#each shown as g, i (i)}<li><span class="gn num" aria-hidden="true">{i + 1}</span><span>{t(g)}</span></li>{/each}
		</ol>
		{#if more.length}
			<details class="more">
				<summary>{t({ en: more.length === 1 ? 'One more goal' : `${more.length} more goals`, zh: `另外 ${more.length} 个目标` })}</summary>
				<ol start={4}>
					{#each more as g, i (i)}<li><span class="gn num" aria-hidden="true">{i + 4}</span><span>{t(g)}</span></li>{/each}
				</ol>
			</details>
		{/if}
	</div>
{/if}
{#if prequestion}
	<div class="beat" role="group" aria-labelledby="{gid}-q">
		<p class="beat-k"><Icon d={help} size={16} />{t({ en: 'Before you start', zh: '开始之前' })}</p>
		<p class="q" id="{gid}-q">{t(prequestion.question)}</p>
		<div class="opts" role="radiogroup" aria-labelledby="{gid}-q">
			{#each prequestion.options as o, i (i)}
				<button type="button" class="opt" role="radio" aria-checked={saved === i} onclick={() => savePrediction(prequestion.id, i)}>
					<span class="kbd hint" aria-hidden="true">{i + 1}</span><span>{t(o)}</span>{#if saved === i}<Icon d={check} size={18} />{:else}<span></span>{/if}
				</button>
			{/each}
		</div>
		<p class="note" aria-live="polite">
			{saved === undefined
				? t({ en: 'Answer now; the lesson shows you, and the last step reveals the answer.', zh: '先作答；课程会一步步展示，最后一步揭晓答案。' })
				: t({ en: 'Saved. The last step reveals the answer.', zh: '已记下。最后一步揭晓答案。' })}
		</p>
	</div>
{/if}

<style>
	.ctitle {
		margin: -4px 0 0;
		font-size: var(--t-2xl);
		line-height: var(--lh-2xl);
		font-weight: var(--w-display);
		letter-spacing: -0.02em;
	}
	:global(html[lang='zh-CN']) .ctitle {
		letter-spacing: 0;
		line-height: 1.35;
	}
	.say {
		margin: 0;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	:global(html[lang='zh-CN']) .say {
		line-height: 1.8;
	}
	.resume {
		margin: 0;
		color: var(--ink-2);
		font-size: var(--t-sm);
	}
	.h {
		margin: 0 0 6px;
		font-size: var(--t-sm);
		font-weight: var(--w-title);
		color: var(--ink-2);
	}
	.more {
		margin-top: 8px;
	}
	.more summary {
		width: fit-content;
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		cursor: pointer;
	}
	.goals ol {
		display: flex;
		flex-direction: column;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.goals li {
		display: grid;
		grid-template-columns: 24px 1fr;
		gap: 10px;
		align-items: start;
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.more ol {
		margin-top: 8px;
	}
	.gn {
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
		border: 1px solid var(--rule-strong);
		border-radius: 50%;
		color: var(--ink-2);
		font-size: var(--t-xs);
	}
	.beat {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 14px 16px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
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
		font-weight: var(--w-title);
	}
	.q {
		font-weight: var(--w-ui);
	}
	.opts {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.note {
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
