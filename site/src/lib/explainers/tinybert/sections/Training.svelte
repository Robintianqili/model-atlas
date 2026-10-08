<!--
  tinybert: the page's Training section (replaces the auto list of raw metric keys and shares such as "0.7241", teaching
  review). Every number is read from facts.json (pipeline/tinybert/stats.py, train.py) or spec.training.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { AtlasSpec } from '../../../atlas/types/index.ts';
	import type { AtlasStore } from '../../../atlas/ui/runtime.svelte.ts';
	import { fmtInt, fmtPct } from '../../../atlas/core/format.ts';

	let { store, spec }: { store: AtlasStore | null; spec: AtlasSpec | null } = $props();
	const tr = $derived((spec?.training ?? {}) as { optimizer?: string });
	const f = (k: string) => store?.rt.num(`fact:${k}`) ?? NaN;
	const p = (k: string) => fmtPct(f(k));
	const whole = (k: string) => `${Math.round(f(k) * 100)}%`;
	const rows = $derived([
		{ en: 'Data', zh: '数据', v: t({ en: 'TinyStories, with TinyGPT’s tokenizer: the first 100 MiB of its training stories. Every number below is measured on the validation stories, which training never saw.', zh: 'TinyStories，用 TinyGPT 的分词器：训练故事的前 100 MiB。下面每个数都是在训练从没见过的验证故事上量出来的。' }) },
		{ en: 'The puzzles', zh: '训练题', v: t({ en: `In each text, ${whole('maskRate')} of the tokens are chosen (at least one); a chosen token becomes [MASK] ${whole('maskShare')} of the time, a random token ${whole('randomShare')}, and stays as it is ${whole('keepShare')}. Only the chosen tokens count in the loss.`, zh: `每段文字选中 ${whole('maskRate')} 的词元（至少一个）；被选中的词元有 ${whole('maskShare')} 换成 [MASK]，${whole('randomShare')} 换成随机词元，${whole('keepShare')} 保持不变。损失只算被选中的词元。` }) },
		{ en: 'Training', zh: '训练', v: t({ en: `${fmtInt(f('steps'))} steps of ${fmtInt(f('batch'))} texts, ${fmtInt(f('examples'))} texts in all`, zh: `${fmtInt(f('steps'))} 步，每步 ${fmtInt(f('batch'))} 段文字，一共 ${fmtInt(f('examples'))} 段` }) },
		{ en: 'Recipe', zh: '训练方法', v: tr.optimizer ?? '—' },
		{ en: 'Chosen tokens guessed right', zh: '被选中的词元猜对的比例', v: t({ en: `${p('valAcc')} of ${fmtInt(f('valN'))} (among the top 5 guesses: ${p('valTop5')}); of those shown as [MASK], ${p('valAccMasked')}`, zh: `${fmtInt(f('valN'))} 个中猜对 ${p('valAcc')}（正确答案在前 5 名里：${p('valTop5')}）；其中显示为 [MASK] 的猜对 ${p('valAccMasked')}` }) },
		{ en: 'One hidden token per text', zh: '每段文字只藏一个词元', v: t({ en: `${fmtInt(f('cmpN'))} gaps: TinyBERT right ${p('accBoth')} with both sides, ${p('accLeft')} when the text stops right after the gap; TinyGPT, reading the left side (its own task), ${p('accGpt')}`, zh: `${fmtInt(f('cmpN'))} 个空缺：TinyBERT 读两侧时猜对 ${p('accBoth')}，文字在空缺处就结束时 ${p('accLeft')}；TinyGPT 读左边（它本来的任务）猜对 ${p('accGpt')}` }) }
	]);
</script>

<dl class="facts">
	{#each rows as r (r.en)}
		<div><dt>{t(r)}</dt><dd>{r.v}</dd></div>
	{/each}
</dl>

<style>
	.facts {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 8px;
		margin: 0;
	}
	.facts div {
		display: grid;
		grid-template-columns: minmax(0, 14em) minmax(0, 1fr);
		gap: 4px 16px;
	}
	.facts dt {
		color: var(--ink-2);
		font-size: var(--fs-sm);
	}
	.facts dd {
		margin: 0;
	}
	@media (max-width: 560px) {
		.facts div {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
