<!--
  kit.probbars: the kit's ProbBars on a class or vocabulary output (manifest output.view, design §5.1). Scores that are
  not yet a distribution are turned into one with a softmax (f64) first, and the bars say so. [T, V] outputs show the
  prediction at one row (K2: props.row, else the package's answer row, manifest output.row, else the last; the chips
  pick another). props: layer, k (default 10), row, chosen (K1: the
  id of the item that was picked, e.g. a sampled token; a class output marks its top class, a vocabulary output marks
  nothing by default, since the most likely next token is not necessarily the one chosen). Items with a name (a class,
  a vocabulary word) show no '#id', which read as a rank.
-->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import Notice from '../kit.parts/Notice.svelte';
	import { outLabels, stageModel, stageTrace, tokenIds, tokenTexts } from '../kit.parts/stage.ts';
	import { distOf } from './dist.ts';

	let P: StageProps = $props();
	let rowPick = $derived.by<number | null>(() => {
		void P.at;
		return null;
	});
	const D = $derived(distOf(P, rowPick));
	const model = $derived(stageModel(P));
	const ids = $derived(tokenIds(stageTrace(P), model));
	const tokens = $derived(D && D.rows > 1 ? tokenTexts(P, D.rows, ids) : []);
	const labels = $derived(outLabels(P, model));
	const nameOf = (i: number) => {
		const l = labels?.(i);
		return l === undefined ? `#${i}` : typeof l === 'string' ? l : t(l);
	};
	const items = $derived(D ? Array.from(D.p, (p, i) => ({ label: nameOf(i), p, id: i, color: D.axis === 'class' ? categorical(i, themeState.resolved) : undefined })) : []);
	$effect(() => {
		if (P.autoplay && D) P.onend();
	});
</script>

{#if !D}
	<Notice text={{ en: 'No class or vocabulary output to show.', zh: '没有可显示的类别或词表输出。' }} />
{:else}
	<div class="pb">
		{#if tokens.length > 1}
			<TokenChips tokens={tokens.map((s, i) => ({ text: s, id: ids ? ids[i] : undefined }))} selected={D.row} label={D.next ? { en: 'Prediction after which token', zh: '在哪个词元之后的预测' } : { en: 'Prediction at which token', zh: '在哪个词元位置上的预测' }} onselect={(i) => (rowPick = i)} />
		{/if}
		<ProbBars {items} k={typeof P.props.k === 'number' ? P.props.k : 10} chosen={typeof P.props.chosen === 'number' ? P.props.chosen : D.axis === 'class' ? D.top : null} tokens={D.axis === 'vocab'} ids={!labels} label={D.label} />
		{#if D.softmaxed}
			<p class="muted small">{t({ en: 'These were scores (logits); the bars show their softmax.', zh: '这些原本是分数（logits）；条形显示的是它们的 softmax。' })}</p>
		{/if}
	</div>
{/if}

<style>
	.pb {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.small {
		margin: 0;
		font-size: var(--fs-xs);
	}
</style>
