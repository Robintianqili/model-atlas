<!--
  Output: the softmax over the 10 classes (live), with a jump into the softmax arithmetic.
-->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { ProbItem } from '$lib/kit/types';
	import type { CnnStore } from './store.svelte';

	let { S, onzoom }: { S: CnnStore; onzoom: () => void } = $props();

	const items = $derived(
		S.probs ? Array.from(S.probs, (p, i) => ({ label: t(S.spec.classes[i]), p, id: i, color: categorical(i, themeState.resolved) })) : []
	);
	const second = $derived.by(() => {
		const p = S.probs;
		if (!p) return -1;
		let k = S.top === 0 ? 1 : 0;
		for (let i = 0; i < p.length; i++) if (i !== S.top && p[i] > p[k]) k = i;
		return k;
	});
	const gap = $derived(S.trace && second >= 0 ? S.trace.t.dense.data[S.top] - S.trace.t.dense.data[second] : 0);
	function pick(it: ProbItem) {
		S.select('softmax', Number(it.id));
		onzoom();
	}
	const ood = $derived(S.source.kind === 'preset' && S.source.heldOut);
</script>

{#if !S.trace}
	<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
{:else}
	<div class="out">
		<ProbBars {items} k={10} chosen={S.top} label={t({ en: 'Probability per class', zh: '各类别的概率' })} onselect={pick} />
		<div class="side">
			<p class="ans">
				{t({ en: 'Answer:', zh: '回答：' })} <strong>{t(S.spec.classes[S.top])}</strong> <span class="num">{fmtPct(S.probs?.[S.top] ?? 0)}</span>
			</p>
			<p class="small">
				<Lang>
					{#snippet en()}Its <Term id="logit" /> is {fmt3(gap)} above the runner-up ({t(S.spec.classes[second])}), so softmax gives it about e<sup>{fmt3(gap)}</sup> ≈ {fmt3(Math.exp(gap))} times the probability.{/snippet}
					{#snippet zh()}它的 <Term id="logit" /> 比第二名（{t(S.spec.classes[second])}）高 {fmt3(gap)}，所以 softmax 给它的概率大约是第二名的 e<sup>{fmt3(gap)}</sup> ≈ {fmt3(Math.exp(gap))} 倍。{/snippet}
				</Lang>
			</p>
			{#if ood}
				<p class="warn">{t({ en: 'This drawing is not one of the 10 classes: whatever the bars say, the answer is wrong.', zh: '这幅画不属于这 10 个类别：无论条形图怎么显示，这个回答都是错的。' })}</p>
			{/if}
			<p class="small muted">{t({ en: 'Click a bar to see its softmax arithmetic.', zh: '点击某个条形，查看它的 softmax 计算。' })}</p>
		</div>
	</div>
{/if}

<style>
	.out {
		display: grid;
		grid-template-columns: minmax(0, 560px) minmax(0, 1fr);
		gap: 16px 32px;
		align-items: start;
	}
	@media (max-width: 760px) {
		.out {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.side p {
		margin: 0 0 8px;
	}
	.ans {
		font-size: var(--fs-lg);
	}
	.small {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.warn {
		font-size: var(--fs-sm);
		padding: 8px 10px;
		border-left: 3px solid var(--illus);
		background: var(--illus-soft);
	}
</style>
