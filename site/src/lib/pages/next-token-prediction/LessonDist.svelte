<!--
  Lesson step 3: for the followed window, the final model's probability for every one of the
  1,024 tokens (live, fp32): all bars sorted, the top guesses as labeled bars, and their sum.
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import LessonWindow from './LessonWindow.svelte';
	import DistChart from './DistChart.svelte';
	import TopBars from './TopBars.svelte';
	import type { NtpLesson } from './lessonState.svelte';
	import { FACTS } from './facts';

	let { L }: { L: NtpLesson } = $props();
	const S = $derived(L.S);
	const d = $derived(L.dist('final'));
	let log = $state(false);

</script>

<div class="ds">
	<div class="top">
		<LessonWindow {L} />
		<ProvenanceBadge mode="live" compact precision="fp32" tolerance={FACTS.tolerance} against="PyTorch fp32" />
	</div>
	{#if L.pos < 1}
		<p class="muted">{t({ en: 'Type at least two tokens.', zh: '至少输入两个词元。' })}</p>
	{:else if !d}
		<p class="muted">{t({ en: 'TinyGPT is reading this window…', zh: 'TinyGPT 正在读这个窗口…' })}</p>
	{:else}
		<DistChart dist={d} disp={(id) => S.disp(id)} bind:log />
		<div class="cols">
			<TopBars dist={d} disp={(id) => S.disp(id)} k={6} label={{ en: 'The top guesses, with the real next token', zh: '排名最前的猜测，以及真实的下一个词元' }} />
			<div class="facts small">
				<p>
					{t({ en: 'Green: the token that really came next.', zh: '绿色的是真正出现的下一个词元。' })}
					{#if d.rank > 6}
						{t({ en: `It is not among the top 6: it is number ${d.rank}.`, zh: `它不在前 6 名里，而是排第 ${d.rank}。` })}
					{/if}
				</p>
				<p>
					{t({ en: 'Sum of all 1,024 probabilities:', zh: '1,024 个概率之和：' })}
					<strong class="num">{d.sum.toFixed(3)}</strong>
				</p>
				<p>
					{t({ en: 'Spread (entropy):', zh: '分散程度（熵）：' })} <Num value={d.entropy} />
					{t({ en: 'bits', zh: '比特' })}
					<span class="muted">{t({ en: '(0 = completely sure of one token, 10 = all 1,024 equally likely)', zh: '（0 = 完全确定是某一个词元，10 = 1,024 个同样可能）' })}</span>
				</p>
			</div>
		</div>
	{/if}
</div>

<style>
	.ds {
		container: ds / inline-size;
		display: grid;
		gap: 12px;
		min-width: 0;
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 8px;
		align-items: flex-start;
		justify-content: flex-end;
	}
	.top > :global(.wp) {
		flex: 1 1 280px;
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
		gap: 14px;
		align-items: start;
	}
	@container ds (max-width: 520px) {
		.cols {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.facts {
		display: grid;
		gap: 6px;
	}
	.facts p {
		margin: 0;
	}
	.small {
		font-size: var(--fs-xs);
	}
</style>
