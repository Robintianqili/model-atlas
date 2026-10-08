<!--
  Lesson step 11: after the last token, the output head gives 1,024 logits; softmax (with the
  temperature, after top-k) turns them into probabilities. The live logits of the top tokens, the
  formula with the top token's numbers, and the page's own sampling controls and bars (lesson state).
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import Sampling from './Sampling.svelte';
	import type { TGState } from './state.svelte';
	import { V } from './state.svelte';
	import { row, topIndices } from './analysis';

	interface Props {
		LS: TGState;
		dist: Float64Array | null;
		temperature: number;
		topK: number;
	}
	let { LS, dist, temperature = $bindable(), topK = $bindable() }: Props = $props();

	const f = $derived(LS.run?.forward);
	const T = $derived(LS.T);
	const z = $derived(f ? row(f.t.logits.data, T - 1, V) : null);
	const pm = $derived(f ? row(f.t.probs.data, T - 1, V) : null);
	/** tokens listed and rows per bar chart (6 keeps the slider and both charts within about one phone screen) */
	const N = 6;
	const top = $derived(z ? topIndices(z, N) : []);
	const best = $derived(top[0] ?? 0);
	const massN = $derived.by(() => {
		if (!pm) return 0;
		let s = 0;
		for (const i of topIndices(pm, N)) s += pm[i];
		return s;
	});
	const tex = $derived(
		temperature <= 0
			? `${sym('prob', 'p_i')} = \\begin{cases} 1 & i = \\arg\\max_j ${sym('act', 'z_j')} \\\\ 0 & \\text{otherwise} \\end{cases}`
			: `\\begin{aligned}${sym('prob', 'p_i')} &= \\frac{e^{${sym('act', 'z_i')}/\\tau}}{\\sum_{j} e^{${sym('act', 'z_j')}/\\tau}} \\\\ ${sym('prob', 'p')}_{\\text{top}} &= \\frac{e^{ {{z}} / {{tau}} }}{\\sum_j e^{z_j / {{tau}} }} = {{p}}\\end{aligned}`
	);
</script>

{#if z && pm}
	<div class="ls">
		<p class="head">
			{t({ en: 'After the last token', zh: '在最后一个词元' })} <span class="tok">{LS.label(T - 1)}</span>{t({ en: ', the output head gives one score (logit) to each of the 1,024 tokens. The highest:', zh: ' 之后，输出头给 1024 个词元各打一个分数（logit）。最高的几个：' })}
		</p>
		<ul class="logits">
			{#each top as id (id)}
				<li><span class="tok">{LS.idLabel(id)}</span> <span class="num">z = {fmt3(z[id])}</span></li>
			{/each}
		</ul>
		<FormulaCard
			{tex}
			values={{ z: z[best], tau: temperature, p: dist ? dist[best] : 0 }}
			caption={{ en: `Softmax with temperature τ, for the top token “${LS.idLabel(best)}”`, zh: `带温度 τ 的 softmax，以第一名词元“${LS.idLabel(best)}”为例` }}
			where={[
				{ tex: sym('act', 'z_i'), text: { en: 'logit of token i (sum over the kept tokens only, when top-k is on)', zh: '词元 i 的 logit（开启 top-k 时只对保留下来的词元求和）' } },
				{ tex: '\\tau', text: { en: 'temperature: 1 = the model as trained', zh: '温度：1 = 训练出来的原样' } }
			]}
			provenance="live"
		/>
		<Sampling S={LS} {dist} bind:temperature bind:topK topP={1} parts={['controls', 'bars', 'stats']} knobs={['temperature', 'topk']} rows={N} ids={false} />
		<p class="note muted">
			{t({
				en: `At temperature 1 these ${N} most likely tokens hold ${fmtPct(massN)} of the probability; the other ${(V - N).toLocaleString('en-US')} share the remaining ${fmtPct(1 - massN)}.`,
				zh: `在温度 1 时，这 ${N} 个最可能的词元占了 ${fmtPct(massN)} 的概率；其余 ${V - N} 个词元分剩下的 ${fmtPct(1 - massN)}。`
			})}
		</p>
	</div>
{/if}

<style>
	.ls {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
	}
	.head {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 4px;
		white-space: pre;
		font-variant-ligatures: none;
	}
	.logits {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
	}
	.logits .num {
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	.note {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.ls :global(.smp .bars) {
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
	}
	/* narrower label and badge columns, so the bars themselves get the width (two charts side by side) */
	.ls :global(.smp .pb .row) {
		grid-template-columns: minmax(3.5em, 6.5em) minmax(40px, 1fr) 3.8em auto;
	}
</style>
