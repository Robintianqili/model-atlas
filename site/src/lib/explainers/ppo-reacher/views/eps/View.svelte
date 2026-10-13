<!-- ppo-reacher 'eps': ε buttons (they change the learner's input, so the train sources start again) and two kit
     loss-curve charts, your ε against ε = 0.2: the average episode return, and the KL of every update (log scale). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import ViewSlot from '../../../../atlas/ui/ViewSlot.svelte';
	import { EPS_CHOICES, epsOf, targetOf } from '../../sim.ts';
	import { curveProps, epsText } from './meta.ts';

	let P: StageProps = $props();
	const eps = $derived(epsOf(P.rt.input));
	const ret = $derived({ ...P, props: curveProps(P.rt, 'epRet', false) }), kl = $derived({ ...P, props: curveProps(P.rt, 'kl', true) });
	const pick = (e: number) => (P.setInput ?? P.rt.setInput)({ target: targetOf(P.rt.input), eps: e });
</script>

<div class="ep">
	<div class="row" role="group" aria-label={t({ en: 'Clip ε', zh: '截断 ε' })}>
		<span class="small">{t({ en: 'Clip ε:', zh: '截断 ε：' })}</span>
		{#each EPS_CHOICES as e (e)}<button type="button" class="btn" aria-pressed={e === eps} onclick={() => pick(e)}>{t(epsText(e))}</button>{/each}
	</div>
	<p class="small">{t({ en: 'Average episode return of each iteration’s batch (higher is better):', zh: '每次迭代那一批回合的平均回报（越高越好）：' })}</p>
	<ViewSlot view="loss-curve" stage={ret} text={false} />
	<p class="small">{t({ en: 'KL: how far each iteration has moved the policy from the one that played its batch (log scale):', zh: 'KL：每次迭代结束时，策略离玩这批回合的旧策略有多远（对数坐标）：' })}</p>
	<ViewSlot view="loss-curve" stage={kl} text={false} />
</div>

<style>
	.ep { display: grid; gap: 6px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.small { margin: 0; font-size: var(--fs-xs); }
</style>
