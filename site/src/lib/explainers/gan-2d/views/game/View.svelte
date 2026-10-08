<!-- gan-2d/game (props: meta.ts): the kit's loss-curve on D's and G's losses, and the link for the comparison step. -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import ViewSlot from '../../../../atlas/ui/ViewSlot.svelte';
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import { curveProps } from './meta.ts';

	let { rt, source, at, focus, sel, ctx, provenance, autoplay, onend }: StageProps = $props();
	const stage: StageProps = $derived({ rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props: curveProps });
</script>

<div class="game">
	<ViewSlot view="loss-curve" {stage} text={false} />
	<p class="small">
		{t({ en: 'Equilibrium values (D says ½ everywhere): D’s loss 2 ln 2 ≈ 1.39, G’s loss ln 2 ≈ 0.693.', zh: '平衡时的值（D 处处说 ½）：D 的损失 2 ln 2 ≈ 1.39，G 的损失 ln 2 ≈ 0.693。' })}
		<a href="{base}/m/flow-2d/">{t({ en: 'Compare: 2-D flow matching, one network, one falling loss →', zh: '对比：二维流匹配，一个网络，一个下降的损失 →' })}</a>
	</p>
</div>

<style>
	.game { display: grid; gap: 8px; min-width: 0; }
</style>
