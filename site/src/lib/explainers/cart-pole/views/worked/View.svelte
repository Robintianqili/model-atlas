<!-- cart-pole 'worked': a view (props.view, its props in props.inner) with the worked numbers under it: a line
     (props.note) and a formula (props.tex, props.values, props.where). -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n, StageProps } from '../../../../atlas/types/index.ts';
	import ViewSlot from '../../../../atlas/ui/ViewSlot.svelte';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';

	let P: StageProps = $props();
	const inner = $derived({ ...P, props: (P.props.inner ?? {}) as Record<string, unknown> });
	const note = $derived(P.props.note as I18n | undefined);
	const tex = $derived(P.props.tex as string | undefined);
	const where = $derived((P.props.where ?? []) as { tex: string; text: I18n }[]);
</script>

<div class="wk">
	<ViewSlot view={String(P.props.view)} stage={inner} text={false} />
	{#if note}<p class="small">{t(note)}</p>{/if}
	{#if tex}<FormulaCard tex={alignTex(tex)} values={texValues((P.props.values ?? {}) as Record<string, number | string>)} {where} provenance={P.provenance} />{/if}
</div>

<style>
	.wk { display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.small { margin: 0; font-size: var(--fs-sm); }
</style>
