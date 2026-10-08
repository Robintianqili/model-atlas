<!--
  The op's own formula (OpDef.formula, {{name}} = explain().values) in a FormulaCard, broken into aligned lines at
  its top-level "=" signs so it fits a phone, followed by the self-check: the value recomputed by explain() next to
  the value the forward pass stored.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import Num from './Num.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Provenance, UnitExplanation } from '../../../types/index.ts';
	import { alignTex, texValues } from './tex.ts';

	interface Props {
		tex: string | undefined;
		e: UnitExplanation;
		caption?: Text;
		provenance?: Provenance;
		where?: { tex: string; text: Text }[];
		/** extra values on top of e.values */
		values?: Record<string, number>;
		check?: boolean;
	}
	let { tex, e, caption, provenance, where = [], values = {}, check = true }: Props = $props();
	const vals = $derived<Record<string, number>>({ ...e.values, ...values });
	const same = $derived(e.result === e.traced || (Number.isNaN(e.result) && Number.isNaN(e.traced)));
</script>

{#if tex}
	<FormulaCard tex={alignTex(tex)} values={texValues(vals)} {caption} {where} {provenance} />
{/if}
{#if check}
	<p class="ck">
		{t({ en: 'Recomputed here:', zh: '这里重新算出：' })} <Num value={e.result} />
		· {t({ en: 'stored by the forward pass:', zh: '前向传播存下的值：' })} <Num value={e.traced} />
		<span class="muted">({t({ en: 'difference', zh: '差' })} {#if same}0{:else}<Num value={Math.abs(e.result - e.traced)} />{/if})</span>
	</p>
{/if}

<style>
	.ck {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
</style>
