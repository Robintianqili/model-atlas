<!--
  zoom.formula: one formula with live numbers, in a FormulaCard. Where the formula comes from, in order:
    1. props.tex (+ props.values, props.where, props.caption): an author's own formula;
    2. the frame's phase formula (PhaseDef.formula: tex + values(frame)) on process and training tapes (kinds B and C);
       a tape loaded from a file carries the tex only, and then the frame's metrics fill it;
    3. the op formula of the stage's unit explanation (kind A), with explain() values.
  Under it: the numbers that filled it (so nothing is only in the picture) and, on tapes, where the frame is.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import Notice from '../kit.parts/Notice.svelte';
	import { alignTex, missingNames, texValues } from '../kit.parts/tex.ts';
	import { formulaOf } from './formula.ts';

	let P: StageProps = $props();
	const F = $derived(formulaOf(P));
	const missing = $derived(F ? missingNames(F.tex, F.values) : []);
	// K1: the numbers of the formula again, under it, only when the step names them (props.names: { key: { en, zh } }) or
	// asks for them (props.chips: true): the bare metric keys with 3 digits repeated what the formula shows
	const names = $derived((P.props.names ?? {}) as Record<string, Text>);
	const chips = $derived(P.props.chips === true || Object.keys(names).length > 0);
	const shown = $derived(F && chips ? Object.entries(F.values).filter(([k]) => F.tex.includes(`{{${k}}}`) || F.tex.includes(`{{ ${k} }}`)) : []);
	$effect(() => {
		if (P.autoplay && F) P.onend();
	});
</script>

{#if !F}
	<Notice text={{ en: 'No formula for this stage: give the step props.tex, or point it at a frame whose phase has a formula.', zh: '这一步没有公式：请给步骤设置 props.tex，或让它指向一个其阶段带有公式的帧。' }} />
{:else}
	<div class="fz">
		{#if F.where}<p class="wh">{t(F.where)}</p>{/if}
		<FormulaCard tex={alignTex(F.tex)} values={texValues(F.values)} caption={F.caption} where={F.legend as { tex: string; text: Text }[]} provenance={P.provenance} />
		{#if shown.length}
			<dl class="vals num" aria-label={t({ en: 'The numbers in the formula', zh: '公式中的数' })}>
				{#each shown as [k, v] (k)}
					<div><dt>{names[k] ? t(names[k]) : k}</dt><dd>{typeof v === 'number' ? (Number.isInteger(v) ? String(v) : fmt3(v, 4)) : v}</dd></div>
				{/each}
			</dl>
		{/if}
		{#if missing.length}
			<p class="bad num">{t({ en: 'Not filled:', zh: '没有填上：' })} {missing.join(', ')}</p>
		{/if}
	</div>
{/if}

<style>
	.fz {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.wh {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.vals {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		margin: 0;
		font-size: var(--fs-xs);
	}
	.vals div {
		display: flex;
		gap: 6px;
	}
	dt {
		color: var(--muted);
	}
	dd {
		margin: 0;
	}
	.bad {
		margin: 0;
		color: var(--div-neg-text);
		font-size: var(--fs-xs);
	}
</style>
