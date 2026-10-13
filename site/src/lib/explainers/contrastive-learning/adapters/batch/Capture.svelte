<!--
  contrastive-learning Capture for the local 'batch' adapter: the batch of pairs (thumbnail, editable caption, remove), a
  framed pad to draw a pair of your own and add it, ready-made pairs to add, three starting batches, and the temperature
  slider (log scale, 0.01 to 2) with a button for the τ TinyCLIP learned. The compact copy beside the lesson shows the
  slider and the batch, and opens the editor on request. Reports { pairs, tau }.
-->
<script lang="ts">
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import type { Stroke } from '$lib/kit/draw';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { fmtSetting } from '../../../../atlas/core/format.ts';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { render } from '../../scenes.ts';
	import { scaleIn } from '../../batch.ts';
	import { BATCHES, MAX_PAIRS, MIN_PAIRS, POOL, TAU_RANGE, pairOf, pairsOf, tauOf, type BatchRaw, type PairRaw } from './adapter.ts';

	let { raw, onchange, compact = false, rt }: CaptureProps<BatchRaw> = $props();
	const L = (en: string, zh: string) => t({ en, zh });
	const pairs = $derived(pairsOf(raw));
	const tau = $derived(tauOf(raw));
	const send = (p: Partial<BatchRaw>, live = false) => onchange?.({ pairs, tau, ...p }, { live });

	// τ on a log slider: 0..100 ↦ 0.01..2
	const [a, b] = TAU_RANGE.map(Math.log10);
	const toTau = (v: number) => Number((10 ** (a + ((b - a) * v) / 100)).toPrecision(2));
	const toPos = (x: number) => Math.round(((Math.log10(x) - a) / (b - a)) * 100);
	const learned = $derived(1 / scaleIn(rt?.sources.clip?.model));

	const setCaption = (i: number, c: string, live: boolean) => send({ pairs: pairs.map((p, k) => (k === i ? { ...p, caption: c } : p)) }, live);
	const remove = (i: number) => send({ pairs: pairs.filter((_, k) => k !== i) });
	const add = (p: PairRaw) => pairs.length < MAX_PAIRS && send({ pairs: [...pairs, p] });
	const pool = $derived(POOL.filter((f) => !pairs.some((p) => p.from === f.id)));

	let open = $state(false);
	let strokes = $state.raw<Stroke[]>([]);
	let caption = $state('');
	const drawn = $derived(strokes.some((s) => s.points.length));
	function addOwn() {
		add({ strokes: strokes.map((s) => s.points.map((q) => ({ x: q.x, y: q.y }))), caption: caption.trim() });
		strokes = [];
		caption = '';
	}
</script>

<div class="cap" class:compact>
	<label class="tau">
		<span>{L('Temperature τ', '温度 τ')}</span>
		<input type="range" min="0" max="100" step="1" value={toPos(tau)} oninput={(e) => send({ tau: toTau(Number(e.currentTarget.value)) }, true)} onchange={(e) => send({ tau: toTau(Number(e.currentTarget.value)) })} />
		<output class="num">{fmtSetting(tau)}</output>
	</label>
	{#if Number.isFinite(learned)}
		<button type="button" class="btn small" onclick={() => send({ tau: Number(learned.toPrecision(3)) })}>{L(`Use the τ TinyCLIP learned (${fmt3(learned)})`, `用 TinyCLIP 学到的 τ（${fmt3(learned)}）`)}</button>
	{/if}

	<ul class="pairs" aria-label={L('The batch', '这一批')}>
		{#each pairs as p, i (i)}
			<li>
				<span class="ix num">{i}</span>
				<TensorGrid data={render(p.strokes)} rows={28} cols={28} kind="sequential" domain={[0, 1]} cell={compact ? 1 : 2} interactive={false} gridLines={false} labels={false} label={{ en: `drawing ${i}`, zh: `画 ${i}` }} />
				{#if compact && !open}
					<span class="ct">“{p.caption}”</span>
				{:else}
					<input type="text" value={p.caption} maxlength="80" spellcheck="false" aria-label={L(`Caption ${i}`, `描述 ${i}`)} oninput={(e) => setCaption(i, e.currentTarget.value, true)} onchange={(e) => setCaption(i, e.currentTarget.value, false)} />
					<button type="button" class="btn x" disabled={pairs.length <= MIN_PAIRS} aria-label={L(`Remove pair ${i}`, `删除第 ${i} 对`)} onclick={() => remove(i)}>×</button>
				{/if}
			</li>
		{/each}
	</ul>

	{#if !compact || open}
		<div class="own">
			<p class="h">{L('Draw a pair of your own', '画一对你自己的')}</p>
			<div class="frame">
				<DrawPad bind:strokes brush={0.045} size={compact ? 160 : 200} showBrush={false} label={L('Drawing pad with a frame', '带画框的画板')} />
			</div>
			<label class="in"><span class="muted">{L('Its caption', '它的描述')}</span><input type="text" bind:value={caption} maxlength="80" spellcheck="false" /></label>
			<button type="button" class="btn" disabled={!drawn || !caption.trim() || pairs.length >= MAX_PAIRS} onclick={addOwn}>{L('Add to the batch', '加入这一批')}</button>
			{#if pairs.length >= MAX_PAIRS}<p class="muted">{L(`The batch is full (${MAX_PAIRS} pairs): remove one first.`, `这一批已满（${MAX_PAIRS} 对）：先删掉一对。`)}</p>{/if}
		</div>
		{#if pool.length && pairs.length < MAX_PAIRS}
			<div class="chips" role="group" aria-label={L('Ready-made pairs to add', '可以加入的现成图文对')}>
				{#each pool as f (f.id)}<button type="button" class="chip" onclick={() => add(pairOf(f))}>+ {t(f.label)}</button>{/each}
			</div>
		{/if}
		<div class="chips" role="group" aria-label={L('Start again from', '重新开始的批次')}>
			{#each BATCHES as s (s.id)}<button type="button" class="btn small" aria-pressed={raw?.preset === s.id} onclick={() => onchange?.(structuredClone(s.raw))}>{t(compact ? s.short : s.label)}</button>{/each}
		</div>
	{/if}
	{#if compact}<button type="button" class="btn small" aria-expanded={open} onclick={() => (open = !open)}>{open ? L('Close the editor', '收起编辑器') : L('Edit the batch', '编辑这一批')}</button>{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; width: 100%; max-width: 560px; }
	.tau { display: grid; grid-template-columns: minmax(0, 8.5em) minmax(0, 1fr) 3.4em; gap: 8px; align-items: center; font-size: var(--fs-sm); }
	.compact .tau { grid-template-columns: minmax(0, 1fr) 3.4em; }
	.compact .tau > span { grid-column: 1 / -1; }
	.tau input { width: 100%; min-width: 0; accent-color: var(--accent); }
	.tau output { text-align: right; }
	.pairs { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
	.pairs li { display: grid; grid-template-columns: 1.2em auto minmax(0, 1fr) auto; gap: 6px; align-items: center; font-size: var(--fs-sm); }
	.ix { font-size: var(--fs-xs); color: var(--ink-2); }
	.ct { overflow-wrap: anywhere; font-size: var(--fs-xs); grid-column: 3 / -1; }
	.pairs input, .in input { min-width: 0; width: 100%; min-height: 32px; }
	.x { min-width: 32px; min-height: 32px; padding: 0 8px; cursor: pointer; }
	.own { display: grid; gap: 6px; justify-items: start; border-top: 1px solid var(--rule); padding-top: 8px; }
	.h { margin: 0; font-weight: 600; font-size: var(--fs-sm); }
	.frame :global(canvas) { outline: 3px solid var(--ink); outline-offset: -3px; }
	.in { display: grid; grid-template-columns: 6em minmax(0, 1fr); gap: 6px; align-items: center; width: 100%; font-size: var(--fs-sm); }
	.chips { display: flex; flex-wrap: wrap; gap: 4px 6px; }
	.chip { cursor: pointer; font-size: var(--fs-xs); padding: 2px 8px; }
	.small { min-height: 32px; padding: 0 12px; font-size: var(--fs-sm); justify-self: start; }
	.btn[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); }
	.muted { margin: 0; font-size: var(--fs-xs); }
</style>
