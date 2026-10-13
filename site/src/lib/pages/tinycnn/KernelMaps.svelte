<!--
  Lesson stage "Eight kernels make eight feature maps": each Conv 1-1 kernel (its 9 weights) next to
  the map it makes from the learner's drawing, plus how strongly it responds to a thin stroke at four
  angles (strokeResponses: computed from the weights, live). Clicking a card shows its numbers.
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import NumGrid from './NumGrid.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { channelDomain } from './data';
	import { STROKE_GLYPHS, STROKE_NAMES, strokeResponses } from './lesson';
	import type { CnnStore, SelView } from './store.svelte';

	let { S, view }: { S: CnnStore; view: SelView } = $props();

	const C = 8;
	const H = 26;
	const W = 26;
	const chans = Array.from({ length: C }, (_, c) => c);
	const w = $derived(S.model?.weights['conv1_1.weight'].data ?? null);
	const b = $derived(S.model?.weights['conv1_1.bias'].data ?? null);
	const maps = $derived(S.trace?.t.conv1_1.data ?? null);
	const kDom = $derived.by<[number, number]>(() => {
		let m = 0;
		if (w) for (let i = 0; i < w.length; i++) m = Math.max(m, Math.abs(w[i]));
		return [-(m || 1), m || 1];
	});
	const mDom = $derived(S.domainFor('conv1_1', () => (maps ? channelDomain(maps, 0, H * W) : [-1, 1])));
	const probes = $derived(w && b ? chans.map((c) => strokeResponses(w.subarray(c * 9, c * 9 + 9), b[c])) : []);
	const pMax = $derived(Math.max(1e-6, ...probes.flat()));
	const sel = $derived(view.sel.layer === 'conv1_1' ? view.sel.ch : 0);
	const best = (p: number[]) => p.indexOf(Math.max(...p));
	let cell = $state<{ row: number; col: number; value: number } | null>(null);
</script>

{#if w && b && maps}
	<div class="km">
		<div class="cards" role="group" aria-label={t({ en: 'The 8 kernels of Conv 1-1 and their maps', zh: '卷积 1-1 的 8 个卷积核及其特征图' })}>
			{#each chans as c (c)}
				{@const p = probes[c]}
				<button type="button" class="card" class:on={c === sel} aria-pressed={c === sel} onclick={() => view.select('conv1_1', c)}>
					<span class="hd num">{t({ en: 'kernel', zh: '卷积核' })} {c}</span>
					<span class="pair">
						<MapCanvas data={w.subarray(c * 9, c * 9 + 9)} rows={3} cols={3} domain={kDom} size={34} />
						<span class="arr" aria-hidden="true">→</span>
						<MapCanvas data={maps.subarray(c * H * W, (c + 1) * H * W)} rows={H} cols={W} domain={mDom} size={74} />
					</span>
					<span class="probe" aria-hidden="true">
						{#each p as v, i (i)}
							<span class="pb" class:top={i === best(p)}>
								<span class="pbar"><span style="height: {Math.max(4, (Math.max(0, v) / pMax) * 100)}%"></span></span>
								<span class="pg">{STROKE_GLYPHS[i]}</span>
							</span>
						{/each}
					</span>
					<span class="visually-hidden">{t({ en: `responds most to ${STROKE_NAMES[best(p)].en} strokes`, zh: `对${STROKE_NAMES[best(p)].zh}笔画反应最强` })}</span>
				</button>
			{/each}
		</div>

		<div class="detail">
			<NumGrid values={w.subarray(sel * 9, sel * 9 + 9)} rows={3} cols={3} domain={kDom} role="param" cell={54} label={t({ en: `kernel ${sel}: its 9 weights`, zh: `卷积核 ${sel}：它的 9 个权重` })} />
			<figure>
				<figcaption>{t({ en: `its map on your drawing`, zh: `它在你的画上得到的特征图` })} <span class="num muted">[26×26]</span></figcaption>
				<TensorGrid
					data={maps.subarray(sel * H * W, (sel + 1) * H * W)}
					rows={H}
					cols={W}
					domain={mDom}
					maxCell={8}
					readout={false}
					label={t({ en: `Conv 1-1 map ${sel}`, zh: `卷积 1-1 特征图 ${sel}` })}
					onhover={(c) => (cell = c)}
				/>
				<span class="ro num">
					{#if cell}({cell.row}, {cell.col}) = {fmt3(cell.value)}{:else}{t({ en: 'hover or tap the map', zh: '悬停或点按特征图' })}{/if}
				</span>
			</figure>
			<table class="pt num">
				<caption>{t({ en: 'Thin stroke at each angle', zh: '各个角度的细笔画' })}</caption>
				<tbody>
					{#each probes[sel] as v, i (i)}
						<tr class:top={i === best(probes[sel])}><th>{STROKE_GLYPHS[i]}</th><td>{fmt3(v)}</td></tr>
					{/each}
					<tr class="blank"><th>{t({ en: 'blank', zh: '空白' })}</th><td>{fmt3(b[sel])}</td></tr>
				</tbody>
			</table>
		</div>
		<p class="cap">
			{t({
				en: 'Bars: the strongest output of each kernel for a thin test stroke at four angles (window × kernel + bias, computed from the 9 weights). “Blank” is the output on empty paper: just the bias.',
				zh: '小条：用四个角度的细测试笔画算出的每个卷积核的最大输出（窗口 × 卷积核 + 偏置，由这 9 个权重直接算出）。“空白”是在空白纸上的输出，也就是偏置本身。'
			})}
		</p>
	</div>
{/if}

<style>
	.km {
		display: grid;
		gap: 12px;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
		gap: 6px;
	}
	.card {
		display: grid;
		gap: 4px;
		justify-items: start;
		padding: 6px 8px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		text-align: left;
		cursor: pointer;
	}
	.card:hover {
		border-color: var(--rule-strong);
	}
	.card.on {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.hd {
		font-size: var(--t-xs);
		color: var(--ink-2);
	}
	.pair {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.arr {
		color: var(--muted);
		font-size: 12px;
	}
	.probe {
		display: flex;
		gap: 6px;
		align-items: flex-end;
	}
	.pb {
		display: grid;
		justify-items: center;
		gap: 1px;
	}
	.pbar {
		display: flex;
		align-items: flex-end;
		width: 9px;
		height: 22px;
		background: var(--surface-2);
		border-radius: 2px;
	}
	.pbar span {
		display: block;
		width: 100%;
		background: var(--div-pos);
		opacity: 0.55;
		border-radius: 2px;
	}
	.pb.top .pbar span {
		opacity: 1;
	}
	.pg {
		font-family: var(--font-mono);
		font-size: var(--t-xs);
		line-height: 1;
		color: var(--muted);
	}
	.pb.top .pg {
		color: var(--ink);
		font-weight: 700;
	}
	.detail {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 20px;
		align-items: flex-start;
		padding-top: 10px;
		border-top: 1px solid var(--rule);
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	figcaption,
	.ro {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.pt {
		border-collapse: collapse;
		font-size: var(--fs-xs);
	}
	.pt caption {
		caption-side: top;
		padding-bottom: 4px;
		font-family: var(--font-body);
		font-size: var(--fs-xs);
		color: var(--ink-2);
		text-align: left;
	}
	.pt th,
	.pt td {
		padding: 1px 8px;
		text-align: right;
		border-bottom: 1px solid var(--rule);
	}
	.pt th {
		font-weight: 600;
		color: var(--ink-2);
		text-align: center;
	}
	.pt tr.top td {
		font-weight: 700;
		color: var(--div-pos-text);
	}
	.pt tr.blank th {
		font-family: var(--font-body);
		font-weight: 400;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
		max-width: var(--measure);
	}
</style>
