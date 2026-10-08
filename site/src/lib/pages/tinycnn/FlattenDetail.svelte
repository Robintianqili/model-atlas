<!--
  Flatten zoom: the 16 maps of 4x4 laid end to end as 256 numbers (channel by channel, row by row).
  Hovering a cell on either side highlights the same number on the other (flattenIndexToCHW).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CnnStore } from './store.svelte';

	type Part = 'maps' | 'formula' | 'note';
	let { S, parts = ['maps', 'formula', 'note'] }: { S: CnnStore; parts?: Part[] } = $props();
	const show = (p: Part) => parts.includes(p);

	const src = $derived(S.layer(S.layer('flatten').in[0]));
	const C = $derived(src.outShape[0]);
	const H = $derived(src.outShape[1]);
	const W = $derived(src.outShape[2]);
	const flat = $derived(S.trace?.t.flatten.data ?? null);
	const dom = $derived(S.domainFor('flatten', () => [-1, 1]));
	const maxIdx = $derived.by(() => {
		if (!flat) return 0;
		let b = 0;
		for (let i = 1; i < flat.length; i++) if (flat[i] > flat[b]) b = i;
		return b;
	});
	let picked = $state<number | null>(null);
	const idx = $derived(picked ?? maxIdx);
	const chw = $derived(S.model ? S.model.flattenIndexToCHW(idx) : { c: 0, y: 0, x: 0 });
	const cols = 16;

	function setIdx(i: number) {
		picked = i;
		const p = S.model?.flattenIndexToCHW(i);
		if (p) S.hoverRF = S.receptiveField(src.id, p.y, p.x);
	}
</script>

{#if flat && S.trace}
	{#if show('maps')}
	<div class="fl">
		<figure class="maps">
			<figcaption>{t(src.label)} <span class="num muted">[{C}×{H}×{W}]</span></figcaption>
			<div class="grid16">
				{#each Array.from({ length: C }, (_, c) => c) as c (c)}
					<div class="mc">
						<span class="cl num">c={c}</span>
						<TensorGrid
							data={S.trace.t[src.id].data.subarray(c * H * W, (c + 1) * H * W)}
							rows={H}
							cols={W}
							domain={dom}
							cell={12}
							gridLines={true}
							readout={false}
							highlights={chw.c === c ? [{ row: chw.y, col: chw.x, color: 'var(--accent)' }] : []}
							label="{t(src.label)} {c}"
							onhover={(e) => e && setIdx(c * H * W + e.row * W + e.col)}
						/>
					</div>
				{/each}
			</div>
		</figure>
		<div class="arrow" aria-hidden="true">→</div>
		<figure class="vec">
			<figcaption>
				{t({ en: 'Flatten: 256 numbers, shown 16 per row', zh: '展平：256 个数，每行显示 16 个' })}
				<span class="num muted">[256]</span>
			</figcaption>
			<TensorGrid
				data={flat}
				rows={256 / cols}
				{cols}
				domain={dom}
				maxCell={20}
				highlights={[{ row: Math.floor(idx / cols), col: idx % cols, color: 'var(--accent)' }]}
				label={t({ en: 'Flatten vector', zh: '展平后的向量' })}
				onhover={(e) => e && setIdx(e.row * cols + e.col)}
			/>
		</figure>
	</div>
	{/if}
	{#if show('formula')}
	<FormulaCard
		tex={`\\begin{aligned} i &= ${H * W}\\,c + ${W}\\,y + x \\\\ &= ${H * W}\\cdot{{c}} + ${W}\\cdot{{y}} + {{x}} = {{i}} \\\\ ${sym('act', 'v_{i}')} &= {{v}} \\end{aligned}`}
		values={{ c: String(chw.c), y: String(chw.y), x: String(chw.x), i: String(idx), v: flat[idx] }}
		provenance="live"
		caption={t({ en: 'Where number i comes from (channel c, row y, column x)', zh: '第 i 个数来自哪里（通道 c，行 y，列 x）' })}
	/>
	{/if}
	{#if show('note')}
	<p class="note">
		{t({
			en: 'Nothing is computed here: the same 256 numbers are just put in one line, so the dense layer can give each of them its own weight. The order is fixed, so position 37 always means "channel 2, row 1, column 1".',
			zh: '这里没有任何计算：同样的 256 个数只是排成一行，好让全连接层给每个数一个自己的权重。顺序是固定的，所以第 37 个永远表示“通道 2、第 1 行、第 1 列”。'
		})}
	</p>
{/if}
{/if}

<style>
	.fl {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 20px;
		align-items: center;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.vec {
		flex: 1 1 260px;
		max-width: 340px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.grid16 {
		display: grid;
		grid-template-columns: repeat(8, auto);
		gap: 6px 8px;
	}
	@media (max-width: 520px) {
		.grid16 {
			grid-template-columns: repeat(4, auto);
		}
	}
	.mc {
		display: grid;
		gap: 1px;
	}
	.cl {
		font-size: 10px;
		color: var(--muted);
	}
	.arrow {
		color: var(--muted);
		font-size: var(--fs-xl);
	}
	.note {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
	}
</style>
