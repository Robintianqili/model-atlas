<!--
  Input zoom: the 784 numbers the model receives, one pixel at a time.
  compact (guided lesson): the grid, the selected pixel, its 5×5 neighbourhood as numbers, and the count
  of inked pixels; view: whose selection to show (default: the page store).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import Num from '$lib/kit/Num.svelte';
	import NumGrid from './NumGrid.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CnnStore, SelView } from './store.svelte';

	let { S, view, compact = false }: { S: CnnStore; view?: SelView; compact?: boolean } = $props();
	const V = $derived(view ?? S);
	const v = $derived(S.input[V.sel.y * 28 + V.sel.x]);
	const inked = $derived.by(() => {
		let n = 0;
		for (let i = 0; i < S.input.length; i++) if (S.input[i] > 0) n++;
		return n;
	});
	// the 5×5 block around the selected pixel (kept inside the 28×28 grid)
	const y0 = $derived(Math.max(0, Math.min(23, V.sel.y - 2)));
	const x0 = $derived(Math.max(0, Math.min(23, V.sel.x - 2)));
	const patch = $derived.by(() => {
		const out = new Float32Array(25);
		for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) out[r * 5 + c] = S.input[(y0 + r) * 28 + x0 + c];
		return out;
	});
	const mark = $derived((V.sel.y - y0) * 5 + (V.sel.x - x0));
	const highlights = $derived(
		compact
			? [
					{ row: y0, col: x0, h: 5, w: 5, style: 'dashed' as const, color: 'var(--accent)' },
					{ row: V.sel.y, col: V.sel.x, color: 'var(--accent)' }
				]
			: [{ row: V.sel.y, col: V.sel.x, color: 'var(--accent)' }]
	);
</script>

<div class="id" class:compact>
	<div class="g">
		<TensorGrid
			data={S.input}
			rows={28}
			cols={28}
			kind="sequential"
			domain={[0, 1]}
			maxCell={compact ? 15 : 18}
			{highlights}
			label={t({ en: 'Model input', zh: '模型输入' })}
			bind:cursor={() => ({ row: V.sel.y, col: V.sel.x }), (c) => { if (c) V.setCell(c.row, c.col); }}
			onhover={(c) => c && V.setCell(c.row, c.col)}
		/>
	</div>
	<div class="txt">
		<p>
			{t({ en: `Pixel (${V.sel.y}, ${V.sel.x}) =`, zh: `像素 (${V.sel.y}, ${V.sel.x}) =` })}
			<Num value={v} />
		</p>
		{#if compact}
			<NumGrid values={patch} rows={5} cols={5} kind="sequential" domain={[0, 1]} {mark} cell={48} label={t({ en: 'The 5×5 block around it (dashed box)', zh: '它周围的 5×5 区块（虚线框）' })} />
			<p class="muted">
				{t({
					en: `${inked} of the 784 numbers are above 0 (ink); the other ${784 - inked} are exactly 0 (blank paper).`,
					zh: `784 个数中有 ${inked} 个大于 0（有墨迹）；其余 ${784 - inked} 个恰好是 0（白纸）。`
				})}
			</p>
		{:else}
			<p class="muted">
				{t({
					en: `${inked} of the 784 pixels have some ink. Every later number on this page is computed from these 784 values and the 6,722 learned parameters, nothing else.`,
					zh: `784 个像素中有 ${inked} 个带墨迹。本页之后的每一个数，都只由这 784 个值和 6,722 个学到的参数算出来。`
				})}
			</p>
			<p class="muted">
				{t({
					en: 'The training drawings were framed the same way (long side 26 pixels, centered, brightest pixel = 1), so your drawing arrives in the format the model knows.',
					zh: '训练用的画也是这样取景的（长边 26 像素、居中、最亮的像素为 1），所以你的画会以模型熟悉的格式到达。'
				})}
			</p>
		{/if}
	</div>
</div>

<style>
	.id {
		display: grid;
		grid-template-columns: minmax(0, 520px) minmax(0, 1fr);
		gap: 16px 24px;
		align-items: start;
	}
	.id.compact {
		grid-template-columns: minmax(0, 430px) minmax(0, 1fr);
	}
	@media (max-width: 760px) {
		.id,
		.id.compact {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.txt p {
		margin: 0 0 8px;
		font-size: var(--fs-sm);
	}
	.compact .txt {
		display: grid;
		gap: 8px;
		align-content: start;
	}
	.compact .txt p {
		margin: 0;
	}
</style>
