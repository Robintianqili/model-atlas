<!--
  Lesson stage "Block 2 sees bigger parts": how far one cell of each layer can see (receptive
  fields from the model, drawn to scale on the 28×28 grid), where a Conv 2-2 channel fires on the
  learner's drawing (live) and the test drawings that excite it most (precomputed, channels.json).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import ChannelResponds from './ChannelResponds.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { argmaxCell } from './data';
	import { PART_CHANNELS } from './lesson';
	import type { CnnStore, SelView } from './store.svelte';

	let { S, view }: { S: CnnStore; view: SelView } = $props();

	const LAYER = 'relu2_2';
	const H = 8;
	const W = 8;
	const ch = $derived(view.sel.layer === LAYER ? view.sel.ch : 6);
	const data = $derived(S.trace?.t[LAYER].data ?? null);
	const map = $derived(data ? data.subarray(ch * H * W, (ch + 1) * H * W) : null);
	const peak = $derived(data ? argmaxCell(data, ch, H, W) : { y: 0, x: 0 });
	const peakV = $derived(map ? map[peak.y * W + peak.x] : 0);
	const dom = $derived(S.domainFor(LAYER, () => [-1, 1]));
	const rf = $derived(S.receptiveField(LAYER, peak.y, peak.x));

	// receptive field of one cell, layer by layer (sizes from the model)
	const ladder = $derived(
		S.model
			? ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2'].map((id) => {
					const b = S.receptiveField(id, 0, 0)!;
					return { id, label: S.layer(id).label, size: b.y1 - b.y0 + 1 };
				})
			: []
	);
	const PX = 6; // px per input pixel in the ladder drawing
	const peakOf = (c: number) => {
		if (!data) return 0;
		let m = -Infinity;
		for (let i = 0; i < H * W; i++) m = Math.max(m, data[c * H * W + i]);
		return m;
	};
</script>

{#if S.trace && map && rf}
	<div class="gs">
		<div class="left">
			<figure class="ladder">
				<figcaption>{t({ en: 'How much of the 28×28 drawing one cell can see', zh: '一个格子能看到 28×28 画面中的多大区域' })}</figcaption>
				<svg viewBox="0 0 {28 * PX + 2} {28 * PX + 2}" width={28 * PX + 2} height={28 * PX + 2} role="img" aria-label={ladder.map((l) => `${t(l.label)} ${l.size}×${l.size}`).join(', ')}>
					<rect x="1" y="1" width={28 * PX} height={28 * PX} fill="var(--surface-2)" stroke="var(--rule-strong)" />
					{#each ladder as l, i (l.id)}
						{@const o = 1 + ((28 - l.size) / 2) * PX}
						<rect x={o} y={o} width={l.size * PX} height={l.size * PX} fill="none" stroke="var(--accent)" stroke-width={l.id === 'conv2_2' ? 2.5 : 1.2} opacity={0.35 + (0.65 * (i + 1)) / ladder.length} stroke-dasharray={l.id.startsWith('pool') ? '4 3' : undefined} />
					{/each}
				</svg>
				<ol class="sizes num">
					{#each ladder as l (l.id)}
						<li class:hl={l.id === 'conv2_2'}><span>{t(l.label)}</span><span>{l.size}×{l.size}</span></li>
					{/each}
				</ol>
			</figure>
		</div>

		<div class="right">
			<div class="chips" role="group" aria-label={t({ en: 'Conv 2-2 channel', zh: '卷积 2-2 的通道' })}>
				{#each PART_CHANNELS as p (p.c)}
					<button type="button" class="chip-btn" class:on={p.c === ch} aria-pressed={p.c === ch} onclick={() => view.select(LAYER, p.c)}>
						<span class="num">{t({ en: 'channel', zh: '通道' })} {p.c}</span> · {t(p.name)}
					</button>
				{/each}
			</div>
			<div class="live">
				<figure>
					<figcaption>
						{t({ en: `Channel ${ch} on your drawing`, zh: `通道 ${ch} 在你的画上` })} <span class="num muted">[8×8]</span>
						<ProvenanceBadge mode="live" compact />
					</figcaption>
					<TensorGrid data={map} rows={H} cols={W} domain={dom} maxCell={22} readout={false} highlights={[{ row: peak.y, col: peak.x, color: 'var(--accent)' }]} label={t({ en: `ReLU 2-2 channel ${ch}`, zh: `ReLU 2-2 通道 ${ch}` })} />
				</figure>
				<figure>
					<figcaption>{t({ en: 'Its strongest cell sees this patch', zh: '它最强的格子看到的区域' })}</figcaption>
					<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={120} boxes={[{ ...rf, color: 'var(--accent)' }]} />
					<span class="pv num">{t({ en: 'peak', zh: '峰值' })} {fmt3(peakV)} · {rf.y1 - rf.y0 + 1}×{rf.x1 - rf.x0 + 1}</span>
				</figure>
			</div>
			<ChannelResponds {S} {view} layerId="conv2_2" channel={ch} compact />
			<p class="others num">
				{t({ en: 'Peak on your drawing: ', zh: '在你的画上的峰值：' })}{#each PART_CHANNELS as p, i (p.c)}{i ? ' · ' : ''}<span class="nw">{t({ en: 'ch', zh: '通道' })} {p.c} = {fmt3(peakOf(p.c))}</span>{/each}
			</p>
		</div>
	</div>
{/if}

<style>
	.gs {
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		gap: 16px 22px;
		align-items: start;
	}
	@media (max-width: 640px) {
		.gs {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 6px;
	}
	.ladder svg {
		max-width: 100%;
		height: auto;
	}
	.sizes {
		list-style: none;
		margin: 0;
		padding: 0;
		font-size: var(--t-xs);
		display: grid;
		gap: 1px;
		max-width: 200px;
	}
	.sizes li {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		color: var(--ink-2);
	}
	.sizes li.hl {
		color: var(--ink);
		font-weight: 700;
	}
	.right {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.chip-btn {
		min-height: 32px;
		padding: 0 12px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r-pill);
		background: var(--surface);
		font-size: var(--fs-sm);
		cursor: pointer;
	}
	.chip-btn.on {
		border-color: var(--accent);
		background: var(--accent-soft);
		font-weight: 700;
	}
	.live {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 18px;
		align-items: flex-start;
	}
	.pv,
	.others {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin: 0;
	}
	.nw {
		white-space: nowrap;
	}
	.right :global(.cr) {
		margin-top: 0;
	}
</style>
