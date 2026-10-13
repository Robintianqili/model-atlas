<!--
  Attention explorer (BertViz-style): all 4 blocks × 4 heads as small maps with a summary number,
  click one to zoom; the zoomed map, the text shaded by one query's attention row, and the
  dot-product arithmetic for one query/key pair. All live.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import MiniHeat from './MiniHeat.svelte';
	import AttnText from './AttnText.svelte';
	import DotProduct from './DotProduct.svelte';
	import AttentionMatrix from '$lib/kit/AttentionMatrix.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { TGState } from './state.svelte';
	import { H, NL } from './state.svelte';
	import { headMetrics, type HeadMetric } from './analysis';

	interface Props {
		S: TGState;
		metric?: HeadMetric;
	}
	let { S, metric = $bindable('prev') }: Props = $props();

	const f = $derived(S.run?.forward);
	const T = $derived(S.T);
	const metrics = $derived(f ? headMetrics(f, NL, H) : null);
	const vals = $derived(metrics ? metrics[metric] : null);
	const best = $derived.by(() => {
		if (!vals) return -1;
		let b = 0;
		for (let i = 1; i < vals.length; i++) if (vals[i] > vals[b]) b = i;
		return Number.isNaN(vals[b]) ? -1 : b;
	});
	const metricNames: Record<HeadMetric, { en: string; zh: string }> = {
		prev: { en: 'previous token', zh: '前一个词元' },
		first: { en: 'first token', zh: '第一个词元' },
		self: { en: 'itself', zh: '自己' }
	};
	const METRICS: HeadMetric[] = ['prev', 'first', 'self'];
	let cells: HTMLButtonElement[] = $state([]);

	function choose(b: number, h: number) {
		S.block = b;
		S.head = h;
		S.keySel = null;
		S.layerId = `L${b}.attn.weights`;
	}
	function onkeydown(e: KeyboardEvent, b: number, h: number) {
		let nb = b;
		let nh = h;
		if (e.key === 'ArrowRight') nh = Math.min(H - 1, h + 1);
		else if (e.key === 'ArrowLeft') nh = Math.max(0, h - 1);
		else if (e.key === 'ArrowDown') nb = Math.min(NL - 1, b + 1);
		else if (e.key === 'ArrowUp') nb = Math.max(0, b - 1);
		else return;
		e.preventDefault();
		choose(nb, nh);
		cells[nb * H + nh]?.focus();
	}

	// zoomed matrix: row = query (S.pos), col = key (S.key); clicks in the matrix select both
	let qrow = $state<number | null>(null);
	let kcol = $state<number | null>(null);
	let synced = { r: -1, c: -1 }; // last pair copied either way (not reactive on purpose)
	$effect(() => {
		const r = S.pos;
		const c = S.key;
		qrow = r;
		kcol = c;
		synced = { r, c };
	});
	$effect(() => {
		const r = qrow;
		const c = kcol;
		if (r === null || c === null || (r === synced.r && c === synced.c)) return;
		synced = { r, c };
		untrack(() => {
			S.selectPos(r);
			if (c <= r) S.selectKey(c);
		});
	});
	const weights = $derived(f ? f.t[`L${S.block}.attn.weights`].data.subarray(S.head * T * T, (S.head + 1) * T * T) : null);
</script>

{#if f && metrics && vals}
	<div class="ax">
		<div class="gridside">
			<div class="mhead">
				<span class="lbl">{t({ en: 'Number under each map: average weight on', zh: '每张图下方的数字：平均注意力落在' })}</span>
				<span class="segmented" role="group" aria-label={t({ en: 'Summary', zh: '统计' })}>
					{#each METRICS as m (m)}
						<button type="button" aria-pressed={metric === m} onclick={() => (metric = m)}>{t(metricNames[m])}</button>
					{/each}
				</span>
			</div>
			<div class="grid" role="group" aria-label={t({ en: 'Attention maps: 4 blocks × 4 heads', zh: '注意力图：4 块 × 4 头' })}>
				<div class="corner" aria-hidden="true"></div>
				{#each Array.from({ length: H }, (_, h) => h) as h (h)}
					<div class="colh" aria-hidden="true">{t({ en: 'Head', zh: '头' })} {h + 1}</div>
				{/each}
				{#each Array.from({ length: NL }, (_, b) => b) as b (b)}
					<div class="rowh" aria-hidden="true">{t({ en: 'Block', zh: '块' })} {b + 1}</div>
					{#each Array.from({ length: H }, (_, h) => h) as h (h)}
						{@const v = vals[b * H + h]}
						<button
							bind:this={cells[b * H + h]}
							type="button"
							class="cell"
							class:cur={S.block === b && S.head === h}
							class:best={best === b * H + h}
							tabindex={S.block === b && S.head === h ? 0 : -1}
							aria-label="{t({ en: `Block ${b + 1}, head ${h + 1}`, zh: `第 ${b + 1} 块第 ${h + 1} 头` })}: {fmt3(v)}"
							aria-pressed={S.block === b && S.head === h}
							onclick={() => choose(b, h)}
							onkeydown={(e) => onkeydown(e, b, h)}
						>
							<MiniHeat
								data={f.t[`L${b}.attn.weights`].data.subarray(h * T * T, (h + 1) * T * T)}
								rows={T}
								cols={T}
								width={64}
								height={64}
								kind="sequential"
								domain={[0, 1]}
								causal
								hlRow={S.pos}
								hlColor="--q"
								label="L{b + 1} H{h + 1}"
							/>
							<span class="val num">{Number.isNaN(v) ? '—' : fmt3(v, 2)}{best === b * H + h ? ' ★' : ''}</span>
						</button>
					{/each}
				{/each}
			</div>
			<ColorLegend kind="sequential" domain={[0, 1]} label={{ en: 'attention weight', zh: '注意力权重' }} width={160} />
			<p class="note muted">
				{t({
					en: `★ = the head with the highest average weight on the ${metricNames[metric].en}, for this text. Rows of each map are queries, columns are keys; the hatched upper triangle is masked.`,
					zh: `★ = 对这段文本而言，平均注意力落在“${metricNames[metric].zh}”上最多的头。每张图的行是查询，列是键；右上的阴影三角是被掩码的部分。`
				})}
			</p>
		</div>
		<div class="zoom">
			{#if weights}
				<AttentionMatrix
					{weights}
					queries={S.labels}
					causal
					bind:row={qrow}
					bind:col={kcol}
					maxCell={30}
					label={{ en: `Block ${S.block + 1}, head ${S.head + 1}: attention weights [${T}×${T}]`, zh: `第 ${S.block + 1} 块第 ${S.head + 1} 头：注意力权重 [${T}×${T}]` }}
				/>
			{/if}
		</div>
		<div class="textrow">
			<AttnText {S} />
		</div>
		<details class="dot" open>
			<summary>{t({ en: 'One score by hand: q · k / √19, then softmax', zh: '手算一个分数：q · k / √19，再做 softmax' })}</summary>
			<DotProduct {S} />
		</details>
	</div>
{/if}

<style>
	.ax {
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		gap: 18px 24px;
		align-items: start;
	}
	.gridside {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.mhead {
		display: grid;
		gap: 4px;
	}
	.lbl {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.grid {
		display: grid;
		grid-template-columns: auto repeat(4, auto);
		gap: 4px;
		align-items: center;
		justify-content: start;
	}
	.colh,
	.rowh {
		font-size: var(--t-xs);
		color: var(--muted);
		text-align: center;
	}
	.rowh {
		writing-mode: vertical-rl;
		transform: rotate(180deg);
	}
	.cell {
		display: grid;
		justify-items: center;
		gap: 2px;
		padding: 4px;
		border: 1.5px solid var(--rule);
		border-radius: var(--r1);
		background: var(--surface);
		color: var(--ink);
	}
	.cell:hover {
		border-color: var(--ink-2);
	}
	.cell.cur {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent);
	}
	.cell.best .val {
		font-weight: 700;
		color: var(--accent-strong);
	}
	.val {
		font-size: var(--t-xs);
	}
	.note {
		font-size: var(--fs-xs);
		margin: 0;
		max-width: 360px;
	}
	.zoom {
		min-width: 0;
	}
	.textrow,
	.dot {
		grid-column: 1 / -1;
		min-width: 0;
	}
	.dot summary {
		cursor: pointer;
		font-weight: 650;
		margin-bottom: 8px;
	}
	@media (max-width: 860px) {
		.ax {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 420px) {
		.cell {
			padding: 2px;
		}
	}
</style>
