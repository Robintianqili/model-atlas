<!--
  zoom.lookup: one output number of a table lookup (embedding, position_embedding, local cls_token). Modeled on
  TinyGPT's embedding stage: the tokens with the unit's position marked, the id it holds, a few rows of the learned
  table around that row with the copied row outlined, and the copied row as bars with the unit's entry marked. No
  arithmetic happens: the output is the table entry itself, which the formula and the stored-value check show.
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import TokenChips from '$lib/kit/TokenChips.svelte';
	import RoleFrame from '$lib/kit/RoleFrame.svelte';
	import Num from '../kit.parts/Num.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../types/view.ts';
	import UnitHead from '../kit.parts/UnitHead.svelte';
	import OpFormula from '../kit.parts/OpFormula.svelte';
	import RowBars from '../kit.parts/RowBars.svelte';
	import Notice from '../kit.parts/Notice.svelte';
	import { axesOfId, domainOf, layerLabel, resolveUnit, tokenIds, tokenTexts, unitWhere, vocabText } from '../kit.parts/stage.ts';

	let P: StageProps = $props();

	let pick = $derived.by<number[] | null>(() => {
		void P.at;
		void P.props.layer;
		return null;
	});
	const R = $derived(resolveUnit(P, pick));
	const e = $derived(R.e);
	const id = $derived(e?.layer ?? R.id ?? '');
	const out = $derived(R.trace && id ? R.trace.t[id] : null);
	const axes = $derived(out ? axesOfId(R.model, id, out.shape) : []);
	const T = $derived(out && axes[0] === 't' ? out.shape[0] : 0);
	const ids = $derived(tokenIds(R.trace, R.model));
	const tokens = $derived(T ? tokenTexts(P, T, ids) : []);
	const where = $derived(e ? unitWhere(e.unit, axes, tokens) : null);
	const tableName = $derived(e?.reads[1]?.tensor ?? '');
	const table = $derived(R.model && tableName ? R.model.weights[tableName] : null);
	const row = $derived(e ? (e.values.row ?? e.values.id ?? 0) : 0);
	const pos = $derived(e ? (e.values.t ?? e.unit[0]) : 0);
	const d = $derived(e ? (e.values.d ?? e.unit[e.unit.length - 1]) : 0);
	const isPos = $derived(e ? e.values.id === undefined : false);
	const Dm = $derived(table ? table.shape[1] : 0);
	const Vn = $derived(table ? table.shape[0] : 0);
	const r0 = $derived(Math.max(0, Math.min(Vn - 5, row - 2)));
	const r1 = $derived(Math.min(Vn, r0 + 5));
	const win = $derived(table ? table.data.subarray(r0 * Dm, r1 * Dm) : null);
	const tDom = $derived(table ? domainOf('diverging', table.data) : ([-1, 1] as [number, number]));
	const cellPx = $derived(Math.max(10, Math.min(18, Math.floor(320 / Math.max(1, Dm)))));
	const rowData = $derived(table ? table.data.subarray(row * Dm, (row + 1) * Dm) : null);
	const rowLabel = (r: number) => (isPos ? `${t({ en: 'position', zh: '位置' })} ${r}` : `${r}${P.props.vocab ? ` ${vocabText(P, r)}` : ''}`);
	const chips = $derived(tokens.map((s, i) => ({ text: s, id: ids ? ids[i] : undefined })));
	$effect(() => {
		if (P.autoplay && e) P.onend();
	});
</script>

{#if !e}
	<Notice error={R.error} />
{:else}
	<div class="lk">
		<UnitHead layer={layerLabel(R.model, id)} op={R.op?.def.label} where={where ?? undefined} value={e.result} />
		{#if chips.length}
			<TokenChips tokens={chips} selected={pos} showIds={!isPos} label={{ en: 'Tokens: pick one', zh: '词元：选一个' }} onselect={(i) => (pick = [i, d])} />
		{/if}
		<p class="line">
			{#if isPos}
				{t({ en: `Position ${pos} reads row ${pos} of the position table (counting from 0)`, zh: `位置 ${pos} 读取位置表的第 ${pos} 行（从 0 数起）` })}
			{:else}
				{t({ en: `Position ${pos} holds id ${row}, so it reads row ${row} of the table (counting from 0)`, zh: `位置 ${pos} 的编号是 ${row}，所以读取表的第 ${row} 行（从 0 数起）` })}
			{/if}
			<span class="muted">({t({ en: `${Vn} rows of ${Dm} learned numbers`, zh: `共 ${Vn} 行，每行 ${Dm} 个学来的数` })})</span>
		</p>
		{#if win && table}
			<div class="tbl" use:keyScroll>
				<RoleFrame role="param" label={{ en: `${tableName}: rows ${r0}–${r1 - 1}`, zh: `${tableName}：第 ${r0}–${r1 - 1} 行` }}>
					<div class="grid">
						<div class="rl num" aria-hidden="true" style="--rh: {cellPx}px">
							{#each Array.from({ length: r1 - r0 }, (_, k) => r0 + k) as r (r)}
								<span class:on={r === row}>{rowLabel(r)}</span>
							{/each}
						</div>
						<TensorGrid
							data={win}
							rows={r1 - r0}
							cols={Dm}
							domain={tDom}
							cell={cellPx}
							readout={false}
							highlights={[
								{ row: row - r0, col: 0, h: 1, w: Dm, color: 'var(--accent)' },
								{ row: row - r0, col: d, color: 'var(--highlight)' }
							]}
							label={{ en: 'table rows', zh: '表的若干行' }}
							onselect={(c) => {
								if (c.row + r0 === row) pick = [pos, c.col];
							}}
						/>
					</div>
				</RoleFrame>
			</div>
		{/if}
		{#if rowData}
			<RowBars data={rowData} selected={d} name={isPos ? `P[${pos}]` : `E[${row}]`} label={{ en: 'the copied row', zh: '被复制的那一行' }} onselect={(i) => (pick = [pos, i])} />
			<p class="line num">
				{t({ en: `entry ${d} of the row`, zh: `这一行的第 ${d} 个数` })} = <strong><Num value={e.result} /></strong>
				<span class="muted">· {t({ en: 'copied unchanged: nothing is computed', zh: '原样复制：没有任何计算' })}</span>
			</p>
		{/if}
		<OpFormula tex={R.op?.def.formula} {e} caption={where ? { en: `${layerLabel(R.model, id).en}, ${where.en}`, zh: `${layerLabel(R.model, id).zh ?? id}，${where.zh}` } : undefined} provenance={P.provenance} />
	</div>
{/if}

<style>
	.lk {
		display: grid;
		gap: 12px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.line {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.tbl {
		min-width: 0;
		max-width: 100%;
		overflow-x: auto;
	}
	.grid {
		display: flex;
		gap: 6px;
		align-items: flex-start;
	}
	.rl {
		display: grid;
		font-size: 9.5px;
		line-height: 1;
		color: var(--muted);
		text-align: right;
		white-space: pre;
	}
	.rl span {
		display: flex;
		align-items: center;
		justify-content: flex-end;
		height: var(--rh, 16px);
	}
	.rl span.on {
		color: var(--ink);
		font-weight: 700;
	}
</style>
