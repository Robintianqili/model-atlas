<!--
  hmm Tables: the model's three tables (start row π, transition table A, emission table B) as probabilities. Read-only
  (cells colored by the sequential ramp) unless `set` is given: then every cell has a slider (steps of 0.05, each entry at
  least 0.05), and moving one entry shares the rest of its row among the others so the row still sums to 1 (setProb).
  Used by the Capture (the side panel) and by the lattice view's task step. `set` gets the whole new raw input.
-->
<script lang="ts" module>
	// a new input re-creates the stage, so the slider being moved would lose the keyboard focus: the new one takes it back
	// (the same snippet as kalman-filter's track view)
	let refocus: { key: string; at: number } | null = null;
</script>

<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { ACTS, MINP, percentOf, rawOf, setProb, STATES, type HmmRaw } from '../../hmm.ts';

	interface Props { raw: unknown; set?: (raw: HmmRaw) => void; compact?: boolean }
	let { raw, set, compact = false }: Props = $props();
	const tb = $derived(percentOf(raw));
	const scale = $derived((void themeState.resolved, makeScale('sequential', [0, 1])));
	const p = (v: number) => String(v / 100);
	type Which = 'pi' | 'A' | 'B';
	function edit(which: Which, i: number, j: number, v: number) {
		if (!set) return;
		const next = { pi: tb.pi, A: tb.A.map((r) => r.slice()), B: tb.B.map((r) => r.slice()) };
		if (which === 'pi') next.pi = setProb(tb.pi, j, v);
		else next[which][i] = setProb(tb[which][i], j, v);
		refocus = { key: `${which}${i}${j}`, at: Date.now() };
		set({ ...rawOf(raw), ...next });
	}
	let el = $state<HTMLDivElement>();
	$effect(() => {
		if (!el || !refocus) return;
		if (Date.now() - refocus.at < 3000) el.querySelector<HTMLInputElement>(`input[data-k="${refocus.key}"]`)?.focus();
		refocus = null;
	});
	const tables = $derived([
		{ which: 'pi' as Which, title: { en: 'Start: the weather on day 0', zh: '初始：第 0 天的天气' }, rows: [{ head: null, vals: tb.pi }], cols: STATES },
		{ which: 'A' as Which, title: { en: 'Transition: today’s weather (row) → tomorrow’s (column)', zh: '转移：今天的天气（行）→ 明天的天气（列）' }, rows: tb.A.map((r, i) => ({ head: STATES[i], vals: r })), cols: STATES },
		{ which: 'B' as Which, title: { en: 'Emission: the weather (row) → her activity (column)', zh: '发射：天气（行）→ 她的活动（列）' }, rows: tb.B.map((r, i) => ({ head: STATES[i], vals: r })), cols: ACTS }
	]);
	const name = (which: Which, i: number, j: number) =>
		which === 'pi' ? t({ en: `P(day 0 is ${STATES[j].en})`, zh: `P(第 0 天是${STATES[j].zh})` })
		: which === 'A' ? t({ en: `P(tomorrow ${STATES[j].en} | today ${STATES[i].en})`, zh: `P(明天${STATES[j].zh} | 今天${STATES[i].zh})` })
		: t({ en: `P(${ACTS[j].en} | ${STATES[i].en})`, zh: `P(${ACTS[j].zh} | ${STATES[i].zh})` });
</script>

<div class="tabs" class:compact bind:this={el}>
	{#each tables as tbl (tbl.which)}
		<table>
			<caption>{t(tbl.title)}</caption>
			<thead><tr>{#if tbl.which !== 'pi'}<td></td>{/if}{#each tbl.cols as c, j (j)}<th scope="col">{t(c)}</th>{/each}</tr></thead>
			<tbody>
				{#each tbl.rows as row, i (i)}
					<tr>
						{#if row.head}<th scope="row">{t(row.head)}</th>{/if}
						{#each row.vals as v, j (j)}
							{#if set}
								<td class="ed"><label><span class="num">{p(v)}</span><input type="range" min={MINP} max={100 - MINP * (row.vals.length - 1)} step="5" value={v} data-k="{tbl.which}{i}{j}" aria-label={name(tbl.which, i, j)} aria-valuetext={p(v)} onchange={(e) => edit(tbl.which, i, j, +e.currentTarget.value)} /></label></td>
							{:else}
								<td class="ro" style:background={scale.css(v / 100)} style:color={scale.ink(v / 100)}>{p(v)}</td>
							{/if}
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	{/each}
</div>

<style>
	.tabs { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: flex-start; font-size: var(--fs-sm); min-width: 0; }
	table { border-collapse: separate; border-spacing: 3px; }
	caption { text-align: left; font-size: var(--fs-xs); color: var(--muted); padding-bottom: 2px; caption-side: top; max-width: 30ch; }
	th { font-weight: 600; font-size: var(--fs-xs); text-align: center; padding: 0 4px; }
	th[scope='row'] { text-align: right; }
	.ro { min-width: 44px; text-align: center; padding: 4px 6px; border-radius: 4px; font-variant-numeric: tabular-nums; }
	.ed label { display: grid; gap: 0; justify-items: center; }
	.ed input { width: 84px; }
	.compact .ed input { width: 70px; }
	.num { font-variant-numeric: tabular-nums; }
</style>
