<!--
  One attention score, done by hand: query q_i (blue) · key k_j (red), 19 products, the sum,
  the 1/√19 scaling, then the softmax over the row. Every number is read from the live trace,
  and the result is checked against the score the model itself computed.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { sym } from '$lib/kit/tex';
	import { makeScale, domainOf } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { TGState } from './state.svelte';
	import { D, DH } from './state.svelte';

	type Part = 'table' | 'score' | 'softmax' | 'check';
	interface Props {
		S: TGState;
		/** which pieces to show (lesson: no softmax yet); default all */
		parts?: Part[];
	}
	let { S, parts = ['table', 'score', 'softmax', 'check'] }: Props = $props();
	const has = (p: Part) => parts.includes(p);

	const f = $derived(S.run?.forward);
	const T = $derived(S.T);
	const b = $derived(S.block);
	const h = $derived(S.head);
	const i = $derived(S.pos);
	const j = $derived(Math.min(S.key, i));
	const scale = 1 / Math.sqrt(DH);

	const calc = $derived.by(() => {
		if (!f) return null;
		const q = f.t[`L${b}.attn.q`].data;
		const k = f.t[`L${b}.attn.k`].data;
		const qv = Array.from({ length: DH }, (_, c) => q[i * D + h * DH + c]);
		const kv = Array.from({ length: DH }, (_, c) => k[j * D + h * DH + c]);
		const prod = qv.map((x, c) => x * kv[c]);
		const dot = prod.reduce((a, x) => a + x, 0);
		const s = dot * scale;
		const masked = f.t[`L${b}.attn.masked`].data;
		const o = (h * T + i) * T;
		let m = -Infinity;
		for (let jj = 0; jj <= i; jj++) m = Math.max(m, masked[o + jj]);
		let Z = 0;
		for (let jj = 0; jj <= i; jj++) Z += Math.exp(masked[o + jj] - m);
		const num = Math.exp(s - m);
		const a = num / Z;
		const tracedS = f.t[`L${b}.attn.scores`].data[o + j];
		const tracedA = f.t[`L${b}.attn.weights`].data[o + j];
		return { qv, kv, prod, dot, s, m, Z, num, a, tracedS, tracedA, maxAbs: domainOf('diverging', qv, kv)[1], prodMax: domainOf('diverging', prod)[1] };
	});
	const sc = $derived(calc ? makeScale('diverging', [-calc.maxAbs, calc.maxAbs], themeState.resolved) : null);
	const scp = $derived(calc ? makeScale('diverging', [-calc.prodMax, calc.prodMax], themeState.resolved) : null);

	const tex1 = `\\begin{aligned}${sym('accent', 's_{ij}')} &= \\frac{${sym('q', 'q_i')} \\cdot ${sym('k', 'k_j')}}{\\sqrt{19}} = \\frac{\\sum_{c=0}^{18} ${sym('q', 'q_{i,c}')}\\,${sym('k', 'k_{j,c}')}}{\\sqrt{19}} \\\\ &= \\frac{ {{dot}} }{ {{sq}} } = {{s}}\\end{aligned}`;
	const tex2 = `\\begin{aligned}${sym('prob', 'a_{ij}')} &= \\frac{e^{${sym('accent', 's_{ij}')} - m}}{\\sum_{j' \\le i} e^{s_{ij'} - m}} \\\\ &= \\frac{ {{num}} }{ {{Z}} } = {{a}}\\end{aligned}`;
</script>

{#if calc && sc && scp}
	<div class="dpw">
	<div class="dp" class:one={!has('table')}>
		<p class="head">
			{t({ en: `Block ${b + 1}, head ${h + 1}: query`, zh: `第 ${b + 1} 块，第 ${h + 1} 头：查询` })} <span class="qtok">{S.label(i)}</span>
			{t({ en: `(position ${i}) · key`, zh: `（位置 ${i}）· 键` })} <span class="ktok">{S.label(j)}</span>
			{t({ en: `(position ${j})`, zh: `（位置 ${j}）` })}
		</p>
		{#if has('table')}
		<div class="tablewrap">
			<table class="num">
				<caption class="visually-hidden">{t({ en: 'The 19 products of the query and key entries', zh: '查询与键的 19 个分量乘积' })}</caption>
				<thead>
					<tr>
						<th scope="col">c</th>
						<th scope="col" class="qh">q<sub>i,c</sub></th>
						<th scope="col" class="kh">k<sub>j,c</sub></th>
						<th scope="col">q·k</th>
					</tr>
				</thead>
				<tbody>
					{#each calc.prod as p, c (c)}
						<tr>
							<td class="c">{c}</td>
							<td style="background: {sc.css(calc.qv[c])}; color: {sc.ink(calc.qv[c])}">{fmt3(calc.qv[c])}</td>
							<td style="background: {sc.css(calc.kv[c])}; color: {sc.ink(calc.kv[c])}">{fmt3(calc.kv[c])}</td>
							<td style="background: {scp.css(p)}; color: {scp.ink(p)}">{fmt3(p)}</td>
						</tr>
					{/each}
				</tbody>
				<tfoot>
					<tr>
						<th scope="row" colspan="3">{t({ en: 'sum', zh: '求和' })} Σ</th>
						<td><Num value={calc.dot} /></td>
					</tr>
				</tfoot>
			</table>
		</div>
		{/if}
		<div class="formulas">
			{#if has('score')}
			<FormulaCard
				tex={tex1}
				values={{ dot: calc.dot, sq: Math.sqrt(DH), s: calc.s }}
				caption={{ en: 'Scaled dot product', zh: '缩放点积' }}
				where={[
					{ tex: '\\sqrt{19}', text: { en: '≈ 4.36, with 19 = the numbers per head: keeps scores from growing with the vector length', zh: '≈ 4.36，19 = 每个头的数的个数：防止分数随向量维度变大' } }
				]}
				provenance="live"
			/>
			{/if}
			{#if has('softmax')}
			<FormulaCard
				tex={tex2}
				values={{ num: calc.num, Z: calc.Z, a: calc.a }}
				caption={{ en: 'Softmax over the row (keys 0…i)', zh: '对这一行做 softmax（键 0…i）' }}
				where={[{ tex: 'm', text: { en: `largest score in the row (${fmt3(calc.m)}), subtracted for numerical safety`, zh: `这一行的最大分数（${fmt3(calc.m)}），减去它只是为了数值稳定` } }]}
				provenance="live"
			/>
			{/if}
			{#if has('check')}
			<p class="check">
				{t({ en: 'Check against the model’s own trace:', zh: '与模型自己的计算结果核对：' })}
				s = <Num value={calc.tracedS} />{#if has('softmax')}, a = <Num value={calc.tracedA} />{/if}
				<span class="muted">(|Δs| = {fmt3(Math.abs(calc.tracedS - calc.s))})</span>
			</p>
			{/if}
		</div>
	</div>
	</div>
{/if}

<style>
	.dpw {
		container: dp / inline-size;
	}
	.dp {
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		gap: 16px;
		align-items: start;
	}
	.head {
		grid-column: 1 / -1;
		margin: 0;
		font-size: var(--fs-sm);
	}
	.qtok,
	.ktok {
		font-family: var(--font-mono);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
		background: var(--surface-2);
	}
	.qtok {
		box-shadow: inset 0 0 0 2px var(--q);
	}
	.ktok {
		box-shadow: inset 0 0 0 2px var(--k);
	}
	.tablewrap {
		max-height: 470px;
		overflow: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
	}
	table {
		border-collapse: collapse;
		font-size: 12px;
	}
	th,
	td {
		padding: 2px 8px;
		text-align: right;
		border-bottom: 1px solid var(--surface);
	}
	thead th {
		position: sticky;
		top: 0;
		background: var(--surface-2);
	}
	.qh {
		box-shadow: inset 0 -2px 0 var(--q);
	}
	.kh {
		box-shadow: inset 0 -2px 0 var(--k);
	}
	.c {
		color: var(--muted);
	}
	/* the sum row stays visible while the 19 products scroll */
	tfoot th,
	tfoot td {
		position: sticky;
		bottom: 0;
		border-top: 1.5px solid var(--ink-2);
		background: var(--surface);
	}
	.formulas {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.check {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.dp.one {
		grid-template-columns: minmax(0, 1fr);
	}
	@container dp (max-width: 600px) {
		.dp {
			grid-template-columns: minmax(0, 1fr);
		}
		.tablewrap {
			max-height: 300px;
		}
	}
</style>
