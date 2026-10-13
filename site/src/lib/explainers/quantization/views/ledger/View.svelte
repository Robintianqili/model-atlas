<!--
  'quantization/ledger': the rounding ledger (meta.ts has the props). One number line per block of each setting: the
  levels the codes stand for as ticks, each value as a dot joined to the level it is stored as. Then the table, one row per
  value (the row of the frame's value outlined), and a summary line.
-->
<script lang="ts">
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { ledger, type Ledger, type LedgerOpt } from '../../quant.ts';
	import { optOf, settingText, valsAt } from './meta.ts';

	let { at, props }: StageProps = $props();

	// the toy's own settings (props.controls)
	let mine = $state<LedgerOpt>({ bits: 4, scheme: 'absmax', clip: 'none', block: 8 });
	const vals = $derived(valsAt({ at }));
	const run = $derived(optOf({ at }));
	const opt = $derived<LedgerOpt>(props.controls ? mine : run);
	const L = $derived(vals ? ledger(vals, opt) : null);
	const A = $derived(vals && props.alt && !props.controls && run.alt ? ledger(vals, { ...run, ...run.alt }) : null);
	const cur = $derived(at.frame?.phase === 'round' ? (at.frame.step ?? -1) : -1);
	const both = $derived(L ? (A ? [L, A] : [L]) : []);

	// one domain for every line: the values and every level
	const dom = $derived.by(() => {
		let lo = 0;
		let hi = 0;
		for (const G of both) {
			for (const v of G.x) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
			for (const b of G.blocks) [lo, hi] = [Math.min(lo, (b.qlo - b.z) * b.s), Math.max(hi, (b.qhi - b.z) * b.s)];
		}
		const pad = (hi - lo || 1) * 0.04;
		return [lo - pad, hi + pad];
	});
	const [W, RH, M] = [440, 46, 12];
	const sx = (v: number) => M + ((v - dom[0]) / (dom[1] - dom[0])) * (W - 2 * M);
	const levels = (G: Ledger, bi: number) => {
		const b = G.blocks[bi];
		return Array.from({ length: b.qhi - b.qlo + 1 }, (_, k) => (b.qlo + k - b.z) * b.s);
	};
	const lines = $derived(both.flatMap((G, gi) => G.blocks.map((b, bi) => ({ G, gi, bi, b }))));

	const choose = <K extends keyof LedgerOpt>(k: K, v: LedgerOpt[K]) => (mine = { ...mine, [k]: v });
	const CTL = [
		{ k: 'bits' as const, label: { en: 'Bits', zh: '位数' }, opts: [[8, 'int8'], [4, 'int4']] as [number, string][] },
		{ k: 'scheme' as const, label: { en: 'Scheme', zh: '方案' }, opts: [['absmax', 'absmax'], ['affine', { en: 'affine', zh: '仿射' }]] as [string, string | { en: string; zh: string }][] },
		{ k: 'clip' as const, label: { en: 'Clip', zh: '截断' }, opts: [['none', { en: 'no', zh: '不截断' }], ['second', { en: 'at 2nd largest', zh: '第二大处' }]] as [string, string | { en: string; zh: string }][] },
		{ k: 'block' as const, label: { en: 'Block size', zh: '块大小' }, opts: [[8, '8'], [4, '4'], [2, '2']] as [number, string][] }
	];
	const lab = (x: string | { en: string; zh: string }) => (typeof x === 'string' ? x : t(x));
</script>

{#if L}
	<div class="ledger">
		{#if props.controls}
			<div class="ctl">
				{#each CTL as c (c.k)}
					<div class="seg" role="group" aria-label={t(c.label)}>
						<span class="seglab">{t(c.label)}</span>
						{#each c.opts as [v, name] (String(v))}
							<button type="button" class="btn" aria-pressed={mine[c.k] === v} disabled={c.k === 'clip' && mine.scheme === 'affine'} onclick={() => choose(c.k, v as never)}>{lab(name)}</button>
						{/each}
					</div>
				{/each}
			</div>
		{/if}
		<svg viewBox="0 0 {W} {lines.length * RH + 8}" role="img" aria-label={t({ en: 'Each value and the level it is stored as, one ruler per block', zh: '每个值和它被存成的刻度，每一块一把尺子' })}>
			{#each lines as ln, k (k)}
				{@const y = 8 + k * RH + RH / 2}
				<text class="lbl" x={M} y={y - 12}>{t(settingText(ln.G.opt, true))}{ln.G.blocks.length > 1 ? ` · ${t({ en: 'block', zh: '块' })} ${ln.bi}` : ''} · s = {fmt3(ln.b.s)}</text>
				<line class="axis" x1={sx(ln.b.lo === ln.b.hi ? dom[0] : (ln.b.qlo - ln.b.z) * ln.b.s)} x2={sx((ln.b.qhi - ln.b.z) * ln.b.s)} y1={y} y2={y} />
				{#each levels(ln.G, ln.bi) as v, j (j)}
					<line class="tick" x1={sx(v)} x2={sx(v)} y1={y - 4} y2={y + 4} />
				{/each}
				<line class="zero" x1={sx(0)} x2={sx(0)} y1={y - 7} y2={y + 7} />
				{#each ln.G.x as v, i (i)}
					{#if ln.G.blockOf[i] === ln.bi}
						<line class="link" class:cur={i === cur && ln.gi === 0} x1={sx(v)} x2={sx(ln.G.xh[i])} y1={y + 10} y2={y + 1} />
						<circle class="dot" class:cur={i === cur && ln.gi === 0} class:alt={ln.gi === 1} cx={sx(v)} cy={y + 10} r="3.5" />
					{/if}
				{/each}
			{/each}
		</svg>
		<div class="tbl" use:keyScroll>
			<table>
				<thead>
					<tr>
						<th scope="col">i</th>
						<th scope="col">x</th>
						<th scope="col">x / s{L.opt.scheme === 'affine' ? ' + z' : ''}</th>
						<th scope="col">q</th>
						<th scope="col">x̂</th>
						<th scope="col">{t({ en: 'error', zh: '误差' })}</th>
						{#if A}<th scope="col">q′</th><th scope="col">{t({ en: 'error′', zh: '误差′' })}</th>{/if}
					</tr>
				</thead>
				<tbody>
					{#each L.x as v, i (i)}
						<tr class:cur={i === cur}>
							<td>{i}</td>
							<td class="num">{fmt3(v)}</td>
							<td class="num">{fmt3(L.r[i])}</td>
							<td class="num">{L.q[i]}{#if L.clipped.includes(i)}<span class="note"> {t({ en: 'clipped', zh: '截断' })}</span>{/if}</td>
							<td class="num">{fmt3(L.xh[i])}</td>
							<td class="num">{fmt3(L.err[i])}</td>
							{#if A}<td class="num">{A.q[i]}</td><td class="num">{fmt3(A.err[i])}</td>{/if}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="sum">
			{t({ en: 'Largest error', zh: '最大误差' })} <strong class="num">{fmt3(L.maxErr)}</strong>, {t({ en: 'mean', zh: '平均' })} <span class="num">{fmt3(L.meanErr)}</span>, <span class="num">{fmt3(L.bitsPerValue)}</span> {t({ en: 'bits per value with the scales', zh: '位/值（含缩放比例）' })}{#if A}<br />′ = {t(settingText(A.opt))}{colon()} {t({ en: 'largest error', zh: '最大误差' })} <strong class="num">{fmt3(A.maxErr)}</strong>, {t({ en: 'mean', zh: '平均' })} <span class="num">{fmt3(A.meanErr)}</span>, <span class="num">{fmt3(A.bitsPerValue)}</span> {t({ en: 'bits per value', zh: '位/值' })}{/if}
		</p>
	</div>
{/if}

<style>
	.ledger {
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.ctl {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 14px;
	}
	.seg {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px;
	}
	.seglab {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.seg .btn {
		font-size: var(--fs-xs);
	}
	.seg .btn[aria-pressed='true'] {
		outline: 2px solid var(--accent);
		outline-offset: -2px;
		font-weight: 600;
	}
	svg {
		width: 100%;
		max-width: 620px;
		height: auto;
		display: block;
	}
	.lbl {
		font-size: 12px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.axis {
		stroke: var(--rule-strong);
		stroke-width: 1;
	}
	.tick {
		stroke: var(--ink-2);
		stroke-width: 1;
	}
	.zero {
		stroke: var(--ink);
		stroke-width: 1.5;
	}
	.link {
		stroke: var(--accent);
		stroke-width: 1.2;
	}
	.dot {
		fill: var(--accent);
	}
	.dot.alt {
		fill: var(--div-pos);
	}
	.link.cur,
	.dot.cur {
		stroke: var(--highlight);
		stroke-width: 2.5;
	}
	.tbl {
		overflow-x: auto;
		min-width: 0;
	}
	table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		width: 100%;
	}
	th,
	td {
		padding: 2px 8px;
		text-align: right;
		border-bottom: 1px solid var(--rule);
		white-space: nowrap;
	}
	th {
		color: var(--ink-2);
		font-weight: 500;
	}
	tr.cur td {
		outline: 2px solid var(--highlight);
		outline-offset: -2px;
	}
	.note {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.sum {
		margin: 0;
		font-size: var(--fs-sm);
	}
</style>
