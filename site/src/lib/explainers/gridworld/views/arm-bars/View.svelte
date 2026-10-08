<!-- gridworld 'arm-bars'. props.mode: 'you' (Pull buttons, through bus.ts), 'agent' (a player's frame: its pick, UCB's
     bonus, ε-greedy's u against ε) or 'regret' (the race). props.reveal: the hidden win chances; props.formula. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Frame, StageProps, Tensor } from '../../../../atlas/types/index.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { ARM_NAMES, MAX_PULLS, PROBS, toRaw } from '../../world.ts';
	import { edit } from '../../bus.ts';
	import { POLICY_NAMES } from './meta.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'agent'));
	const m = $derived(P.at.frame?.metrics ?? {});
	const st = $derived((P.at.tape && P.at.index >= 0 ? P.at.tape.state(P.at.index) : {}) as { Q?: Tensor; N?: Tensor; R?: Tensor });
	const bonus = $derived(P.at.frame?.phase === 'score' ? [0, 1, 2, 3].map((k) => Math.max(0, m[`b${k}`])) : [0, 0, 0, 0]);
	const top = $derived(Math.max(1, ...[0, 1, 2, 3].map((k) => (st.Q?.data[k] ?? 0) + bonus[k])));
	const formula = $derived.by(() => {
		const f = P.at.frame;
		const ph = f ? (P.at.tape?.phases?.[f.phase] as { formula?: { tex: string; values: (x: Frame) => Record<string, number> } } | undefined) : undefined;
		return P.props.formula && f && ph?.formula ? { tex: alignTex(ph.formula.tex), values: texValues(ph.formula.values(f)) } : null;
	});
	function pull(k: number) {
		const add = (r: ReturnType<typeof toRaw>) => ({ ...r, pulls: [...r.pulls, k].slice(0, MAX_PULLS) });
		if (!edit(add)) P.rt.setInput(add(toRaw(P.rt.input)));
	}
	const pulls = $derived(toRaw(P.rt.input).pulls.length);
	// the race: regret of each strategy, pull by pull, up to this frame
	const curves = $derived(mode === 'regret' && P.at.tape ? [0, 1, 2, 3].map((k) => Array.from(P.at.tape!.metric(`r${k}`)).slice(0, P.at.index + 1)) : []);
	const rMax = $derived(Math.max(1, ...curves.map((c) => c[c.length - 1] ?? 0)));
	const xOf = (i: number, n: number) => 44 + (i / Math.max(1, n - 1)) * 540;
	const yOf = (v: number) => 214 - (v / rMax) * 200;
</script>

<div class="ab">
	{#if mode === 'regret'}
		<svg viewBox="0 0 600 240" role="img" aria-label={t({ en: 'Regret curves', zh: '遗憾曲线' })}>
			<line x1="44" y1="214" x2="584" y2="214" class="axis" /><line x1="44" y1="14" x2="44" y2="214" class="axis" />
			<text x="40" y="20" class="tick end">{fmt3(rMax)}</text><text x="52" y="20" class="tick">{t({ en: 'regret (coins)', zh: '遗憾（金币）' })}</text><text x="40" y="214" class="tick end">0</text><text x="584" y="234" class="tick end">{t({ en: `pull ${m.n}`, zh: `第 ${m.n} 次` })}</text>
			{#each curves as c, k (k)}
				<polyline points={c.map((v, i) => `${xOf(i, c.length)},${yOf(v)}`).join(' ')} style:stroke="var(--cat-{k})" class="line" />
			{/each}
		</svg>
		<ul class="legend num">{#each curves as c, k (k)}<li><i style:background="var(--cat-{k})"></i>{t(POLICY_NAMES[k])} {fmt3(c[c.length - 1] ?? 0)}</li>{/each}</ul>
	{:else}
		<div class="arms">
			{#each ARM_NAMES as a, k (a)}
				{@const q = st.Q?.data[k] ?? 0}
				<div class="arm" class:pick={m.arm === k && P.at.frame?.phase !== 'init'}>
					<b>{a}</b>
					<div class="bar" title="Q">
						<span class="q" style:height="{(q / top) * 100}%"></span>
						{#if bonus[k] > 0}<span class="bonus" style:height="{(bonus[k] / top) * 100}%" style:bottom="{(q / top) * 100}%"></span>{/if}
						{#if P.props.reveal}<i style:bottom="{(PROBS[k] / top) * 100}%" title={fmtPct(PROBS[k])}></i>{/if}
					</div>
					<span class="num">{fmt3(q)}</span>
					<span class="num small">{st.R?.data[k] ?? 0}/{st.N?.data[k] ?? 0}</span>
					{#if bonus[k] > 0}<span class="num small b">+{fmt3(bonus[k])}</span>{/if}
					{#if mode === 'you'}<button type="button" class="btn" disabled={pulls >= MAX_PULLS} onclick={() => pull(k)}>{t({ en: 'Pull', zh: '拉' })}</button>{/if}
				</div>
			{/each}
		</div>
		<p class="muted small">{[{ en: 'Bar: the average payout Q; numbers: coins won / pulls.', zh: '柱子：平均回报 Q；数字：赢到的金币 / 拉的次数。' }, P.at.frame?.phase === 'score' && { en: 'Striped: the UCB bonus.', zh: '斜纹：UCB 奖励分。' }, P.props.reveal && { en: 'Line: the hidden win chance.', zh: '横线：隐藏的中奖机会。' }].filter((x): x is { en: string; zh: string } => !!x).map((x) => t(x)).join(t({ en: ' ', zh: '' }))}</p>
		{#if m.u >= 0}
			<div class="gauge" aria-hidden="true"><span class="eps" style:width="{m.eps * 100}%"></span><span class="u" style:left="{m.u * 100}%"></span></div>
			<p class="num small">u = {fmt3(m.u)} {m.u < m.eps ? '<' : '≥'} ε = {+m.eps.toFixed(4)} → {t(m.explore ? { en: 'explore: a random machine', zh: '探索：随机选一台' } : { en: 'exploit: the best average', zh: '利用：选平均值最高的' })}</p>
		{/if}
	{/if}
	{#if P.props.reveal}<p class="num small">{t({ en: 'Hidden win chances: ', zh: '隐藏的中奖机会：' })}{ARM_NAMES.map((a, k) => `${a} ${fmtPct(PROBS[k], 0)}`).join(' · ')}</p>{/if}
	{#if formula}<FormulaCard tex={formula.tex} values={formula.values} provenance={P.provenance} />{/if}
</div>

<style>
	.ab { display: grid; gap: 10px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.arms { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
	.arm { display: grid; justify-items: center; gap: 4px; padding: 8px 4px; border: 1px solid var(--rule); border-radius: var(--r2); background: var(--surface); }
	.arm.pick { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
	.bar { position: relative; width: 60%; height: 110px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
	.bar .q, .bar .bonus { position: absolute; left: 0; right: 0; bottom: 0; background: var(--div-pos); }
	.bar .bonus { background: repeating-linear-gradient(45deg, var(--accent) 0 4px, transparent 4px 8px); }
	.bar i { position: absolute; left: -2px; right: -2px; height: 3px; background: var(--ink); } .b { color: var(--accent-strong); }
	.btn { min-height: 34px; width: 100%; } svg { width: 100%; max-width: 720px; height: auto; } .axis { stroke: var(--rule-strong); }
	.line { fill: none; stroke-width: 3; } .tick { font: 18px var(--font-mono); fill: var(--ink-2); } .end { text-anchor: end; }
	.gauge { position: relative; height: 14px; border-radius: 7px; background: var(--surface-2); border: 1px solid var(--rule); overflow: hidden; }
	.gauge .eps { position: absolute; inset: 0 auto 0 0; background: color-mix(in srgb, var(--pre) 45%, transparent); }
	.gauge .u { position: absolute; top: 0; bottom: 0; width: 3px; margin-left: -1px; background: var(--ink); }
	.small { margin: 0; font-size: var(--fs-xs); } .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 0; padding: 0; list-style: none; font-size: var(--fs-sm); }
	.legend i { display: inline-block; width: 14px; height: 4px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }
</style>
