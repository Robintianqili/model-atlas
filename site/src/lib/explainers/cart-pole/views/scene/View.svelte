<!-- cart-pole 'scene': the cart, the pole and the four numbers at the step's moment, the p(left) gauge on an 'act' frame,
     and the pole's angle over the episode (the ±12° lines end it). On a train tape: the policy at those weights plays from
     your start, 50 steps a second (real time); with reduced motion, or with props.curve (the average episode length of
     every round up to this one, drawn above), it shows the whole episode at once. -->
<script lang="ts">
	import { untrack } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { deg, lenText, TH_LIMIT, X_LIMIT } from '../../sim.ts';
	import { sceneOf } from './meta.ts';

	let P: StageProps = $props();
	const sc = $derived(sceneOf(P));
	const PL = $derived(t({ en: 'p(left)', zh: 'p(左)' }));
	const still = $derived(!sc?.policy || !!P.props.curve || !!P.ctx?.reduced);
	let k = $state(0);
	let playing = $state(false);
	$effect(() => {
		const s = sceneOf(P);
		k = s && !still ? 0 : (s?.t ?? 0);
		playing = !still;
	});
	$effect(() => {
		if (!playing || !sc) return;
		const k0 = untrack(() => k), t0 = performance.now(), n = sc.len;
		let raf = 0;
		const tick = (now: number) => {
			k = Math.min(n, k0 + Math.floor((now - t0) / 20));
			if (k >= n) playing = false;
			else raf = requestAnimationFrame(tick);
		};
		raf = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(raf);
	});
	const cur = $derived(sc ? Math.min(k, sc.states.length - 1) : 0);
	const s = $derived(sc?.states[cur] ?? [0, 0, 0, 0]);
	const PX = 520 / (2 * X_LIMIT), CY = 150, POLE = 2 * 0.5 * PX;
	const cx = $derived(300 + Math.max(-1.05 * X_LIMIT, Math.min(1.05 * X_LIMIT, s[0])) * PX);
	const span = $derived(Math.max(20, sc?.len ?? 0));
	const xT = (i: number) => 44 + (i / span) * 540;
	const yT = (th: number) => 45 - Math.max(-1.6, Math.min(1.6, th / TH_LIMIT)) * 32;
	const END = [null, { en: 'the pole fell past 12°', zh: '杆倒过了 12°' }, { en: 'the cart left the track', zh: '小车离开了轨道' }, { en: 'the time limit', zh: '到了步数上限' }];
	const arrow = $derived(sc?.act ? (sc.act.left ? -1 : 1) : Math.sign(sc?.force ?? 0));
	const m = (v: number) => `${v < 0 ? '−' : ''}${Math.abs(v)} m`;
	const xR = (r: number) => 44 + (r / Math.max(1, sc?.rounds ?? 1)) * 540;
	const yR = (v: number) => 92 - (v / (sc?.maxLen ?? 1)) * 80;
</script>

{#if sc}
	<div class="sc">
		{#if sc.curve}
			<svg viewBox="0 0 600 112" role="img" aria-label={t({ en: 'Average episode length per round', zh: '每一轮的平均回合长度' })}>
				<line x1="44" y1="92" x2="584" y2="92" class="track" /><line x1="44" y1="12" x2="584" y2="12" class="limit" />
				<text x="40" y="17" class="tick end">{sc.maxLen}</text><text x="40" y="96" class="tick end">0</text>
				<polyline points={sc.curve.map((v, r) => `${xR(r)},${yR(v)}`).join(' ')} class="th" />
				<text x="584" y="108" class="tick end">{t(sc.curve.length ? { en: `round ${sc.curve.length - 1}: average ${lenText(sc.curve[sc.curve.length - 1])} steps per episode`, zh: `第 ${sc.curve.length - 1} 轮：平均每个回合 ${lenText(sc.curve[sc.curve.length - 1])} 步` } : { en: 'before training: no round played yet', zh: '训练之前：还没有玩过一轮' })}</text>
			</svg>
		{/if}
		<svg viewBox="0 0 600 190" role="img" aria-label={t({ en: 'The cart and the pole', zh: '小车和杆' })}>
			<line x1="40" y1={CY + 22} x2="560" y2={CY + 22} class="track" />
			{#each [-X_LIMIT, 0, X_LIMIT] as v (v)}<line x1={300 + v * PX} y1={CY + 16} x2={300 + v * PX} y2={CY + 28} class="track" /><text x={300 + v * PX} y={CY + 40} class="tick mid">{m(v)}</text>{/each}
			{#each [-TH_LIMIT, TH_LIMIT] as a (a)}<line x1={cx} y1={CY - 12} x2={cx + Math.sin(a) * POLE * 1.15} y2={CY - 12 - Math.cos(a) * POLE * 1.15} class="limit" />{/each}
			<rect x={cx - 30} y={CY - 12} width="60" height="30" rx="4" class="cart" />
			<line x1={cx} y1={CY - 12} x2={cx + Math.sin(s[2]) * POLE} y2={CY - 12 - Math.cos(s[2]) * POLE} class="pole" />
			<circle cx={cx} cy={CY - 12} r="4" class="pivot" />
			{#if arrow}
				<path d="M {cx + arrow * 34} {CY + 3} l {arrow * 34} 0 m {-arrow * 9} -7 l {arrow * 9} 7 l {-arrow * 9} 7" class="push" />
				<text x={cx + arrow * 74} y={CY + 8} class="tick" text-anchor={arrow < 0 ? 'end' : 'start'}>{t(arrow < 0 ? { en: '10 N left', zh: '向左 10 N' } : { en: '10 N right', zh: '向右 10 N' })}</text>
			{/if}
		</svg>
		<dl class="num">
			<div><dt>x =</dt><dd>{fmt3(s[0])} m</dd></div>
			<div><dt>ẋ =</dt><dd>{fmt3(s[1])} m/s</dd></div>
			<div><dt>θ =</dt><dd>{fmt3(s[2])} rad ({fmt3(deg(s[2]))}°)</dd></div>
			<div><dt>θ̇ =</dt><dd>{fmt3(s[3])} rad/s</dd></div>
		</dl>
		{#if sc.act ?? sc.first}
			{@const g = (sc.act ?? sc.first)!}
			<div class="gauge" aria-hidden="true"><span class="p" style:width="{g.p * 100}%"></span><span class="u" style:left="{g.u * 100}%"></span></div>
			<p class="num small">{sc.act ? '' : t({ en: 'First push: ', zh: '第一推：' })}{PL} = {fmt3(g.p)} · u = {fmt3(g.u)} {g.u < g.p ? '<' : '≥'} {PL} → {t(g.left ? { en: 'push left', zh: '向左推' } : { en: 'push right', zh: '向右推' })}</p>
		{/if}
		<svg viewBox="0 0 600 112" role="img" aria-label={t({ en: 'The pole angle at every step', zh: '每一步杆的角度' })}>
			<line x1="44" y1={yT(TH_LIMIT)} x2="584" y2={yT(TH_LIMIT)} class="limit" /><line x1="44" y1={yT(-TH_LIMIT)} x2="584" y2={yT(-TH_LIMIT)} class="limit" />
			<line x1="44" y1={yT(0)} x2="584" y2={yT(0)} class="track" />
			<text x="40" y={yT(TH_LIMIT) + 5} class="tick end">12°</text><text x="40" y={yT(-TH_LIMIT) + 5} class="tick end">−12°</text>
			<polyline points={sc.states.slice(0, cur + 1).map((v, i) => `${xT(i)},${yT(v[2])}`).join(' ')} class="th" />
			<circle cx={xT(cur)} cy={yT(s[2])} r="4" class="pivot" />
			<text x="584" y="108" class="tick end">{t(sc.end ? { en: `angle θ · step ${cur} of ${sc.len}`, zh: `角度 θ · 第 ${cur} / ${sc.len} 步` } : { en: `angle θ · step ${cur}`, zh: `角度 θ · 第 ${cur} 步` })}{sc.end && cur === sc.len ? ` · ${t(END[sc.end]!)}` : ''}</text>
		</svg>
		{#if sc.policy && !P.props.curve}
			<div class="row">
				<button type="button" class="btn" onclick={() => ((k = 0), (playing = true))}>{t({ en: 'Play again', zh: '再放一遍' })}</button>
				<input type="range" min="0" max={sc.len} value={cur} aria-label={t({ en: 'Step of the episode', zh: '回合中的第几步' })} oninput={(e) => ((playing = false), (k = Number(e.currentTarget.value)))} />
			</div>
		{/if}
	</div>
{:else}
	<p class="muted">{t({ en: 'Computing…', zh: '计算中…' })}</p>
{/if}

<style>
	.sc { display: grid; gap: 6px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	svg { width: 100%; height: auto; max-width: 720px; }
	.track { stroke: var(--rule-strong); stroke-width: 2; } .limit { stroke: var(--div-neg); stroke-dasharray: 5 4; stroke-width: 1.5; }
	.cart { fill: var(--surface-2); stroke: var(--ink-2); stroke-width: 2; } .pole { stroke: var(--accent-strong, var(--accent)); stroke-width: 7; stroke-linecap: round; }
	.pivot { fill: var(--ink); } .push { fill: none; stroke: var(--ink); stroke-width: 3; } .th { fill: none; stroke: var(--accent); stroke-width: 2.5; }
	.tick { font: 17px var(--font-mono); fill: var(--ink-2); } .mid { text-anchor: middle; } .end { text-anchor: end; }
	dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(12em, 1fr)); gap: 4px 12px; margin: 0; font-size: var(--fs-sm); }
	dl div { display: flex; gap: 6px; } dt { font-weight: 600; white-space: nowrap; } dd { margin: 0; white-space: nowrap; }
	.gauge { position: relative; height: 14px; border-radius: 7px; background: color-mix(in srgb, var(--div-pos) 45%, transparent); border: 1px solid var(--rule); overflow: hidden; }
	.gauge .p { position: absolute; inset: 0 auto 0 0; background: color-mix(in srgb, var(--div-neg) 80%, transparent); }
	.gauge .u { position: absolute; top: 0; bottom: 0; width: 3px; margin-left: -1px; background: var(--ink); }
	.small { margin: 0; font-size: var(--fs-xs); } .row { display: flex; gap: 8px; align-items: center; } .row input { flex: 1; min-width: 0; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
</style>
