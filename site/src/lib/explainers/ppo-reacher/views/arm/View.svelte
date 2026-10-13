<!-- ppo-reacher 'arm': the arm from above (shoulder at the origin, the shaded half disc is where training puts targets),
     the target, the fingertip's path and the four numbers of the arm; on an 'act' frame the two torques' bell curves; the
     distance to the target at every step. props.curve: the average episode return per iteration above it (and the link to
     cart-pole's REINFORCE); props.eval: the 16 test targets of the limits step and where the trained fingertip ended. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { elbow, tip, type Pt } from '../../sim.ts';
	import { sceneOf } from './meta.ts';

	let P: StageProps = $props();
	const sc = $derived(sceneOf(P));
	const K = 260, X0 = 300, y1 = 1.08;
	// the view reaches down to the lowest fingertip (all the way for the limits step's mirrored targets)
	const y0 = $derived(sc?.eval ? -1.08 : Math.max(-1.08, Math.min(-0.22, ...(sc?.states ?? []).map((s) => tip(s)[1] - 0.08))));
	const H = $derived((y1 - y0) * K);
	const X = (q: Pt) => X0 + q[0] * K, Y = (q: Pt) => (y1 - q[1]) * K;
	const last = $derived(sc ? sc.states[sc.states.length - 1] : null);
	const trail = $derived(sc ? sc.states.map((s) => `${X(tip(s))},${Y(tip(s))}`).join(' ') : '');
	// the bell curves of the two torques: N(μ, σ²) drawn on [−3, 3]
	const bell = (mu: number, s: number, x0: number) => Array.from({ length: 61 }, (_, i) => -3 + i / 10).map((a) => `${x0 + (a + 3) * 45},${92 - 70 * Math.exp(-0.5 * ((a - mu) / s) ** 2)}`).join(' ');
	const ax = (a: number, x0: number) => x0 + (Math.max(-3, Math.min(3, a)) + 3) * 45;
	const dmax = $derived(Math.max(1, ...(sc?.d ?? [0])));
	const lo = $derived(sc?.curve?.length ? Math.min(-1, ...sc.curve.filter(Number.isFinite)) : -1);
	const xC = (i: number) => 60 + (i / Math.max(1, sc?.iters ?? 1)) * 520;
	const yC = (v: number) => 12 + (v / lo) * 80;
	const TQ = [{ en: 'shoulder torque a1', zh: '肩部力矩 a1' }, { en: 'elbow torque a2', zh: '肘部力矩 a2' }];
</script>

{#if sc && last}
	<div class="arm">
		{#if sc.curve}
			<svg viewBox="0 0 600 122" role="img" aria-label={t({ en: 'Average episode return per iteration', zh: '每次迭代的平均回合回报' })}>
				<line x1="60" y1="92" x2="580" y2="92" class="axis" /><line x1="60" y1="12" x2="60" y2="92" class="axis" />
				<text x="54" y="17" class="tick end">0</text><text x="54" y="96" class="tick end">{fmt3(lo)}</text>
				<polyline points={sc.curve.map((v, i) => `${xC(i)},${yC(v)}`).join(' ')} class="ret" />
				<text x="60" y="114" class="tick">{t({ en: 'average episode return, per iteration', zh: '平均回合回报，每次迭代一个点' })}</text>
				<text x="580" y="114" class="tick end">{t({ en: `iteration ${sc.curve.length - 1}`, zh: `第 ${sc.curve.length - 1} 次迭代` })}</text>
			</svg>
		{/if}
		<svg viewBox="0 0 600 {H}" role="img" aria-label={t({ en: 'The arm seen from above', zh: '从上往下看机械臂' })}>
			<path d="M {X0 - K} {y1 * K} A {K} {K} 0 0 1 {X0 + K} {y1 * K} Z" class="area" />
			<circle cx={X0} cy={y1 * K} r={K} class="reach" />
			<line x1="0" y1={y1 * K} x2="600" y2={y1 * K} class="axis" />
			{#if sc.eval}
				{#each sc.eval as e, i (i)}
					<line x1={X(e.g)} y1={Y(e.g)} x2={X(e.p)} y2={Y(e.p)} class="miss" />
					<circle cx={X(e.g)} cy={Y(e.g)} r="9" class={e.up ? 'goal' : 'goal down'} />
					<circle cx={X(e.p)} cy={Y(e.p)} r="4" class="dot" />
				{/each}
			{:else}
				<polyline points={trail} class="trail" />
				<polyline points="{X0},{y1 * K} {X(elbow(last))},{Y(elbow(last))} {X(tip(last))},{Y(tip(last))}" class="link" />
				<circle cx={X0} cy={y1 * K} r="7" class="dot" /><circle cx={X(elbow(last))} cy={Y(elbow(last))} r="5" class="dot" />
				<circle cx={X(sc.target)} cy={Y(sc.target)} r="11" class="goal" />
				<path d="M {X(sc.target) - 6} {Y(sc.target)} h 12 M {X(sc.target)} {Y(sc.target) - 6} v 12" class="cross" />
				<circle cx={X(tip(last))} cy={Y(tip(last))} r="6" class="fin" />
			{/if}
			<text x="8" y={y1 * K - 6} class="tick">{t({ en: 'shoulder line (y = 0)', zh: '肩膀那条线（y = 0）' })}</text>
		</svg>
		{#if sc.eval}
			<p class="small">{t({ en: 'Rings: test targets (solid above the shoulder line, dashed below it). Dots: where the trained fingertip ended; each line is a miss.', zh: '圆圈：测试目标（线以上为实线，线以下为虚线）。圆点：训练好的指尖最后停在哪里；每条连线就是指尖差了多远。' })}</p>
		{:else}
			{@const rows = [['q1, q2', `${fmt3(last[0])}, ${fmt3(last[1])} rad`], ['ω1, ω2', `${fmt3(last[2])}, ${fmt3(last[3])} rad/s`], [t({ en: 'distance', zh: '距离' }), `${fmt3(Math.hypot(sc.target[0] - tip(last)[0], sc.target[1] - tip(last)[1]))} m`], [t({ en: 'step', zh: '步' }), `${sc.states.length - 1}`]]}
			<dl>{#each rows as [k, v] (k)}<div><dt>{k} =</dt><dd>{v}</dd></div>{/each}</dl>
		{/if}
		{#if sc.act}
			{@const A = sc.act}
			<svg viewBox="0 0 600 120" role="img" aria-label={t({ en: 'The two torques: bell curves and the sampled values', zh: '两个力矩：钟形曲线和采样到的值' })}>
				{#each [0, 1] as j (j)}
					<line x1={12 + j * 300} y1="92" x2={282 + j * 300} y2="92" class="axis" />
					<polyline points={bell(A.mu[j], A.sigma[j], 12 + j * 300)} class="bell" />
					<line x1={ax(A.mu[j], 12 + j * 300)} y1="18" x2={ax(A.mu[j], 12 + j * 300)} y2="92" class="mu" />
					<line x1={ax(A.a[j], 12 + j * 300)} y1="10" x2={ax(A.a[j], 12 + j * 300)} y2="96" class="draw" />
					<text x={147 + j * 300} y="114" class="tick mid">{t(TQ[j])} = {fmt3(A.a[j])}</text>
					<text x={12 + j * 300} y="106" class="tick">−3</text><text x={282 + j * 300} y="106" class="tick end">3</text>
				{/each}
			</svg>
			<p class="small">μ = ({fmt3(A.mu[0])}, {fmt3(A.mu[1])}) · σ = ({fmt3(A.sigma[0])}, {fmt3(A.sigma[1])}) · z = ({fmt3(A.z[0])}, {fmt3(A.z[1])}) · log π(a | s) = {fmt3(A.logp)}</p>
			<p class="small">{t({ en: 'Solid line: the mean μ; dashed line: the sampled torque a = μ + σ·z.', zh: '实线：均值 μ；虚线：采样得到的力矩 a = μ + σ·z。' })}</p>
		{/if}
		{#if sc.d.length && !sc.eval}
			<svg viewBox="0 0 600 96" role="img" aria-label={t({ en: 'Distance to the target at every step', zh: '每一步离目标的距离' })}>
				<line x1="60" y1="72" x2="580" y2="72" class="axis" /><line x1="60" y1="10" x2="60" y2="72" class="axis" />
				<text x="54" y="15" class="tick end">{fmt3(dmax)}</text><text x="54" y="76" class="tick end">0</text>
				<polyline points={sc.d.map((v, i) => `${60 + ((i + 1) / 40) * 520},${72 - (v / dmax) * 62}`).join(' ')} class="ret" />
				<text x="60" y="90" class="tick">{t({ en: 'distance to the target (m), step by step', zh: '离目标的距离（m），逐步' })}</text>
			</svg>
		{/if}
		{#if sc.curve}<p class="small"><a href="{base}/m/cart-pole/">{t({ en: 'Compare: cart-pole’s REINFORCE, one update per batch and no critic →', zh: '对比：倒立摆的 REINFORCE，每批数据只更新一次，也没有评论家 →' })}</a></p>{/if}
	</div>
{:else}
	<p class="muted">{t({ en: 'Computing…', zh: '计算中…' })}</p>
{/if}

<style>
	.arm { display: grid; gap: 6px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	svg { width: 100%; height: auto; max-width: 640px; }
	.area { fill: var(--accent-soft); } .reach { fill: none; stroke: var(--rule-strong); stroke-dasharray: 4 4; }
	.axis { stroke: var(--rule-strong); stroke-width: 1.5; } .trail { fill: none; stroke: var(--accent); stroke-width: 2; }
	.link { fill: none; stroke: var(--ink-2); stroke-width: 9; stroke-linecap: round; stroke-linejoin: round; }
	.dot { fill: var(--ink); } .fin { fill: var(--accent); stroke: var(--ink); stroke-width: 2; }
	.goal { fill: none; stroke: var(--div-pos); stroke-width: 3; } .goal.down { stroke: var(--div-neg); stroke-dasharray: 4 3; } .cross { stroke: var(--div-pos); stroke-width: 2; }
	.miss { stroke: var(--ink-2); stroke-width: 1.5; } .ret { fill: none; stroke: var(--accent); stroke-width: 2.5; }
	.bell { fill: none; stroke: var(--accent); stroke-width: 2; } .mu { stroke: var(--ink); stroke-width: 2; } .draw { stroke: var(--div-pos); stroke-width: 2; stroke-dasharray: 3 2; }
	.tick { font: 15px var(--font-mono); fill: var(--ink-2); } .mid { text-anchor: middle; } .end { text-anchor: end; }
	dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(12em, 1fr)); gap: 4px 12px; margin: 0; font-size: var(--fs-sm); }
	dl div { display: flex; gap: 6px; } dt { font-weight: 600; white-space: nowrap; } dd { margin: 0; white-space: nowrap; }
	.small { margin: 0; font-size: var(--fs-xs); }
</style>
