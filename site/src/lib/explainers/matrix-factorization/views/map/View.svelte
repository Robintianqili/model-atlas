<!-- matrix-factorization 'map': the two numbers of every movie (dots colored by kind) and person (gray dots), and an arrow
     to yours, on one plane. The scale is fixed by the run's last frame, so playing the run shows the dots moving. On a
     'nudge' frame: that step's person and movie, pale before and solid after the update, with the phase formula. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Frame, StageProps, TrainState } from '../../../../atlas/types/index.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { KINDS, MOVIES, N_OTHERS, YOU, factorsOf, kindColor, vecU, vecV } from '../../world.ts';
	import { closest, dotText, factorsAt, num, sameKindNeighbours } from '../../picture.ts';

	let P: StageProps = $props();
	const W = 340, PAD = 18;
	const m = $derived(P.at.frame?.metrics ?? {});
	const nudge = $derived('u1' in m);
	const f = $derived(factorsAt(P.at));
	const end = $derived.by(() => {
		const tp = P.at.tape;
		return tp && tp.length ? factorsOf((tp.state(tp.length - 1) as Partial<TrainState>).params) : null;
	});
	const R = $derived.by(() => {
		const xs = nudge ? [m.u1, m.u2, m.v1, m.v2, m.nu1 ?? 0, m.nu2 ?? 0, m.nv1 ?? 0, m.nv2 ?? 0] : Array.from((end ?? f)?.U ?? []).concat(Array.from((end ?? f)?.V ?? []));
		return Math.max(0.25, ...xs.map(Math.abs)) * 1.12;
	});
	const X = (x: number) => W / 2 + (x / R) * (W / 2 - PAD);
	const Y = (y: number) => W / 2 - (y / R) * (W / 2 - PAD);
	const you = $derived(f ? vecU(f, YOU) : null);
	const movie = $derived(nudge ? m.item : f && P.props.movie !== undefined ? (P.props.movie === 'closest' ? closest(f, P.rt.input).i : Number(P.props.movie)) : -1);
	const shadow = $derived.by(() => {
		if (!f || !you || movie < 0 || !P.props.dot) return null;
		const v = vecV(f, movie), k = (you[0] * v[0] + you[1] * v[1]) / (you[0] ** 2 + you[1] ** 2 || 1);
		return { v, foot: [k * you[0], k * you[1]] };
	});
	const formula = $derived.by(() => {
		const fr = P.at.frame;
		const ph = fr ? (P.at.tape?.phases?.[fr.phase] as { formula?: { tex: string; values: (x: Frame) => Record<string, number> } } | undefined) : undefined;
		// lr as written in the code (0.05, not 0.0500)
		return P.props.formula && fr && ph?.formula ? { tex: alignTex(ph.formula.tex), values: { ...texValues(ph.formula.values(fr)), lr: String(fr.metrics?.lr) } } : null;
	});
	const curve = $derived.by(() => {
		const tp = P.at.tape;
		if (!P.props.loss || !tp) return [];
		const out: [number, number, number][] = [];
		for (let i = 0; i <= P.at.index; i++) {
			const fr = tp.frame(i);
			if (fr.phase !== 'step' && typeof fr.metrics?.mse === 'number') out.push([fr.step ?? 0, fr.metrics.mse, fr.metrics.you ?? NaN]);
		}
		return out;
	});
	const lmax = $derived(Math.max(1e-9, ...curve.map((c) => c[1])));
	const steps = $derived(Number((P.at.tape?.opt as { steps?: number } | undefined)?.steps ?? 1));
	// the error caption: at the start and now (the numbers step 7 narrates), over all known ratings and over yours
	const errText = $derived.by(() => {
		if (!curve.length) return [];
		const [a, z] = [curve[0], curve[curve.length - 1]], n = P.at.frame?.metrics?.n ?? 0, st = (s: number) => s.toLocaleString('en-US');
		const span = (i: 1 | 2) => (z[0] > a[0] ? { en: `${num(a[i])} at step ${st(a[0])}, ${num(z[i])} at step ${st(z[0])}`, zh: `第 ${st(a[0])} 步 ${num(a[i])}，第 ${st(z[0])} 步 ${num(z[i])}` } : { en: `${num(a[i])} at step ${st(a[0])}`, zh: `第 ${st(a[0])} 步 ${num(a[i])}` });
		const out = [{ en: `average squared error over the ${n} known ratings: ${span(1).en}`, zh: `${n} 个已知评分的平均平方误差：${span(1).zh}` }];
		if (Number.isFinite(z[2])) out.push({ en: `over your ratings: ${span(2).en}`, zh: `你的评分上：${span(2).zh}` });
		return out;
	});
</script>

{#snippet label(x: number, y: number, text: string)}
	<text x={X(x) + (X(x) > W * 0.6 ? -10 : 10)} y={Y(y) + (Y(y) < 24 ? 16 : -8)} class="lab" text-anchor={X(x) > W * 0.6 ? 'end' : 'start'}>{text}</text>
{/snippet}

{#snippet arrow(x: number, y: number, cls: string, label: string)}
	{@const a = Math.atan2(Y(y) - Y(0), X(x) - X(0))}
	<line x1={X(0)} y1={Y(0)} x2={X(x)} y2={Y(y)} class="arr {cls}" />
	<polygon points="0,0 -9,-4.5 -9,4.5" transform="translate({X(x)} {Y(y)}) rotate({(a * 180) / Math.PI})" class="head {cls}" />
	{#if label}<text x={X(x) + 6} y={Y(y) - 6} class="lab {cls}">{label}</text>{/if}
{/snippet}

<div class="map">
	<svg viewBox="0 0 {W} {W}" role="img" aria-label={t({ en: 'Map of the two numbers', zh: '两个数的地图' })}>
		<line x1={PAD / 2} y1={Y(0)} x2={W - PAD / 2} y2={Y(0)} class="axis" /><line x1={X(0)} y1={PAD / 2} x2={X(0)} y2={W - PAD / 2} class="axis" />
		{#if nudge}
			{@const lab = m.user === YOU ? t({ en: 'you', zh: '你' }) : t({ en: `person ${m.user}`, zh: `第 ${m.user} 个人` })}
			{@render arrow(m.u1, m.u2, 'nu1' in m ? 'pale' : 'you', 'nu1' in m ? '' : lab)}
			{#if 'nu1' in m}{@render arrow(m.nu1, m.nu2, 'you', lab)}{/if}
			<circle cx={X(m.v1)} cy={Y(m.v2)} r="7" class:pale={'nu1' in m} style:fill={kindColor(MOVIES[m.item].kind)} />
			{#if 'nu1' in m}<line x1={X(m.v1)} y1={Y(m.v2)} x2={X(m.nv1)} y2={Y(m.nv2)} class="move" /><circle cx={X(m.nv1)} cy={Y(m.nv2)} r="7" style:fill={kindColor(MOVIES[m.item].kind)} />{/if}
			{@render label('nv1' in m ? m.nv1 : m.v1, 'nv2' in m ? m.nv2 : m.v2, t(MOVIES[m.item].title))}
		{:else if f}
			{#if P.props.people}{#each { length: N_OTHERS } as _, u (u)}<circle cx={X(f.U[2 * u])} cy={Y(f.U[2 * u + 1])} r="3" class="person" />{/each}{/if}
			{#if shadow}<line x1={X(shadow.v[0])} y1={Y(shadow.v[1])} x2={X(shadow.foot[0])} y2={Y(shadow.foot[1])} class="drop" /><line x1={X(0)} y1={Y(0)} x2={X(shadow.foot[0])} y2={Y(shadow.foot[1])} class="shadow" />{/if}
			{#each MOVIES as mv, i (i)}<circle cx={X(f.V[2 * i])} cy={Y(f.V[2 * i + 1])} r={i === movie ? 8 : 5} class:ring={i === movie} style:fill={kindColor(mv.kind)}><title>{t(mv.title)}</title></circle>{/each}
			{#if movie >= 0}{@render label(f.V[2 * movie], f.V[2 * movie + 1], t(MOVIES[movie].title))}{/if}
			{#if you}{@render arrow(you[0], you[1], 'you', String(P.props.who ? t(P.props.who as { en: string; zh: string }) : t({ en: 'you', zh: '你' })))}{/if}
		{/if}
	</svg>
	<div class="side">
		<ul class="legend">
			{#each KINDS as k, i (i)}<li><i style:background={kindColor(i)}></i>{t(k)}</li>{/each}
			{#if P.props.people && !nudge}<li><i class="gray"></i>{t({ en: 'other people', zh: '其他人' })}</li>{/if}
		</ul>
		<p class="muted small">{t({ en: 'The colors are only for you: training never sees a movie’s kind.', zh: '颜色只是给你看的：训练从来看不到电影的类型。' })}</p>
		{#if you && !nudge}<p class="num small">{t(P.props.who ? (P.props.who as { en: string; zh: string }) : { en: 'you', zh: '你' })} = ({num(you[0])}, {num(you[1])})</p>{/if}
		{#if shadow && you}<p class="num small">{t(MOVIES[movie].title)} = ({num(shadow.v[0])}, {num(shadow.v[1])})<br /><span class="answer">{t({ en: 'prediction', zh: '预测' })} = {dotText(you, shadow.v)}</span></p>{/if}
		{#if curve.length}
			{#if curve.length > 1}<svg viewBox="0 0 200 70" class="spark" role="img" aria-label={t({ en: 'Error over the training steps', zh: '误差随训练步数的变化' })}>
				<polyline points={curve.map(([s, v]) => `${4 + (s / steps) * 192},${64 - (v / lmax) * 58}`).join(' ')} />
			</svg>{/if}
			{#each errText as line, i (i)}<p class="num small">{t(line)}</p>{/each}
		{/if}
		{#if P.props.kinds && f && !nudge}{@const k = sameKindNeighbours(f)}<p class="num small">{t({ en: `movies whose nearest movie is of the same kind: ${k} of ${MOVIES.length}`, zh: `离它最近的电影与它同类的：${MOVIES.length} 部里有 ${k} 部` })}</p>{/if}
	</div>
	{#if formula}<div class="fm"><FormulaCard tex={formula.tex} values={formula.values} provenance={P.provenance} /></div>{/if}
</div>

<style>
	.map { display: grid; gap: 10px 16px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	@media (min-width: 720px) { .map { grid-template-columns: minmax(0, 360px) minmax(0, 1fr); } .fm { grid-column: 1 / -1; } }
	svg { display: block; width: 100%; max-width: 360px; height: auto; }
	.axis { stroke: var(--rule); } .person { fill: var(--muted); opacity: 0.55; }
	circle { stroke: var(--surface); stroke-width: 1; } circle.ring { stroke: var(--ink); stroke-width: 2.5; } circle.pale { opacity: 0.35; }
	.arr { stroke-width: 3; } .arr.you { stroke: var(--accent); } .head.you { fill: var(--accent); }
	.arr.pale { stroke: var(--muted); stroke-dasharray: 4 3; } .head.pale { fill: var(--muted); }
	.lab { font: 12px var(--font-body); fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 3px; } .lab.you { font-weight: 700; fill: var(--accent-strong); }
	.drop { stroke: var(--ink-2); stroke-dasharray: 3 3; } .shadow { stroke: var(--div-pos); stroke-width: 6; opacity: 0.45; }
	.move { stroke: var(--ink-2); stroke-dasharray: 2 2; }
	.side { display: grid; gap: 6px; align-content: start; min-width: 0; }
	.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 0; padding: 0; list-style: none; font-size: var(--fs-sm); }
	.legend i { display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 50%; } .legend i.gray { background: var(--muted); opacity: 0.55; }
	.small { margin: 0; font-size: var(--fs-xs); }
	/* the answer to step 4's predict question stays hidden while the kit's predict gate makes the stage inert */
	:global([inert]) .answer { visibility: hidden; }
	.spark { max-width: 240px; } .spark polyline { fill: none; stroke: var(--accent); stroke-width: 2; }
</style>
