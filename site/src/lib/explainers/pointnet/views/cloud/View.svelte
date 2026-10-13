<!-- pointnet 'cloud' view: the model's input points on the plane, in five modes (meta.ts). In 'critical' with
     props.task the buttons erase the critical points (setInput), or put the points back. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { I18n, StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { T, argmax, crit, cutRaw, label, maxDiff, variants } from '../../cloud.ts';
	import type { CloudRaw } from '../../adapters/points2d/adapter.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'points'));
	const x = $derived(T(P.rt, 'input'));
	const c = $derived(crit(P.rt));
	const V = $derived(mode === 'order' || mode === 'stray' ? variants(P.rt) : null);
	const probs = $derived(T(P.rt, 'softmax')?.data);
	const h = $derived(T(P.rt, 'relu2'));
	const L = (en: string, zh: string) => t({ en, zh });
	const ans = (p: ArrayLike<number> | undefined) => (p ? `${t(label(argmax(p)))} ${fmtPct(p[argmax(p)])}` : '…');
	let picked = $state<number | null>(null);
	const f = $derived(Math.min((c?.F ?? 1) - 1, Math.max(0, picked ?? Number(P.props.f ?? 0))));
	$effect(() => {
		void P.props.f;
		picked = null;
	});
	const scale = $derived(makeScale('sequential', [0, Math.max(1e-9, c?.max[f] ?? 1)], themeState.resolved));
	const val = (i: number) => (h ? h.data[i * h.shape[1] + f] : 0);
	const raw = $derived((P.rt.input ?? {}) as CloudRaw);
	const set = (r: CloudRaw) => (P.setInput ? P.setInput(r) : P.rt.setInput(r));
	const cut = () => c && probs && set(cutRaw(raw, c.pts, argmax(probs)));
	const back = () => raw.cut?.back && set({ points: raw.cut.back });
	type Pane = { xy: ArrayLike<number>; title: I18n; answer: string; line?: boolean; ring?: number[] };
	const panes = $derived.by((): Pane[] => {
		if (!x) return [];
		const own: Pane = { xy: x.data, title: { en: 'Your points', zh: '你的点' }, answer: ans(probs), line: mode === 'order' };
		if (mode === 'order' && V?.shuf) return [{ ...own, title: { en: 'Table order', zh: '表中的顺序' } }, { xy: V.shuf.xy, title: { en: 'Shuffled', zh: '打乱之后' }, answer: ans(V.shuf.p), line: true }];
		if (mode === 'stray' && V?.stray) return [own, { xy: V.stray.xy, title: { en: 'Plus one stray point', zh: '加上一个离群点' }, answer: ans(V.stray.p), ring: [V.stray.xy.length / 2 - 1] }];
		if (mode === 'feature' && c) return [{ ...own, ring: [c.win[f] >= 0 ? c.win[f] : -1] }];
		return [own];
	});
	const isCrit = $derived(new Set(c?.pts ?? []));
	const fill = (i: number) => (mode === 'feature' ? scale.css(val(i)) : mode === 'critical' && isCrit.has(i) ? 'var(--accent)' : 'var(--muted)');
	const rad = (i: number) => (mode === 'critical' && c && isCrit.has(i) ? 0.035 + 0.012 * Math.sqrt(c.wins[i]) : 0.026);
	const firstRows = $derived(x ? Array.from({ length: Math.min(3, x.shape[0]) }, (_, i) => `${i}: (${fmt3(x.data[2 * i])}, ${fmt3(x.data[2 * i + 1])})`).join('; ') + (x.shape[0] > 3 ? '; …' : '') : '');
	const poly = (xy: ArrayLike<number>) => Array.from({ length: xy.length / 2 }, (_, i) => `${xy[2 * i]},${-xy[2 * i + 1]}`).join(' ');
</script>

{#if !x || !c}
	<p class="muted">{L('Draw some points, or pick a shape, to see what the network does with them.', '画几个点，或者选一个形状，看看网络怎么处理它们。')}</p>
{:else}
	<div class="cloud">
		<div class="panes" class:two={panes.length > 1}>
			{#each panes as p, k (k)}
				<figure>
					<figcaption>{t(p.title)}</figcaption>
					<svg viewBox="-1.3 -1.3 2.6 2.6" role="img" aria-label="{t(p.title)}: {p.xy.length / 2} {L('points', '个点')}. {L('Answer', '答案')}: {p.answer}">
						<line class="axis" x1="-1.25" x2="1.25" y1="0" y2="0" /><line class="axis" y1="-1.25" y2="1.25" x1="0" x2="0" />
						<circle class="unit" cx="0" cy="0" r="1" />
						{#if p.line}<polyline class="order" points={poly(p.xy)} />{/if}
						{#each { length: p.xy.length / 2 } as _, i (i)}
							<circle class="pt" cx={p.xy[2 * i]} cy={-p.xy[2 * i + 1]} r={k === 0 ? rad(i) : 0.026} fill={k === 0 ? fill(i) : 'var(--muted)'} />
						{/each}
						{#each p.ring ?? [] as i (i)}
							{#if i >= 0}<circle class="ring" cx={p.xy[2 * i]} cy={-p.xy[2 * i + 1]} r="0.075" />{/if}
						{/each}
					</svg>
					<p class="ans">{L('Answer', '答案')}: <b>{p.answer}</b></p>
				</figure>
			{/each}
		</div>
		{#if mode === 'points'}
			<p class="small">{L('The cross is 0; the dashed circle has radius 1. Table rows (counting from 0):', '十字处是 0；虚线圆的半径是 1。表的前几行（从 0 数起）：')} <code>{firstRows}</code></p>
		{:else if mode === 'order' && V?.shuf}
			<p class="small">{L(`The line joins the points in table order. The ${c.F} numbers after the max: largest difference ${fmt3(maxDiff(T(P.rt, 'pool')?.data, V.shuf.pool))}.`, `线按表中的顺序连接各点。取最大值之后的 ${c.F} 个数：最大差 ${fmt3(maxDiff(T(P.rt, 'pool')?.data, V.shuf.pool))}。`)}</p>
		{:else if mode === 'feature'}
			<div class="row">
				<button type="button" class="btn ghost" disabled={f <= 0} onclick={() => (picked = f - 1)} aria-label={L('Previous feature', '上一个特征')}>◀</button>
				<button type="button" class="btn ghost" disabled={f >= c.F - 1} onclick={() => (picked = f + 1)} aria-label={L('Next feature', '下一个特征')}>▶</button>
				<span aria-live="polite">{L(`Feature ${f} (counting from 0): `, `特征 ${f}（从 0 数起）：`)}{c.win[f] >= 0 ? L(`largest ${fmt3(c.max[f])}, at point ${c.win[f]} (ringed)`, `最大 ${fmt3(c.max[f])}，在点 ${c.win[f]}（加圈）`) : L('0 at every point: no winner', '每个点都是 0：没有赢家')}</span>
			</div>
			<ColorLegend kind="sequential" domain={scale.domain} label={{ en: "the chosen feature at each point", zh: "每个点上所选特征的值" }} />
		{:else if mode === 'critical'}
			<p class="small"><span class="key"></span>{L(`${c.pts.length} critical point${c.pts.length === 1 ? '' : 's'} of ${c.n} (larger: wins more features)`, `${c.n} 个点中有 ${c.pts.length} 个关键点（越大：赢的特征越多）`)}</p>
			{#if P.props.task}
				<div class="row">
					<button type="button" class="btn" disabled={!c.pts.length} onclick={cut}>{L('Remove the critical points', '删掉这些关键点')}</button>
					<button type="button" class="btn ghost" disabled={!raw.cut?.back} onclick={back}>{L('Put them back', '放回去')}</button>
					<span aria-live="polite">{L(`Presses: ${raw.cut?.n ?? 0}; points left: ${c.n}`, `已按：${raw.cut?.n ?? 0} 次；剩下 ${c.n} 个点`)}{raw.cut ? L(`; first answer: ${t(label(raw.cut.cls))}`, `；最初的答案：${t(label(raw.cut.cls))}`) : ''}</span>
				</div>
			{/if}
		{/if}
	</div>
{/if}

<style>
	.cloud { display: grid; gap: var(--s2); min-width: 0; }
	.panes { display: grid; gap: var(--s2); grid-template-columns: minmax(0, 340px); }
	.panes.two { grid-template-columns: repeat(2, minmax(0, 300px)); }
	figure { margin: 0; min-width: 0; }
	figcaption, .ans { font-size: var(--fs-sm); color: var(--ink-2); margin: 0; }
	svg { display: block; width: 100%; height: auto; border: 1px solid var(--rule); border-radius: var(--r1); background: var(--surface); }
	.axis { stroke: var(--rule); stroke-width: 0.01; }
	.unit { fill: none; stroke: var(--rule-strong); stroke-width: 0.012; stroke-dasharray: 0.05 0.04; }
	.order { fill: none; stroke: var(--muted); stroke-width: 0.008; }
	.pt { stroke: var(--ink-2); stroke-width: 0.006; }
	.ring { fill: none; stroke: var(--accent); stroke-width: 0.02; }
	.row { display: flex; flex-wrap: wrap; gap: var(--s2); align-items: center; }
	.key { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--accent); margin-right: 6px; }
	p { margin: 0; }
	code { font-size: var(--fs-sm); }
</style>
