<!--
  Part 2 (precomputed, finetune.json): test accuracy vs drawings per class (log x axis) for
  head only / full fine-tune / from scratch, mean ± std over 5 seeds, plus the learner's live run.
  Pointer, touch and arrow keys pick a training-set size; the readout lists every seed.
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { linear, linePath, bandPath, markerPath, clamp } from './chart';
	import { METHODS, type FinetuneJson, type Method } from './data';
	import { lab, type CurveSel } from './state.svelte';

	interface Props {
		finetune: FinetuneJson;
		/** selection state (training-set size, zoom, seeds); the guided lesson passes its own */
		sel?: CurveSel;
		/** show the learner's live Part 1 run as a diamond (default true) */
		showLive?: boolean;
		/** lesson view: no options row, no parameter bars, no table */
		compact?: boolean;
	}
	let { finetune, sel = lab, showLive = true, compact = false }: Props = $props();
	const gid = $props.id();

	const SHAPE: Record<Method, 'circle' | 'square' | 'triangle'> = { head: 'circle', full: 'square', scratch: 'triangle' };
	const SLOT: Record<Method, number> = { head: 2, full: 6, scratch: 3 };
	const color = (m: Method) => categorical(SLOT[m], themeState.resolved);
	const LABEL: Record<Method, { en: string; zh: string }> = {
		head: { en: 'Head only (frozen body)', zh: '只训练输出头（冻结主体）' },
		full: { en: 'Full fine-tune', zh: '全量微调' },
		scratch: { en: 'From scratch', zh: '从零训练' }
	};

	let w = $state(600);
	const H = $derived(w < 480 ? 260 : 300);
	const M = $derived({ l: 44, r: 16, t: 14, b: 44 });
	const ns = $derived(finetune.ns);
	const idx = $derived(Math.max(0, ns.indexOf(sel.part2N)));
	const x = $derived(linear([0, Math.log2(ns[ns.length - 1])], [M.l, w - M.r]));
	const yDom = $derived<[number, number]>(sel.part2Zoom ? [0.9, 1] : [0.4, 1]);
	const y = $derived(linear(yDom, [H - M.b, M.t]));
	const yTicks = $derived(sel.part2Zoom ? [0.9, 0.92, 0.94, 0.96, 0.98, 1] : [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]);
	const xs = $derived(ns.map((n) => x(Math.log2(n))));
	const narrow = $derived(w < 460);
	// the gap between 128 and 512 (256 and 1024 were not run): dashed connector
	const gapAt = $derived(ns.findIndex((n, i) => i > 0 && n / ns[i - 1] > 2));

	function series(m: Method) {
		const r = finetune.results[m];
		return {
			mean: r.map((v) => y(clamp(v.mean, yDom[0] - 0.05, 1))),
			lo: r.map((v) => y(clamp(v.mean - v.std, yDom[0] - 0.05, 1))),
			hi: r.map((v) => y(clamp(v.mean + v.std, yDom[0] - 0.05, 1)))
		};
	}
	const S = $derived(Object.fromEntries(METHODS.map((m) => [m, series(m)])) as Record<Method, ReturnType<typeof series>>);

	function pick(e: PointerEvent, svg: SVGSVGElement) {
		const r = svg.getBoundingClientRect();
		const px = ((e.clientX - r.left) / r.width) * w;
		let best = 0;
		for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - px) < Math.abs(xs[best] - px)) best = i;
		sel.part2N = ns[best];
	}
	function key(e: KeyboardEvent) {
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') sel.part2N = ns[Math.min(ns.length - 1, idx + 1)];
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') sel.part2N = ns[Math.max(0, idx - 1)];
		else if (e.key === 'Home') sel.part2N = ns[0];
		else if (e.key === 'End') sel.part2N = ns[ns.length - 1];
		else return;
		e.preventDefault();
	}
	const live = $derived(
		showLive && lab.liveRun && lab.liveRun.preset && lab.liveRun.testAcc !== null && lab.liveRun.nA === lab.liveRun.nB ? lab.liveRun : null
	);
	const p1 = (v: number) => (v * 100).toFixed(1);
	const p2 = (v: number) => (v * 100).toFixed(2);
</script>

<div class="cc">
	<div class="top">
		<ul class="legend" aria-label={t({ en: 'Legend', zh: '图例' })}>
			{#each METHODS as m (m)}
				<li>
					<svg width="18" height="14" aria-hidden="true"><path d={markerPath(SHAPE[m], 5)} transform="translate(9,7)" fill={color(m)} /></svg>
					<span>{t(LABEL[m])}</span>
					<span class="muted num">{fmtInt(finetune.methods[m].trainableParams)} {t({ en: 'trainable', zh: '可训练' })}</span>
				</li>
			{/each}
			{#if live}
				<li>
					<svg width="18" height="14" aria-hidden="true"><path d={markerPath('diamond', 5)} transform="translate(9,7)" class="livemk" /></svg>
					<span>{t({ en: 'Your live run (Part 1)', zh: '你的实时运行（第 1 部分）' })}</span>
				</li>
			{/if}
		</ul>
		<div class="opts">
			<label><input type="checkbox" bind:checked={sel.part2Zoom} /> {t({ en: 'Zoom to 90–100%', zh: '放大到 90–100%' })}</label>
			<label><input type="checkbox" bind:checked={sel.part2Seeds} /> {t({ en: 'Show each seed', zh: '显示每个种子' })}</label>
			<ProvenanceBadge mode="precomputed" compact />
		</div>
	</div>

	<div class="plot" bind:clientWidth={w}>
		<svg
			viewBox="0 0 {w} {H}"
			width={w}
			height={H}
			tabindex="0"
			role="slider"
			aria-orientation="horizontal"
			aria-valuemin={ns[0]}
			aria-valuemax={ns[ns.length - 1]}
			aria-valuenow={sel.part2N}
			aria-valuetext={t({ en: `${sel.part2N} drawings per class`, zh: `每类 ${sel.part2N} 张` })}
			aria-label={t({
				en: 'Test accuracy vs drawings per class: choose a training-set size (arrow keys)',
				zh: '测试正确率与每类图数：选择训练集大小（方向键）'
			})}
			onpointermove={(e) => pick(e, e.currentTarget)}
			onpointerdown={(e) => pick(e, e.currentTarget)}
			onkeydown={key}
		>
			<defs>
				<clipPath id="{gid}-clip"><rect x={M.l} y={M.t} width={Math.max(0, w - M.l - M.r)} height={H - M.t - M.b} /></clipPath>
			</defs>
			{#each yTicks as v (v)}
				<line class="grid" x1={M.l} x2={w - M.r} y1={y(v)} y2={y(v)} />
				<text class="tick" x={M.l - 6} y={y(v) + 4} text-anchor="end">{(v * 100).toFixed(0)}%</text>
			{/each}
			{#if !sel.part2Zoom}
				<line class="chance" x1={M.l} x2={w - M.r} y1={y(0.5)} y2={y(0.5)} />
				<text class="tick" x={w - M.r - 4} y={y(0.5) - 5} text-anchor="end">{t({ en: 'chance 50%', zh: '瞎猜 50%' })}</text>
			{/if}
			{#each ns as n, i (n)}
				{#if !narrow || [1, 4, 16, 64, 512, 2048].includes(n)}
					<text class="tick" x={xs[i]} y={H - M.b + 16} text-anchor="middle">{fmtInt(n)}</text>
				{/if}
			{/each}
			<text class="axt" x={(M.l + w - M.r) / 2} y={H - 6} text-anchor="middle">{t({ en: 'drawings per class (log scale)', zh: '每类图数（对数刻度）' })}</text>
			<line class="axis" x1={M.l} x2={w - M.r} y1={H - M.b} y2={H - M.b} />

			<g clip-path="url(#{gid}-clip)">
				{#each METHODS as m (m)}
					<path d={bandPath(xs, S[m].lo, S[m].hi)} fill={color(m)} opacity="0.12" />
				{/each}
				{#each METHODS as m (m)}
					{@const pts = xs.map((xx, i) => [xx, S[m].mean[i]] as [number, number])}
					<path class="ln" d={linePath(gapAt > 0 ? pts.slice(0, gapAt) : pts)} stroke={color(m)} />
					{#if gapAt > 0}
						<path class="ln dash" d={linePath(pts.slice(gapAt - 1, gapAt + 1))} stroke={color(m)} />
						<path class="ln" d={linePath(pts.slice(gapAt))} stroke={color(m)} />
					{/if}
					{#if sel.part2Seeds}
						{#each finetune.results[m] as r, i (i)}
							{#each r.acc as a, s (s)}
								<circle cx={xs[i] + (METHODS.indexOf(m) - 1) * 4} cy={y(a)} r="2.2" fill={color(m)} opacity="0.75" />
							{/each}
						{/each}
					{/if}
					{#each xs as xx, i (i)}
						<path class="mk" d={markerPath(SHAPE[m], i === idx ? 5.5 : 4)} transform="translate({xx},{S[m].mean[i]})" fill={color(m)} />
					{/each}
				{/each}
				{#if live}
					<path class="mk livemk" d={markerPath('diamond', 6)} transform="translate({x(Math.log2(live.nA))},{y(live.testAcc!)})" />
				{/if}
			</g>
			<line class="cursor" x1={xs[idx]} x2={xs[idx]} y1={M.t} y2={H - M.b} />
		</svg>
	</div>

	<!-- in the lesson the size is stepped by a player (the lesson announces its own steps): no live region there -->
	<div class="readout" aria-live={compact ? 'off' : 'polite'}>
		<p class="rt">
			<strong>{t({ en: `${fmtInt(sel.part2N)} drawing${sel.part2N === 1 ? '' : 's'} per class`, zh: `每类 ${fmtInt(sel.part2N)} 张` })}</strong>
			<span class="muted">{t({ en: '· test accuracy on 3,000 drawings, mean ± std over 5 seeds', zh: '· 在 3,000 张测试图上的正确率，5 个种子的均值 ± 标准差' })}</span>
		</p>
		<table>
			<tbody>
				{#each METHODS as m (m)}
					{@const r = finetune.results[m][idx]}
					<tr>
						<th scope="row"><span class="mth"><svg width="14" height="12" aria-hidden="true"><path d={markerPath(SHAPE[m], 4.5)} transform="translate(7,6)" fill={color(m)} /></svg><span>{t(LABEL[m])}</span></span></th>
						<td class="num big">{p1(r.mean)}%</td>
						<td class="num muted">± {p1(r.std)}</td>
						<td class="num seeds muted">{r.acc.map((a) => p1(a)).join(' · ')}</td>
					</tr>
				{/each}
				{#if live && live.nA === sel.part2N}
					<tr>
						<th scope="row"><span class="mth"><svg width="14" height="12" aria-hidden="true"><path d={markerPath('diamond', 4.5)} transform="translate(7,6)" class="livemk" /></svg><span>{t({ en: "Your live head (Part 1)", zh: "你的实时输出头（第 1 部分）" })}</span></span></th>
						<td class="num big">{p1(live.testAcc!)}%</td>
						<td class="num muted" colspan="2">{t({ en: `on ${live.testN} test drawings, gradient descent, seed-0 drawings`, zh: `${live.testN} 张测试图，梯度下降，种子 0 的那组图` })}</td>
					</tr>
				{/if}
			</tbody>
		</table>
		<p class="muted small">{t({ en: 'Small dots (when shown) are the 5 seeds; each seed uses a different random set of training drawings.', zh: '小圆点（打开时）是 5 个种子；每个种子用的是一组不同的随机训练图。' })}</p>
	</div>

	{#if !compact}
	<div class="params">
		<p class="ph">{t({ en: 'What each method is allowed to change (out of 4,666 parameters)', zh: '每种方法允许改动的参数（共 4,666 个）' })}</p>
		{#each METHODS as m (m)}
			{@const tp = finetune.methods[m].trainableParams}
			<div class="prow">
				<span class="pl">{t(LABEL[m])}</span>
				<div class="bar" role="img" aria-label={t({ en: `${fmtInt(tp)} of 4,666 trainable`, zh: `4,666 个中有 ${fmtInt(tp)} 个可训练` })}>
					{#if m === 'head'}
						<span class="frozen" style:width="{((4666 - tp) / 4666) * 100}%"><svg class="lock" viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><rect x="2" y="5.2" width="8" height="5.6" rx="1.2" fill="currentColor" /><path d="M3.8 5.2V3.8a2.2 2.2 0 0 1 4.4 0v1.4" fill="none" stroke="currentColor" stroke-width="1.4" /></svg></span>
					{/if}
					<span class="train" style:width="{(tp / 4666) * 100}%" style:background={color(m)}></span>
				</div>
				<span class="pv num">{fmtInt(tp)}</span>
			</div>
		{/each}
		<p class="muted small">{t({ en: 'Head only: 4,152 frozen + 514 trainable. Full fine-tune: all 4,666 start from the pre-trained values. From scratch: all 4,666 start random.', zh: '只训练输出头：4,152 个冻结 + 514 个可训练。全量微调：全部 4,666 个从预训练值出发。从零训练：全部 4,666 个从随机值出发。' })}</p>
	</div>

	<details class="tbl">
		<summary>{t({ en: 'All numbers as a table', zh: '以表格查看全部数字' })}</summary>
		<div class="scroll">
			<table>
				<thead>
					<tr>
						<th scope="col">{t({ en: 'per class', zh: '每类' })}</th>
						{#each METHODS as m (m)}<th scope="col">{t(LABEL[m])}</th>{/each}
					</tr>
				</thead>
				<tbody>
					{#each ns as n, i (n)}
						<tr>
							<th scope="row" class="num">{fmtInt(n)}</th>
							{#each METHODS as m (m)}<td class="num">{p2(finetune.results[m][i].mean)} ± {p2(finetune.results[m][i].std)}</td>{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</details>
	{/if}
</div>

<style>
	.cc {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		min-width: 0;
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 8px 16px;
		align-items: flex-start;
	}
	.legend {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		font-size: var(--fs-sm);
	}
	.legend li {
		display: inline-flex;
		gap: 6px;
		align-items: center;
	}
	.opts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 14px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.opts input {
		accent-color: var(--accent);
	}
	.plot {
		min-width: 0;
	}
	svg {
		display: block;
		max-width: 100%;
		height: auto;
		touch-action: pan-y;
	}
	.grid {
		stroke: var(--rule);
	}
	.axis {
		stroke: var(--rule-strong);
	}
	.chance {
		stroke: var(--rule-strong);
		stroke-dasharray: 4 4;
	}
	.tick {
		font-size: 11px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.axt {
		font-size: 12px;
		fill: var(--ink-2);
	}
	.ln {
		fill: none;
		stroke-width: 2;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.ln.dash {
		stroke-dasharray: 3 4;
	}
	.mk {
		stroke: var(--surface);
		stroke-width: 2;
		paint-order: stroke;
	}
	.livemk {
		fill: var(--surface);
		stroke: var(--ink);
		stroke-width: 2;
	}
	.cursor {
		stroke: var(--ink-2);
		stroke-dasharray: 3 3;
	}
	.readout table,
	.tbl table {
		border-collapse: collapse;
		font-size: var(--fs-sm);
	}
	.readout th,
	.readout td {
		padding: 3px 10px 3px 0;
		text-align: left;
		font-weight: 400;
		vertical-align: baseline;
	}
	.mth {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.mth svg {
		flex: 0 0 auto;
	}
	.big {
		font-weight: 650;
	}
	.seeds {
		font-size: var(--fs-xs);
	}
	.rt {
		margin: 0 0 4px;
		font-size: var(--fs-sm);
	}
	.small {
		font-size: var(--fs-xs);
		margin: 4px 0 0;
	}
	.params {
		display: grid;
		gap: 6px;
		padding: 12px 14px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.ph {
		margin: 0;
		font-weight: 600;
		font-size: var(--fs-sm);
	}
	.prow {
		display: grid;
		grid-template-columns: minmax(0, 11em) minmax(0, 1fr) 4em;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.bar {
		display: flex;
		height: 16px;
		border-radius: 3px;
		overflow: hidden;
		background: var(--surface-3);
		gap: 2px;
	}
	.frozen {
		display: grid;
		place-items: center;
		background: repeating-linear-gradient(135deg, var(--surface-3) 0 4px, var(--rule) 4px 7px);
		color: var(--muted);
	}
	.train {
		display: block;
	}
	.pv {
		text-align: right;
	}
	.tbl summary {
		cursor: pointer;
		font-size: var(--fs-sm);
		color: var(--accent);
	}
	.scroll {
		overflow-x: auto;
	}
	.tbl th,
	.tbl td {
		padding: 3px 10px;
		text-align: right;
		border-bottom: 1px solid var(--rule);
		white-space: nowrap;
	}
	@media (max-width: 520px) {
		.seeds {
			display: none;
		}
		.prow {
			grid-template-columns: minmax(0, 7.5em) minmax(0, 1fr) 3.5em;
		}
	}
</style>
