<!--
  Intermediate-task training in our BERT reproduction (precomputed, research/bert_improve.json):
  RTE dev accuracy after fine-tuning BERT-base on RTE directly vs on MNLI first, then RTE.
  5 seeds, paired (same seed on both sides). `compact` (the guided lesson) shows only the
  paired-seed chart; the full version (Part 3) adds the per-epoch means, the recipe and one dev pair.
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { linear, linePath, markerPath } from './chart';
	import { MNLI_RTE as M } from './bert';

	let { compact = false }: { compact?: boolean } = $props();

	const cDirect = $derived(categorical(6, themeState.resolved));
	const cMnli = $derived(categorical(8, themeState.resolved));
	const zh = $derived(i18n.lang === 'zh');
	const SRC = {
		en: 'From our BERT reproduction: research/bert_improve.json (its runs tagged repro and mnli_inter, 5 paired seeds). RTE dev set, 277 pairs.',
		zh: '来自我们的 BERT 复现：research/bert_improve.json（其中标记为 repro 和 mnli_inter 的运行，5 组配对种子）。RTE 验证集，277 对。'
	};

	// ------------------------------------------------ paired seeds (slope chart)
	let w = $state(520);
	const H = $derived(compact ? 230 : 250);
	const pad = { l: 44, r: 16, t: 26, b: 34 };
	const y = $derived(linear([56, 80], [H - pad.b, pad.t]));
	const narrow = $derived(w < 440);
	const colX = $derived(narrow ? [pad.l + (w - pad.l - pad.r) * 0.36, pad.l + (w - pad.l - pad.r) * 0.72] : [pad.l + (w - pad.l - pad.r) * 0.24, pad.l + (w - pad.l - pad.r) * 0.76]);
	// identical values (three seeds at 78.0) are spread sideways so every seed stays visible
	const off = (i: number) => (i - 2) * 7;
	let hi = $state<number | null>(null);
	function pickSeed(e: PointerEvent, svg: SVGSVGElement) {
		const r = svg.getBoundingClientRect();
		const py = ((e.clientY - r.top) / r.height) * H;
		const px = ((e.clientX - r.left) / r.width) * w;
		// distance to each seed's line segment, in px
		let best = -1;
		let bd = 14;
		for (let i = 0; i < 5; i++) {
			const x0 = colX[0] + off(i);
			const x1 = colX[1] + off(i);
			const y0 = y(M.direct.perSeed[i]);
			const y1 = y(M.mnliFirst.perSeed[i]);
			const tt = Math.max(0, Math.min(1, ((px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2)));
			const d = Math.hypot(px - (x0 + (x1 - x0) * tt), py - (y0 + (y1 - y0) * tt));
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		hi = best >= 0 ? best : null;
	}
	const gains = M.direct.perSeed.map((v, i) => +(M.mnliFirst.perSeed[i] - v).toFixed(1));

	// ------------------------------------------------ per-epoch means (full only)
	let w2 = $state(320);
	const H2 = 170;
	const p2 = { l: 40, r: 30, t: 14, b: 30 };
	const x2 = $derived(linear([1, 3], [p2.l + 26, w2 - p2.r]));
	const y2 = linear([55, 80], [H2 - p2.b, p2.t]);
	const ePts = (arr: number[]) => arr.map((v, i) => [x2(i + 1), y2(v)] as [number, number]);
</script>

<div class="mr" class:compact>
	<figure class="fig">
		<figcaption>
			<span>{t({ en: 'RTE dev accuracy (%), BERT-base, 5 seeds: fine-tuned on RTE directly vs on MNLI first, then RTE', zh: 'RTE 验证集正确率（%），BERT-base，5 个种子：直接在 RTE 上微调 vs 先 MNLI、再 RTE' })}</span>
			<ProvenanceBadge mode="precomputed" compact note={SRC} />
		</figcaption>
		<div class="slope" bind:clientWidth={w}>
			<svg
				viewBox="0 0 {w} {H}"
				width={w}
				height={H}
				role="img"
				aria-label={t({
					en: `RTE directly: ${M.direct.perSeed.map((v) => v.toFixed(1)).join(', ')} (mean ${M.direct.mean.toFixed(1)} ± ${M.direct.std.toFixed(1)}). MNLI first: ${M.mnliFirst.perSeed.map((v) => v.toFixed(1)).join(', ')} (mean ${M.mnliFirst.mean.toFixed(1)} ± ${M.mnliFirst.std.toFixed(1)}). Higher on ${M.higherOn} of 5 seeds, +${M.gain} points on average.`,
					zh: `直接训练 RTE：${M.direct.perSeed.map((v) => v.toFixed(1)).join('、')}（均值 ${M.direct.mean.toFixed(1)} ± ${M.direct.std.toFixed(1)}）。先 MNLI：${M.mnliFirst.perSeed.map((v) => v.toFixed(1)).join('、')}（均值 ${M.mnliFirst.mean.toFixed(1)} ± ${M.mnliFirst.std.toFixed(1)}）。5 个种子全部更高，平均高 ${M.gain} 个百分点。`
				})}
				onpointermove={(e) => pickSeed(e, e.currentTarget)}
				onpointerdown={(e) => pickSeed(e, e.currentTarget)}
				onpointerleave={() => (hi = null)}
			>
				{#each [60, 65, 70, 75, 80] as v (v)}
					<line class="grid" x1={pad.l} x2={w - pad.r} y1={y(v)} y2={y(v)} />
					<text class="tick" x={pad.l - 6} y={y(v) + 4} text-anchor="end">{v}</text>
				{/each}
				<text class="colh" x={narrow ? pad.l : colX[0]} y={14} text-anchor={narrow ? "start" : "middle"} fill={cDirect}>{t({ en: 'RTE directly', zh: '直接训练 RTE' })}</text>
				<text class="colh" x={narrow ? w - pad.r : colX[1]} y={14} text-anchor={narrow ? "end" : "middle"} fill={cMnli}>{t({ en: 'MNLI first, then RTE', zh: '先 MNLI，再 RTE' })}</text>
				{#each M.direct.perSeed as v, i (i)}
					<path class="seed" class:hi={hi === i} class:lo={hi !== null && hi !== i} d={linePath([[colX[0] + off(i), y(v)], [colX[1] + off(i), y(M.mnliFirst.perSeed[i])]])} />
				{/each}
				{#each M.direct.perSeed as v, i (i)}
					<path class="mk" d={markerPath('square', hi === i ? 5.5 : 4.2)} transform="translate({colX[0] + off(i)},{y(v)})" fill={cDirect} />
					<path class="mk" d={markerPath('diamond', hi === i ? 5.5 : 4.2)} transform="translate({colX[1] + off(i)},{y(M.mnliFirst.perSeed[i])})" fill={cMnli} />
				{/each}
				<!-- means ± sample std -->
				{#each [[M.direct, 0, cDirect, -1], [M.mnliFirst, 1, cMnli, 1]] as const as [s, c, col, side] (c)}
					{@const xm = colX[c] + side * (narrow ? 24 : 30)}
					<line class="whisk" x1={xm} x2={xm} y1={y(s.mean - s.std)} y2={y(s.mean + s.std)} stroke={col} />
					<line class="meanl" x1={xm - 7} x2={xm + 7} y1={y(s.mean)} y2={y(s.mean)} stroke={col} />
					<text class="val strong" x={xm + side * 10} y={y(s.mean) + 4} text-anchor={side < 0 ? 'end' : 'start'}>{s.mean.toFixed(1)}</text>
					<text class="val" x={xm + side * 10} y={y(s.mean) + 17} text-anchor={side < 0 ? 'end' : 'start'}>± {s.std.toFixed(1)}</text>
				{/each}
				<text class="gain" x={(colX[0] + colX[1]) / 2} y={y((M.direct.mean + M.mnliFirst.mean) / 2) - 8} text-anchor="middle">+{M.gain}</text>
				{#if hi !== null}
					<text class="tip" x={w / 2} y={H - 10} text-anchor="middle">
						{t({ en: `seed ${M.seeds[hi]}: ${M.direct.perSeed[hi].toFixed(1)} → ${M.mnliFirst.perSeed[hi].toFixed(1)} (+${gains[hi].toFixed(1)})`, zh: `种子 ${M.seeds[hi]}：${M.direct.perSeed[hi].toFixed(1)} → ${M.mnliFirst.perSeed[hi].toFixed(1)}（+${gains[hi].toFixed(1)}）` })}
					</text>
				{:else}
					<text class="tick" x={w / 2} y={H - 10} text-anchor="middle">{t({ en: 'one line per seed (same seed on both sides)', zh: '每条线是一个种子（两边用同一个种子）' })}</text>
				{/if}
			</svg>
		</div>
		{#if compact}
			<p class="cap">
				{t({
					en: `Mean ± sample std over 5 seeds: ${M.direct.mean.toFixed(1)} ± ${M.direct.std.toFixed(1)} → ${M.mnliFirst.mean.toFixed(1)} ± ${M.mnliFirst.std.toFixed(1)}. Higher on ${M.higherOn} of 5 seeds. MNLI has ${fmtInt(M.mnli.trainPairs)} training pairs, RTE ${fmtInt(M.rte.trainPairs)}.`,
					zh: `5 个种子的均值 ± 样本标准差：${M.direct.mean.toFixed(1)} ± ${M.direct.std.toFixed(1)} → ${M.mnliFirst.mean.toFixed(1)} ± ${M.mnliFirst.std.toFixed(1)}。5 个种子全部更高。MNLI 有 ${fmtInt(M.mnli.trainPairs)} 对训练样本，RTE 只有 ${fmtInt(M.rte.trainPairs)} 对。`
				})}
			</p>
		{:else}
			<div class="tscroll" use:keyScroll>
			<table class="seeds">
				<caption class="visually-hidden">{t({ en: 'RTE dev accuracy per seed', zh: '每个种子的 RTE 验证集正确率' })}</caption>
				<thead>
					<tr>
						<th scope="col">{t({ en: 'seed', zh: '种子' })}</th>
						{#each M.seeds as s, i (s)}<th scope="col" class="num" class:on={hi === i}>{s}</th>{/each}
						<th scope="col">{t({ en: 'mean ± std', zh: '均值 ± 标准差' })}</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<th scope="row"><span class="sw" style:background={cDirect}></span>{t({ en: 'RTE directly', zh: '直接训练 RTE' })}</th>
						{#each M.direct.perSeed as v, i (i)}<td class="num" class:on={hi === i}>{v.toFixed(1)}</td>{/each}
						<td class="num">{M.direct.mean.toFixed(1)} ± {M.direct.std.toFixed(1)}</td>
					</tr>
					<tr>
						<th scope="row"><span class="sw dia" style:background={cMnli}></span>{t({ en: 'MNLI first', zh: '先 MNLI' })}</th>
						{#each M.mnliFirst.perSeed as v, i (i)}<td class="num" class:on={hi === i}>{v.toFixed(1)}</td>{/each}
						<td class="num">{M.mnliFirst.mean.toFixed(1)} ± {M.mnliFirst.std.toFixed(1)}</td>
					</tr>
					<tr class="gainrow">
						<th scope="row">{t({ en: 'gain', zh: '提升' })}</th>
						{#each gains as g, i (i)}<td class="num" class:on={hi === i}>+{g.toFixed(1)}</td>{/each}
						<td class="num">+{M.gain}</td>
					</tr>
				</tbody>
			</table>
			</div>
		{/if}
	</figure>

	{#if !compact}
		<div class="two">
			<figure class="fig">
				<figcaption>
					<span>{t({ en: 'Mean RTE dev accuracy after each epoch (%)', zh: '每个 epoch 之后的 RTE 验证集平均正确率（%）' })}</span>
					<ProvenanceBadge mode="precomputed" compact note={SRC} />
				</figcaption>
				<div bind:clientWidth={w2}>
					<svg
						viewBox="0 0 {w2} {H2}"
						width={w2}
						height={H2}
						role="img"
						aria-label={t({
							en: `After epochs 1, 2, 3: RTE directly ${M.direct.perEpoch.map((v) => v.toFixed(1)).join(', ')}; MNLI first ${M.mnliFirst.perEpoch.map((v) => v.toFixed(1)).join(', ')}.`,
							zh: `第 1、2、3 个 epoch 之后：直接训练 RTE ${M.direct.perEpoch.map((v) => v.toFixed(1)).join('、')}；先 MNLI ${M.mnliFirst.perEpoch.map((v) => v.toFixed(1)).join('、')}。`
						})}
					>
						{#each [60, 70, 80] as v (v)}
							<line class="grid" x1={p2.l} x2={w2 - p2.r} y1={y2(v)} y2={y2(v)} />
							<text class="tick" x={p2.l - 6} y={y2(v) + 4} text-anchor="end">{v}</text>
						{/each}
						{#each [1, 2, 3] as e (e)}
							<text class="tick" x={x2(e)} y={H2 - 10} text-anchor="middle">{t({ en: `epoch ${e}`, zh: `epoch ${e}` })}</text>
						{/each}
						<path class="ln" d={linePath(ePts(M.direct.perEpoch))} stroke={cDirect} />
						<path class="ln" d={linePath(ePts(M.mnliFirst.perEpoch))} stroke={cMnli} />
						{#each M.direct.perEpoch as v, i (i)}
							<path class="mk" d={markerPath('square', 4)} transform="translate({x2(i + 1)},{y2(v)})" fill={cDirect} />
							<text class="val" x={x2(i + 1)} y={y2(v) + 16} text-anchor="middle">{v.toFixed(1)}</text>
						{/each}
						{#each M.mnliFirst.perEpoch as v, i (i)}
							<path class="mk" d={markerPath('diamond', 4)} transform="translate({x2(i + 1)},{y2(v)})" fill={cMnli} />
							<text class="val" x={x2(i + 1)} y={y2(v) - 9} text-anchor="middle">{v.toFixed(1)}</text>
						{/each}
					</svg>
				</div>
				<p class="cap muted">{t({ en: 'After one epoch on RTE, the MNLI-first model is already at 75.5; the direct one is at 57.8 and stops at 63.0.', zh: '在 RTE 上只训练 1 个 epoch，先 MNLI 的模型就已到 75.5；直接训练的模型只有 57.8，最后停在 63.0。' })}</p>
			</figure>

			<figure class="fig">
				<figcaption><span>{t({ en: 'The recipe', zh: '做法' })}</span></figcaption>
				<ol class="recipe">
					<li>
						<strong>{t({ en: 'Fine-tune pre-trained BERT-base on MNLI', zh: '在 MNLI 上微调预训练的 BERT-base' })}</strong>
						<span class="muted">{t({ en: `3 classes (entails, contradicts, neither), ${fmtInt(M.mnli.trainPairs)} pairs, 3 epochs = ${fmtInt(M.mnli.steps)} steps, ${M.mnli.minutes} minutes on one laptop GPU. Dev accuracy ${M.mnli.devMatched} matched / ${M.mnli.devMismatched} mismatched.`, zh: `3 个类别（蕴含、矛盾、都不是），${fmtInt(M.mnli.trainPairs)} 对样本，3 个 epoch 共 ${fmtInt(M.mnli.steps)} 步，在一块笔记本 GPU 上用了 ${M.mnli.minutes} 分钟。验证集正确率：匹配 ${M.mnli.devMatched}，不匹配 ${M.mnli.devMismatched}。` })}</span>
					</li>
					<li>
						<strong>{t({ en: 'Swap the head', zh: '换掉输出头' })}</strong>
						<span class="muted">{t({ en: 'Throw away the 3-way MNLI head and attach a new 2-way head (entailed / not entailed), just as TinyCNN’s 10-way head was set aside for a 2-way one.', zh: '扔掉 MNLI 的三分类输出头，接上一个新的二分类输出头（蕴含 / 不蕴含），就像 TinyCNN 把 10 类输出头换成二分类输出头一样。' })}</span>
					</li>
					<li>
						<strong>{t({ en: 'Fine-tune on RTE exactly as before', zh: '完全照旧在 RTE 上微调' })}</strong>
						<span class="muted">{t({ en: `${fmtInt(M.rte.trainPairs)} pairs, learning rate 2e-5, 3 epochs, batch 32 = ${M.rte.updates} updates. Only the starting weights differ.`, zh: `${fmtInt(M.rte.trainPairs)} 对样本，学习率 2e-5，3 个 epoch，批大小 32，共 ${M.rte.updates} 次更新。唯一的区别是起点权重。` })}</span>
					</li>
				</ol>
				<div class="sizes" role="img" aria-label={t({ en: `Training pairs: MNLI ${fmtInt(M.mnli.trainPairs)}, RTE ${fmtInt(M.rte.trainPairs)} (${M.mnli.ratio} times fewer)`, zh: `训练样本：MNLI ${fmtInt(M.mnli.trainPairs)} 对，RTE ${fmtInt(M.rte.trainPairs)} 对（少 ${M.mnli.ratio} 倍）` })}>
					<div class="srow"><span class="sl">MNLI</span><span class="sbar" style:width="100%" style:background={cMnli}></span><span class="num sv">{fmtInt(M.mnli.trainPairs)}</span></div>
					<div class="srow"><span class="sl">RTE</span><span class="sbar" style:width="{Math.max(0.6, (M.rte.trainPairs / M.mnli.trainPairs) * 100)}%" style:background={cDirect}></span><span class="num sv">{fmtInt(M.rte.trainPairs)}</span></div>
					<p class="cap muted">{t({ en: `Training pairs per task: MNLI has ${M.mnli.ratio}× more.`, zh: `各任务的训练样本对数：MNLI 是 RTE 的 ${M.mnli.ratio} 倍。` })}</p>
				</div>
			</figure>
		</div>

		<figure class="fig ex">
			<figcaption>
				<span>{t({ en: `One RTE dev pair (#${M.example.devIndex}), read by both seed-1 models`, zh: `一对 RTE 验证集样本（第 ${M.example.devIndex} 对），由两个种子 1 的模型来判断` })}</span>
				<ProvenanceBadge mode="precomputed" compact note={SRC} />
			</figcaption>
			<div class="pair" lang="en">
				<p><span class="tag">A</span> {M.example.a}</p>
				<p><span class="tag">B</span> {M.example.b}</p>
			</div>
			{#if zh}
				<!-- the model reads the English pair; Chinese readers also get a translation -->
				<p class="gloss muted" lang="zh-CN">（译文）A：{M.example.aZh} B：{M.example.bZh}</p>
			{/if}
			<p class="small">{t({ en: 'Correct answer: A entails B.', zh: '正确答案：A 蕴含 B。' })}</p>
			<div class="pbars">
				{#each [[M.example.directP, cDirect, { en: 'RTE directly', zh: '直接训练 RTE' }, false], [M.example.mnliP, cMnli, { en: 'MNLI first', zh: '先 MNLI' }, true]] as const as [p, col, lab, right] (p)}
					<div class="pb">
						<span class="pl">{t(lab)}</span>
						<span class="track"><span class="fill" style:width="{p * 100}%" style:background={col}></span><span class="half"></span></span>
						<span class="num pv">p = {p.toFixed(2)}</span>
						<span class="verdict">{right ? t({ en: '✓ says “entailed”: right', zh: '✓ 判为“蕴含”：对' }) : t({ en: '✗ says “not entailed”: wrong', zh: '✗ 判为“不蕴含”：错' })}</span>
					</div>
				{/each}
			</div>
			<p class="cap muted">{t({ en: `p = the model’s probability for “entailed”; the mark in the middle of each bar is 0.5. Over all ${M.rte.devPairs} dev pairs, the MNLI-first model gets ${M.example.fixes} pairs right that the direct model got wrong, and ${M.example.breaks} the other way round.`, zh: `p = 模型给“蕴含”的概率；每个条中间的竖线是 0.5。在全部 ${M.rte.devPairs} 对验证样本中，先 MNLI 的模型答对了 ${M.example.fixes} 对直接训练的模型答错的样本，反过来的情况有 ${M.example.breaks} 对。` })}</p>
		</figure>
	{/if}
</div>

<style>
	.mr {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
		min-width: 0;
	}
	.fig {
		margin: 0;
		padding: 12px 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 6px;
		min-width: 0;
	}
	.compact .fig {
		padding: 10px 12px;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 6px 12px;
		align-items: center;
		font-weight: 600;
		font-size: var(--fs-sm);
	}
	.slope {
		max-width: 640px;
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
	.tick {
		font-size: 11px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.colh {
		font-size: 12px;
		font-weight: 700;
	}
	.seed {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 1.5;
		opacity: 0.55;
	}
	.seed.hi {
		stroke: var(--ink);
		stroke-width: 2.5;
		opacity: 1;
	}
	.seed.lo {
		opacity: 0.25;
	}
	.mk {
		stroke: var(--surface);
		stroke-width: 1.5;
		paint-order: stroke;
	}
	.whisk {
		stroke-width: 2;
	}
	.meanl {
		stroke-width: 3;
	}
	.val {
		font-size: 11px;
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.val.strong {
		font-size: 13px;
		font-weight: 700;
		fill: var(--ink);
	}
	.gain {
		font-size: 15px;
		font-weight: 700;
		fill: var(--ink);
		font-family: var(--font-mono);
		paint-order: stroke;
		stroke: var(--surface);
		stroke-width: 4px;
	}
	.tip {
		font-size: 12px;
		fill: var(--ink);
		font-family: var(--font-mono);
	}
	.ln {
		fill: none;
		stroke-width: 2;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.tscroll {
		overflow-x: auto;
	}
	.seeds {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		width: max-content;
		max-width: 100%;
	}
	.seeds th,
	.seeds td {
		padding: 2px 8px 2px 0;
		text-align: right;
		font-weight: 400;
		white-space: nowrap;
	}
	.seeds th[scope='row'] {
		text-align: left;
	}
	.seeds thead th {
		color: var(--muted);
		font-size: var(--fs-xs);
	}
	.seeds .on {
		font-weight: 700;
		color: var(--ink);
	}
	.gainrow td,
	.gainrow th {
		border-top: 1px solid var(--rule);
	}
	.sw {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 2px;
		margin-right: 6px;
	}
	.sw.dia {
		transform: rotate(45deg) scale(0.85);
	}
	.two {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
		gap: var(--s4);
	}
	.recipe {
		margin: 0;
		padding-left: 1.3em;
		display: grid;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.recipe li span {
		display: block;
	}
	.sizes {
		display: grid;
		gap: 4px;
		margin-top: 4px;
	}
	.srow {
		display: grid;
		grid-template-columns: 3.5em minmax(0, 1fr) 5.5em;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.sbar {
		display: block;
		height: 10px;
		border-radius: 3px;
		justify-self: start;
	}
	.sv {
		text-align: right;
	}
	.pair {
		display: grid;
		gap: 4px;
		padding: 8px 10px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.pair p {
		margin: 0;
	}
	.tag {
		display: inline-grid;
		place-items: center;
		width: 1.4em;
		height: 1.4em;
		margin-right: 4px;
		border-radius: 4px;
		background: var(--surface-3);
		font-size: var(--fs-xs);
		font-weight: 700;
	}
	.gloss {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.small {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.pbars {
		display: grid;
		gap: 6px;
	}
	.pb {
		display: grid;
		grid-template-columns: minmax(0, 8em) minmax(60px, 1fr) auto;
		gap: 4px 10px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.track {
		position: relative;
		height: 12px;
		border-radius: 3px;
		background: var(--surface-3);
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
	}
	.half {
		position: absolute;
		left: 50%;
		top: 0;
		bottom: 0;
		border-left: 2px solid var(--ink);
	}
	.pv {
		white-space: nowrap;
	}
	.verdict {
		grid-column: 2 / -1;
		font-size: var(--fs-xs);
		color: var(--ink);
		font-weight: 600;
	}
</style>
