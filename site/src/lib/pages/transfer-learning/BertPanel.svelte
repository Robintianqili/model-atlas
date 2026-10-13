<!--
  Part 3 (precomputed): real-scale evidence from our BERT reproduction (research/bert_facts.json and
  research/bert_improve.json). Frozen features vs full fine-tuning, which layer's features, the
  learning-rate search, BERT-Large instability on RTE, and intermediate-task training (MNLI first,
  then RTE: 63.0 ± 2.7 → 77.3 ± 0.9 over 5 seeds).
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import Article from './ArticlePart.svelte';
	import MnliRte from './MnliRte.svelte';
	import { linear, linePath, markerPath, ticks } from './chart';
	import { BERT_SLIDES, FEATURES_VS_FT, GLUE_TRAIN, LARGE_RTE, LR_SEARCH } from './bert';

	const cFrozen = $derived(categorical(2, themeState.resolved));
	const cFull = $derived(categorical(6, themeState.resolved));

	let w1 = $state(560);
	let w2 = $state(560);
	let w3 = $state(280);
	let w4 = $state(560);

	// (a) dumbbell
	const dbM = { l: 8, r: 52, t: 8, b: 26 };
	const dbRow = 46;
	const dbH = $derived(FEATURES_VS_FT.rows.length * dbRow + dbM.t + dbM.b);
	const dbX = $derived(linear([70, 100], [dbM.l, w1 - dbM.r]));

	// (b) per layer
	const lyM = { l: 40, r: 92, t: 12, b: 34 };
	const lyH = 250;
	const lyX = $derived(linear([0, 12], [lyM.l, w2 - lyM.r]));
	const lyY = linear([45, 95], [lyH - lyM.b, lyM.t]);
	let layer = $state(12);
	function pickLayer(e: PointerEvent, svg: SVGSVGElement) {
		const r = svg.getBoundingClientRect();
		const px = ((e.clientX - r.left) / r.width) * w2;
		layer = Math.max(0, Math.min(12, Math.round(lyX.invert(px))));
	}
	function layerKey(e: KeyboardEvent) {
		if (e.key === 'ArrowRight' || e.key === 'ArrowUp') layer = Math.min(12, layer + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') layer = Math.max(0, layer - 1);
		else return;
		e.preventDefault();
	}
	const meanPts = $derived(FEATURES_VS_FT.perLayerMean.map((v, i) => [lyX(i), lyY(v)] as [number, number]));
	const clsPts = $derived(FEATURES_VS_FT.perLayerCls.map((v, i) => [lyX(i), lyY(v)] as [number, number]));

	// (c) learning rates
	const lrM = { l: 36, r: 10, t: 22, b: 30 };
	const lrH = 190;
	function lrScales(task: 'mrpc' | 'rte', width: number) {
		const d = LR_SEARCH[task];
		const lo = Math.floor(Math.min(...d.points.map((p) => p.mean - p.std), d.paper) - 0.5);
		const hi = Math.ceil(Math.max(...d.points.map((p) => p.mean + p.std), d.paper) + 0.5);
		return {
			x: (i: number) => lrM.l + ((i + 0.5) / d.points.length) * (width - lrM.l - lrM.r),
			y: linear([lo, hi], [lrH - lrM.b, lrM.t]),
			ticks: ticks(lo, hi, 4)
		};
	}

	// (d) BERT-Large stability
	const stM = { l: 8, r: 12, t: 10, b: 28 };
	const stRow = 44;
	const stH = 2 * stRow + stM.t + stM.b;
	const stX = $derived(linear([40, 80], [stM.l + 150, w4 - stM.r]));
	const stXn = $derived(linear([40, 80], [stM.l + 8, w4 - stM.r]));
	const narrow4 = $derived(w4 < 520);
	const sx = $derived(narrow4 ? stXn : stX);
	const collapsed = (v: number) => LARGE_RTE.collapseValues.includes(v);
	function jitter(arr: number[]) {
		// stack identical values vertically so all 6 seeds stay visible
		const seen = new Map<number, number>();
		return arr.map((v) => {
			const k = seen.get(v) ?? 0;
			seen.set(v, k + 1);
			return k;
		});
	}
</script>

<div class="bert">
	<Article part="compare-intro" />

	<!-- (a) frozen vs fine-tuned -->
	<Article part="compare-features" />
	<figure class="fig">
		<figcaption>
			<span>{t({ en: 'Frozen features + linear probe vs full fine-tuning (dev set)', zh: '冻结特征 + 线性探针 vs 全量微调（验证集）' })}</span>
			<ProvenanceBadge mode="precomputed" compact note={{ en: 'From our BERT reproduction (research/bert_facts.json); the NER row is the BERT paper’s Table 7.', zh: '来自我们的 BERT 复现（research/bert_facts.json）；NER 一行来自 BERT 论文表 7。' }} />
		</figcaption>
		<ul class="legend">
			<li><svg width="14" height="12" aria-hidden="true"><path d={markerPath('circle', 4.5)} transform="translate(7,6)" fill={cFrozen} /></svg>{t({ en: 'frozen features (best probe)', zh: '冻结特征（最佳探针）' })}</li>
			<li><svg width="14" height="12" aria-hidden="true"><path d={markerPath('square', 4.5)} transform="translate(7,6)" fill={cFull} /></svg>{t({ en: 'full fine-tuning', zh: '全量微调' })}</li>
		</ul>
		<div bind:clientWidth={w1}>
			<svg viewBox="0 0 {w1} {dbH}" width={w1} height={dbH} role="img" aria-label={FEATURES_VS_FT.rows.map((r) => `${t(r.task)}: ${r.frozen} vs ${r.ft}`).join('; ')}>
				{#each [70, 80, 90, 100] as v (v)}
					<line class="grid" x1={dbX(v)} x2={dbX(v)} y1={dbM.t} y2={dbH - dbM.b} />
					<text class="tick" x={dbX(v)} y={dbH - 8} text-anchor="middle">{v}</text>
				{/each}
				{#each FEATURES_VS_FT.rows as r, i (r.id)}
					{@const yy = dbM.t + i * dbRow + 28}
					<text class="rl" x={dbM.l} y={yy - 12}>{t(r.task)} <tspan class="muted-t">· {r.metric === 'F1' ? 'F1' : t({ en: 'accuracy %', zh: '正确率 %' })}</tspan></text>
					<line class="conn" x1={dbX(r.frozen)} x2={dbX(r.ft)} y1={yy} y2={yy} />
					<path class="mk" d={markerPath('circle', 5.5)} transform="translate({dbX(r.frozen)},{yy})" fill={cFrozen} />
					<path class="mk" d={markerPath('square', 5.5)} transform="translate({dbX(r.ft)},{yy})" fill={cFull} />
					<text class="val" x={dbX(r.frozen) - 9} y={yy + 4} text-anchor="end">{r.frozen.toFixed(1)}</text>
					<text class="gap" x={w1 - dbM.r + 6} y={yy + 4}>−{r.gap.toFixed(1)}</text>
				{/each}
			</svg>
		</div>
		<p class="cap muted">{t({ en: 'Right column: how much lower the frozen features score. Fine-tuned values: NER 96.4, SST-2 93.0 ± 0.2, MRPC 84.1.', zh: '右侧一列：冻结特征低了多少。微调的数值：NER 96.4，SST-2 93.0 ± 0.2，MRPC 84.1。' })}</p>
	</figure>

	<!-- (b) which layer -->
	<Article part="compare-layers" />
	<figure class="fig">
		<figcaption>
			<span>{t({ en: 'SST-2: a linear probe on each layer’s frozen features', zh: 'SST-2：在每一层的冻结特征上训练线性探针' })}</span>
			<ProvenanceBadge mode="precomputed" compact />
		</figcaption>
		<ul class="legend">
			<li><svg width="18" height="12" aria-hidden="true"><path d="M1 6h16" stroke={cFrozen} stroke-width="2" /><path d={markerPath('circle', 3.5)} transform="translate(9,6)" fill={cFrozen} /></svg>{t({ en: 'mean of all tokens', zh: '所有词元的平均' })}</li>
			<li><svg width="18" height="12" aria-hidden="true"><path d="M1 6h16" stroke={cFrozen} stroke-width="2" stroke-dasharray="4 3" /><path d={markerPath('triangle', 3.5)} transform="translate(9,6)" fill={cFrozen} /></svg>{t({ en: '[CLS] token only', zh: '只用 [CLS] 词元' })}</li>
			<li><svg width="18" height="12" aria-hidden="true"><path d="M1 6h16" stroke={cFull} stroke-width="2" /></svg>{t({ en: 'full fine-tuning (93.0)', zh: '全量微调（93.0）' })}</li>
		</ul>
		<div bind:clientWidth={w2}>
			<svg
				viewBox="0 0 {w2} {lyH}"
				width={w2}
				height={lyH}
				tabindex="0"
				role="slider"
				aria-orientation="horizontal"
				aria-valuemin={0}
				aria-valuemax={12}
				aria-valuenow={layer}
				aria-valuetext={t({ en: `layer ${layer}: mean pooling ${FEATURES_VS_FT.perLayerMean[layer]}%, [CLS] ${FEATURES_VS_FT.perLayerCls[layer]}%`, zh: `第 ${layer} 层：平均池化 ${FEATURES_VS_FT.perLayerMean[layer]}%，[CLS] ${FEATURES_VS_FT.perLayerCls[layer]}%` })}
				aria-label={t({ en: 'Probe accuracy by layer: choose a layer (arrow keys)', zh: '各层探针正确率：选择一层（方向键）' })}
				onpointermove={(e) => pickLayer(e, e.currentTarget)}
				onpointerdown={(e) => pickLayer(e, e.currentTarget)}
				onkeydown={layerKey}
			>
				{#each [50, 60, 70, 80, 90] as v (v)}
					<line class="grid" x1={lyM.l} x2={w2 - lyM.r} y1={lyY(v)} y2={lyY(v)} />
					<text class="tick" x={lyM.l - 6} y={lyY(v) + 4} text-anchor="end">{v}%</text>
				{/each}
				{#each [0, 2, 4, 6, 8, 10, 12] as v (v)}
					<text class="tick" x={lyX(v)} y={lyH - lyM.b + 16} text-anchor="middle">{v}</text>
				{/each}
				<text class="axt" x={(lyM.l + w2 - lyM.r) / 2} y={lyH - 4} text-anchor="middle">{t({ en: 'layer (0 = embedding output)', zh: '层（0 = 嵌入层输出）' })}</text>
				<rect x={lyM.l} y={lyY(FEATURES_VS_FT.ftSst2 + FEATURES_VS_FT.ftSst2Std)} width={Math.max(0, w2 - lyM.l - lyM.r)} height={lyY(FEATURES_VS_FT.ftSst2 - FEATURES_VS_FT.ftSst2Std) - lyY(FEATURES_VS_FT.ftSst2 + FEATURES_VS_FT.ftSst2Std)} fill={cFull} opacity="0.15" />
				<line x1={lyM.l} x2={w2 - lyM.r} y1={lyY(FEATURES_VS_FT.ftSst2)} y2={lyY(FEATURES_VS_FT.ftSst2)} stroke={cFull} stroke-width="2" />
				<text class="lab" x={w2 - lyM.r + 6} y={lyY(FEATURES_VS_FT.ftSst2) + 4}>{t({ en: 'fine-tuned', zh: '微调' })} 93.0</text>
				<line class="chance" x1={lyM.l} x2={w2 - lyM.r} y1={lyY(FEATURES_VS_FT.majority)} y2={lyY(FEATURES_VS_FT.majority)} />
				<text class="lab" x={w2 - lyM.r + 6} y={lyY(FEATURES_VS_FT.majority) + 4}>{t({ en: 'majority', zh: '多数类' })} 50.9</text>
				<path class="ln" d={linePath(meanPts)} stroke={cFrozen} />
				<path class="ln" d={linePath(clsPts)} stroke={cFrozen} stroke-dasharray="5 4" />
				{#each meanPts as p, i (i)}<path class="mk" d={markerPath('circle', i === layer ? 5 : 3.5)} transform="translate({p[0]},{p[1]})" fill={cFrozen} />{/each}
				{#each clsPts as p, i (i)}<path class="mk" d={markerPath('triangle', i === layer ? 5 : 3.5)} transform="translate({p[0]},{p[1]})" fill={cFrozen} />{/each}
				<path class="mk best" d={markerPath('diamond', 5.5)} transform="translate({w2 - lyM.r + 14},{lyY(FEATURES_VS_FT.bestFrozen)})" fill={cFrozen} />
				<text class="lab" x={w2 - lyM.r + 24} y={lyY(FEATURES_VS_FT.bestFrozen) + 4}>86.8</text>
				<line class="cursor" x1={lyX(layer)} x2={lyX(layer)} y1={lyM.t} y2={lyH - lyM.b} />
			</svg>
		</div>
		<p class="cap" aria-live="polite">
			<strong>{t({ en: `Layer ${layer}`, zh: `第 ${layer} 层` })}:</strong>
			{t({ en: 'mean of tokens', zh: '词元平均' })} <span class="num">{FEATURES_VS_FT.perLayerMean[layer].toFixed(1)}%</span>,
			[CLS] <span class="num">{FEATURES_VS_FT.perLayerCls[layer].toFixed(1)}%</span>.
			<span class="muted">{t({ en: '◆ 86.8 = last four layers concatenated (the best of 27 choices).', zh: '◆ 86.8 = 最后四层拼接（27 种选择中最好的）。' })}</span>
		</p>
	</figure>

	<!-- (c) learning-rate search -->
	<Article part="compare-lr" />
	<figure class="fig">
		<figcaption>
			<span>{t({ en: 'BERT-base fine-tuned with each of the paper’s 4 learning rates (dev accuracy, mean ± std)', zh: '用论文的 4 个学习率分别微调 BERT-base（验证集正确率，均值 ± 标准差）' })}</span>
			<ProvenanceBadge mode="precomputed" compact />
		</figcaption>
		<div class="lrs">
			{#each ['mrpc', 'rte'] as const as task (task)}
				{@const d = LR_SEARCH[task]}
				{@const sc = lrScales(task, w3)}
				<div class="lr" bind:clientWidth={w3}>
					<p class="lrt">{task === 'mrpc' ? `MRPC · ${fmtInt(GLUE_TRAIN.MRPC)}` : `RTE · ${fmtInt(GLUE_TRAIN.RTE)}`} {t({ en: 'training pairs', zh: '对训练样本' })}</p>
					<svg viewBox="0 0 {w3} {lrH}" width={w3} height={lrH} role="img" aria-label={`${task.toUpperCase()}: ${d.points.map((p) => `${p.lr} ${p.mean} ± ${p.std}`).join(', ')}; ${t({ en: 'paper', zh: '论文' })} ${d.paper}`}>
						{#each sc.ticks as v (v)}
							<line class="grid" x1={lrM.l} x2={w3 - lrM.r} y1={sc.y(v)} y2={sc.y(v)} />
							<text class="tick" x={lrM.l - 5} y={sc.y(v) + 4} text-anchor="end">{v}</text>
						{/each}
						<line class="paper" x1={lrM.l} x2={w3 - lrM.r} y1={sc.y(d.paper)} y2={sc.y(d.paper)} />
						<text class="lab" x={w3 - lrM.r} y={sc.y(d.paper) - 5} text-anchor="end">{t(d.paperIsTest ? { en: `paper ${d.paper} (test set)`, zh: `论文 ${d.paper}（测试集）` } : { en: `paper ${d.paper} (dev)`, zh: `论文 ${d.paper}（验证集）` })}</text>
						{#each d.points as p, i (p.lr)}
							{@const best = p.lr === d.best}
							<line class="whisk" x1={sc.x(i)} x2={sc.x(i)} y1={sc.y(p.mean - p.std)} y2={sc.y(p.mean + p.std)} stroke={cFull} />
							<path d={markerPath('square', best ? 6 : 4.5)} transform="translate({sc.x(i)},{sc.y(p.mean)})" style:fill={best ? cFull : 'var(--surface)'} style:stroke={cFull} style:stroke-width="2" />
							<text class="val" x={sc.x(i) + 9} y={sc.y(p.mean) + 4}>{p.mean.toFixed(1)}</text>
							<text class="tick" x={sc.x(i)} y={lrH - 10} text-anchor="middle">{p.lr}</text>
						{/each}
					</svg>
					<p class="cap muted">{t({ en: `Best: ${d.best} (filled), +${d.gain.toFixed(1)} over 2e-5.`, zh: `最佳：${d.best}（实心），比 2e-5 高 ${d.gain.toFixed(1)}。` })}</p>
				</div>
			{/each}
		</div>
	</figure>

	<!-- (d) small data fragility -->
	<Article part="compare-small" />
	<figure class="fig">
		<figcaption>
			<span>{t({ en: 'BERT-Large on RTE, 6 seeds each (dev accuracy %)', zh: 'BERT-Large 在 RTE 上，各 6 个种子（验证集正确率 %）' })}</span>
			<ProvenanceBadge mode="precomputed" compact />
		</figcaption>
		<div bind:clientWidth={w4}>
			<svg viewBox="0 0 {w4} {stH + (narrow4 ? 48 : 0)}" width={w4} height={stH + (narrow4 ? 48 : 0)} role="img" aria-label={t({ en: `Original optimizer: ${LARGE_RTE.original.perSeed.join(', ')} (4 of 6 collapsed). With bias correction: ${LARGE_RTE.adamw.perSeed.join(', ')} (0 collapsed).`, zh: `原版优化器：${LARGE_RTE.original.perSeed.join('、')}（6 次中 4 次崩溃）。加偏差校正：${LARGE_RTE.adamw.perSeed.join('、')}（0 次崩溃）。` })}>
				{#each [40, 50, 60, 70, 80] as v (v)}
					<line class="grid" x1={sx(v)} x2={sx(v)} y1={stM.t} y2={stH + (narrow4 ? 48 : 0) - stM.b} />
					<text class="tick" x={sx(v)} y={stH + (narrow4 ? 48 : 0) - 8} text-anchor="middle">{v}</text>
				{/each}
				{#each [['original', LARGE_RTE.original, { en: 'Original BERT optimizer', zh: '原版 BERT 优化器' }], ['adamw', LARGE_RTE.adamw, { en: 'AdamW, bias-corrected', zh: 'AdamW（含偏差校正）' }]] as const as [id, row, lab], r (id)}
					{@const yy = stM.t + r * (stRow + (narrow4 ? 24 : 0)) + (narrow4 ? 40 : 22)}
					{@const stack = jitter([...row.perSeed])}
					<text class="rl" x={narrow4 ? stM.l + 8 : stM.l} y={narrow4 ? yy - 25 : yy + 4}>{t(lab)}</text>
					<line class="meanl" x1={sx(row.mean)} x2={sx(row.mean)} y1={yy - 13} y2={yy + 13} />
					{#each row.perSeed as v, i (i)}
						<path
							d={markerPath('circle', 5)}
							transform="translate({sx(v)},{yy + [0, -10, 10, -20][stack[i]]})"
							style:fill={id === 'original' && collapsed(v) ? 'var(--surface)' : cFull}
							style:stroke={id === 'original' && collapsed(v) ? 'var(--ink)' : 'var(--surface)'}
							style:stroke-width="2"
						/>
					{/each}
					<text class="val" x={sx(row.mean) + 6} y={yy + 18}>{t({ en: 'mean', zh: '均值' })} {row.mean.toFixed(1)}</text>
				{/each}
			</svg>
		</div>
		<p class="cap muted">{t({ en: 'Hollow dots: collapsed runs (one class for every pair, so the score equals that class’s share: 47.3 or 52.7). Vertical tick: the mean.', zh: '空心点：崩溃的运行（对所有样本都给同一个类别，所以得分等于该类别的占比：47.3 或 52.7）。竖线：均值。' })}</p>
	</figure>

	<!-- (e) MNLI → RTE: intermediate-task training (research/bert_improve.json) -->
	<Article part="compare-mnli" />
	<MnliRte />
	<Article part="compare-mnli-after" />

	<p class="slides">
		<a href={BERT_SLIDES} rel="noopener">{t({ en: 'Slides of our BERT reproduction (all runs, methods and caveats) ↗', zh: '我们的 BERT 复现幻灯片（全部实验、方法与注意事项）↗' })}</a>
	</p>
</div>

<style>
	.bert {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.fig {
		margin: 0;
		padding: 12px 14px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		display: grid;
		gap: 6px;
		min-width: 0;
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
	.legend {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
	}
	.legend li {
		display: inline-flex;
		align-items: center;
		gap: 5px;
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
	.chance,
	.paper {
		stroke: var(--rule-strong);
		stroke-dasharray: 4 4;
	}
	.paper {
		stroke: var(--ink-2);
	}
	.tick {
		font-size: var(--t-xs);
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.axt {
		font-size: 12px;
		fill: var(--ink-2);
	}
	.rl {
		font-size: 12px;
		fill: var(--ink);
		font-weight: 600;
	}
	.muted-t {
		fill: var(--muted);
		font-weight: 400;
	}
	.val,
	.lab,
	.gap {
		font-size: var(--t-xs);
		fill: var(--ink-2);
		font-family: var(--font-mono);
	}
	.gap {
		fill: var(--ink);
		font-weight: 700;
	}
	.conn {
		stroke: var(--rule-strong);
		stroke-width: 3;
	}
	.ln {
		fill: none;
		stroke-width: 2;
	}
	.mk {
		stroke: var(--surface);
		stroke-width: 2;
		paint-order: stroke;
	}
	.whisk {
		stroke-width: 2;
	}
	.meanl {
		stroke: var(--ink);
		stroke-width: 2;
	}
	.cursor {
		stroke: var(--ink-2);
		stroke-dasharray: 3 3;
	}
	.cap {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.lrs {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
		gap: 12px;
	}
	.lr {
		min-width: 0;
	}
	.lrt {
		margin: 0;
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.slides {
		margin: 0;
	}
</style>
