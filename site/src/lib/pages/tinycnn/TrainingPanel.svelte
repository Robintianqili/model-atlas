<!--
  Training facts (precomputed, from spec.json): dataset and recipe, test accuracy, per-class
  accuracy, the confusion matrix, the training curve, the parameter budget per layer, and how
  the drawing-input paths were checked.
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { categorical, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt, fmtPct, fmtBytes } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { layerParams, type TinyCNNSpec, type TrainingFacts } from './data';

	let { spec }: { spec: TinyCNNSpec } = $props();
	const tr = $derived(spec.training as unknown as TrainingFacts);
	const m = $derived(tr.metrics);
	const classes = $derived(spec.classes);
	const name = (k: number) => t(classes[k]);

	// ---- confusion matrix
	const conf = $derived(m.confusion);
	const offMax = $derived(Math.max(1, ...conf.flatMap((r, i) => r.filter((_, j) => j !== i))));
	const seq = $derived(makeScale('sequential', [0, offMax], themeState.resolved));
	let hov = $state<{ i: number; j: number } | null>(null);
	const rowSum = (i: number) => conf[i].reduce((a, b) => a + b, 0);
	const worst = $derived.by(() => {
		let b = { i: 0, j: 1, v: -1 };
		conf.forEach((r, i) => r.forEach((v, j) => i !== j && v > b.v && (b = { i, j, v })));
		return b;
	});

	// ---- parameters
	const convIds = $derived(spec.layers.filter((l) => l.op === 'conv2d' || l.op === 'dense').map((l) => l.id));
	const params = $derived(
		convIds.map((id) => {
			const p = layerParams(spec, id)!;
			const ws = p.wShape;
			const formula = ws.length === 4 ? `${ws[0]} × (${ws[1]} × 3 × 3) + ${ws[0]}` : `${ws[0]} × ${ws[1]} + ${ws[0]}`;
			return { id, label: spec.layers.find((l) => l.id === id)!.label, formula, n: p.weight + p.bias };
		})
	);
	const total = $derived(params.reduce((a, b) => a + b.n, 0));
	const denseEquiv = $derived.by(() => {
		const c = spec.layers.find((l) => l.id === 'conv1_1')!;
		const outs = c.outShape.reduce((a, b) => a * b, 1);
		return { outs, n: 28 * 28 * outs + outs };
	});

	// ---- curve
	const hist = $derived(tr.history);
	const CW = 300;
	const CH = 120;
	const accLo = 0.7;
	const xOf = (e: number) => 30 + ((e - 1) / Math.max(1, hist.length - 1)) * (CW - 40);
	const yAcc = (a: number) => 10 + (1 - (a - accLo) / (1 - accLo)) * (CH - 30);
	const lossMax = $derived(Math.max(...hist.map((h) => h.train_loss)));
	const yLoss = (l: number) => 10 + (1 - l / lossMax) * (CH - 30);
	const accPath = $derived(hist.map((h, i) => `${i ? 'L' : 'M'}${xOf(h.epoch).toFixed(1)} ${yAcc(h.val_acc).toFixed(1)}`).join(''));
	const lossPath = $derived(hist.map((h, i) => `${i ? 'L' : 'M'}${xOf(h.epoch).toFixed(1)} ${yLoss(h.train_loss).toFixed(1)}`).join(''));
	const best = $derived(hist.find((h) => h.epoch === tr.bestEpoch));

	// Chinese for the English technical strings stored in spec.json (falls back to English if they change)
	const SPEC_ZH: Record<string, string> = {
		'Quick, Draw! numpy bitmaps (first 15,000 drawings per class)': 'Quick, Draw! numpy 位图（每类前 15,000 幅）',
		'AdamW lr 0.003 one-cycle, weight decay 0.0005 (weights only)': 'AdamW，学习率 0.003，one-cycle 调度，权重衰减 0.0005（只作用于权重）',
		'random shift (up to 12% of half-width), scale 0.8-1.1, rotation +-12 deg, stroke thickness jitter (blend toward 3x3 dilation/erosion), max renormalised to 1':
			'随机平移（最多为半宽的 12%）、缩放 0.8–1.1、旋转 ±12°、笔画粗细扰动（向 3×3 膨胀或腐蚀混合），最后把最大值重新归一化为 1'
	};
	const tz = (s: string) => t({ en: s, zh: SPEC_ZH[s] ?? s });
	const pathRows = $derived(Object.entries(m.drawingPath?.accuracy ?? {}));
	const PATH_ZH: Record<string, string> = {
		raw: '原始位图',
		'soft brush, alpha canvas': '软笔刷，透明画布',
		'dark ink on white (opaque)': '白底深色墨（不透明）',
		'thick brush (+8 px)': '粗笔刷（+8 像素）',
		'small drawing (60-100 px, stroke scaled with the drawing)': '小画（60–100 像素，笔画随画缩放）',
		'dark-mode pad: light ink on #1e1e1e (opaque)': '深色模式画板：#1e1e1e 上的浅色墨',
		'cream pad: blue ink on #faf5e1 (opaque)': '米色画板：#faf5e1 上的蓝色墨',
		'fixed 15 px brush, drawing 140-260 px': '固定 15 像素笔刷，画 140–260 像素',
		'fixed 15 px brush, small drawing (60-100 px)': '固定 15 像素笔刷，小画（60–100 像素）',
		'photo of a sketch on paper (480x360, noisy, uneven light)': '纸上草图的照片（480×360，有噪声、光照不均）'
	};
</script>

<div class="trn">
	<div class="head">
		<div class="kpi">
			<span class="big num">{fmtPct(m.test_acc, 2)}</span>
			<span>{t({ en: `test accuracy on ${fmtInt(m.test_n)} drawings the model never saw in training`, zh: `在 ${fmtInt(m.test_n)} 幅训练中没见过的画上的测试准确率` })}</span>
		</div>
		<div class="kpi small">
			<span class="num">{fmtPct(m.val_acc, 2)}</span><span>{t({ en: 'validation', zh: '验证集' })}</span>
		</div>
		<div class="kpi small">
			<span class="num">{fmtPct(m.test_acc_augmented, 2)}</span><span>{t({ en: 'test, randomly shifted / scaled / rotated', zh: '测试集，随机平移 / 缩放 / 旋转后' })}</span>
		</div>
		<ProvenanceBadge mode="precomputed" precision={spec.precision} note={{ en: 'Measured offline with PyTorch on the same weights (CPU, fp32).', zh: '用同一组权重在 PyTorch 中离线测得（CPU，fp32）。' }} />
	</div>

	<dl class="facts">
		<div><dt>{t({ en: 'Data', zh: '数据' })}</dt><dd>{tz(tr.dataset)} · <a href={tr.datasetUrl} rel="noopener">{tr.license}</a></dd></div>
		<div>
			<dt>{t({ en: 'Split per class', zh: '每类划分' })}</dt>
			<dd class="num">{fmtInt(tr.split.perClass.train)} / {fmtInt(tr.split.perClass.val)} / {fmtInt(tr.split.perClass.test)} <span class="muted">(<Term id="dataset-split">{t({ en: 'train / val / test', zh: '训练 / 验证 / 测试' })}</Term>)</span></dd>
		</div>
		<div><dt>{t({ en: 'Training', zh: '训练' })}</dt><dd>{t({ en: `${tr.epochs} epochs (${fmtInt(tr.steps)} steps of ${tr.batchSize} drawings); weights from epoch ${tr.bestEpoch}, the best on validation`, zh: `${tr.epochs} 轮（${fmtInt(tr.steps)} 步，每步 ${tr.batchSize} 幅画）；取验证集上最好的第 ${tr.bestEpoch} 轮的权重` })}</dd></div>
		<div><dt>{t({ en: 'Optimizer', zh: '优化器' })}</dt><dd class="mono">{tz(tr.optimizer)}</dd></div>
		<div><dt><Term id="data-augmentation" /></dt><dd class="mono">{tz(tr.augmentation)}</dd></div>
		<div><dt>{t({ en: 'Loss', zh: '损失' })}</dt><dd><Term id="cross-entropy" /> · {tr.hardware}</dd></div>
	</dl>

	<div class="row">
		<figure class="curve">
			<figcaption>{t({ en: 'Learning curve', zh: '学习曲线' })} · <span class="acc">{t({ en: 'validation accuracy', zh: '验证准确率' })}</span> · <span class="loss">{t({ en: 'training loss', zh: '训练损失' })}</span></figcaption>
			<svg viewBox="0 0 {CW} {CH}" width="100%" role="img" aria-label={t({ en: `Validation accuracy rises from ${fmtPct(hist[0].val_acc)} after epoch 1 to ${fmtPct(best?.val_acc ?? 0)} at epoch ${tr.bestEpoch}`, zh: `验证准确率从第 1 轮的 ${fmtPct(hist[0].val_acc)} 升到第 ${tr.bestEpoch} 轮的 ${fmtPct(best?.val_acc ?? 0)}` })}>
				{#each [0.7, 0.8, 0.9, 1] as g (g)}
					<line x1="30" x2={CW - 10} y1={yAcc(g)} y2={yAcc(g)} class="grid" />
					<text x="26" y={yAcc(g) + 3} class="ax" text-anchor="end">{Math.round(g * 100)}%</text>
				{/each}
				<path d={lossPath} class="lossl" />
				<path d={accPath} class="accl" />
				{#if best}
					<circle cx={xOf(best.epoch)} cy={yAcc(best.val_acc)} r="3.5" class="best" />
					<text x={xOf(best.epoch)} y={yAcc(best.val_acc) - 7} class="ax" text-anchor="middle">{t({ en: 'best', zh: '最好' })} ({best.epoch})</text>
				{/if}
				<text x="30" y={CH - 4} class="ax">{t({ en: 'epoch 1', zh: '第 1 轮' })}</text>
				<text x={CW - 10} y={CH - 4} class="ax" text-anchor="end">{hist.length}</text>
			</svg>
		</figure>

		<figure class="pcls">
			<figcaption>{t({ en: 'Test accuracy per class', zh: '各类别测试准确率' })}</figcaption>
			{#each classes as c, k (c.id)}
				{@const a = m.per_class_test_acc[c.id]}
				<div class="pc">
					<span class="cn"><span class="sw" style="background: {categorical(k, themeState.resolved)}"></span>{name(k)}</span>
					<span class="bar"><span style="width: {((a - 0.9) / 0.1) * 100}%; background: {categorical(k, themeState.resolved)}"></span></span>
					<span class="num v">{fmtPct(a)}</span>
				</div>
			{/each}
			<span class="muted tiny">{t({ en: 'bars start at 90%', zh: '条形从 90% 开始' })}</span>
		</figure>
	</div>

	<figure class="cm">
		<figcaption>
			<Term id="confusion-matrix" /> · {t({ en: 'rows: true class, columns: what the model said; 1,500 test drawings per row', zh: '行：真实类别，列：模型的回答；每行 1,500 幅测试画' })}
		</figcaption>
		<div class="cm-wrap" use:keyScroll>
			<table class="num">
				<thead>
					<tr>
						<th></th>
						{#each classes as c, j (c.id)}<th scope="col" class:hl={hov?.j === j}><span class="vlab">{name(j)}</span></th>{/each}
					</tr>
				</thead>
				<tbody>
					{#each conf as r, i (i)}
						<tr>
							<th scope="row" class:hl={hov?.i === i}>{name(i)}</th>
							{#each r as v, j (j)}
								<td
									class:diag={i === j}
									class:on={hov?.i === i && hov?.j === j}
									style={i === j ? '' : `background: ${seq.css(v)}; color: ${seq.ink(v)}`}
									onpointerenter={() => (hov = { i, j })}
									onpointerleave={() => (hov = null)}>{v}</td
								>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="cm-read">
			{#if hov}
				{#if hov.i === hov.j}
					{t({ en: `${name(hov.i)}: ${conf[hov.i][hov.j]} of ${fmtInt(rowSum(hov.i))} right (${fmtPct(conf[hov.i][hov.j] / rowSum(hov.i))})`, zh: `${name(hov.i)}：${fmtInt(rowSum(hov.i))} 幅中答对 ${conf[hov.i][hov.j]} 幅（${fmtPct(conf[hov.i][hov.j] / rowSum(hov.i))}）` })}
				{:else}
					{t({ en: `${conf[hov.i][hov.j]} ${name(hov.i)} drawings were called “${name(hov.j)}”`, zh: `有 ${conf[hov.i][hov.j]} 幅${name(hov.i)}被说成了“${name(hov.j)}”` })}
				{/if}
			{:else}
				{t({ en: `Diagonal: correct answers. The most common mix-up: ${worst.v} ${name(worst.i)} drawings called “${name(worst.j)}”. Off-diagonal colors run from 0 to ${offMax}.`, zh: `对角线：答对的数量。最常见的混淆：${worst.v} 幅${name(worst.i)}被说成“${name(worst.j)}”。非对角线的颜色从 0 到 ${offMax}。` })}
			{/if}
		</p>
	</figure>

	<div class="params">
		<h3>{t({ en: `${fmtInt(total)} parameters, layer by layer`, zh: `${fmtInt(total)} 个参数，逐层统计` })}</h3>
		<table class="ptable">
			<thead>
				<tr><th>{t({ en: 'layer', zh: '层' })}</th><th>{t({ en: 'out × (in × 3 × 3) + biases', zh: '输出 × (输入 × 3 × 3) + 偏置' })}</th><th class="r">{t({ en: 'count', zh: '数量' })}</th><th class="share"></th></tr>
			</thead>
			<tbody>
				{#each params as p (p.id)}
					<tr>
						<td>{t(p.label)}</td>
						<td class="num">{p.formula}</td>
						<td class="num r">{fmtInt(p.n)}</td>
						<td class="share"><div class="shw"><span class="sbar"><span style="width: {(p.n / total) * 100}%"></span></span><span class="num muted">{fmtPct(p.n / total, 0)}</span></div></td>
					</tr>
				{/each}
				<tr class="tot"><td>{t({ en: 'total', zh: '合计' })}</td><td class="num">{t({ en: `× 4 bytes (fp32) = ${fmtBytes(total * 4)}`, zh: `× 4 字节（fp32）= ${fmtBytes(total * 4)}` })}</td><td class="num r">{fmtInt(total)}</td><td></td></tr>
			</tbody>
		</table>
		<p class="why">
			<Lang>
				{#snippet en()}Why so few? A <Term id="kernel">kernel</Term> is <em>shared</em>: the same 9 weights are reused at all 676 positions of a 26×26 map. If Conv 1-1 were a <Term id="dense-layer">dense layer</Term> connecting all 784 pixels to its {fmtInt(denseEquiv.outs)} outputs, it would need {fmtInt(denseEquiv.n)} parameters instead of 80. The pooling layers have none at all.{/snippet}
				{#snippet zh()}为什么这么少？因为<Term id="kernel">卷积核</Term>是<em>共享</em>的：同样的 9 个权重在 26×26 图的全部 676 个位置上重复使用。如果卷积 1-1 换成把 784 个像素全部连到 {fmtInt(denseEquiv.outs)} 个输出的<Term id="dense-layer">全连接层</Term>，就需要 {fmtInt(denseEquiv.n)} 个参数，而不是 80 个。池化层则完全没有参数。{/snippet}
			</Lang>
		</p>
	</div>

	{#if pathRows.length}
		<details class="paths">
			<summary>{t({ en: 'How the drawing input was checked (accuracy of each input path)', zh: '如何检查绘画输入（各输入路径的准确率）' })}</summary>
			<p class="muted tiny">{t({ en: '1,500 test drawings (150 per class) turned into simulated pad drawings or photos, then preprocessed and classified.', zh: '把 1,500 幅测试画（每类 150 幅）模拟成画板绘画或照片，再预处理并分类。' })}</p>
			<table class="ptable">
				<tbody>
					{#each pathRows as [k, v] (k)}
						<tr><td>{t({ en: k, zh: PATH_ZH[k] ?? k })}</td><td class="num r">{fmtPct(v, 2)}</td></tr>
					{/each}
				</tbody>
			</table>
			<p class="muted tiny">{t({ en: 'The weak row (fixed brush, small drawing) is why this page feeds the model your strokes, not a picture of the pad.', zh: '表现差的那一行（固定笔刷、小画）就是本页把你的笔画而不是画板截图交给模型的原因。' })}</p>
		</details>
	{/if}
</div>

<style>
	.trn {
		display: grid;
		gap: 20px;
	}
	.head {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 28px;
		align-items: center;
	}
	.kpi {
		display: grid;
		gap: 0;
		max-width: 300px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.kpi .big {
		font-size: var(--fs-2xl);
		font-weight: 700;
		color: var(--ink);
	}
	.kpi.small .num {
		font-size: var(--fs-lg);
		color: var(--ink);
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr));
		gap: 10px 24px;
		margin: 0;
	}
	dt {
		font-size: var(--fs-xs);
		color: var(--muted);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
	dd {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.mono {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.row {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
		gap: 20px 32px;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.curve svg {
		max-width: 460px;
	}
	.grid {
		stroke: var(--rule);
	}
	.ax {
		font-size: 8.5px;
		fill: var(--muted);
		font-family: var(--font-mono);
	}
	.accl {
		fill: none;
		stroke: var(--accent);
		stroke-width: 2;
	}
	.lossl {
		fill: none;
		stroke: var(--rule-strong);
		stroke-width: 1.5;
		stroke-dasharray: 4 3;
	}
	.acc {
		color: var(--accent);
		font-weight: 650;
	}
	.loss {
		color: var(--muted);
	}
	.best {
		fill: var(--accent);
	}
	.pc {
		display: grid;
		grid-template-columns: 6.5em 1fr 3.6em;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-xs);
	}
	.cn {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
	}
	.bar,
	.sbar {
		height: 9px;
		background: var(--surface-2);
		border-radius: 0 2px 2px 0;
		overflow: hidden;
	}
	.bar span,
	.sbar span {
		display: block;
		height: 100%;
	}
	.sbar span {
		background: var(--param);
	}
	.v {
		text-align: right;
	}
	.tiny {
		font-size: var(--fs-xs);
	}
	.cm-wrap {
		overflow-x: auto;
		max-width: 100%;
	}
	.cm table {
		border-collapse: separate;
		border-spacing: 2px;
		font-size: 11px;
	}
	.cm th {
		font-family: var(--font-body);
		font-weight: 500;
		color: var(--ink-2);
		padding: 0 6px;
		text-align: right;
		white-space: nowrap;
	}
	.cm thead th {
		height: 70px;
		vertical-align: bottom;
		padding: 0;
	}
	.vlab {
		display: inline-block;
		writing-mode: vertical-rl;
		transform: rotate(180deg);
		text-align: left;
	}
	.cm th.hl {
		color: var(--accent);
		font-weight: 700;
	}
	.cm td {
		width: 40px;
		height: 26px;
		text-align: center;
		border-radius: 2px;
		cursor: default;
	}
	.cm td.diag {
		background: var(--surface-3);
		font-weight: 700;
		color: var(--ink);
	}
	.cm td.on {
		outline: 2px solid var(--highlight);
	}
	.cm-read {
		margin: 4px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		min-height: 1.6em;
	}
	.params h3 {
		font-size: var(--fs-lg);
		margin: 0 0 8px;
	}
	.ptable {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		width: 100%;
		max-width: 640px;
	}
	.ptable th,
	.ptable td {
		padding: 4px 8px;
		border-bottom: 1px solid var(--rule);
		text-align: left;
	}
	.ptable th {
		font-size: var(--fs-xs);
		color: var(--muted);
		font-weight: 500;
	}
	.ptable .r {
		text-align: right;
	}
	.share {
		width: 34%;
	}
	.shw {
		display: flex;
		gap: 6px;
		align-items: center;
		min-width: 90px;
	}
	.shw .sbar {
		flex: 1;
	}
	.tot td {
		font-weight: 700;
		border-bottom: 0;
	}
	.why {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		max-width: var(--measure);
		margin: 10px 0 0;
	}
	.paths summary {
		cursor: pointer;
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.paths p {
		margin: 6px 0;
	}
	@media (max-width: 560px) {
		.ptable th:nth-child(2),
		.ptable td:nth-child(2) {
			font-size: 11px;
		}
		.share {
			display: none !important;
		}
	}
</style>
