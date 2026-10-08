<!--
  Part 1 network with semantic zoom: overview (frozen body, new trainable head) → one block's
  layers → the head's arithmetic for one unit, all for the inspected drawing and the head at the
  current training step. Every number is live.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import ModelOverview from '$lib/kit/ModelOverview.svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TensorView from '$lib/kit/TensorView.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import RoleChip from '$lib/kit/RoleChip.svelte';
	import RoleFrame from '$lib/kit/RoleFrame.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import ColorLegend from '$lib/kit/ColorLegend.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { domainOf } from '$lib/kit/color';
	import { sym } from '$lib/kit/tex';
	import { fmtInt } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { LayerNode, CellValue } from '$lib/kit/types';
	import type { Head } from '$lib/models/tinycnn_finetune';
	import Thumb from './Thumb.svelte';
	import { LAYER_PARAMS, PARAMS } from './data';

	interface TraceT {
		order: string[];
		t: Record<string, { shape: number[]; data: Float32Array }>;
	}
	interface Props {
		trace: TraceT | null;
		head: Head | null;
		/** class names for the new head */
		names: [string, string];
		epoch: number;
		/** training steps done so far (0 = the untrained head) */
		trained: number;
		/** what is being inspected, e.g. "your drawing" */
		inspectedLabel: string;
		current?: string;
	}
	let { trace, head, names, epoch, trained, inspectedLabel, current = $bindable('head') }: Props = $props();

	const nodes: LayerNode[] = [
		{ id: 'input', label: { en: 'Input', zh: '输入' }, op: 'drawing', outShape: [1, 28, 28] },
		{ id: 'block1', label: { en: 'Conv block 1', zh: '卷积块 1' }, op: 'conv·conv·pool', outShape: [8, 12, 12], status: 'frozen' },
		{ id: 'block2', label: { en: 'Conv block 2', zh: '卷积块 2' }, op: 'conv·conv·pool', outShape: [16, 4, 4], status: 'frozen' },
		{ id: 'features', label: { en: 'Features', zh: '特征' }, op: 'flatten', outShape: [256], status: 'frozen' },
		{ id: 'head', label: { en: 'New head', zh: '新输出头' }, op: 'dense 256→2', outShape: [2], status: 'trainable' },
		{ id: 'softmax', label: 'Softmax', op: 'softmax', outShape: [2] }
	];

	const SUBS: Record<string, { id: string; label: { en: string; zh: string } }[]> = {
		block1: [
			{ id: 'conv1_1', label: { en: 'Conv 1-1', zh: '卷积 1-1' } },
			{ id: 'relu1_1', label: { en: 'ReLU', zh: 'ReLU' } },
			{ id: 'conv1_2', label: { en: 'Conv 1-2', zh: '卷积 1-2' } },
			{ id: 'relu1_2', label: { en: 'ReLU', zh: 'ReLU' } },
			{ id: 'pool1', label: { en: 'Max pool', zh: '最大池化' } }
		],
		block2: [
			{ id: 'conv2_1', label: { en: 'Conv 2-1', zh: '卷积 2-1' } },
			{ id: 'relu2_1', label: { en: 'ReLU', zh: 'ReLU' } },
			{ id: 'conv2_2', label: { en: 'Conv 2-2', zh: '卷积 2-2' } },
			{ id: 'relu2_2', label: { en: 'ReLU', zh: 'ReLU' } },
			{ id: 'pool2', label: { en: 'Max pool', zh: '最大池化' } }
		]
	};
	let sub = $state<Record<string, string>>({ block1: 'pool1', block2: 'pool2' });
	let channel = $state(0);
	let unit = $state<0 | 1>(0);
	let cellHover = $state<{ row: number; col: number } | null>(null);

	// best channel (largest total activation) of a [C,H,W] tensor, for the overview thumbnails
	function bestChannel(x: { shape: number[]; data: Float32Array }) {
		const [C, H, W] = x.shape;
		let best = 0;
		let bs = -Infinity;
		for (let c = 0; c < C; c++) {
			let s = 0;
			for (let i = 0; i < H * W; i++) s += x.data[c * H * W + i];
			if (s > bs) {
				bs = s;
				best = c;
			}
		}
		return x.data.subarray(best * H * W, (best + 1) * H * W);
	}

	const feat = $derived(trace?.t.flatten?.data ?? null);
	// The head reads features divided by s (RMS feature norm of the training set).
	const fs = $derived(feat && head ? Float32Array.from(feat, (v) => v / head.scale) : null);
	const Wk = (k: number) => (head ? head.W.subarray(k * 256, (k + 1) * 256) : null);
	const logits = $derived.by(() => {
		if (!fs || !head) return null;
		const z = [0, 0];
		for (let k = 0; k < 2; k++) {
			let s = head.b[k];
			for (let d = 0; d < 256; d++) s += head.W[k * 256 + d] * fs[d];
			z[k] = s;
		}
		return z;
	});
	const probs = $derived.by(() => {
		if (!logits) return null;
		const m = Math.max(...logits);
		const e = logits.map((z) => Math.exp(z - m));
		const s = e[0] + e[1];
		return [e[0] / s, e[1] / s];
	});
	const products = $derived.by(() => {
		const w = Wk(unit);
		if (!w || !fs) return null;
		return Float32Array.from(w, (v, d) => v * fs[d]);
	});
	const wDom = $derived(head ? domainOf('diverging', head.W) : ([-1, 1] as [number, number]));
	const top3 = $derived.by(() => {
		if (!products) return [];
		return Array.from(products, (v, d) => ({ d, v }))
			.sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
			.slice(0, 3);
	});
	const sumProducts = $derived(products ? products.reduce((a, v) => a + v, 0) : 0);
	const hl = $derived(cellHover ? [{ row: cellHover.row, col: cellHover.col, color: 'var(--highlight)' }] : []);
	const onCell = (c: CellValue | null) => (cellHover = c ? { row: c.row, col: c.col } : null);
	const dIndex = $derived(cellHover ? cellHover.row * 16 + cellHover.col : null);

	const zTex = $derived(
		`\\begin{aligned} ${sym('out', `z_{${unit === 0 ? 'A' : 'B'}}`)} &= \\sum_{d=0}^{255} ${sym('param', `W_{${unit === 0 ? 'A' : 'B'},d}`)}\\,\\frac{${sym('act', 'f_d')}}{s} + ${sym('param', `b_{${unit === 0 ? 'A' : 'B'}}`)} \\\\ &= {{sum}} + ({{b}}) = {{z}} \\end{aligned}`
	);
	const pTex = `\\begin{aligned} ${sym('prob', 'p_A')} &= \\frac{e^{${sym('out', 'z_A')}}}{e^{${sym('out', 'z_A')}} + e^{${sym('out', 'z_B')}}} = \\frac{1}{1 + e^{-(${sym('out', 'z_A')} - ${sym('out', 'z_B')})}} \\\\ &= \\frac{1}{1 + e^{-({{zA}} - ({{zB}}))}} = {{pA}} \\end{aligned}`;
	const paramTex = $derived(
		`\\underbrace{256}_{\\text{${t({ en: 'features', zh: '特征' })}}} \\times \\underbrace{2}_{\\text{${t({ en: 'classes', zh: '类别' })}}} + \\underbrace{2}_{\\text{${t({ en: 'biases', zh: '偏置' })}}} = ${sym('param', '514')}`
	);
</script>

<div class="net">
	<div class="ov-row">
		<ModelOverview layers={nodes} bind:current label={{ en: 'Transfer network: frozen body, new head', zh: '迁移网络：冻结主体 + 新输出头' }}>
			{#snippet thumb(l)}
				{#if trace}
					{#if l.id === 'input'}
						<Thumb data={trace.t.input.data} size={36} alt="" />
					{:else if l.id === 'block1'}
						<TensorGrid data={bestChannel(trace.t.pool1)} rows={12} cols={12} cell={3} interactive={false} readout={false} gridLines={false} label="pool1" />
					{:else if l.id === 'block2'}
						<TensorGrid data={bestChannel(trace.t.pool2)} rows={4} cols={4} cell={9} interactive={false} readout={false} gridLines={false} label="pool2" />
					{:else if l.id === 'features'}
						<TensorGrid data={trace.t.flatten.data} rows={16} cols={16} cell={2} interactive={false} readout={false} gridLines={false} label="features" />
					{:else if l.id === 'head' && logits}
						<TensorGrid data={logits} rows={1} cols={2} cell={14} interactive={false} readout={false} label="logits" />
					{:else if l.id === 'softmax' && probs}
						<TensorGrid data={probs} rows={1} cols={2} cell={14} kind="sequential" domain={[0, 1]} interactive={false} readout={false} label="probabilities" />
					{/if}
				{/if}
			{/snippet}
		</ModelOverview>
		<p class="legend-line">
			<RoleChip role="frozen" label={{ en: `frozen: ${fmtInt(PARAMS.frozenBody)} parameters, never updated`, zh: `冻结：${fmtInt(PARAMS.frozenBody)} 个参数，从不更新` }} />
			<RoleChip role="trainable" label={{ en: `trainable: ${PARAMS.newHead} parameters (the new head)`, zh: `可训练：${PARAMS.newHead} 个参数（新输出头）` }} />
			<span class="muted">{t({ en: `Showing: ${inspectedLabel}`, zh: `正在查看：${inspectedLabel}` })} · {t({ en: `head after step ${epoch} of ${trained}`, zh: `输出头：第 ${epoch} / ${trained} 步之后` })}</span>
		</p>
	</div>

	<div class="zoom" aria-live="off">
		{#if !trace}
			<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
		{:else if current === 'input'}
			<h4>{t({ en: 'Input: what the network receives', zh: '输入：网络实际收到的东西' })}</h4>
			<div class="two">
				<TensorGrid data={trace.t.input.data} rows={28} cols={28} kind="sequential" domain={[0, 1]} maxCell={12} label={{ en: 'input 28×28', zh: '输入 28×28' }} />
				<p class="note">
					{t({
						en: '784 numbers between 0 and 1 (1 = ink). Every drawing, yours included, is framed like the Quick, Draw! training drawings: the ink is scaled so its long side spans 26 pixels, centered, with thin strokes. If new inputs were prepared differently from the training data, the frozen features would mean less.',
						zh: '784 个 0 到 1 之间的数（1 表示墨迹）。每张画（包括你的）都按 Quick, Draw! 训练图的方式取景：把墨迹缩放到长边 26 像素、居中，线条细。如果新输入的处理方式和训练数据不同，冻结特征就没那么可靠了。'
					})}
				</p>
			</div>
		{:else if current === 'block1' || current === 'block2'}
			{@const subs = SUBS[current]}
			{@const sid = sub[current]}
			<h4>
				{t(current === 'block1' ? { en: 'Conv block 1 (frozen)', zh: '卷积块 1（冻结）' } : { en: 'Conv block 2 (frozen)', zh: '卷积块 2（冻结）' })}
				<RoleChip role="frozen" iconOnly />
			</h4>
			<div class="segmented" role="group" aria-label={t({ en: 'Layer inside the block', zh: '块内的层' })}>
				{#each subs as s (s.id)}
					<button type="button" aria-pressed={sid === s.id} onclick={() => (sub[current] = s.id)}>{t(s.label)}</button>
				{/each}
			</div>
			<p class="params num">
				{#if LAYER_PARAMS[sid] !== undefined}
					{sid}: {fmtInt(LAYER_PARAMS[sid])} {t({ en: 'parameters, frozen', zh: '个参数，已冻结' })}
				{:else}
					{sid}: {t({ en: 'no parameters', zh: '没有参数' })}
				{/if}
				· {t({ en: 'output', zh: '输出' })} [{trace.t[sid].shape.join('×')}]
			</p>
			<TensorView tensor={trace.t[sid]} label={sid} bind:channel />
			<p class="note">
				{t({
					en: 'Frozen means these numbers are the same before and after you train the head: training reads this block’s output but never changes its kernels. They were learned on 10 other classes, yet they still respond to strokes, corners and crossings in a ladder.',
					zh: '冻结的意思是：训练输出头前后，这些数一模一样。训练只读取这一块的输出，从不修改它的卷积核。这些卷积核是在另外 10 个类别上学到的，却同样会对梯子里的笔画、拐角和交叉作出反应。'
				})}
				<a href="{base}/m/tinycnn/">{t({ en: 'Every multiply-add of these layers is on the TinyCNN page →', zh: '这些层的每一次乘加，都可以在 TinyCNN 页面上看到 →' })}</a>
			</p>
		{:else if current === 'features'}
			<h4>{t({ en: 'Features: the 256 numbers the head reads', zh: '特征：输出头读取的 256 个数' })} <RoleChip role="frozen" iconOnly /></h4>
			<div class="two">
				<div>
					<TensorGrid data={trace.t.flatten.data} rows={16} cols={16} kind="diverging" maxCell={18} label={{ en: 'features, 16 channels × 16 positions', zh: '特征：16 个通道 × 16 个位置' }} onhover={onCell} highlights={hl} />
					<ColorLegend kind="diverging" domain={domainOf('diverging', trace.t.flatten.data)} />
				</div>
				<p class="note">
					{t({
						en: 'Block 2 ends with 16 channels of 4 × 4. Flattening lays them out as one list of 256 numbers (here: one row per channel). This list is the frozen feature vector. The old 10-way head read it; the new 2-way head reads the same list. Nothing before this point changes during training.',
						zh: '卷积块 2 最后输出 16 个 4 × 4 的通道。展平把它们排成一条 256 个数的列表（这里每行是一个通道）。这就是冻结的特征向量。旧的 10 类输出头读它，新的二分类输出头读的也是它。在这之前的部分，训练时都不会变。'
					})}
				</p>
			</div>
		{:else if current === 'head'}
			<h4>{t({ en: 'New head: one dense layer, 256 → 2', zh: '新输出头：一个全连接层，256 → 2' })} <RoleChip role="trainable" iconOnly /></h4>
			<FormulaCard tex={paramTex} display caption={{ en: 'Trainable parameters', zh: '可训练参数' }} />
			{#if head && fs && products && logits}
				<div class="segmented" role="group" aria-label={t({ en: 'Output unit', zh: '输出单元' })}>
					<button type="button" aria-pressed={unit === 0} onclick={() => (unit = 0)}><span class="dot a"></span>{t({ en: `Unit A: ${names[0]}`, zh: `单元 A：${names[0]}` })}</button>
					<button type="button" aria-pressed={unit === 1} onclick={() => (unit = 1)}><span class="dot b"></span>{t({ en: `Unit B: ${names[1]}`, zh: `单元 B：${names[1]}` })}</button>
				</div>
				<div class="triple">
					<figure>
						<figcaption>f / s · {t({ en: 'features ÷ s', zh: '特征 ÷ s' })}</figcaption>
						<TensorGrid data={fs} rows={16} cols={16} maxCell={11} readout={false} label={{ en: 'scaled features', zh: '缩放后的特征' }} onhover={onCell} highlights={hl} />
					</figure>
					<span class="op" aria-hidden="true">⊙</span>
					<div class="wf"><RoleFrame role="param" status="trainable" label={unit === 0 ? 'W_A [16×16]' : 'W_B [16×16]'}>
						<TensorGrid data={Wk(unit)!} rows={16} cols={16} domain={wDom} maxCell={11} readout={false} label={{ en: `weights of unit ${unit === 0 ? 'A' : 'B'}`, zh: `单元 ${unit === 0 ? 'A' : 'B'} 的权重` }} onhover={onCell} highlights={hl} />
					</RoleFrame></div>
					<span class="op eq" aria-hidden="true">=</span>
					<figure class="prod">
						<figcaption>{t({ en: 'products', zh: '逐项乘积' })}</figcaption>
						<TensorGrid data={products} rows={16} cols={16} maxCell={11} readout={false} label={{ en: 'products W·f/s', zh: '乘积 W·f/s' }} onhover={onCell} highlights={hl} />
					</figure>
				</div>
				<p class="readout num" aria-live="polite">
					{#if dIndex !== null}
						d = {dIndex} ({t({ en: 'channel', zh: '通道' })} {cellHover!.row}, {t({ en: 'position', zh: '位置' })} {cellHover!.col}):
						W = <Num value={Wk(unit)![dIndex]} signed /> × f/s = <Num value={fs[dIndex]} /> → <Num value={products[dIndex]} signed />
					{:else}
						{t({ en: 'Hover or tap a cell to see one multiply. Largest contributions:', zh: '悬停或轻点一个格子，查看一次乘法。贡献最大的几项：' })}
						{#each top3 as c, i (c.d)}{i ? ', ' : ' '}d={c.d}: <Num value={c.v} signed />{/each}
					{/if}
				</p>
				<FormulaCard
					tex={zTex}
					values={{ sum: sumProducts, b: head.b[unit], z: logits[unit] }}
					provenance="live"
					where={[
						{ tex: sym('act', 'f_d'), text: { en: 'feature d of the inspected drawing (frozen)', zh: '所查看画作的第 d 个特征（冻结）' } },
						{ tex: sym('param', 'W, b'), text: { en: `weights and bias of the new head after step ${epoch}`, zh: `新输出头在第 ${epoch} 步之后的权重与偏置` } },
						{ tex: 's', text: { en: `= ${head.scale.toFixed(2)}, the average feature length of the training examples`, zh: `= ${head.scale.toFixed(2)}，训练样本特征的平均长度` } }
					]}
				/>
				{#if epoch === 0}
					<p class="note warn">
						{t({
							en: 'This is the head before training: small random weights, so both scores are close to 0 and it says about 50/50 for every drawing. Press Train, or scrub the training steps, and watch the weights grow.',
							zh: '这是训练前的输出头：权重是很小的随机数，所以两个分数都接近 0，对每张画都差不多是五五开。点“训练”，或者拖动训练步数，看权重怎样长起来。'
						})}
					</p>
				{/if}
			{:else}
				<p class="muted">{t({ en: 'Add at least one example to each class to create the head.', zh: '每个类别至少放一个样本，才能建立输出头。' })}</p>
			{/if}
			<p class="note">
				{t({
					en: `The old head (dense 256 → 10, ${fmtInt(PARAMS.oldHead)} parameters) is set aside, not deleted: the 10-class model still works exactly as in the first section.`,
					zh: `旧的输出头（全连接 256 → 10，${fmtInt(PARAMS.oldHead)} 个参数）只是被放在一边，没有删除：10 类模型仍然和第一节里完全一样。`
				})}
			</p>
		{:else if current === 'softmax'}
			<h4>{t({ en: 'Softmax: two scores become two probabilities', zh: 'Softmax：两个分数变成两个概率' })}</h4>
			{#if logits && probs}
				<FormulaCard tex={pTex} values={{ zA: logits[0], zB: logits[1], pA: probs[0] }} provenance="live" />
				<ProbBars
					items={[
						{ label: names[0], p: probs[0], id: 0, color: 'var(--div-pos)' },
						{ label: names[1], p: probs[1], id: 1, color: 'var(--div-neg)' }
					]}
					sort={false}
					chosen={probs[0] >= probs[1] ? 0 : 1}
				/>
				<p class="note">
					{t({
						en: 'With two classes only the difference z_A − z_B matters. Positive: class A (orange). Negative: class B (blue). That difference is the head’s margin, drawn as the boundary on the map below.',
						zh: '只有两个类别时，只有差值 z_A − z_B 才重要。为正：类别 A（橙色）；为负：类别 B（蓝色）。这个差值就是输出头的“间隔”，也就是下面地图上那条分界线。'
					})}
				</p>
			{/if}
		{/if}
	</div>
</div>

<style>
	.net {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
	}
	.legend-line {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 12px;
		align-items: center;
		margin: 6px 0 0;
		font-size: var(--fs-sm);
	}
	.zoom {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		padding: var(--s4);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
		min-width: 0;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.two {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
		align-items: start;
	}
	@media (min-width: 760px) {
		.two {
			grid-template-columns: minmax(0, auto) minmax(0, 1fr);
		}
	}
	.note {
		margin: 0;
		color: var(--ink-2);
		max-width: var(--measure);
		font-size: var(--fs-sm);
	}
	.note.warn {
		padding: 8px 12px;
		border-left: 3px solid var(--accent);
		background: var(--accent-soft);
		border-radius: var(--r1);
		color: var(--ink);
	}
	.params {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	/* f/s ⊙ W = products: one row on wide screens; on phones "f/s ⊙ W" then "= products" */
	.triple {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
		align-items: center;
		gap: 8px 10px;
		max-width: 760px;
	}
	.triple .eq {
		grid-column: 2;
	}
	.triple .prod {
		grid-column: 3;
	}
	@media (min-width: 700px) {
		.triple {
			grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
		}
		.triple .eq,
		.triple .prod {
			grid-column: auto;
		}
	}
	.triple figure {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.wf {
		min-width: 0;
	}
	.wf :global(figure) {
		display: flex;
		width: 100%;
		box-sizing: border-box;
	}
	.triple figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		font-family: var(--font-mono);
	}
	.op {
		font-size: var(--fs-xl);
		color: var(--muted);
	}
	.readout {
		margin: 0;
		font-size: var(--fs-sm);
		min-height: 1.6em;
	}
	.dot {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		margin-right: 6px;
		vertical-align: middle;
	}
	.dot.a {
		background: var(--div-pos);
	}
	.dot.b {
		background: var(--div-neg);
		border-radius: 2px;
	}
</style>
