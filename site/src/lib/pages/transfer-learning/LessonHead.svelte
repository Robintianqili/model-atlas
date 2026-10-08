<!--
  Lesson step 4: the new head (dense 256 → 2, 514 trainable parameters) before any training:
  its seeded random weights, its answer for the learner's drawing and its accuracy on the 200
  held-out test drawings. All live (the lesson's run, step 0).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import RoleFrame from '$lib/kit/RoleFrame.svelte';
	import RoleChip from '$lib/kit/RoleChip.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { domainOf } from '$lib/kit/color';
	import { sym } from '$lib/kit/tex';
	import { t } from '$lib/i18n/i18n.svelte';
	import { clsName, headOf, type LessonState } from './lessonState.svelte';
	import { predictHead } from '$lib/models/tinycnn_finetune';

	let { L }: { L: LessonState } = $props();

	const h0 = $derived(L.run ? headOf(L.run, 0) : null);
	// colored on the scale of the trained head (step 200), so "almost 0" looks almost gray
	const dom = $derived(L.run ? domainOf('diverging', headOf(L.run, L.run.snaps.length - 1).W) : ([-1, 1] as [number, number]));
	const mine = $derived(h0 && L.feat ? predictHead(h0, L.feat) : null);
	const paramTex = $derived(
		`\\underbrace{256}_{\\text{${t({ en: 'features', zh: '特征' })}}} \\times \\underbrace{2}_{\\text{${t({ en: 'outputs', zh: '输出' })}}} + \\underbrace{2}_{\\text{${t({ en: 'biases', zh: '偏置' })}}} = ${sym('param', '514')}`
	);
	const pct = (v: number) => (v * 100).toFixed(1) + '%';
</script>

{#if L.run && h0}
	{@const run = L.run}
	<div class="lh">
		<div class="top">
			<FormulaCard tex={paramTex} display caption={{ en: 'Trainable parameters of the new head', zh: '新输出头的可训练参数' }} />
			<p class="chips">
				<RoleChip role="frozen" label={{ en: '4,152 frozen (body)', zh: '4,152 个冻结（主体）' }} />
				<RoleChip role="trainable" label={{ en: '514 trainable (new head)', zh: '514 个可训练（新输出头）' }} />
			</p>
		</div>
		<div class="ws">
			{#each [0, 1] as const as k (k)}
				<figure>
					<RoleFrame role="param" status="trainable" label={k === 0 ? 'W_A [16×16]' : 'W_B [16×16]'}>
						<TensorGrid
							data={h0.W.subarray(k * 256, (k + 1) * 256)}
							rows={16}
							cols={16}
							domain={dom}
							maxCell={10}
							readout={false}
							label={{ en: `weights of the ${clsName(k).en} output, before training`, zh: `“${clsName(k).zh}”输出的权重（训练前）` }}
						/>
					</RoleFrame>
					<figcaption>
						<span class="dot {k === 0 ? 'a' : 'b'}"></span>{k === 0 ? 'A' : 'B'} = {t(clsName(k))} · <span class="num">b<sub>{k === 0 ? 'A' : 'B'}</sub> = <Num value={h0.b[k]} /></span>
					</figcaption>
				</figure>
			{/each}
			<p class="note">
				{t({
					en: `Each grid holds one output’s 256 weights, one per feature, laid out like the features in step 2. Every weight starts as a small random number between −0.01 and 0.01 (seed 0, a fixed random start, so everyone sees the same numbers); the 2 biases start at 0. Drawn on the color scale of the trained head (±${dom[1].toFixed(2)} after step 200), the grids are almost gray.`,
					zh: `每个网格是一个输出的 256 个权重，每个特征一个，排法和第 2 步的特征一样。每个权重都从 −0.01 到 0.01 之间的一个小随机数开始（种子 0：固定的随机起点，所以每个人看到的数都一样）；2 个偏置从 0 开始。按训练好的输出头的色阶（第 200 步之后为 ±${dom[1].toFixed(2)}）来画，网格几乎是灰的。`
				})}
			</p>
		</div>
		<div class="res">
			<div class="mine">
				<div class="hd">
					<h4>{t({ en: 'This drawing, before training', zh: '这张画（训练前）' })}</h4>
					<ProvenanceBadge mode="live" compact note={{ en: 'The new head is created right here in your browser (seeded); the features come from the shipped TinyCNN.', zh: '新输出头就在你的浏览器里创建（固定种子）；特征来自发布的 TinyCNN。' }} />
				</div>
				{#if mine}
					<ProbBars
						items={[
							{ label: t(clsName(0)), p: mine.probs[0], id: 0, color: 'var(--div-pos)' },
							{ label: t(clsName(1)), p: mine.probs[1], id: 1, color: 'var(--div-neg)' }
						]}
						sort={false}
						chosen={mine.top}
					/>
					<p class="small num">{t({ en: 'scores:', zh: '分数：' })} z<sub>A</sub> = <Num value={mine.logits[0]} signed />, z<sub>B</sub> = <Num value={mine.logits[1]} signed /></p>
				{/if}
			</div>
			<div class="stat">
				<span class="lbl">{t({ en: 'Held-out test drawings right (step 0)', zh: '留出测试图判对（第 0 步）' })}</span>
				<span class="val num">{pct(run.testAcc[0])}</span>
				<span class="sub num">{t(clsName(0))} {run.testPer[0][0]}/{run.testN[0]} · {t(clsName(1))} {run.testPer[0][1]}/{run.testN[1]}</span>
			</div>
		</div>
	</div>
{:else if L.presetError}
	<p class="muted">{t({ en: 'Could not compute the features. Please reload the page.', zh: '特征计算失败，请刷新页面。' })}</p>
{:else}
	<p class="muted">{t({ en: 'Computing the frozen features of the training drawings…', zh: '正在计算训练图的冻结特征…' })}</p>
{/if}

<style>
	.lh {
		container: lh / inline-size;
		display: grid;
		gap: 12px;
		min-width: 0;
	}
	.top {
		display: grid;
		gap: 6px;
	}
	.chips {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 6px 10px;
	}
	.ws {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 16px;
		align-items: flex-start;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
		flex: 1 1 130px;
		max-width: 190px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.dot {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
		margin-right: 4px;
		vertical-align: middle;
	}
	.dot.a {
		background: var(--div-pos);
	}
	.dot.b {
		background: var(--div-neg);
		border-radius: 2px;
	}
	.note {
		flex: 1 1 200px;
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.res {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px 16px;
		align-items: start;
	}
	@container lh (min-width: 520px) {
		.res {
			grid-template-columns: minmax(0, 1fr) auto;
		}
	}
	.hd {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 4px 10px;
		margin-bottom: 4px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.small {
		margin: 4px 0 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.stat {
		display: grid;
		gap: 2px;
		padding: 8px 12px;
		border: 1px solid var(--accent);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.lbl {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.val {
		font-size: var(--fs-xl);
		font-weight: 650;
	}
	.sub {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>
