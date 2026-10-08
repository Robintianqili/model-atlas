<!--
  The TinyCNN guided lesson: ten steps from the learner's drawing to the answer, one focused view at
  a time (kit Walkthrough). Every stage reuses the page's own zoom components with the lesson's own
  selection (LessonView), so the lesson and the "Explore freely" views never move each other. All
  numbers come from the live forward pass of the current input; redrawing updates the current step.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import MapCanvas from './MapCanvas.svelte';
	import InputDetail from './InputDetail.svelte';
	import ConvDetail from './ConvDetail.svelte';
	import KernelMaps from './KernelMaps.svelte';
	import ReluDetail from './ReluDetail.svelte';
	import PoolDetail from './PoolDetail.svelte';
	import GrowStage from './GrowStage.svelte';
	import FlattenDetail from './FlattenDetail.svelte';
	import DenseDetail from './DenseDetail.svelte';
	import SoftmaxDetail from './SoftmaxDetail.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { WalkContext, WalkStep } from '$lib/kit/types';
	import { argmaxCell } from './data';
	import { PART_CHANNELS, argminCell, negativeShare } from './lesson';
	import { FISH, HOUSE, SUN, cloneStrokes } from './sketches';
	import { LessonView, type CnnStore } from './store.svelte';

	let { S }: { S: CnnStore } = $props();
	const LV = new LessonView(untrack(() => S));

	let index = $state(0);
	/** the Conv 1-1 channel followed through steps 2–4: fixed when step 2 begins (until then, the strongest one) */
	let c11fixed = $state(-1);
	const c11 = $derived(c11fixed >= 0 ? c11fixed : S.strongestChannel('conv1_1'));
	let padOpen = $state(false);

	/** the layer each step's stage needs in the lesson selection (set by its onEnter hook) */
	const NEEDS: Record<string, string> = {
		pixels: 'input',
		slide: 'conv1_1',
		kernels: 'conv1_1',
		relu: 'relu1_1',
		combine: 'conv1_2',
		pool: 'pool1',
		grow: 'relu2_2',
		dense: 'dense',
		softmax: 'softmax'
	};
	const stageReady = (id: string) => !NEEDS[id] || LV.sel.layer === NEEDS[id];

	const cls = (k: number) => t(S.spec.classes[k]);
	/** a softmax share is never exactly 100% (fp32 rounds 0.99999994 up to 1): say ">99.9%" instead of "100.0%" */
	const pct = (p: number) => (p >= 0.9995 ? '>99.9%' : fmtPct(p));
	const topName = $derived(S.top >= 0 ? cls(S.top) : '');
	const topP = $derived(S.probs && S.top >= 0 ? pct(S.probs[S.top]) : '');
	const topPzh = $derived(topP.startsWith('>') ? `占比超过 ${topP.slice(1)}` : `占 ${topP}`);

	/** move the lesson's cell to the strongest cell of its current map */
	function toPeak() {
		const l = S.layer(LV.sel.layer);
		const d = S.trace?.t[l.id]?.data;
		if (!d) return;
		const p = argmaxCell(d, LV.sel.ch, l.outShape[1], l.outShape[2]);
		LV.setCell(p.y, p.x);
	}

	// dense / softmax steps follow the model's answer while the learner redraws, until they pick a class
	$effect(() => {
		const id = steps[index]?.id;
		const top = S.top;
		if ((id === 'dense' || id === 'softmax') && top >= 0 && !untrack(() => LV.picked) && untrack(() => LV.sel.ch) !== top) LV.set(id, top);
	});

	const steps: WalkStep[] = [
		{
			id: 'pixels',
			title: { en: 'Your drawing becomes 784 numbers', zh: '你的画变成 784 个数' },
			narration: {
				en: 'The model never sees your strokes, only this grid. Your drawing is cropped, scaled so its long side spans 26 pixels, centered, and redrawn as 28 × 28 = 784 numbers: 0 is blank paper, 1 is full ink. Everything that follows is arithmetic on these numbers.',
				zh: '模型看不到你的笔画，只看到这个网格。你的画被裁剪、缩放到长边 26 像素、居中，再重画成 28 × 28 = 784 个数：0 是白纸，1 是满墨。后面的一切都是对这些数的计算。'
			},
			look: {
				en: 'Hover or tap any cell to see its value, its position (row, column; counted from 0 at the top left) and the 5×5 block of numbers around it next to the grid.',
				zh: '悬停或点按任意格子，旁边会显示它的数值、位置（行，列；从左上角的 0 开始数）以及它周围 5×5 区块的数。'
			},
			onEnter: () => LV.set('input', 0, argmaxCell(S.input, 0, 28, 28))
		},
		{
			id: 'slide',
			title: { en: 'One 3×3 kernel slides over it', zh: '一个 3×3 卷积核在上面滑动' },
			narration: {
				en: 'A kernel is 9 learned numbers (weights). The first layer, Conv 1-1, slides it over your drawing one cell at a time. At each stop the 3×3 window is multiplied by the kernel cell by cell, and the 9 products plus one more learned number, the bias, are added up: that is one cell of the 26×26 output map.',
				zh: '卷积核就是 9 个学来的数（权重）。第一层“卷积 1-1”把它在你的画上一格一格地滑动。每停一次，把 3×3 窗口和卷积核逐格相乘，再把 9 个乘积和另一个学来的数（偏置）加起来，就得到 26×26 输出图里的一个格子。'
			},
			look: () => ({
				en: `Kernel ${c11} reacts most to your drawing. Hatched cells are not computed yet. A new cell turns orange where the window covers ink in the kernel’s pattern, blue where it covers the opposite. Afterwards the numbers below show the strongest cell; hover or tap the output map for any other.`,
				zh: `卷积核 ${c11} 对你的画反应最强。带斜线的格子还没算到。窗口盖住的墨迹符合卷积核的模式时，新格子变成橙色；正好相反时变成蓝色。滑完后，下方的数字显示最强的格子；悬停或点按输出图可以查看别的格子。`
			}),
			waitForDone: true,
			linger: 3500,
			onEnter: () => {
				c11fixed = S.strongestChannel('conv1_1');
				LV.set('conv1_1', c11fixed, { y: 0, x: 0 });
			}
		},
		{
			id: 'kernels',
			title: { en: 'Eight kernels make eight feature maps', zh: '8 个卷积核得到 8 张特征图' },
			narration: () => ({
				en: `Conv 1-1 has 8 different kernels, so your one drawing becomes 8 feature maps, also called channels: kernel ${c11} makes channel ${c11}. Orange weights reward ink and blue weights punish it, so each kernel lights up where strokes run in the direction it prefers. Nobody designed these kernels: training found them.`,
				zh: `卷积 1-1 有 8 个不同的卷积核，所以你的一幅画变成 8 张特征图，也叫 8 个通道：卷积核 ${c11} 得到通道 ${c11}。橙色权重奖励墨迹，蓝色权重惩罚墨迹，所以每个卷积核会在笔画走向符合它偏好的地方亮起来。这些卷积核不是人设计的，是训练找到的。`
			}),
			look: () => ({
				en: `Kernel ${c11} (outlined) is the one that just slid. The four small bars on each card show how strongly it responds to a thin stroke at each angle (—, \\, |, /); kernel 4 prefers “/”, kernel 7 prefers “\\”. Click a card for its 9 weights and its map.`,
				zh: `外框高亮的卷积核 ${c11} 就是刚才滑动的那个。每张卡片上的四个小条表示它对各个角度（—、\\、|、/）细笔画的反应有多强；卷积核 4 偏爱“/”，卷积核 7 偏爱“\\”。点一张卡片，查看它的 9 个权重和特征图。`
			}),
			onEnter: () => LV.set('conv1_1', c11)
		},
		{
			id: 'relu',
			title: { en: 'ReLU keeps the positive evidence', zh: 'ReLU 只保留正的证据' },
			narration: () => {
				const d = S.trace?.t.conv1_1.data;
				const p = d ? fmtPct(negativeShare(d, c11, 26 * 26)) : '–';
				return {
					en: `ReLU replaces every negative number with 0 and leaves the positive ones alone. In kernel ${c11}’s map, ${p} of the cells were negative (blue) and are now 0. The next layer only sees where each pattern is present, not where it is absent.`,
					zh: `ReLU 把所有负数换成 0，正数保持不变。在卷积核 ${c11} 的特征图里，${p} 的格子是负数（蓝色），现在都变成了 0。下一层只看到每种模式“在哪里出现”，而不是“在哪里没有”。`
				};
			},
			look: {
				en: 'Before and after ReLU, on the same color scale. The selected cell (here the most negative one) is outlined on both maps and drawn as a dot on the ReLU line, which stays flat at 0 for every negative input. Hover or tap a cell to move it.',
				zh: 'ReLU 之前和之后，用同一个色阶。选中的格子（这里是最负的那个）在两张图上都有方框，并在 ReLU 折线上画成圆点：输入为负时，折线一直平躺在 0。悬停或点按格子可以移动它。'
			},
			onEnter: () => {
				const d = S.trace?.t.conv1_1.data;
				LV.set('relu1_1', c11, d ? argminCell(d, c11, 26, 26) : undefined);
			}
		},
		{
			id: 'combine',
			title: { en: 'Conv 1-2 combines all eight maps', zh: '卷积 1-2 把 8 张图合在一起' },
			narration: {
				en: 'Conv 1-2 works on the 8 maps that came out of ReLU. It also has 8 kernels, but each one is 8 slices deep: one 3×3 slice for each input map. For one output cell it takes the same 3×3 window in all 8 maps, multiplies each by its slice, and adds everything plus one bias: 8 × 9 = 72 products for one number.',
				zh: '卷积 1-2 处理的是 ReLU 之后的 8 张图。它也有 8 个卷积核，但每个都有 8 层切片：每张输入图对应一块 3×3 切片。算一个输出格时，它在全部 8 张图中取同一个 3×3 窗口，各自乘以对应的切片，再全部加起来并加上一个偏置：一个数要用 8 × 9 = 72 次乘法。'
			},
			look: () => {
				const l = LV.sel;
				const d = l.layer === 'conv1_2' && S.model && S.trace ? S.model.convDetail(S.trace, 'conv1_2', l.ch, l.y, l.x) : null;
				const y = d ? fmt3(d.total) : '–';
				return {
					en: `The small maps are the 8 inputs, each with the same 3×3 window outlined. The bars s0 … s7 show how much each map pushed this cell up (orange) or down (blue); with the bias they add up to y = ${y}, cell (${l.y}, ${l.x}) of Conv 1-2’s map ${l.ch}. Click a map to see its 9 products.`,
					zh: `小图是 8 张输入图，每张都框出同一个 3×3 窗口。条形 s0 … s7 表示每张图把这个格子推高（橙色）或压低（蓝色）了多少；加上偏置，合起来就是 y = ${y}，也就是卷积 1-2 第 ${l.ch} 张图的格子 (${l.y}, ${l.x})。点一张图，查看它的 9 个乘积。`
				};
			},
			onEnter: () => LV.set('conv1_2', S.strongestChannel('conv1_2'))
		},
		{
			id: 'pool',
			title: { en: 'Max pooling keeps the strongest response', zh: '最大池化保留最强的响应' },
			narration: {
				en: 'First, Conv 1-2’s maps go through ReLU again (ReLU 1-2). Then max pooling cuts each map into 2×2 blocks and keeps only the largest number of each block, so the 24×24 map becomes 12×12. The network keeps “this pattern was here, roughly” and forgets the exact pixel, so a stroke drawn one pixel to the side changes little.',
				zh: '卷积 1-2 的输出先再经过一次 ReLU（ReLU 1-2）。然后最大池化把每张图切成 2×2 的小块，每块只保留最大的那个数，于是 24×24 的图变成 12×12。网络记住“这种模式大约在这里出现过”，忘掉精确的像素位置，所以笔画偏一个像素，影响也不大。'
			},
			look: {
				en: 'Hatched output cells are not filled yet. The dashed cell inside the window is the largest of its 4 numbers: that one is copied to the output.',
				zh: '带斜线的输出格还没填上。窗口里的虚线格子是 4 个数中最大的那个，它被复制到输出。'
			},
			waitForDone: true,
			linger: 3000,
			onEnter: () => LV.set('pool1', S.strongestChannel('pool1'), { y: 0, x: 0 })
		},
		{
			id: 'grow',
			title: { en: 'Block 2 sees bigger parts', zh: '第二组看到更大的部件' },
			narration: {
				en: 'Block 2 repeats the recipe on the pooled maps: conv, ReLU, conv, ReLU, pool. The maps are now half as wide, so every 3×3 window covers more of the drawing: one Conv 2-2 cell sees a 14×14 patch (its receptive field), half the width of the drawing. Its channels respond to parts, not single strokes.',
				zh: '第二组在池化后的图上重复同样的步骤：卷积、ReLU、卷积、ReLU、池化。图已经缩小一半，所以每个 3×3 窗口覆盖的画面更多：卷积 2-2 的一个格子能看到 14×14 的区域（它的“感受野”），相当于画面宽度的一半。它的通道对“部件”有反应，而不是单个笔画。'
			},
			look: {
				en: 'The nested squares show how much of the drawing one cell of each layer sees. Pick a channel to compare where it fires on your drawing (live) with the test drawings that excite it most (precomputed). A channel fires wherever something like its part appears, even in a drawing of another class.',
				zh: '一层套一层的方框表示每一层的一个格子能看到画面的多大区域。选一个通道，比较它在你的画上亮起的位置（实时）和最能激活它的测试画（预先算好）。只要出现类似的部件，通道就会亮起来，即使画的是别的类别。'
			},
			onEnter: () => {
				const d = S.trace?.t.relu2_2.data;
				let best: number = PART_CHANNELS[1].c;
				if (d) {
					let bv = -Infinity;
					for (const p of PART_CHANNELS)
						for (let i = 0; i < 64; i++)
							if (d[p.c * 64 + i] > bv) {
								bv = d[p.c * 64 + i];
								best = p.c;
							}
				}
				LV.set('relu2_2', best);
			}
		},
		{
			id: 'flatten',
			title: { en: 'Flatten: 16 × 4 × 4 becomes 256 numbers', zh: '展平：16 × 4 × 4 变成 256 个数' },
			narration: {
				en: 'After the second pooling, 16 maps of 4×4 are left: 256 numbers. Flatten lays them in one row, map by map and row by row, so number i comes from map c, row y, column x with i = 16c + 4y + x. Nothing is computed here; only the order changes.',
				zh: '第二次池化后剩下 16 张 4×4 的图，共 256 个数。展平把它们排成一行：一张图接一张图，一行接一行，所以第 i 个数来自第 c 张图的第 y 行第 x 列，i = 16c + 4y + x。这里没有任何计算，只是换了排列方式。'
			},
			look: { en: 'Hover or tap a cell on either side and the same number lights up on the other side; the formula shows its index.', zh: '悬停或点按任意一边的格子，另一边的同一个数会亮起，公式显示它的序号。' }
		},
		{
			id: 'dense',
			title: { en: 'Dense: 256 numbers vote for each class', zh: '全连接层：256 个数为每个类别投票' },
			narration: {
				en: 'Each class has its own 256 weights. Every input is multiplied by its weight: orange products are votes for the class, blue products are votes against. All 256 votes plus a bias give the class its score, called a logit.',
				zh: '每个类别都有自己的 256 个权重。每个输入乘以对应的权重：橙色的乘积是赞成票，蓝色的是反对票。256 票加起来再加上偏置，就是这个类别的分数，叫做 logit。'
			},
			look: () => ({
				en: `The contributions grid (W·x) shows the 256 votes for “${cls(LV.sel.ch)}”. Pick another class to see how its votes differ.`,
				zh: `“贡献 W·x”网格就是“${cls(LV.sel.ch)}”得到的 256 票。换一个类别，看看票数有什么不同。`
			}),
			onEnter: () => LV.set('dense', Math.max(0, S.top))
		},
		{
			id: 'softmax',
			title: { en: 'Softmax turns 10 scores into the answer', zh: 'Softmax 把 10 个分数变成答案' },
			narration: () => ({
				en: `The 10 logits (z in the table) can be any size. Softmax subtracts the largest one (m), raises e ≈ 2.718 to the power of each result, and divides by the total, so all 10 become positive and add up to 100% (p). The model answers “${topName}” with ${topP}: a share among the 10 classes it knows, not a guarantee that it is right.`,
				zh: `10 个 logit（表中的 z）可以是任意大小。Softmax 先减去其中最大的一个（m），再以 e ≈ 2.718 为底对每个结果取幂，最后除以总和，于是 10 个数都变成正数，加起来正好 100%（p）。模型的回答是“${topName}”，${topPzh}：这是它认识的 10 个类别之间的份额，并不保证答对。`
			}),
			look: { en: 'Watch the bars change in three steps: raw scores, then e to the power, then shares of 100%.', zh: '看条形分三步变化：原始分数，取 e 的幂，再变成 100% 的份额。' },
			waitForDone: true,
			linger: 4000,
			onEnter: () => LV.set('softmax', Math.max(0, S.top))
		}
	];

	function peakThenDone(ctx: WalkContext) {
		return () => {
			toPeak();
			ctx.done();
		};
	}
</script>

<Walkthrough {steps} bind:index ready={!!S.trace} label={{ en: 'TinyCNN, step by step', zh: 'TinyCNN 分步课程' }} stageMinHeight={380}>
	{#snippet stage(step, ctx)}
		{#if !S.trace}
			<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
		{:else if stageReady(step.id)}
			<!-- step 7 mixes live views with the precomputed top test drawings, which carry their own badge -->
			<div class="badge"><ProvenanceBadge mode="live" compact precision={S.spec.precision} note={step.id === 'grow' ? { en: 'The sizes, the map and the box are computed live; the top test drawings (marked precomputed) were computed offline with the same weights.', zh: '尺寸、特征图和方框都是实时算出的；标着“预计算”的最强激活测试画是用同一组权重离线算出的。' } : undefined} /></div>
			{#if step.id === 'pixels'}
				<InputDetail {S} view={LV} compact />
			{:else if step.id === 'slide'}
				<ConvDetail {S} view={LV} parts={['slide', 'arith']} autoplay={ctx.shown && !ctx.reduced} interval={14} fill pair onend={peakThenDone(ctx)} title={{ en: 'Watch the kernel slide over your drawing', zh: '看卷积核在你的画上滑动' }} />
			{:else if step.id === 'kernels'}
				<KernelMaps {S} view={LV} />
			{:else if step.id === 'relu'}
				<ReluDetail {S} view={LV} parts={['maps', 'formula']} />
			{:else if step.id === 'combine'}
				<ConvDetail {S} view={LV} parts={['channels', 'arith']} />
			{:else if step.id === 'pool'}
				<PoolDetail {S} view={LV} parts={['slide', 'formula']} autoplay={ctx.shown && !ctx.reduced} interval={70} fill pair onend={peakThenDone(ctx)} title={{ en: 'Watch the 2×2 window slide', zh: '看 2×2 窗口滑动' }} />
			{:else if step.id === 'grow'}
				<GrowStage {S} view={LV} />
			{:else if step.id === 'flatten'}
				<FlattenDetail {S} parts={['maps', 'formula']} />
			{:else if step.id === 'dense'}
				<DenseDetail {S} view={LV} parts={['picker', 'grids', 'formula']} />
			{:else if step.id === 'softmax'}
				<SoftmaxDetail {S} view={LV} parts={['player', 'table']} autoplay={ctx.shown && !ctx.reduced} interval={1600} onend={ctx.done} />
			{/if}
		{/if}
	{/snippet}

	{#snippet aside()}
		<div class="you">
			<div class="you-h">
				<h4>{t({ en: 'Your drawing', zh: '你的画' })}</h4>
				{#if S.trace}
					<span class="ans"
						><span class="sw" style="background: {categorical(S.top, themeState.resolved)}"></span><strong>{topName}</strong>
						<span class="num">{topP}</span></span
					>
				{/if}
			</div>
			<div class="you-row">
				<MapCanvas data={S.input} rows={28} cols={28} kind="sequential" domain={[0, 1]} size={56} alt={t({ en: 'What the model sees', zh: '模型看到的' })} />
				<p class="hint">
					{#if S.source.kind === 'preset'}
						{t({ en: 'A preset drawing is loaded.', zh: '当前是一幅预设画。' })}
					{:else if S.source.kind === 'photo'}
						{t({ en: 'Your photo is loaded.', zh: '当前是你上传的照片。' })}
					{/if}
					{t({ en: 'Redraw at any time: the step you are on updates at once.', zh: '随时可以重画：你所在的这一步会立刻更新。' })}
				</p>
			</div>
			{#if padOpen}
				<DrawPad
					bind:strokes={S.strokes}
					brush={0.055}
					size={190}
					showBrush={false}
					presets={[
						{ name: { en: 'House', zh: '房子' }, strokes: cloneStrokes(HOUSE) },
						{ name: { en: 'Fish', zh: '鱼' }, strokes: cloneStrokes(FISH) },
						{ name: { en: 'Sun', zh: '太阳' }, strokes: cloneStrokes(SUN) }
					]}
					onchange={(s) => S.setStrokes(s)}
					ondraw={(s) => S.drawing(s)}
					label={t({ en: 'Lesson drawing pad', zh: '课程画板' })}
				/>
			{/if}
			<button type="button" class="btn small" aria-expanded={padOpen} onclick={() => (padOpen = !padOpen)}>
				<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M11 2.5 13.5 5 6 12.5H3.5V10z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" /></svg>
				{padOpen ? t({ en: 'Hide the pad', zh: '收起画板' }) : t({ en: 'Redraw here', zh: '在这里重画' })}
			</button>
		</div>
	{/snippet}

	{#snippet end()}
		<div class="fin">
			<p>
				<strong>{t({ en: 'That is the whole forward pass.', zh: '这就是完整的前向传播。' })}</strong>
				{t({ en: 'Draw something else and play the lesson again, or explore every layer at once below.', zh: '画点别的再播放一遍，或者在下面同时查看每一层。' })}
			</p>
			<a class="btn primary" href="#explore">{t({ en: 'Explore freely', zh: '自由探索' })} ↓</a>
		</div>
	{/snippet}
</Walkthrough>
<p class="foot muted">
	{t({
		en: 'Every number in the lesson is computed in your browser from your drawing, except the top test drawings in step 7, which were computed offline with the same weights. The lesson keeps its own selection, so the views below stay where you left them.',
		zh: '课程中的每个数都由你的画在浏览器中实时算出，只有第 7 步里最强激活的测试画是用同一组权重离线算好的。课程使用自己的选中状态，所以下面的视图会停在你离开时的位置。'
	})}
</p>

<style>
	.badge {
		display: flex;
		justify-content: flex-end;
		margin: -4px 0 6px;
	}
	.you {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.you :global(.pad) {
		margin-top: 2px;
	}
	.you-h {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 4px 12px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.ans {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-sm);
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
	}
	.you-row {
		display: flex;
		gap: 10px;
		align-items: center;
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.small {
		justify-self: start;
		gap: 6px;
		min-height: 32px;
		padding: 0 12px;
		font-size: var(--fs-sm);
	}
	.you :global(.tools .btn) {
		min-height: 30px;
		padding: 0 8px;
		font-size: var(--fs-xs);
	}
	.fin {
		display: grid;
		gap: 8px;
		justify-items: start;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--live-soft);
	}
	.fin p {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.foot {
		margin: 8px 0 0;
		font-size: var(--fs-xs);
	}
</style>
