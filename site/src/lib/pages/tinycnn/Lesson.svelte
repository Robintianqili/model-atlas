<!--
  The TinyCNN guided lesson: ten steps from the learner's drawing to the answer, one focused view at
  a time (kit Walkthrough, laid out as the lesson app: redesign spec §5). Every stage reuses the page's
  own zoom components with the lesson's own selection (LessonView), so the lesson and the Explore view
  never move each other. All numbers come from the live forward pass of the current input; redrawing
  (the input chip in the stage header opens the input sheet) updates the current step. Chapters:
  Pixels 1 · Convolution 4 · Going deeper 2 · Decision 3. The cover (answer first) is Cover.svelte; the
  end card's recap names three real numbers of this run.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import Cover from './Cover.svelte';
	import { fmt3, fmtInt, fmtPct } from '$lib/kit/format';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { StageProv, WalkContext, WalkPredict, WalkStep } from '$lib/kit/types';
	import { argmaxCell } from './data';
	import { PART_CHANNELS, argminCell, negativeShare } from './lesson';
	import { LessonView, type CnnStore } from './store.svelte';

	let { S }: { S: CnnStore } = $props();
	const LV = new LessonView(untrack(() => S));

	let index = $state(0);
	// the stage views are a separate chunk (stages.ts): they render only once the model has run
	let st = $state.raw<typeof import('./stages') | null>(null);
	onMount(() => {
		import('./stages').then((m) => (st = m));
	});
	/** the Conv 1-1 channel followed through steps 2–4: fixed when step 2 begins (until then, the strongest one) */
	let c11fixed = $state(-1);
	const c11 = $derived(c11fixed >= 0 ? c11fixed : S.strongestChannel('conv1_1'));

	/** the stage header's provenance label: every stage is computed live from the learner's input */
	const live: StageProv = { mode: 'live', precision: untrack(() => S.spec.precision), short: { en: '· computed on your drawing', zh: '· 用你的画现场计算' } };
	const chapters: { title: Text; count: number }[] = [
		{ title: { en: 'Pixels', zh: '像素' }, count: 1 },
		{ title: { en: 'Convolution', zh: '卷积' }, count: 4 },
		{ title: { en: 'Going deeper', zh: '更深的层' }, count: 2 },
		{ title: { en: 'Decision', zh: '做出判断' }, count: 3 }
	];

	/** step 2: the strongest cell of the followed Conv 1-1 map, its arithmetic, and the cell next to it */
	const peak = $derived.by(() => {
		const tr = S.trace;
		const m = S.model;
		if (!tr || !m) return null;
		const [, H, W] = S.layer('conv1_1').outShape;
		const p = argmaxCell(tr.t.conv1_1.data, c11, H, W);
		const d = m.convDetail(tr, 'conv1_1', c11, p.y, p.x);
		const nx = p.x + 1 < W ? p.x + 1 : p.x - 1;
		const nv = tr.t.conv1_1.data[c11 * H * W + p.y * W + nx];
		return { y: p.y, x: p.x, sum: d.total - d.bias, bias: d.bias, out: d.traced, nx, nv, right: nx > p.x };
	});
	/** a number chip in a Look line (spec §3.4): a bound number that moves the stage's ring to its cell (kit ChipText marks) */
	const chip = (ref: string, text: string) => `\uE000${ref}\uE001${text}\uE002`;

	/** step 2's "Your turn": does the output go up or down when the window moves one cell? (answered from the real map) */
	const slideTurn = (): WalkPredict | undefined => {
		const p = peak;
		if (!p) return undefined;
		const answer = p.nv > p.out ? 0 : p.nv < p.out ? 1 : 2;
		const dir = p.right ? { en: 'right', zh: '右' } : { en: 'left', zh: '左' };
		const word = [
			{ en: 'Up', zh: '变大' },
			{ en: 'Down', zh: '变小' },
			{ en: 'It stays the same', zh: '不变' }
		][answer];
		return {
			gate: false,
			question: {
				en: `The window moves one cell ${dir.en}, from (${p.y}, ${p.x}) to (${p.y}, ${p.nx}). Will the output go up or down?`,
				zh: `窗口向${dir.zh}移动一格，从 (${p.y}, ${p.x}) 到 (${p.y}, ${p.nx})。输出会变大还是变小？`
			},
			options: [
				{ en: 'Up', zh: '变大' },
				{ en: 'Down', zh: '变小' },
				{ en: 'It stays the same', zh: '不变' }
			],
			answer,
			explain: {
				en: `${word.en}: (${p.y}, ${p.nx}) gives ${fmt3(p.nv)}, against ${fmt3(p.out)} at (${p.y}, ${p.x}). (${p.y}, ${p.x}) is the strongest cell of the whole map, the window that matches kernel ${c11} best, so no other cell can be higher. Hover or tap (${p.y}, ${p.nx}) on the output map to see its 9 products.`,
				zh: `${word.zh}：(${p.y}, ${p.nx}) 得到 ${fmt3(p.nv)}，而 (${p.y}, ${p.x}) 是 ${fmt3(p.out)}。(${p.y}, ${p.x}) 是整张图里最强的格子，也就是与卷积核 ${c11} 最匹配的窗口，所以别的格子都不可能更高。在输出图上悬停或点按 (${p.y}, ${p.nx})，可以看它的 9 个乘积。`
			}
		};
	};

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
	/** inked pixels of the model input (for the end card's recap) */
	const inked = $derived(S.input.reduce((n, v) => n + (v > 0 ? 1 : 0), 0));

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
			prov: live,
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
			prov: live,
			title: { en: 'One 3×3 kernel slides over it', zh: '一个 3×3 卷积核在上面滑动' },
			narration: {
				en: 'A kernel is 9 learned numbers, its weights. The first layer, Conv 1-1, slides it over your drawing one cell at a time. At each stop the 3×3 window is multiplied by the kernel cell by cell, and the 9 products plus one more learned number, the bias, add up to one cell of the 26×26 map.',
				zh: '卷积核就是 9 个学来的数，也就是它的权重。第一层“卷积 1-1”把它在你的画上一格一格地滑动。每停一次，3×3 窗口和卷积核逐格相乘，9 个乘积再加上另一个学来的数（偏置），就得到 26×26 特征图里的一个格子。'
			},
			look: () => {
				const p = peak;
				if (!p)
					return {
						en: `Kernel ${c11} reacts most to your drawing. Hatched cells are not computed yet; orange cells match the kernel’s pattern, blue cells the opposite.`,
						zh: `卷积核 ${c11} 对你的画反应最强。带斜线的格子还没算到；橙色格子符合卷积核的模式，蓝色格子正好相反。`
					};
				return {
					en: `Kernel ${c11} reacts most at (${p.y}, ${p.x}): 9 products ${fmt3(p.sum)} + bias ${fmt3(p.bias)} = ${fmt3(p.out)}. Hatched cells are not computed yet; orange cells match the kernel’s pattern, blue cells the opposite.`,
					zh: `卷积核 ${c11} 在 (${p.y}, ${p.x}) 反应最强：9 个乘积之和 ${fmt3(p.sum)} + 偏置 ${fmt3(p.bias)} = ${fmt3(p.out)}。带斜线的格子还没算到；橙色格子符合卷积核的模式，蓝色格子正好相反。`
				};
			},
			chips: {
				look: () => {
					const p = peak;
					if (!p) return { en: `Kernel ${c11} reacts most to your drawing.`, zh: `卷积核 ${c11} 对你的画反应最强。` };
					return {
						en: `Kernel ${c11} reacts most at ${chip('peak', `(${p.y}, ${p.x})`)}: 9 products ${fmt3(p.sum)} + bias ${fmt3(p.bias)} = ${chip('peak', fmt3(p.out))}. Hatched cells are not computed yet; orange cells match the kernel’s pattern, blue cells the opposite.`,
						zh: `卷积核 ${c11} 在 ${chip('peak', `(${p.y}, ${p.x})`)} 反应最强：9 个乘积之和 ${fmt3(p.sum)} + 偏置 ${fmt3(p.bias)} = ${chip('peak', fmt3(p.out))}。带斜线的格子还没算到；橙色格子符合卷积核的模式，蓝色格子正好相反。`
					};
				}
			},
			// a chip moves the lesson's window to the strongest cell (the output map rings it, the arithmetic shows it)
			onref: (ref) => {
				if (ref === 'peak' && peak) LV.setCell(peak.y, peak.x);
			},
			predict: slideTurn,
			why: {
				q: { en: 'Why is the map 26 × 26, not 28 × 28?', zh: '为什么特征图是 26 × 26，而不是 28 × 28？' },
				a: {
					en: 'The 3×3 window has to fit inside the 28 × 28 drawing, so its top-left corner can sit at 26 places across (columns 0 to 25) and 26 down: 28 − 3 + 1 = 26. TinyCNN adds no padding around the drawing, so every convolution trims one pixel from each border.',
					zh: '3×3 窗口必须完整地落在 28 × 28 的画里，所以它的左上角横向只有 26 个位置（第 0 到 25 列），纵向也是 26 个：28 − 3 + 1 = 26。TinyCNN 不在画的四周补零（不做填充），所以每做一次卷积，四边各少一个像素。'
				}
			},
			animated: true,
			waitForDone: true,
			linger: 3500,
			onEnter: () => {
				c11fixed = S.strongestChannel('conv1_1');
				LV.set('conv1_1', c11fixed, { y: 0, x: 0 });
			}
		},
		{
			id: 'kernels',
			prov: live,
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
			prov: live,
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
			prov: live,
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
			prov: live,
			title: { en: 'Max pooling keeps the strongest response', zh: '最大池化保留最强的响应' },
			narration: {
				en: 'First, Conv 1-2’s maps go through ReLU again (ReLU 1-2). Then max pooling cuts each map into 2×2 blocks and keeps only the largest number of each block, so the 24×24 map becomes 12×12. The network keeps “this pattern was here, roughly” and forgets the exact pixel, so a stroke drawn one pixel to the side changes little.',
				zh: '卷积 1-2 的输出先再经过一次 ReLU（ReLU 1-2）。然后最大池化把每张图切成 2×2 的小块，每块只保留最大的那个数，于是 24×24 的图变成 12×12。网络记住“这种模式大约在这里出现过”，忘掉精确的像素位置，所以笔画偏一个像素，影响也不大。'
			},
			look: {
				en: 'Hatched output cells are not filled yet. The dashed cell inside the window is the largest of its 4 numbers: that one is copied to the output.',
				zh: '带斜线的输出格还没填上。窗口里的虚线格子是 4 个数中最大的那个，它被复制到输出。'
			},
			animated: true,
			waitForDone: true,
			linger: 3000,
			onEnter: () => LV.set('pool1', S.strongestChannel('pool1'), { y: 0, x: 0 })
		},
		{
			id: 'grow',
			prov: { ...live, note: { en: 'The sizes, the map and the box are computed live; the top test drawings (marked precomputed) were computed offline with the same weights.', zh: '尺寸、特征图和方框都是实时算出的；标着“预计算”的最强激活测试画是用同一组权重离线算出的。' } },
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
			prov: live,
			title: { en: 'Flatten: 16 × 4 × 4 becomes 256 numbers', zh: '展平：16 × 4 × 4 变成 256 个数' },
			narration: {
				en: 'After the second pooling, 16 maps of 4×4 are left: 256 numbers. Flatten lays them in one row, map by map and row by row, so number i comes from map c, row y, column x with i = 16c + 4y + x. Nothing is computed here; only the order changes.',
				zh: '第二次池化后剩下 16 张 4×4 的图，共 256 个数。展平把它们排成一行：一张图接一张图，一行接一行，所以第 i 个数来自第 c 张图的第 y 行第 x 列，i = 16c + 4y + x。这里没有任何计算，只是换了排列方式。'
			},
			look: { en: 'Hover or tap a cell on either side and the same number lights up on the other side; the formula shows its index.', zh: '悬停或点按任意一边的格子，另一边的同一个数会亮起，公式显示它的序号。' }
		},
		{
			id: 'dense',
			prov: live,
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
			prov: live,
			title: { en: 'Softmax turns 10 scores into the answer', zh: 'Softmax 把 10 个分数变成答案' },
			narration: () => ({
				en: `The 10 logits (z in the table) can be any size. Softmax subtracts the largest one (m), raises e ≈ 2.718 to the power of each result, and divides by the total, so all 10 become positive and add up to 100% (p). The model answers “${topName}” with ${topP}: a share among the 10 classes it knows, not a guarantee that it is right.`,
				zh: `10 个 logit（表中的 z）可以是任意大小。Softmax 先减去其中最大的一个（m），再以 e ≈ 2.718 为底对每个结果取幂，最后除以总和，于是 10 个数都变成正数，加起来正好 100%（p）。模型的回答是“${topName}”，${topPzh}：这是它认识的 10 个类别之间的份额，并不保证答对。`
			}),
			look: { en: 'Watch the bars change in three steps: raw scores, then e to the power, then shares of 100%.', zh: '看条形分三步变化：原始分数，取 e 的幂，再变成 100% 的份额。' },
			animated: true,
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

<Walkthrough {steps} bind:index ready={!!S.trace} label={{ en: 'Follow your drawing through a real CNN', zh: '跟着你的画走过一个真实的 CNN' }} {chapters} stageMinHeight={380}>
	{#snippet cover(ctx)}<Cover {S} compact={ctx.density === 'compact'} />{/snippet}

	{#snippet stage(step, ctx)}
		{#if !S.trace || !st}
			<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
		{:else if stageReady(step.id)}
			{#if step.id === 'pixels'}
				<st.InputDetail {S} view={LV} compact />
			{:else if step.id === 'slide'}
				<st.ConvDetail {S} view={LV} parts={['slide', 'arith']} compact={ctx.density === 'compact'} autoplay={ctx.shown && !ctx.reduced} interval={14} fill pair onend={peakThenDone(ctx)} title={{ en: 'Watch the kernel slide over your drawing', zh: '看卷积核在你的画上滑动' }} />
			{:else if step.id === 'kernels'}
				<st.KernelMaps {S} view={LV} />
			{:else if step.id === 'relu'}
				<st.ReluDetail {S} view={LV} parts={['maps', 'formula']} />
			{:else if step.id === 'combine'}
				<st.ConvDetail {S} view={LV} parts={['channels', 'arith']} compact={ctx.density === 'compact'} />
			{:else if step.id === 'pool'}
				<st.PoolDetail {S} view={LV} parts={['slide', 'formula']} autoplay={ctx.shown && !ctx.reduced} interval={70} fill pair onend={peakThenDone(ctx)} title={{ en: 'Watch the 2×2 window slide', zh: '看 2×2 窗口滑动' }} />
			{:else if step.id === 'grow'}
				<st.GrowStage {S} view={LV} />
			{:else if step.id === 'flatten'}
				<st.FlattenDetail {S} parts={['maps', 'formula']} />
			{:else if step.id === 'dense'}
				<st.DenseDetail {S} view={LV} parts={['picker', 'grids', 'formula']} />
			{:else if step.id === 'softmax'}
				<st.SoftmaxDetail {S} view={LV} parts={['player', 'table']} autoplay={ctx.shown && !ctx.reduced} interval={1600} onend={ctx.done} />
			{/if}
		{/if}
	{/snippet}

	{#snippet recap()}
		<ul class="recap">
			<li>
				{t({
					en: `Your drawing became ${fmtInt(S.input.length)} numbers; ${fmtInt(inked)} of them carry ink.`,
					zh: `你的画变成了 ${fmtInt(S.input.length)} 个数，其中 ${fmtInt(inked)} 个带墨迹。`
				})}
			</li>
			{#if peak}
				<li>
					{t({
						en: `Kernel ${c11} reacted most at (${peak.y}, ${peak.x}): 9 products ${fmt3(peak.sum)} + bias ${fmt3(peak.bias)} = ${fmt3(peak.out)}.`,
						zh: `卷积核 ${c11} 在 (${peak.y}, ${peak.x}) 反应最强：9 个乘积之和 ${fmt3(peak.sum)} + 偏置 ${fmt3(peak.bias)} = ${fmt3(peak.out)}。`
					})}
				</li>
			{/if}
			{#if S.top >= 0}
				<li>
					{t({
						en: `Of the ${S.spec.classes.length} scores, “${topName}” won with ${topP}: a share among the classes it knows, not a guarantee.`,
						zh: `在 ${S.spec.classes.length} 个分数里，“${topName}”以 ${topP} 胜出：这是它认识的类别之间的份额，不是答对的保证。`
					})}
				</li>
			{/if}
		</ul>
	{/snippet}
</Walkthrough>

<style>
	.recap {
		display: grid;
		gap: 6px;
		margin: 0;
		padding-left: 1.2em;
		color: var(--ink-2);
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
</style>
