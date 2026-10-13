// lora: the guided lesson (ten steps; the concept page's primary path). Every number is live: the counts and memory
// are computed from the learner's rank (data.ts), the losses and matrices come from the lora and ranks processes (the
// kit's autograd and Adam on the toy layer).
//   count -> scale -> predict (B = 0) -> start -> first update -> train (play) -> spectrum -> task (smallest rank)
//   -> memory -> limits
import type { At } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { D, GOAL, GPT2, GPT2_PARAMS, LAYERS, MB, MIN_RANK, N, SPECTRUM, floorOf, gpt2Lora, loraCount, memoryOf, metricsOf, pctText, rankAt, rankOf, shareOf } from './data.ts';
import { STARTS } from './adapters/rank/adapter.ts';

const first = (at: At) => at.tape?.frame(0).metrics ?? {};
const ATTN = LAYERS.find((l) => l.id === 'gpt2-attn')!;
/** the adapted matrices of GPT-2 small: query, key, value, output and the two MLP layers of every block */
const PER_BLOCK = 6;
const MATS = PER_BLOCK * GPT2.layers;
const mb = (b: number) => Math.round(MB(b));

const lesson = defineLesson({
	title: { en: 'LoRA, step by step', zh: 'LoRA，分步课程' },
	summary: {
		en: 'Count what LoRA trains, watch B grow from zero on a real layer until it learns the change a new task needs, find the smallest rank that does the job, and see where LoRA saves less than it seems.',
		zh: '数一数 LoRA 训练多少个数，看 B 在一个真实的层上从零长起来，直到学会新任务需要的改变，找出够用的最小秩，再看 LoRA 在哪里省得没有看上去那么多。'
	},
	fixtures: STARTS.map((s) => ({ id: s.id, label: s.label, raw: s.raw })),
	steps: [
		{
			id: 'count',
			title: { en: 'What LoRA trains: two thin matrices', zh: 'LoRA 训练什么：两个细长的矩阵' },
			source: 'fit',
			at: { phase: 'start' },
			view: 'lora/counter',
			provenance: 'live',
			terms: ['fine-tuning', 'frozen-layer', 'lora', 'matrix-rank'],
			narration: {
				en: 'Fine-tuning adapts a trained model to a new task. Full fine-tuning trains all {full} weights of this {d} × {d} layer. LoRA freezes them as W0 and trains two thin matrices instead, B ({d} × r) and A (r × {d}), where r, the rank, is a small number you choose. At r = {r} that is r × ({d} + {d}) = {lora} numbers.',
				zh: '微调就是让一个训练好的模型适应新任务。全量微调要训练这个 {d} × {d} 层的全部 {full} 个权重。LoRA 把它们冻结为 W0，改为训练两个细长的矩阵：B（{d} × r）和 A（r × {d}），其中 r 叫作秩，是一个由你来选的小整数。r = {r} 时，一共是 r × ({d} + {d}) = {lora} 个数。',
				vars: (rt, at) => {
					const r = rankAt({ rt, at });
					return { full: D * D, d: D, r, lora: loraCount(r, D, D) };
				}
			},
			look: { en: 'One square per number: compare the frozen squares of W0 with the trainable squares of B and A.', zh: '每个方块是一个数：比较 W0 中冻结的方块和 B、A 中可训练的方块。' }
		},
		{
			id: 'scale',
			title: { en: 'Bigger layers, bigger savings', zh: '层越大，省得越多' },
			source: 'fit',
			at: { phase: 'start' },
			view: 'lora/counter',
			provenance: 'live',
			terms: ['parameter'],
			narration: {
				en: 'A layer with d_in inputs and d_out outputs has d_in × d_out weights, but LoRA trains only r × (d_in + d_out): a sum grows much more slowly than a product. For one of GPT-2 small\'s {shape} attention projections, rank {r} trains {lora} of {full} numbers, {share}. Adapting {per} matrices in each of its {blocks} blocks, {mats} in all, trains {gpt} numbers, {gptShare} of its {gptFull}.',
				zh: '一个有 d_in 个输入、d_out 个输出的层有 d_in × d_out 个权重，而 LoRA 只训练 r × (d_in + d_out) 个：和比积增长得慢得多。对 GPT-2 small 的一个 {shape} 注意力投影，秩 {r} 只训练 {full} 个数中的 {lora} 个，占 {share}。在它 {blocks} 个块的每一块里适配 {per} 个矩阵，共 {mats} 个，一共训练 {gpt} 个数，占它全部 {gptFull} 个参数的 {gptShare}。',
				vars: (rt, at) => {
					const r = rankAt({ rt, at });
					const lora = loraCount(r, ATTN.dIn, ATTN.dOut);
					const full = ATTN.dIn * ATTN.dOut;
					return { shape: `${ATTN.dOut} × ${ATTN.dIn}`, r, lora, full, share: pctText(lora / full), per: PER_BLOCK, blocks: GPT2.layers, mats: MATS, gpt: gpt2Lora(r), gptShare: pctText(gpt2Lora(r) / GPT2_PARAMS), gptFull: GPT2_PARAMS };
				}
			},
			look: { en: 'The share column of the table: it shrinks as the layers grow.', zh: '表格中的占比一列：层越大，它越小。' }
		},
		{
			id: 'predict',
			title: { en: 'Before the first update', zh: '第一次更新之前' },
			source: 'fit',
			at: { phase: 'start' },
			view: 'lora/matrices',
			provenance: 'live',
			predict: {
				id: 'lora-step0',
				question: {
					en: 'B starts as all zeros and A as small random numbers. Before any training, what does the adapted layer W0 + (α/r)·B·A compute?',
					zh: 'B 一开始全是零，A 是一些小的随机数。在任何训练之前，适配后的层 W0 + (α/r)·B·A 算出的是什么？'
				},
				options: [
					{ en: 'Exactly what the base layer W0 computes', zh: '和基础层 W0 算出的完全一样' },
					{ en: 'W0 plus some random noise from A', zh: 'W0 再加上来自 A 的一些随机噪声' },
					{ en: 'All zeros', zh: '全是零' }
				],
				answer: 0,
				explain: {
					en: 'B·A is a matrix times zero, so it is zero, and W = W0. The adapted model starts exactly as the base model, with the base model\'s loss on the new task; training then grows B away from zero.',
					zh: 'B·A 是一个矩阵乘以零，结果就是零，所以 W = W0。适配后的模型恰好从基础模型出发，在新任务上的损失也和基础模型一样；之后训练才让 B 离开零。'
				}
			},
			narration: {
				en: 'LoRA adds the product B·A to the frozen weights, scaled by a fixed number α divided by the rank: α/r = {alpha}/{r} = {scale}. So W = W0 + (α/r)·B·A, with α = {alpha} for every rank. Answer first, then look at the matrices.',
				zh: 'LoRA 把乘积 B·A 乘上一个缩放系数后加到冻结的权重上，这个系数是固定的数 α 除以秩：α/r = {alpha}/{r} = {scale}。于是 W = W0 + (α/r)·B·A，每个秩都用 α = {alpha}。先回答，再看这些矩阵。',
				vars: (rt, at) => {
					const m = metricsOf(at);
					return { alpha: m.alpha, r: rankAt({ rt, at }), scale: m.scale };
				}
			}
		},
		{
			id: 'start',
			title: { en: 'Step 0 is the base model', zh: '第 0 步就是基础模型' },
			source: 'fit',
			at: { phase: 'start' },
			view: 'lora/matrices',
			provenance: 'live',
			terms: ['loss', 'mse'],
			narration: {
				en: 'With B = 0 the change ΔW = (α/r)·B·A is all zeros, so W is exactly W0. The new task wants, for {n} inputs, the outputs of a changed layer W0 + ΔW*: ΔW* is the change LoRA has to learn. The layer\'s loss, the mean squared error of its {d} outputs over those inputs, is {loss}: the base model\'s loss, where every LoRA run on this page starts.',
				zh: 'B = 0 时，改变量 ΔW = (α/r)·B·A 全是零，所以 W 恰好等于 W0。新任务要求：对 {n} 个输入，给出一个改动过的层 W0 + ΔW* 的输出；ΔW* 就是 LoRA 要学会的改变量。这个层的损失（它在这些输入上 {d} 个输出的均方误差）是 {loss}：这是基础模型的损失，也是本页每次 LoRA 运行的起点。',
				vars: (_rt, at) => ({ d: D, n: N, loss: metricsOf(at).loss })
			},
			look: { en: 'B and ΔW are gray: every number in them is 0.', zh: 'B 和 ΔW 是灰色的：里面每个数都是 0。' }
		},
		{
			id: 'first',
			title: { en: 'The first update moves only B', zh: '第一次更新只动 B' },
			source: 'fit',
			at: { phase: 'step', step: 1 },
			view: 'lora/matrices',
			provenance: 'live',
			terms: ['gradient', 'optimizer'],
			narration: {
				en: 'A\'s gradient is a product that contains B, so while B = 0 it is exactly {gA0}, and the first Adam update leaves A unchanged: its largest change is {dA}. B\'s gradient is not zero (its size is {gB0}), so each number of B moves, by up to {dB}, about the learning rate {lr}. The loss falls from {base|n4} to {loss|n4}.',
				zh: 'A 的梯度是一个含有 B 的乘积，所以只要 B = 0，它就恰好是 {gA0}，第一次 Adam 更新不会改变 A：A 最大的变化是 {dA}。B 的梯度不是零（大小为 {gB0}），所以 B 的每个数都会移动，最多移动 {dB}，大约等于学习率 {lr}。损失从 {base|n4} 降到 {loss|n4}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					const f0 = first(at);
					return { gA0: f0.gA, gB0: f0.gB, dA: m.dA, dB: m.dB, lr: String(m.lr), base: f0.loss, loss: m.loss };
				}
			},
			look: { en: 'B now has color; A looks exactly as it did at step 0.', zh: 'B 现在有了颜色；A 和第 0 步时一模一样。' }
		},
		{
			id: 'train',
			title: { en: 'Train: B and A learn the change', zh: '训练：B 和 A 学会改变量' },
			source: 'fit',
			at: 'last',
			view: { key: 'lora/matrices', props: () => ({ mode: 'target' }) },
			play: { from: 'first', to: 'last', interval: 120, unit: { en: 'update', zh: '次更新' } },
			provenance: 'live',
			narration: {
				en: 'Adam repeats the update {n} times and trains only the {lora} numbers of B and A; W0 never changes. The loss falls from {base} to {loss}. The loss measures what is still missing, ΔW* − ΔW in the third panel, where ΔW* is the change the new task needs: ΔW now captures {covered} of it.',
				zh: 'Adam 把更新重复 {n} 次，只训练 B 和 A 的 {lora} 个数；W0 始终不变。损失从 {base} 降到 {loss}。损失衡量的是还缺的部分，也就是第三个面板里的 ΔW* − ΔW，其中 ΔW* 是新任务需要的改变量：ΔW 已经学到了其中的 {covered}。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { n: m.step, lora: m.trainable, base: m.base, loss: m.loss, covered: pctText(1 - m.loss / m.base) };
				}
			},
			look: { en: 'Compare the first two panels: the strongest patterns of ΔW* show up in ΔW.', zh: '比较前两个面板：ΔW* 中最强的图案也出现在 ΔW 里。' }
		},
		{
			id: 'spectrum',
			title: { en: 'Why a few numbers are enough', zh: '为什么少量的数就够了' },
			source: 'fit',
			at: 'last',
			view: 'lora/spectrum',
			provenance: 'live',
			terms: ['matrix-rank'],
			narration: {
				en: 'Any {d} × {d} change splits into {d} directions, each with a strength. The change this task needs has two strong ones, {s1} and {s2}, and a weak tail. An adapter of rank r can learn at most r directions; at r = {r} they hold {share} of the change (each strength counted squared), so no amount of training gets the loss below {floor}. Your run reached {loss}, {gap} above it.',
				zh: '任何 {d} × {d} 的改变量都能拆成 {d} 个方向，每个方向有一个强度。这个任务需要的改变量有两个强方向，强度为 {s1} 和 {s2}，后面是一串很弱的方向。秩为 r 的适配器最多只能学 r 个方向；r = {r} 时它们占改变量的 {share}（按强度的平方计），所以无论训练多久，损失都不会低于 {floor}。你的运行降到了 {loss}，只比它高 {gap}。',
				vars: (rt, at) => {
					const r = rankAt({ rt, at });
					return { d: D, s1: String(SPECTRUM[0]), s2: String(SPECTRUM[1]), r, share: pctText(shareOf(r)), floor: floorOf(r), loss: metricsOf(at).loss, gap: metricsOf(at).loss - floorOf(r) };
				}
			},
			look: { en: 'The filled bars are the directions your rank can learn; the dashed ones stay out of reach.', zh: '实心的柱子是你的秩能学到的方向；虚线的柱子够不着。' }
		},
		{
			id: 'task',
			title: { en: 'Find the smallest rank that does the job', zh: '找出够用的最小秩' },
			source: 'fit',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: (rt, at) => {
					const r = rankAt({ rt, at });
					return { log: true, refs: [{ y: GOAL, label: { en: 'goal', zh: '目标' } }, { y: floorOf(r), label: { en: `best for rank ${r}`, zh: `秩 ${r} 的最低值` } }] };
				}
			},
			provenance: 'live',
			task: {
				prompt: { en: 'Guess first, then set the rank to the smallest r whose trained loss ends below the goal line.', zh: '先猜一猜，再把秩调到训练后损失能低于目标线的最小 r。' },
				done: (rt) => rankOf(rt.input) === MIN_RANK,
				hint: {
					en: 'The spectrum step tells you how low each rank can get. A rank above the answer also gets under the goal, but trains more numbers than it needs.',
					zh: '上一步的方向强度告诉你每个秩最低能降到多少。比答案大的秩也能低于目标，但要训练的数比需要的多。'
				},
				waiting: { en: 'Not yet: either the loss stays above the goal, or a smaller rank would also do.', zh: '还没有：要么损失还在目标之上，要么更小的秩也够用。' },
				doneText: { en: 'Done: this is the smallest rank that gets under the goal.', zh: '完成：这就是能低于目标的最小秩。' }
			},
			narration: {
				en: 'After {n} updates your rank {r} ends at a loss of {loss}, {verdict} the goal of {goal}. The dashed lines in the stage mark the goal and the lowest loss rank {r} can reach, {floor}; the run ends {gap} above that. A smaller rank trains fewer numbers, so the best choice is the smallest one that still gets under the goal.',
				zh: '更新 {n} 次之后，你的秩 {r} 停在损失 {loss}，{verdict}目标 {goal}。图中的虚线标出了目标，以及秩 {r} 能达到的最低损失 {floor}；运行结束时比它高 {gap}。秩越小，要训练的数越少，所以最好的选择是仍能低于目标的最小秩。',
				vars: (rt, at) => {
					const m = metricsOf(at);
					const r = rankAt({ rt, at });
					return { n: m.step, r, loss: m.loss, goal: String(GOAL), floor: floorOf(r), gap: m.loss - floorOf(r), verdict: Number.isFinite(m.loss) ? pick(m.loss < GOAL, { en: 'below', zh: '低于' }, { en: 'still above', zh: '仍然高于' }) : NaN };
				}
			}
		},
		{
			id: 'memory',
			title: { en: 'Trainable numbers are not memory', zh: '可训练的数不等于内存' },
			source: 'fit',
			at: { phase: 'start' },
			view: 'lora/memory',
			provenance: 'live',
			terms: ['optimizer-state', 'quantization'],
			narration: {
				en: 'Leaving out activations, training holds the weights, plus a gradient and Adam\'s two running averages for every trainable number. For GPT-2 small, full fine-tuning needs {full} MB. LoRA at rank {r} trains {share} of the numbers but still needs {lora} MB, {memShare} of that, because the frozen weights stay in memory. QLoRA stores each frozen weight in 4 bits and needs {qlora} MB in all.',
				zh: '不算激活值，训练时要存下权重，还要为每个可训练的数存一个梯度和 Adam 的两个滑动平均。对 GPT-2 small，全量微调需要 {full} MB。秩为 {r} 的 LoRA 只训练 {share} 的数，却仍需要 {lora} MB，是全量的 {memShare}，因为冻结的权重仍留在内存里。QLoRA 每个冻结权重只用 4 比特存储，一共需要 {qlora} MB。',
				vars: (rt, at) => {
					const r = rankAt({ rt, at });
					const m = memoryOf(r);
					return { full: mb(m.full.total), r, share: pctText(gpt2Lora(r) / GPT2_PARAMS), lora: mb(m.lora.total), memShare: pctText(m.lora.total / m.full.total), qlora: mb(m.qlora.total) };
				}
			},
			look: { en: 'The weights segment of each bar: the one for LoRA is about as long as the one for full fine-tuning.', zh: '看每根条里的权重部分：LoRA 的和全量微调的差不多一样长。' }
		},
		{
			id: 'limits',
			title: { en: 'What LoRA cannot do', zh: 'LoRA 做不到的事' },
			source: 'ranks',
			at: 'last',
			view: {
				key: 'loss-curve',
				props: () => ({ metrics: ['l1', 'lfull'], log: true, labels: { l1: { en: 'rank 1', zh: '秩 1' }, lfull: { en: 'full fine-tuning', zh: '全量微调' } }, refs: [{ y: floorOf(1), label: { en: 'best for rank 1', zh: '秩 1 的最低值' } }] })
			},
			provenance: 'live',
			kindOf: 'limits',
			look: { en: 'The flat line is rank 1, stopped at its floor; the line that keeps falling is full fine-tuning.', zh: '平的那条线是秩 1，停在它的下限；一直往下走的那条是全量微调。' },
			narration: {
				en: 'Both runs start at {base}. After {n} updates rank 1 is stuck at {l1}, its floor, while full fine-tuning, with all {full} numbers, reaches {lfull}. A task whose change has many strong directions would leave every small rank behind like this. And on a layer this small, rank 8 trains {r8} numbers, as many as full fine-tuning: LoRA pays off only when r is much smaller than the layer.',
				zh: '两次运行都从 {base} 出发。更新 {n} 次之后，秩 1 卡在 {l1}，也就是它的下限；而训练全部 {full} 个数的全量微调降到了 {lfull}。如果一个任务的改变量有很多强方向，所有小的秩都会像这样落在后面。而且在这么小的层上，秩 8 要训练 {r8} 个数，和全量微调一样多：只有当 r 远小于层的大小时，LoRA 才划算。',
				vars: (_rt, at) => {
					const m = metricsOf(at);
					return { base: first(at).l1, n: m.step, l1: m.l1, full: D * D, lfull: m.lfull, r8: loraCount(8, D, D) };
				}
			}
		}
	]
});

export default lesson;
