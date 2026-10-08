<!--
  The transfer-learning guided lesson: eight steps from a forced 10-way guess to a trained 2-way
  head and the same idea at BERT scale, one focused view at a time (kit Walkthrough). Steps 1–6
  are live (the learner's drawing, the shipped TinyCNN, a head trained in the browser); steps 7–8
  are precomputed (finetune.json, our BERT reproduction). The lesson keeps its own state
  (lesson.svelte.ts), so it never moves the "Explore freely" parts below.
-->
<script lang="ts">
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState, motion } from '$lib/kit/theme.svelte';
	import { fmtPct, fmtInt } from '$lib/kit/format';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import type { WalkStep } from '$lib/kit/types';
	import type { Stroke } from '$lib/kit/draw';
	import Thumb from './Thumb.svelte';
	import LessonMeet from './LessonMeet.svelte';
	import LessonFeatures from './LessonFeatures.svelte';
	import LessonMap from './LessonMap.svelte';
	import { BASE_CLASSES } from './data';
	import { MNLI_RTE, FEATURES_VS_FT, GLUE_TRAIN } from './bert';
	import { LESSON_N, LESSON_STEPS, clsName, type LessonState } from './lessonState.svelte';
	import type { Component } from 'svelte';

	let { L, tolerance }: { L: LessonState; tolerance?: number } = $props();

	let index = $state(0);

	// Steps 4–8 need the trained head (or are precomputed charts), so their stages are separate
	// chunks: the page shell stays under its JS budget. They are fetched in the background as soon
	// as the model is in, and at once for the step on screen (e.g. a deep link).
	type StageProps = { L: LessonState; autoplay?: boolean; ondone?: () => void };
	const LOADERS: Record<string, () => Promise<{ default: unknown }>> = {
		head: () => import('./LessonHead.svelte'),
		onestep: () => import('./LessonStep.svelte'),
		repeat: () => import('./LessonTrain.svelte'),
		few: () => import('./LessonCurves.svelte'),
		bert: () => import('./LessonBert.svelte')
	};
	let stages = $state.raw<Record<string, Component<StageProps>>>({});
	const loading = new Set<string>();
	function loadStage(id: string) {
		if (stages[id] || loading.has(id) || !LOADERS[id]) return;
		loading.add(id);
		LOADERS[id]().then(
			(m) => (stages = { ...stages, [id]: m.default as Component<StageProps> }),
			() => loading.delete(id)
		);
	}
	$effect(() => {
		const id = steps[index]?.id;
		if (id) loadStage(id);
	});
	$effect(() => {
		if (L.engine) for (const id of Object.keys(LOADERS)) loadStage(id);
	});
	let padOpen = $state(false);
	// the pad strokes live here (the lesson state only receives them through L.drawing)
	let strokes = $state<Stroke[]>([]);

	const classes = $derived(L.engine?.spec.classes ?? BASE_CLASSES);
	const cname = (i: number) => (i18n.lang === 'zh' ? classes[i].zh : classes[i].en);
	const pct = (v: number) => (v * 100).toFixed(1) + '%';
	/** the most common top guess over all 1,500 test drawings of a new class (finetune.json, precomputed) */
	function topGuess(w: 'ladder' | 'envelope') {
		const s = L.finetune?.baseModelGuesses.byHeldOutClass[w]?.top1Share ?? {};
		let best = '';
		for (const k of Object.keys(s)) if (!best || s[k] > s[best]) best = k;
		const c = classes.find((x) => x.id === best);
		return { en: c?.en ?? best, zh: c?.zh ?? best, share: Math.round((s[best] ?? 0) * 100) };
	}
	const ft = (m: 'head' | 'full' | 'scratch') => (L.finetune ? pct(L.finetune.results[m][0].mean) : '–');

	const steps: WalkStep[] = [
		{
			id: 'meet',
			title: { en: 'A model that knows 10 things meets a ladder', zh: '只认识 10 样东西的模型遇到了梯子' },
			narration: () => {
				const o = L.old;
				const s = L.source;
				const said = o ? { en: ` For this drawing it says “${classes[o.top].en}” (${fmtPct(o.p)}).`, zh: `对这张画，它的回答是“${classes[o.top].zh}”（${fmtPct(o.p)}）。` } : { en: '', zh: '' };
				const gl = topGuess('ladder');
				const ge = topGuess('envelope');
				const over =
					s.kind === 'test' && s.cls === 1
						? { en: ` Over all 1,500 test envelopes it says “${ge.en}” ${ge.share}% of the time.`, zh: `在全部 1,500 张测试信封上，它有 ${ge.share}% 的时候回答“${ge.zh}”。` }
						: s.kind === 'test'
							? { en: ` Over all 1,500 test ladders it says “${gl.en}” ${gl.share}% of the time.`, zh: `在全部 1,500 张测试梯子上，它有 ${gl.share}% 的时候回答“${gl.zh}”。` }
							: { en: ` Over all 1,500 test ladders it says “${gl.en}” ${gl.share}% of the time, and over 1,500 envelopes “${ge.en}” ${ge.share}%.`, zh: `在全部 1,500 张测试梯子上，它有 ${gl.share}% 的时候回答“${gl.zh}”；在 1,500 张信封上，有 ${ge.share}% 的时候回答“${ge.zh}”。` };
				return {
					en: `TinyCNN was trained on 10 classes: apple, bicycle, car, clock, fish, house, star, sun, tree and umbrella. It has never seen a ladder or an envelope. But its last layer has exactly 10 outputs, and its last step (softmax) turns them into 10 percentages that add up to 100%, so it must pick one of its 10 classes.${said.en}${over.en}`,
					zh: `TinyCNN 只学过 10 个类别：苹果、自行车、汽车、时钟、鱼、房子、星星、太阳、树和雨伞，从没见过梯子和信封。可它的最后一层恰好有 10 个输出，最后一步（softmax）把它们变成 10 个加起来等于 100% 的百分比，所以它只能在这 10 类里挑一个。${said.zh}${over.zh}`
				};
			},
			look: {
				en: 'Pick a ladder or an envelope in the “Your drawing” panel, or draw your own: steps 1–6 all use that drawing.',
				zh: '在“你的画”面板里选一张梯子或信封，或者自己画一张：第 1–6 步都会用这张画。'
			},
			duration: 20000
		},
		{
			id: 'features',
			title: { en: 'Its frozen body turns any drawing into 256 numbers', zh: '冻结的主体把任何画变成 256 个数' },
			narration: () => {
				let z = 0;
				const f = L.feat;
				if (f) for (let i = 0; i < f.length; i++) if (f[i] === 0) z++;
				return {
					en: `Now look one layer before the answer. Most of the network is two convolution blocks: small pattern detectors (4,152 learned numbers in all) that slide over the drawing looking for strokes, corners and shapes. They turn any drawing into 16 small maps of 4 × 4, which is 256 numbers: the features. For this drawing ${z} of them are exactly 0 (that pattern was not found there). We now freeze these blocks (the locks): nothing in the rest of the lesson changes them, and the old 10-way head is set aside.`,
					zh: `现在往答案前面看一层。网络的大部分是两个卷积块：一组小小的图案探测器（一共 4,152 个学来的数），在画上滑动，寻找笔画、拐角和形状。它们把任何画都变成 16 张 4 × 4 的小图，也就是 256 个数：特征。这张画的特征里有 ${z} 个正好是 0（那里没有找到对应的图案）。现在把这两个卷积块冻结（锁的图标）：课程后面的步骤都不会改动它们，旧的 10 类输出头也放在一边。`
				};
			},
			look: {
				en: 'Hover or tap a cell to read one feature. Pick or draw another drawing: all 256 numbers change, the locked weights do not.',
				zh: '悬停或点按一个格子，读出一个特征。换一张画或自己画：256 个数都会变，锁住的权重不会变。'
			},
			duration: 22000
		},
		{
			id: 'map',
			title: { en: 'Similar drawings land close together', zh: '相似的画落在一起' },
			narration: () => {
				const nb = L.neighbours;
				const list = nb.map((n) => clsName(n.cls));
				return {
					en: `A point on a graph has 2 coordinates, x and y. Read the 256 features the same way: as the coordinates of one point in a space with 256 directions. This map is a flat shadow of that space, along the 2 directions in which drawings differ most. Ladders (orange) and envelopes (blue) fall in different regions, although the network never learned them.${list.length ? ` In all 256 numbers, the 3 drawings closest to this one are: ${list.map((c) => c.en).join(', ')}.` : ''} So a straight cut can separate the two groups, and a few examples are enough to place it.`,
					zh: `图上的一个点有 2 个坐标：x 和 y。256 个特征也可以这样看：它们是一个点在 256 个方向上的坐标。这张地图是那个空间的平面投影，用的是画与画之间差别最大的 2 个方向。梯子（橙色）和信封（蓝色）落在不同的区域，尽管网络从没学过它们。${list.length ? `按全部 256 个数来算，离这张画最近的 3 张画是：${list.map((c) => c.zh).join('、')}。` : ''}所以一条直线就能把两组分开，几个样本就足以确定这条线的位置。`
				};
			},
			look: {
				en: 'The big star-shaped mark is this drawing. Pick another one or draw your own: the star moves, and earlier drawings stay as small numbered × marks.',
				zh: '大的星形记号就是这张画。换一张或者自己画：星形记号会移动，之前的画会留下带编号的小 × 记号。'
			},
			duration: 20000
		},
		{
			id: 'head',
			title: { en: 'A new head: 2 outputs, 514 parameters, starting random', zh: '新的输出头：2 个输出、514 个参数，从随机开始' },
			narration: () => {
				const a = L.run ? pct(L.run.testAcc[0]) : '–';
				return {
					en: `Now attach a new head with 2 outputs, A = ladder and B = envelope. It is one dense layer: each output multiplies each of the 256 features by its own weight, adds up the 256 products and adds a bias. That makes 256 × 2 weights plus 2 biases: 514 parameters, the only numbers that will learn. The weights start as tiny random numbers and the biases at 0, so both scores are near 0 and every drawing gets about 50/50. We test it on 200 drawings it never trains on (held-out drawings): it is right ${a} of the time, no better than a coin flip.`,
					zh: `现在接上一个新的输出头，有 2 个输出：A = 梯子，B = 信封。它是一个全连接层：每个输出把 256 个特征各乘上自己的一个权重，把这 256 个乘积加起来，再加一个偏置。这样就是 256 × 2 个权重加 2 个偏置，共 514 个参数，也是唯一会学习的数。权重从很小的随机数开始，偏置从 0 开始，所以两个分数都接近 0，每张画得到的都差不多是五五开。我们用 200 张它从不拿来训练的画（留出的测试图）来检验：它的正确率是 ${a}，和抛硬币差不多。`
				};
			},
			look: () => {
				const r = L.run;
				if (!r) return { en: '', zh: '' };
				const env = r.testPer[0][1] > r.testPer[0][0];
				const k = env ? clsName(1) : clsName(0);
				return {
					en: `At step 0 it calls almost every test drawing ${env ? 'an' : 'a'} ${k.en} (${clsName(0).en} ${r.testPer[0][0]}/${r.testN[0]} right, ${clsName(1).en} ${r.testPer[0][1]}/${r.testN[1]}): tiny random differences, not knowledge.`,
					zh: `第 0 步时，它几乎把每张测试图都判成${k.zh}（${clsName(0).zh}判对 ${r.testPer[0][0]}/${r.testN[0]}，${clsName(1).zh} ${r.testPer[0][1]}/${r.testN[1]}）：这只是微小的随机差别，不是知识。`
				};
			},
			duration: 24000,
			onEnter: () => (L.headStep = 0)
		},
		{
			id: 'onestep',
			title: { en: 'One training step, by hand', zh: '亲手走一步训练' },
			narration: () => {
				const ex = L.detail ? clsName(L.detail.y) : clsName(0);
				return {
					en: `Training never looks at your drawing, only at the ${2 * LESSON_N} training drawings. Follow one of them, ${ex.en} 1, through one step of gradient descent: forward to two scores, softmax to probabilities, the loss (how wrong the answer is), the gradient (which way each weight should move to lower the loss) and the update. The real step averages the gradients of all ${2 * LESSON_N} training examples, then moves every weight a little against it.`,
					zh: `训练从不看你的画，只看 ${2 * LESSON_N} 张训练图。跟着其中一张（${ex.zh} 1）走完一步梯度下降：前向得到两个分数，softmax 变成概率，算出损失（答案错了多少），求梯度（每个权重往哪边改能让损失变小），最后更新。真正的一步会把全部 ${2 * LESSON_N} 个训练样本的梯度取平均，再让每个权重逆着它挪一小步。`
				};
			},
			look: () => {
				const r = L.run;
				const a0 = r ? pct(r.testAcc[0]) : '–';
				const a1 = r ? pct(r.testAcc[1]) : '–';
				return {
					en: `After this single step, held-out accuracy goes from ${a0} to ${a1}. Switch to the envelope example: now the envelope weights get the push.`,
					zh: `仅仅这一步之后，留出测试图的正确率就从 ${a0} 变成 ${a1}。切换到信封样本：这回被推动的是信封的权重。`
				};
			},
			waitForDone: true,
			linger: 4000
		},
		{
			id: 'repeat',
			title: { en: 'Repeat: the loss falls, held-out accuracy rises', zh: '重复：损失下降，留出准确率上升' },
			narration: () => {
				const r = L.run;
				const v = r
					? { l0: r.loss[0].toFixed(3), l1: r.loss[LESSON_STEPS].toFixed(3), a0: pct(r.testAcc[0]), a1: pct(r.testAcc[LESSON_STEPS]), a10: pct(r.testAcc[10]) }
					: { l0: '–', l1: '–', a0: '–', a1: '–', a10: '–' };
				return {
					en: `Training is that step, repeated: ${LESSON_STEPS} times over the same ${2 * LESSON_N} drawings (${LESSON_N} ladders, ${LESSON_N} envelopes). The loss falls from ${v.l0} to ${v.l1}. What counts is drawings the head never trained on: on the 200 held-out test drawings it goes from ${v.a0} to ${v.a1}. Most of that happens at once (${v.a10} after 10 steps), because the frozen features already did the hard work.`,
					zh: `训练就是把这一步重复下去：在同样的 ${2 * LESSON_N} 张画（${LESSON_N} 张梯子、${LESSON_N} 张信封）上重复 ${LESSON_STEPS} 次。损失从 ${v.l0} 降到 ${v.l1}。真正重要的是输出头从没训练过的画：在 200 张留出的测试图上，正确率从 ${v.a0} 升到 ${v.a1}。大部分提升一下子就完成了（10 步之后已是 ${v.a10}），因为冻结的特征已经把难的部分做完了。`
				};
			},
			look: {
				en: 'Drag across a chart to scrub. The drawing in “Your drawing” is not in the training set: watch its probability follow the replay.',
				zh: '在图表上拖动可以回看。“你的画”里的那张画不在训练集里：看它的概率怎样跟着回放变化。'
			},
			waitForDone: true,
			linger: 3000
		},
		{
			id: 'few',
			title: { en: 'Why freezing helps when examples are few', zh: '为什么样本少时冻结更好' },
			narration: () => ({
				en: `We also trained this task offline in three ways: head only (514 parameters learn), full fine-tuning (all 4,666 learn, starting from the trained weights) and from scratch (all 4,666, starting from random numbers). Each was run 5 times (5 “seeds”: different random starting numbers and different training drawings each time). With 1 drawing per class, head only wins: ${ft('head')} vs ${ft('full')} vs ${ft('scratch')} (average of the 5 runs, on 3,000 test drawings). More trainable parameters means more ways to fit two drawings wrongly. With more drawings the curves come together.`,
				zh: `我们还离线用三种方法训练了同一个任务：只训练输出头（514 个参数在学）、全量微调（全部 4,666 个参数都在学，从训练好的权重出发）和从零训练（全部 4,666 个参数，从随机数出发）。每种方法跑 5 次（5 个“种子”：每次的随机初始值和训练图都不同）。每类只有 1 张图时，只训练输出头胜出：${ft('head')} 对 ${ft('full')} 对 ${ft('scratch')}（5 次的平均，3,000 张测试图）。可训练的参数越多，把两张图拟合“错”的方式也越多。图越多，几条曲线就越靠拢。`
			}),
			look: {
				en: 'The player walks from 1 to 2,048 drawings per class. The shaded bands show how much the 5 runs differ (± one standard deviation).',
				zh: '播放器从每类 1 张一直走到每类 2,048 张。阴影带表示 5 次运行之间的差别有多大（± 1 个标准差）。'
			},
			waitForDone: true,
			linger: 3000,
			onEnter: () => (L.headStep = LESSON_STEPS)
		},
		{
			id: 'bert',
			title: { en: 'The same idea at real scale: our BERT reproduction', zh: '真实规模下的同一思路：我们的 BERT 复现' },
			narration: {
				en: `BERT-base is a language model with 110 million parameters instead of TinyCNN’s 6,722, and the same choices apply. SST-2 asks whether a movie review is positive. A linear probe (a small head on frozen BERT features, like our ladder head) reaches at best ${FEATURES_VS_FT.bestFrozen.toFixed(1)}%; fine-tuning all of BERT on the ${fmtInt(GLUE_TRAIN.SST2)} labeled reviews reaches ${FEATURES_VS_FT.ftSst2.toFixed(1)}%: with a lot of data, fine-tuning everything wins. RTE asks whether sentence A implies sentence B and has only ${fmtInt(GLUE_TRAIN.RTE)} labeled pairs: fine-tuning on it directly gives ${MNLI_RTE.direct.mean.toFixed(1)}% ± ${MNLI_RTE.direct.std.toFixed(1)}. Fine-tuning on MNLI first (${fmtInt(GLUE_TRAIN.MNLI)} pairs, a similar question with 3 answers), then on RTE, gives ${MNLI_RTE.mnliFirst.mean.toFixed(1)}% ± ${MNLI_RTE.mnliFirst.std.toFixed(1)}, higher in all 5 runs. All scores are accuracy on held-out examples.`,
				zh: `BERT-base 是一个语言模型，有 1.1 亿个参数，而 TinyCNN 只有 6,722 个，但同样的取舍依然成立。SST-2 问的是一条影评是不是正面的。线性探针（在冻结的 BERT 特征上训练一个小输出头，就像我们的梯子输出头）最好得到 ${FEATURES_VS_FT.bestFrozen.toFixed(1)}%；用 ${fmtInt(GLUE_TRAIN.SST2)} 条带标签的影评微调整个 BERT，得到 ${FEATURES_VS_FT.ftSst2.toFixed(1)}%：数据很多时，微调全部参数胜出。RTE 问的是句子 A 能否推出句子 B，只有 ${fmtInt(GLUE_TRAIN.RTE)} 对带标签的样本：直接在上面微调得到 ${MNLI_RTE.direct.mean.toFixed(1)}% ± ${MNLI_RTE.direct.std.toFixed(1)}。先在 MNLI（${fmtInt(GLUE_TRAIN.MNLI)} 对，类似的问题，有 3 种答案）上微调、再微调 RTE，得到 ${MNLI_RTE.mnliFirst.mean.toFixed(1)}% ± ${MNLI_RTE.mnliFirst.std.toFixed(1)}，5 次运行全部更高。所有分数都是在留出样本上的正确率。`
			},
			look: {
				en: 'Each gray line is one run (seed): RTE directly on the left, MNLI first on the right. Hover or tap a line to read it.',
				zh: '每条灰线是一次运行（一个种子）：左边是直接训练 RTE，右边是先训练 MNLI。悬停或点按一条线可以读数。'
			},
			duration: 32000,
			onEnter: () => (L.headStep = LESSON_STEPS)
		}
	];

	function pick(cls: 0 | 1) {
		L.pickTest(cls);
		strokes = [];
		L.drawnInput = null;
	}
	function step(d: number) {
		L.stepTest(d);
	}
	const src = $derived(L.source);
	/** narrow layouts: scroll to the narration under the stage (below the sticky lesson bar) */
	function readBelow(e: MouseEvent) {
		const wt = (e.currentTarget as HTMLElement).closest('section.wt');
		const narr = wt?.querySelector<HTMLElement>('.narr');
		if (!wt || !narr) return;
		const bar = wt.querySelector<HTMLElement>('.bar')?.getBoundingClientRect().bottom ?? 0;
		window.scrollTo({ top: window.scrollY + narr.getBoundingClientRect().top - Math.max(0, bar) - 8, behavior: motion.reduced ? 'auto' : 'smooth' });
		narr.querySelector<HTMLElement>('h3')?.setAttribute('tabindex', '-1');
		narr.querySelector<HTMLElement>('h3')?.focus({ preventScroll: true });
	}
</script>

<Walkthrough {steps} bind:index ready={!!L.engine} label={{ en: 'Transfer learning, step by step', zh: '迁移学习分步课程' }} stageMinHeight={400}>
	{#snippet stage(step, ctx)}
		<div class="ntitle">
			<p aria-hidden="true"><span class="num">{ctx.index + 1}/{steps.length}</span> {t(step.title)}</p>
			<button type="button" class="btn ghost" onclick={readBelow}>{t({ en: 'Read', zh: '看说明' })} ↓</button>
		</div>
		{#if !L.engine}
			<p class="muted">{t({ en: 'Loading the model…', zh: '正在加载模型…' })}</p>
		{:else if step.id === 'meet'}
			<LessonMeet {L} {tolerance} />
		{:else if step.id === 'features'}
			<LessonFeatures {L} {tolerance} />
		{:else if step.id === 'map'}
			<LessonMap {L} />
		{:else if stages[step.id]}
			{@const Stage = stages[step.id]}
			<Stage {L} autoplay={ctx.shown && !ctx.reduced} ondone={ctx.done} />
		{:else if LOADERS[step.id]}
			<p class="muted">{t({ en: 'Loading this step…', zh: '正在加载这一步…' })}</p>
		{/if}
	{/snippet}

	{#snippet aside(ctx)}
		<div class="you">
			<div class="you-h">
				<h4>{t({ en: 'Your drawing', zh: '你的画' })}</h4>
				<span class="muted src">{t(L.label())}</span>
			</div>
			<div class="you-row">
				<Thumb data={L.input} size={56} alt={t({ en: 'The drawing the lesson uses, as the model sees it', zh: '课程正在用的画（模型看到的样子）' })} />
				<div class="answers">
					{#if L.old}
						<p>
							<span class="lbl">{t({ en: '10-class model:', zh: '10 类模型：' })}</span>
							<span class="sw" style="background: {categorical(L.old.top, themeState.resolved)}"></span><strong>{cname(L.old.top)}</strong>
							<span class="num">{fmtPct(L.old.p)}</span>
						</p>
					{:else}
						<p class="muted">{t({ en: 'Draw something, or pick a drawing.', zh: '画点什么，或者选一张画。' })}</p>
					{/if}
					{#if ctx.index >= 3 && L.fresh}
						{@const k = L.fresh.top as 0 | 1}
						<p>
							<span class="lbl">{t({ en: `New head, step ${L.headStep}:`, zh: `新输出头，第 ${L.headStep} 步：` })}</span>
							<span class="dot {k === 0 ? 'a' : 'b'}"></span><strong>{t(clsName(k))}</strong>
							<span class="num">{fmtPct(L.fresh.probs[k])}</span>
						</p>
					{/if}
				</div>
			</div>
			<div class="srcs">
				<div class="segmented" role="group" aria-label={t({ en: 'Which drawing the lesson uses', zh: '课程用哪张画' })}>
					<button type="button" aria-pressed={src.kind === 'test' && src.cls === 0} onclick={() => pick(0)}>{t({ en: 'A ladder', zh: '梯子' })}</button>
					<button type="button" aria-pressed={src.kind === 'test' && src.cls === 1} onclick={() => pick(1)}>{t({ en: 'An envelope', zh: '信封' })}</button>
					<button type="button" aria-pressed={src.kind === 'drawn'} aria-expanded={padOpen} onclick={() => (padOpen = !padOpen)}>
						<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M11 2.5 13.5 5 6 12.5H3.5V10z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" /></svg>
						{t({ en: 'Draw', zh: '自己画' })}
					</button>
				</div>
				{#if src.kind === 'test'}
					<span class="nav">
						<button type="button" class="btn ghost icon" onclick={() => step(-1)} aria-label={t({ en: 'Previous test drawing', zh: '上一张测试图' })}>‹</button>
						<span class="num muted">{src.i + 1}/{L.testCount}</span>
						<button type="button" class="btn ghost icon" onclick={() => step(1)} aria-label={t({ en: 'Next test drawing', zh: '下一张测试图' })}>›</button>
					</span>
				{/if}
			</div>
			{#if padOpen}
				<DrawPad
					bind:strokes
					brush={0.055}
					size={190}
					showBrush={false}
					onchange={(s) => L.drawing(s, true)}
					ondraw={(s) => L.drawing(s)}
					label={{ en: 'Lesson drawing pad', zh: '课程画板' }}
				/>
			{/if}
			<p class="hint">
				{t({
					en: 'These ladders and envelopes are test drawings the lesson never trains on (held-out drawings), and your own drawings are never added to training either: the lesson’s head learns from 8 + 8 other drawings.',
					zh: '这里的梯子和信封都是课程从不拿来训练的测试图（留出的测试图），你自己画的也不会加进训练集：课程里的输出头从另外 8 + 8 张画中学习。'
				})}
			</p>
		</div>
	{/snippet}

	{#snippet end()}
		<div class="fin">
			<p>
				<strong>{t({ en: 'That is transfer learning.', zh: '这就是迁移学习。' })}</strong>
				{t({
					en: 'Reuse what a trained network already computes: with few examples, freeze it and train a small head; with more data, fine-tune more; for a small task, a related big task first can help. Below, explore freely: train a head on your own two classes, mislabel examples, read every weight.',
					zh: '复用训练好的网络已经会算的东西：样本少时，冻结它、只训练一个小输出头；数据多时，就微调更多层；任务很小时，先在相关的大任务上训练可能有帮助。下面可以自由探索：用你自己的两个类别训练输出头、故意标错样本、查看每一个权重。'
				})}
			</p>
			<a class="btn primary" href="#explore">{t({ en: 'Explore freely', zh: '自由探索' })} ↓</a>
		</div>
	{/snippet}
</Walkthrough>
<p class="foot muted">
	{t({
		en: `Steps 1–6 are computed in your browser: your drawing through the shipped TinyCNN, and a new head trained on ${LESSON_N} + ${LESSON_N} preset drawings (the map’s axes and gray points and the top-guess shares in step 1 were computed offline from the same model). Steps 7 and 8 are precomputed: TinyCNN fine-tuning runs (5 seeds) and our BERT reproduction. The lesson keeps its own state, so the parts below stay where you left them.`,
		zh: `第 1–6 步都在你的浏览器里实时计算：你的画经过发布的 TinyCNN，新输出头用 ${LESSON_N} + ${LESSON_N} 张预置图训练（地图的坐标轴和灰色参照点，以及第 1 步的首选答案比例，是用同一个模型离线算好的）。第 7、8 步是预计算结果：TinyCNN 的微调实验（5 个种子）和我们的 BERT 复现。课程使用自己的状态，所以下面各部分会停在你离开时的位置。`
	})}
</p>

<style>
	/* only when the lesson is one column (the kit's 860 px container query): title + jump to the narration */
	.ntitle {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		margin: 0 0 10px;
		padding-bottom: 8px;
		border-bottom: 1px solid var(--rule);
	}
	.ntitle p {
		margin: 0;
		font-family: var(--font-display);
		font-weight: 650;
		font-size: var(--fs-md);
		line-height: 1.25;
	}
	.ntitle .num {
		color: var(--accent);
		font-size: var(--fs-sm);
		margin-right: 4px;
	}
	.ntitle .btn {
		flex: none;
		min-height: 32px;
		padding: 0 10px;
		font-size: var(--fs-sm);
	}
	@container wt (min-width: 860px) {
		.ntitle {
			display: none;
		}
	}
	.you {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.you-h {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 2px 12px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.src {
		font-size: var(--fs-xs);
	}
	.you-row {
		display: flex;
		gap: 10px;
		align-items: center;
	}
	.you-row :global(canvas) {
		border: 1px solid var(--rule);
	}
	.answers {
		display: grid;
		gap: 2px;
		font-size: var(--fs-sm);
		min-width: 0;
	}
	.answers p {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 2px 6px;
	}
	.lbl {
		color: var(--ink-2);
		font-size: var(--fs-xs);
	}
	.sw {
		width: 9px;
		height: 9px;
		border-radius: 2px;
	}
	.dot {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
	}
	.dot.a {
		background: var(--div-pos);
	}
	.dot.b {
		background: var(--div-neg);
		border-radius: 2px;
	}
	.srcs {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 10px;
	}
	.srcs .segmented button {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: var(--fs-sm);
	}
	.nav {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: var(--fs-xs);
	}
	.nav .btn {
		min-height: 30px;
		min-width: 30px;
		padding: 0 6px;
	}
	.you :global(.tools .btn) {
		min-height: 30px;
		padding: 0 8px;
		font-size: var(--fs-xs);
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
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
