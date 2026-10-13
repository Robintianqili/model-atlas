<!--
  The next-token-prediction guided lesson: seven steps on the learner's own sentence, one focused view at a time (kit
  Walkthrough, the page's Lesson view; the sentence is edited in the input sheet the stage header's chip opens). Every
  number is TinyGPT's, computed in the browser: the final fp32 weights (steps 1-5, 7) and the fp16 training checkpoints
  (step 6); only the validation losses are precomputed. The lesson follows its own window (NtpLesson), so it never moves
  the Explore views.
-->
<script lang="ts">
	import { untrack, type Component } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import { pct, sci, tailText } from './lessonMath';
	import type { StageProv, WalkContext, WalkStep } from '$lib/kit/types';
	import { FACTS } from './facts';
	import type { NtpState } from './state.svelte';
	import { NtpLesson } from './lessonState.svelte';
	import LessonTokens from './LessonTokens.svelte';
	import LessonWindows from './LessonWindows.svelte';
	import CoverStage from './CoverStage.svelte';

	let { S }: { S: NtpState } = $props();
	const L = new NtpLesson(untrack(() => S));
	let index = $state(0);

	// Steps 3–7 run the model and typeset formulas (KaTeX), so their stages are separate chunks: the lesson shell stays
	// small. They are fetched in the background once the tokenizer is in, and at once for the step on screen (a deep link).
	type StageProps = { L: NtpLesson; ctx: WalkContext };
	const LOADERS: Record<string, () => Promise<{ default: unknown }>> = {
		probs: () => import('./LessonDist.svelte'),
		loss: () => import('./LessonLoss.svelte'),
		average: () => import('./LessonAverage.svelte'),
		training: () => import('./LessonTrain.svelte'),
		sampling: () => import('./LessonSample.svelte')
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
		if (S.tok) for (const id of Object.keys(LOADERS)) loadStage(id);
	});

	// the final model's full distribution for the followed window (cheap: one forward of ≤ 64 tokens)
	$effect(() => {
		void L.ctx;
		void L.target;
		if (!S.client || L.pos < 1) return;
		const h = setTimeout(() => L.request('final'), 100);
		return () => clearTimeout(h);
	});

	/** a token in quotes; a leading space becomes a no-break space so the quote never wraps away from it */
	const q = (s: string) => `“${s.replace(/^ /, '\u00a0')}”`;
	/** the end of the followed window's context, cut at a token boundary (never mid-word) */
	const tail = () => tailText(L.ctx, (ids) => S.decode(ids), 28, (id) => S.disp(id));
	const chars = () => [...S.text].length;

	const steps: WalkStep[] = [
		{
			id: 'tokens',
			title: { en: 'The sentence becomes tokens', zh: '句子变成词元' },
			narration: () => {
				const n = L.n;
				return {
					en: `TinyGPT never sees letters. Its tokenizer cuts your sentence into pieces from a fixed list of 1,024 tokens (whole words, word pieces, single bytes) and replaces each piece with its number. Your ${chars()} characters became ${n} tokens: these ${n} numbers are all the model gets.`,
					zh: `TinyGPT 看不到字母。它的分词器把你的句子切成小块，每一块都来自一张固定的 1,024 词元表（整词、词的一部分或单个字节），再把每块换成它的编号。你的 ${chars()} 个字符变成了 ${n} 个词元：模型拿到的只有这 ${n} 个数。`
				};
			},
			look: {
				en: 'Hover over or tap a token to see its id and bytes. A space belongs to the token after it (shown as ␣).',
				zh: '把鼠标移到词元上或点一下，可以看到它的编号和字节。空格属于它后面的那个词元（显示为 ␣）。'
			}
		},
		{
			id: 'windows',
			title: { en: 'Every position is a lesson', zh: '每个位置都是一道练习题' },
			narration: () => {
				const n = L.n;
				return {
					en: `Nobody labels this text, yet every position holds an exercise: given the tokens so far, which token comes next? The answer is already in the sentence. Your ${n} tokens give ${Math.max(0, n - 1)} of these (context → next token) windows, and the model is graded on every one.`,
					zh: `没有人给这段文本打标签，但每个位置都藏着一道练习题：已知前面的词元，下一个是什么？答案就在句子里。你的 ${n} 个词元构成 ${Math.max(0, n - 1)} 个这样的（上下文 → 下一个词元）窗口，模型在每一个上都要接受评分。`
				};
			},
			look: {
				en: 'Green: the label. Shaded: the context the model may read. Faded: the future it must not see. The staircase shows all windows at once; the model computes them all in one pass. The window you stop on is the one the next steps follow.',
				zh: '绿色是标签，带底色的是模型可以读的上下文，变淡的是它不许看的“未来”。阶梯图同时展示所有窗口，模型一次计算就能全部得到。你停在哪个窗口，后面几步就跟着哪个窗口。'
			},
			waitForDone: true,
			linger: 2500
		},
		{
			id: 'probs',
			title: { en: 'One window, 1,024 probabilities', zh: '一个窗口，1,024 个概率' },
			narration: () => {
				const d = L.dist('final');
				if (!d)
					return {
						en: 'For one window, TinyGPT does not answer with a single token. It outputs a probability for every one of its 1,024 tokens, and together they add up to 1.',
						zh: '对一个窗口，TinyGPT 给出的不是一个词元，而是给全部 1,024 个词元各一个概率，加起来正好是 1。'
					};
				const top = S.disp(d.top[0].id);
				const tgt = S.disp(d.target);
				return {
					en: `For the window ending ${q(tail())}, TinyGPT does not answer with a single token. It outputs a probability for every one of its 1,024 tokens, and together they add up to 1. ${d.rank === 1 ? `Its favorite, ${q(tgt)} with ${pct(d.p)}, is exactly the token that really came next.` : `Its favorite is ${q(top)} (${pct(d.top[0].p)}); the token that really came next, ${q(tgt)}, got ${pct(d.p)}, rank ${d.rank}.`}`,
					zh: `对于以${q(tail())}结尾的窗口，TinyGPT 给出的不是一个词元，而是给全部 1,024 个词元各一个概率，加起来正好是 1。${d.rank === 1 ? `它最看好的${q(tgt)}（${pct(d.p)}）正是真正出现的下一个词元。` : `它最看好${q(top)}（${pct(d.top[0].p)}）；真正出现的下一个词元${q(tgt)}只得到 ${pct(d.p)}，排第 ${d.rank}。`}`
				};
			},
			look: {
				en: 'Left to right: all 1,024 tokens, most likely first. The green line marks the real next token. Switch to the log scale: even the smallest bars are not zero.',
				zh: '从左到右是全部 1,024 个词元，概率最高的在最左边。绿线标出真实的下一个词元。切换到对数刻度：就连最小的条形也不是零。'
			}
		},
		{
			id: 'loss',
			title: { en: 'The loss: −log p of the real next token', zh: '损失：真实下一个词元的 −log p' },
			narration: () => {
				const d = L.dist('final');
				const p = d ? sci(d.p) : 'p';
				const v = d ? fmt3(-Math.log(d.p)) : '…';
				return {
					en: `Training looks at just one of those 1,024 numbers: p, the probability of the token that really came next. The loss for this window is −ln p = −ln(${p}) = ${v} nats (ln is the logarithm with base e; a loss measured with it is counted in “nats”). Sure and right (p near 1) costs almost nothing, a small p costs a lot, and a blind guess (p = 1/1,024) costs ln 1,024 = 6.93.`,
					zh: `训练只看这 1,024 个数里的一个：p，即真正出现的下一个词元的概率。这个窗口的损失是 −ln p = −ln(${p}) = ${v} 奈特（ln 是以 e 为底的对数，用它算出的损失以“奈特”为单位）。有把握又猜对（p 接近 1）几乎没有代价，p 很小就要付出很大代价，盲猜（p = 1/1,024）的代价是 ln 1,024 = 6.93。`
				};
			},
			look: {
				en: 'The green point is this window. The grey points show what the loss would have been if the sentence had continued with one of the model’s other top guesses.',
				zh: '绿点是这个窗口。灰点表示：假如句子接下来是模型的其他热门猜测之一，损失会是多少。'
			}
		},
		{
			id: 'average',
			title: { en: 'Average all windows: loss and perplexity', zh: '所有窗口取平均：损失与困惑度' },
			narration: () => {
				const sc = S.fresh ? S.score : null;
				if (!sc || sc.scored < 1)
					return {
						en: 'Do the same for every window and take the average: that is the loss of your sentence. Raising e to that power gives the perplexity.',
						zh: '对每个窗口都这样算，再取平均，就是你的句子的损失。对它取 e 的幂就得到困惑度。'
					};
				const r = Math.max(1, Math.round(sc.perplexity));
				// the same number the stage's table shows (5.81, not "6"); whole numbers from 1,000 on
				const ppl = sc.perplexity < 1000 || sc.perplexity >= 1e6 ? sci(sc.perplexity) : r.toLocaleString('en-US');
				const head = {
					en: `Do the same for every window and take the average: your sentence costs ${fmt3(sc.meanNats)} nats per token. Raising e to that power gives the perplexity, ${ppl}`,
					zh: `对每个窗口都这样算，再取平均：你的句子平均每个词元的损失是 ${fmt3(sc.meanNats)} 奈特。对它取 e 的幂就是困惑度 ${ppl}`
				};
				// a perplexity above the vocabulary size cannot be read as "choosing among k tokens"
				const mid =
					sc.perplexity <= 1024
						? {
								en: `: on average the model was as unsure as if it were choosing among about ${r.toLocaleString('en-US')} equally likely tokens.`,
								zh: `：平均来说，模型的犹豫程度相当于在大约 ${r.toLocaleString('en-US')} 个同样可能的词元里挑一个。`
							}
						: {
								en: ', more than the 1,024 tokens there are: the model did worse than a blind guess, because it was confidently wrong.',
								zh: '，比词元总数 1,024 还大：模型比盲猜还差，因为它错得很自信。'
							};
				return {
					en: `${head.en}${mid.en} On stories it never saw, TinyGPT averages 1.92 (perplexity 6.81).`,
					zh: `${head.zh}${mid.zh}在没见过的故事上，TinyGPT 的平均损失是 1.92（困惑度 6.81）。`
				};
			},
			look: {
				en: 'Darker tokens surprised the model more. Each bar is one window’s −ln p; the dashed line is their average, built up one window at a time.',
				zh: '颜色越深，模型越意外。每根条形是一个窗口的 −ln p；虚线是它们的平均值，一个窗口一个窗口地累加出来。'
			},
			waitForDone: true,
			linger: 3500
		},
		{
			id: 'training',
			title: { en: 'Training lowers this loss', zh: '训练把这个损失降下来' },
			narration: () => {
				const tgt = L.pos >= 1 ? q(S.disp(L.target)) : '';
				return {
					en: `Training repeats this on about a billion windows: 30,000 steps, each on a batch of 32,768 windows (983,040,000 labels in all). After each batch it nudges all 364,040 weights (the numbers inside the model) a little, so that the real next tokens get a bit more probability. Ten snapshots of the weights, called checkpoints, were saved on the way. At step 0 the weights are random and every token gets about 1/1,024. Pick later checkpoints and watch what happens to ${tgt} in this window.`,
					zh: `训练在大约十亿个窗口上重复这件事：一共 30,000 步，每一步处理一个批次的 32,768 个窗口（总共 983,040,000 个标签）。每个批次之后，它把全部 364,040 个权重（模型内部的数）都微调一点，让真实的下一个词元多得到一点概率。途中保存了 10 份权重快照，叫作检查点。第 0 步时权重是随机的，每个词元只得到大约 1/1,024。选后面的检查点，看看这个窗口里的${tgt}会怎样变化。`
				};
			},
			look: {
				en: 'Each checkpoint is a 728 KB file, downloaded only when you pick it (or press Play training). The chart collects this window’s loss at every checkpoint you have downloaded. The other line is the validation loss: the average loss on stories TinyGPT never trained on, logged during training. It falls at every checkpoint; a single window can go up and down on the way.',
				zh: '每个检查点是一个 728 KB 的文件，只在你选中它（或点“播放训练过程”）时才下载。折线图记录这个窗口在每个已下载检查点上的损失。另一条线是验证损失，即 TinyGPT 在从没训练过的故事上的平均损失，是训练时记录的。它在每个检查点都下降；单个窗口在途中可能时升时降。'
			},
			waitForDone: true,
			duration: 90000,
			linger: 3000
		},
		{
			id: 'sampling',
			title: { en: 'Using the probabilities: sample the next token', zh: '用这些概率：采样下一个词元' },
			narration: {
				en: 'Training is over; now the model writes. It reads your whole sentence, outputs 1,024 probabilities for the next token (the same kind of bars as before) and picks one at random, in proportion to them. A temperature reshapes the bars first: below 1 the favorites get even more, above 1 the bars even out. The picked token is appended and the model runs again.',
				zh: '训练结束了，现在让模型来写。它读完你的整句话，为下一个词元输出 1,024 个概率（和前面一样的条形），再按这些概率随机挑一个。挑之前，温度会先重塑这些条形：低于 1 时热门词元得到更多，高于 1 时条形变得更平。挑出的词元接到末尾，模型再运行一次。'
			},
			look: {
				en: 'Outlined bars: the probabilities as trained (T = 1). Solid bars: after the temperature. Press “Pick the next token”: a random number u lands in one segment of the strip.',
				zh: '空心条形是训练得到的原始概率（T = 1），实心条形是温度调整之后的。点“挑下一个词元”：随机数 u 会落在长条的某一段里。'
			}
		}
	];

	// the stage header's one provenance label per step (the validation losses in steps 5 and 6 carry their own)
	const TOK: StageProv = { mode: 'live', note: { en: 'TinyGPT’s own byte-level BPE tokenizer, running in your browser.', zh: 'TinyGPT 自己的字节级 BPE 分词器，在你的浏览器里运行。' }, short: { en: '· TinyGPT’s tokenizer on your sentence', zh: '· TinyGPT 的分词器处理你的句子' } };
	const FP32: StageProv = { mode: 'live', precision: 'fp32', tolerance: FACTS.tolerance, against: 'PyTorch fp32', short: { en: '· TinyGPT on your sentence', zh: '· TinyGPT 处理你的句子' } };
	const FP16: StageProv = {
		mode: 'live',
		precision: 'fp16',
		note: {
			en: 'Runs the fp16 checkpoint files in your browser. On the same fp16 weights the JS forward pass matches PyTorch within 1.2e-5. The validation losses were logged during training (precomputed).',
			zh: '在你的浏览器里运行 fp16 检查点文件。在同样的 fp16 权重上，JS 前向计算与 PyTorch 的差距在 1.2e-5 以内。验证损失是训练时记录的（预先计算）。'
		},
		short: { en: '· training checkpoints on your sentence', zh: '· 训练检查点处理你的句子' }
	};
	for (const st of steps) st.prov = st.id === 'tokens' || st.id === 'windows' ? TOK : st.id === 'training' ? FP16 : FP32;

	const chapters = [
		{ title: { en: 'Data', zh: '数据' }, count: 2 },
		{ title: { en: 'Loss', zh: '损失' }, count: 3 },
		{ title: { en: 'Training and use', zh: '训练与使用' }, count: 2 }
	];
</script>

<Walkthrough {steps} bind:index {chapters} ready={!!S.tok} label={{ en: 'How a language model learns from plain text', zh: '语言模型怎样从普通文本中学习' }}>
	{#snippet cover()}<CoverStage {S} />{/snippet}
	{#snippet stage(step, ctx)}
		{#if S.error}
			<p class="err">{S.error}</p>
		{:else if !S.tok}
			<p class="muted">{t({ en: 'Loading TinyGPT’s tokenizer…', zh: '正在加载 TinyGPT 的分词器…' })}</p>
		{:else if step.id === 'tokens'}
			<LessonTokens {L} />
		{:else if step.id === 'windows'}
			<LessonWindows {L} {ctx} />
		{:else if stages[step.id]}
			{@const Stage = stages[step.id]}
			<Stage {L} {ctx} />
		{:else if LOADERS[step.id]}
			<p class="muted">{t({ en: 'Loading this step…', zh: '正在加载这一步…' })}</p>
		{/if}
	{/snippet}

	{#snippet recap()}
		{@const sc = S.fresh && S.score && S.score.scored > 0 ? S.score : null}
		<ul class="recap">
			<li>{t({ en: `Your ${L.n} tokens hold ${L.windows} (context → next token) exercises, with no human labels.`, zh: `你的 ${L.n} 个词元里藏着 ${L.windows} 道（上下文 → 下一个词元）练习题，不需要人工标注。` })}</li>
			{#if sc}
				<li>{t({ en: `TinyGPT’s loss on your sentence: ${fmt3(sc.meanNats)} nats per token, perplexity ${sci(sc.perplexity)}.`, zh: `TinyGPT 在你的句子上的损失：每个词元 ${fmt3(sc.meanNats)} 奈特，困惑度 ${sci(sc.perplexity)}。` })}</li>
			{/if}
			<li>{t({ en: `A blind guess costs ln 1,024 = ${FACTS.uniformLoss.toFixed(2)} nats; after ${FACTS.steps.toLocaleString('en-US')} training steps TinyGPT averages ${FACTS.valLoss.toFixed(2)} on stories it never saw.`, zh: `盲猜的代价是 ln 1,024 = ${FACTS.uniformLoss.toFixed(2)} 奈特；训练 ${FACTS.steps.toLocaleString('en-US')} 步之后，TinyGPT 在没见过的故事上平均为 ${FACTS.valLoss.toFixed(2)}。` })}</li>
		</ul>
	{/snippet}

	{#snippet end()}
		<p class="fin">
			<strong>{t({ en: 'That is next-token prediction, end to end.', zh: '这就是下一个词元预测的全过程。' })}</strong>
			{t({
				en: 'The number a GPT’s pre-training pushes down is exactly this one: the average −log p over (context → next token) windows. Try another sentence and play the lesson again, or open Explore to try every part freely.',
				zh: 'GPT 预训练要压低的正是这个数：在（上下文 → 下一个词元）窗口上的平均 −log p。换一句话再播放一遍课程，或者打开“探索”自由地试每个部分。'
			})}
		</p>
	{/snippet}
</Walkthrough>

<style>
	.recap {
		margin: 0;
		padding-left: 1.2em;
		display: grid;
		gap: 4px;
	}
	.err {
		color: var(--no);
	}
	.fin {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
