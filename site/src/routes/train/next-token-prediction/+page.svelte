<!--
  /train/next-token-prediction/ — Next-token prediction and the pre-training loss (concept page), as the lesson app
  (redesign spec §5): the page is the 7-step lesson on the learner's own sentence (with its cover); Explore (the sentence,
  the sliding window and the loss toy, loss and perplexity colored by surprise, the training replay and decoding),
  Pitfalls and Sources are lazy panels one tab away (#explore, #pitfalls, #sources; the old #toy, #real, #compare, #post,
  #links still work). The input chip in the stage header opens the sentence sheet.
  Every number on the page comes from TinyGPT: live in a Web Worker (final fp32 weights and the fp16
  training checkpoints, fetched lazily) or precomputed from the same files (replay.json, curve.json).
  Page code: src/lib/pages/next-token-prediction/. Notes: notes/page-next-token-prediction.md.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import ConceptHost from '$lib/kit/lesson/ConceptHost.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { takeHandoff } from '$lib/kit/handoff';
	import type { PredictQuestion } from '$lib/kit/types';
	import { NtpState } from '$lib/pages/next-token-prediction/state.svelte';
	import { keyOf } from '$lib/pages/next-token-prediction/protocol';
	import { MAX_CHARS } from '$lib/pages/next-token-prediction/facts';
	import type { NtpClient } from '$lib/pages/next-token-prediction/client';
	import Lesson from '$lib/pages/next-token-prediction/Lesson.svelte';
	import InputChip from '$lib/pages/next-token-prediction/InputChip.svelte';
	import InputSheet from '$lib/pages/next-token-prediction/InputSheet.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const S = new NtpState();

	onMount(() => {
		// a sentence handed over by the home-page launcher, or a ?text= link
		const h = takeHandoff('text');
		let q: string | null = null;
		try {
			q = new URLSearchParams(location.search).get('text');
		} catch {
			/* ignore */
		}
		const init = h?.text ?? q;
		if (init) S.text = [...init].slice(0, MAX_CHARS).join('');

		let client: NtpClient | null = null;
		let dead = false;
		(async () => {
			try {
				const modelUrl = new URL(`${base}/models/tinygpt/`, location.href).href;
				const [{ loadTokenizer }, { NtpClient }] = await Promise.all([
					import('$lib/models/bpe'),
					import('$lib/pages/next-token-prediction/client')
				]);
				const tok = await loadTokenizer(fetch, modelUrl);
				if (dead) return;
				S.tok = tok;
				client = new NtpClient(modelUrl, (s) => {
					S.status[keyOf(s.which)] = s;
				});
				S.client = client;
			} catch (e) {
				S.error = t({ en: 'Could not load TinyGPT: ', zh: '无法加载 TinyGPT：' }) + String((e as Error)?.message ?? e);
			}
		})();
		return () => {
			dead = true;
			client?.dispose();
		};
	});

	// the final model scores the learner's text (debounced; runs in the worker)
	$effect(() => {
		const ids = S.ids;
		const c = S.client;
		if (!c) return;
		if (ids.length < 2) {
			S.score = null;
			return;
		}
		const h = setTimeout(() => {
			c.score('final', ids, 'final-score', 5).then(
				(r) => {
					if (r) S.score = r;
				},
				(e: Error) => (S.error = e.message)
			);
		}, 140);
		return () => clearTimeout(h);
	});

	const goals = [
		{
			en: 'Explain how one sentence turns into many (context → next token) training examples, with no human labels.',
			zh: '解释一句话如何变成许多（上下文 → 下一个词元）训练样本，而且完全不需要人工标注。'
		},
		{
			en: "Compute the loss −ln p for one token, and the average loss and perplexity for a whole text; read TinyGPT's validation perplexity of 6.81.",
			zh: '算出单个词元的损失 −ln p，以及整段文本的平均损失和困惑度；读懂 TinyGPT 的验证困惑度 6.81。'
		},
		{
			en: 'Describe how the predictions change during training, from random (loss 6.93) to stories (1.92), and tell the learned probabilities apart from the decoding rule that turns them into text.',
			zh: '描述训练过程中预测如何变化：从随机（损失 6.93）到能写故事（1.92）；并分清学到的概率和把概率变成文字的解码规则。'
		}
	];

	const prequestion: PredictQuestion = {
		id: 'ntp-step0',
		question: {
			en: 'Before any training, TinyGPT’s weights are random numbers. You give this untrained model “Once upon a time, there was a little”. What does it predict next?',
			zh: '训练开始之前，TinyGPT 的权重是随机数。你把“Once upon a time, there was a little”交给这个没训练过的模型，它会预测出什么？'
		},
		options: [
			{
				en: 'Almost nothing useful: all 1,024 tokens get about the same tiny probability, so its samples look like random word pieces.',
				zh: '几乎没用：1,024 个词元都分到差不多一样小的概率，采样出来像随机的词语碎片。'
			},
			{ en: '“girl” or “boy”, just less confidently than after training.', zh: '“girl”或“boy”，只是比训练后更没把握。' },
			{ en: 'Short but correct English sentences; training mostly makes them longer.', zh: '简短但正确的英文句子；训练主要是让句子变长。' },
			{ en: 'Nothing at all: an untrained model cannot output probabilities.', zh: '什么都不输出：没训练过的模型给不出概率。' }
		],
		answer: 0,
		explain: {
			en: 'At step 0 the largest probability is 0.26% (“ little”), barely above 1/1,024 = 0.098%, and the loss is 6.95 nats, almost exactly ln 1024 = 6.93: the cost of guessing blindly. Sampling gives “J jump color picture)u…”. Greedy decoding just repeats the last word (“ little little little…”), a side effect of sharing one matrix for reading and scoring tokens. Words appear around step 100 and simple stories by step 1,000. Replay it in Explore, under “Watch it learn”.',
			zh: '第 0 步时，最大的概率只有 0.26%（“ little”），比 1/1,024 = 0.098% 高不了多少；损失是 6.95 奈特，几乎正好是 ln 1024 = 6.93，也就是盲猜的代价。采样得到的是“J jump color picture)u…”。贪心解码只会重复最后一个词（“ little little little…”），这是读入和打分共用同一个矩阵带来的副作用。大约第 100 步开始出现单词，到第 1,000 步已能写简单的故事。可以在“探索”的“看它学习”里回放。'
		}
	};

	const checks = [
		{
			id: 'ntp-half',
			question: {
				en: 'A model gives the correct next token a probability of 0.5 at every position of a text. What are its loss and perplexity?',
				zh: '某个模型在一段文本的每个位置，都给正确的下一个词元 0.5 的概率。它的损失和困惑度是多少？'
			},
			options: [
				{ en: '0.693 nats per token, perplexity 2', zh: '每个词元 0.693 奈特，困惑度 2' },
				{ en: '0.5 nats per token, perplexity 0.5', zh: '每个词元 0.5 奈特，困惑度 0.5' },
				{ en: '1 nat per token, perplexity e ≈ 2.72', zh: '每个词元 1 奈特，困惑度 e ≈ 2.72' },
				{ en: '0.693 nats per token, perplexity 0.5', zh: '每个词元 0.693 奈特，困惑度 0.5' }
			],
			answer: 0,
			explain: {
				en: '−ln 0.5 = 0.693 at every position, so the average is 0.693 nats (exactly 1 bit) and the perplexity is e^0.693 = 2: as unsure as a choice between two equally likely tokens. Check it in Explore with the loss toy: set p = 0.5.',
				zh: '每个位置都是 −ln 0.5 = 0.693，所以平均是 0.693 奈特（正好 1 比特），困惑度 e^0.693 = 2：相当于在两个同样可能的词元之间犹豫。可以在“探索”的损失小实验里把 p 设成 0.5 验证。'
			}
		},
		{
			id: 'ntp-curve',
			question: {
				en: 'Find it in Explore, under “Watch it learn”: what is the first saved checkpoint whose validation loss is below 3 nats per token?',
				zh: '在“探索”的“看它学习”里找一找：验证损失第一次低于每个词元 3 奈特的是哪个检查点？'
			},
			options: [
				{ en: 'step 100', zh: '第 100 步' },
				{ en: 'step 200', zh: '第 200 步' },
				{ en: 'step 400', zh: '第 400 步' },
				{ en: 'step 1,000', zh: '第 1,000 步' }
			],
			answer: 2,
			explain: {
				en: 'Step 200 is at 3.30 and step 400 at 2.66. Most of the drop happens in the first few hundred of the 30,000 steps.',
				zh: '第 200 步是 3.30，第 400 步是 2.66。大部分下降发生在 30,000 步中的前几百步。'
			}
		},
		{
			id: 'ntp-temp',
			question: {
				en: 'You lower the sampling temperature from 1 to 0.5. What happens to TinyGPT’s validation loss?',
				zh: '你把采样温度从 1 降到 0.5。TinyGPT 的验证损失会怎样？'
			},
			options: [
				{ en: 'It goes down, because the model becomes more confident.', zh: '下降，因为模型更自信了。' },
				{ en: 'It goes up, because the text gets more repetitive.', zh: '上升，因为文本更重复了。' },
				{ en: 'It does not change: temperature is part of decoding, not of the trained model.', zh: '不变：温度属于解码规则，不属于训练好的模型。' }
			],
			answer: 2,
			explain: {
				en: 'The loss is measured on the plain probabilities (T = 1) that training produced. Temperature, top-k and top-p only change how a token is picked when generating text.',
				zh: '损失是在训练得到的原始概率（T = 1）上计算的。温度、top-k 和 top-p 只改变生成文本时如何挑选词元。'
			}
		}
	];

	/** the cover's reading time (spec §6.2, measured at the default inputs): 419 English narration words / 180 + 7 steps × 20 s, rounded */
	const MINUTES = 5;

	// Explore, Pitfalls and Sources are lazy chunks: the lesson shell carries none of them
	const views = {
		explore: () => import('$lib/pages/next-token-prediction/panels/Explore.svelte'),
		pitfalls: () => import('$lib/pages/next-token-prediction/panels/Pitfalls.svelte'),
		sources: () => import('$lib/pages/next-token-prediction/panels/Sources.svelte')
	};
	// the panels get the page's state, the cover question and checks (Pitfalls) and the related lessons (Sources)
	const viewProps = {
		S,
		prequestion,
		checks,
		get related() {
			return data.related;
		}
	};
</script>

{#snippet inputChip(open: () => void)}<InputChip {S} {open} />{/snippet}
{#snippet inputSheet()}<InputSheet {S} />{/snippet}
{#snippet lesson()}<Lesson {S} />{/snippet}
{#snippet about()}
	<p class="about">
		{t({
			en: 'Every number in the lesson is computed in your browser by TinyGPT from your sentence: the final weights (fp32) in steps 3–5 and 7, the fp16 training checkpoints in step 6 (each 728 KB, downloaded only when you pick it). Only the validation losses (steps 5 and 6) were measured offline. The lesson follows its own window, so the Explore views stay where you left them.',
			zh: '课程中的每个数都由 TinyGPT 在你的浏览器里根据你的句子算出：第 3–5 步和第 7 步用最终权重（fp32），第 6 步用 fp16 训练检查点（每个 728 KB，只在你选中时才下载）。只有验证损失（第 5、6 步）是离线测得的。课程跟随自己的窗口，所以“探索”里的视图会停在你离开时的位置。'
		})}
	</p>
{/snippet}

<ConceptHost
	concept={data.conceptCatalog}
	{goals}
	{prequestion}
	summary={{
		en: 'A language model learns from plain text by guessing the next token. Turn your sentence into training examples, measure TinyGPT’s loss on it, and replay its training.',
		zh: '语言模型靠猜下一个词元从普通文本中学习。把你的句子变成训练样本，测一测 TinyGPT 在上面的损失，再回放它的训练过程。'
	}}
	minutes={MINUTES}
	exploreIntro={{
		en: 'Every part of the lesson at full size, with more to try: the sliding window, the loss curve you can drag, your text colored by surprise, the training replay with probe prompts, and decoding controls.',
		zh: '课程的每一部分都在这里完整展开，还有更多可以试：滑动窗口、可拖动的损失曲线、按意外程度着色的文本、带探针提示的训练回放，以及解码控件。'
	}}
	{views}
	{viewProps}
	{inputChip}
	{inputSheet}
	{about}
	{lesson}
/>

<style>
	/* Kit issue (Walkthrough): the Look line's visually hidden "Look:" label is absolutely positioned with no positioned
	   ancestor inside the scrolling narration pane, so when the pane scrolls the label escapes it and makes the page a few
	   px taller than the window in app mode (measured: 688 px at 1280 x 680 on a long step). Anchor it to the pane. */
	:global(section.wt .narr) {
		position: relative;
	}
	.about {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
