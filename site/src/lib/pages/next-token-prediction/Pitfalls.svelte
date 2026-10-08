<!--
  Misconceptions, each with live evidence from TinyGPT:
   1. "The model looks up the right answer" -> it outputs a probability for every token (misconception check quiz).
   2. "Lower loss means it tells the truth" -> a false, story-like sentence scores better than a true one.
   + short notes on perplexity.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct, visibleToken } from '$lib/kit/format';
	import Quiz from './Quiz.svelte';
	import type { NtpState } from './state.svelte';
	import type { ScoreResult } from './protocol';
	import { FACTS, TRUTH_PAIR } from './facts';

	let { S }: { S: NtpState } = $props();

	// ---- 1. a probability for every token
	const sc = $derived(S.fresh ? S.score : null);
	const pos = $derived.by(() => {
		if (!sc || sc.p.length < 2) return null;
		const a = S.active;
		if (a !== null && a >= 1 && a < sc.p.length) return a;
		return sc.p.length - 1;
	});
	const spread = $derived.by(() => {
		if (!sc || pos === null) return null;
		const top = sc.top[pos][0];
		const actual = sc.ids[pos];
		const pA = sc.p[pos];
		const isTop = top.id === actual;
		const rest = 1 - (isTop ? pA : pA + top.p);
		return { top, actual, pA, isTop, rest, others: isTop ? 1023 : 1022 };
	});

	const quiz1 = {
		id: 'ntp-lookup',
		question: {
			en: 'True or false: when TinyGPT continues “Once upon a”, it finds the matching sentence in its training stories and copies the next word.',
			zh: '判断对错：TinyGPT 续写“Once upon a”时，会在训练故事里找到匹配的句子，然后照抄下一个词。'
		},
		options: [
			{ en: 'True: the training text is stored inside the model.', zh: '对：训练文本存在模型里面。' },
			{ en: 'False: it computes a probability for every one of the 1,024 tokens from its weights, every time.', zh: '错：它每次都用权重为 1,024 个词元中的每一个计算一个概率。' }
		],
		answer: 1,
		explain: {
			en: 'The model is 364,040 numbers (1.46 MB), while its training text was 32.5 million tokens (about 100 MB): there is no room for a copy. Every prediction is a fresh calculation that ends in 1,024 probabilities. Training only adjusts the weights so that, on average, the real next token gets more of that probability.',
			zh: '整个模型只有 364,040 个数（1.46 MB），而训练文本有 3,250 万个词元（约 100 MB）：根本放不下一份拷贝。每次预测都是一次全新的计算，结果是 1,024 个概率。训练只是调整权重，让真正的下一个词元平均来说分到更多概率。'
		}
	};

	// ---- 2. lower loss is not truth
	let pair = $state.raw<{ a: ScoreResult; b: ScoreResult } | null>(null);
	let root: HTMLElement | undefined = $state();
	let visible = $state(false);
	onMount(() => {
		let io: IntersectionObserver | null = null;
		if (root && 'IntersectionObserver' in window) {
			io = new IntersectionObserver((es) => es.forEach((e) => (visible = e.isIntersecting)), { rootMargin: '300px 0px' });
			io.observe(root);
		} else visible = true;
		return () => io?.disconnect();
	});
	$effect(() => {
		const c = S.client;
		const tok = S.tok;
		if (!visible || !c || !tok || pair) return;
		const ia = tok.encode(TRUTH_PAIR.falseText);
		const ib = tok.encode(TRUTH_PAIR.trueText);
		Promise.all([c.score('final', ia, 'myth-a', 1), c.score('final', ib, 'myth-b', 1)]).then(([a, b]) => {
			if (a && b) pair = { a, b };
		});
	});
	const bitsScale = $derived(makeScale('sequential', [0, 14], themeState.resolved));
	const total = (r: ScoreResult) => r.meanBits * r.scored;
	const rows = $derived(
		pair
			? [
					{ r: pair.a, label: { en: 'false', zh: '错误' }, text: TRUTH_PAIR.falseText },
					{ r: pair.b, label: { en: 'true', zh: '正确' }, text: TRUTH_PAIR.trueText }
				]
			: []
	);
	const top1Share = $derived.by(() => {
		if (!sc || sc.scored === 0) return null;
		let c = 0;
		for (let i = 1; i < sc.rank.length; i++) if (sc.rank[i] === 1) c++;
		return c / sc.scored;
	});
</script>

<div class="pit" bind:this={root}>
	<section class="myth">
		<h3>
			<span class="badge-m">{t({ en: 'Misconception', zh: '误区' })}</span>
			{t({ en: '“The model looks up the right answer.”', zh: '“模型是去查找正确答案的。”' })}
		</h3>
		<Lang>
			{#snippet en()}
				<p>
					It never outputs one answer. Each time, it outputs a full probability distribution over its vocabulary, computed
					from its weights. Even when it is right, the right token is only the biggest share.
				</p>
			{/snippet}
			{#snippet zh()}
				<p>它从来不只输出一个答案。每一次，它都根据权重算出一整个覆盖全部词表的概率分布。即使猜对了，正确的词元也只是分到了最大的那一份。</p>
			{/snippet}
		</Lang>
		{#if spread && pos !== null && sc}
			<div class="evidence">
				<p class="small">
					{t({ en: `Your text, position ${pos} (select a token above to change it):`, zh: `你的文本，位置 ${pos}（在上方选中别的词元即可切换）：` })}
					<ProvenanceBadge mode="live" compact />
				</p>
				<div class="spreadbar" aria-hidden="true">
					{#if spread.isTop}
						<span class="seg a" style:flex-grow={spread.pA}></span>
					{:else}
						<span class="seg tp" style:flex-grow={spread.top.p}></span>
						<span class="seg a" style:flex-grow={Math.max(spread.pA, 0.004)}></span>
					{/if}
					<span class="seg r" style:flex-grow={spread.rest}></span>
				</div>
				<ul class="small keys">
					{#if spread.isTop}
						<li><i class="sw a"></i>{t({ en: 'the real next token, also its top guess', zh: '真正的下一个词元，也是它的第一猜测' })} <span class="tokc">{visibleToken(S.disp(spread.actual))}</span>: {fmtPct(spread.pA)}</li>
					{:else}
						<li><i class="sw tp"></i>{t({ en: 'its top guess', zh: '它的第一猜测' })} <span class="tokc">{visibleToken(S.disp(spread.top.id))}</span>: {fmtPct(spread.top.p)}</li>
						<li><i class="sw a"></i>{t({ en: 'the real next token', zh: '真正的下一个词元' })} <span class="tokc">{visibleToken(S.disp(spread.actual))}</span>: {fmtPct(spread.pA)}</li>
					{/if}
					<li><i class="sw r"></i>{t({ en: `the other ${spread.others.toLocaleString('en-US')} tokens together`, zh: `其余 ${spread.others.toLocaleString('en-US')} 个词元合计` })}: {fmtPct(spread.rest)}</li>
				</ul>
			</div>
		{/if}
		<Quiz q={quiz1} />
	</section>

	<section class="myth" aria-live="polite">
		<h3>
			<span class="badge-m">{t({ en: 'Misconception', zh: '误区' })}</span>
			{t({ en: '“Lower loss means the model tells the truth.”', zh: '“损失越低，模型说的就越真实。”' })}
		</h3>
		<Lang>
			{#snippet en()}
				<p>
					The loss measures how well text matches the patterns the model has read, not whether it is true. TinyGPT read
					children's stories, not science books. Compare a false sentence with a true one:
				</p>
			{/snippet}
			{#snippet zh()}
				<p>损失衡量的是文本和模型读过的规律有多吻合，而不是它真不真。TinyGPT 读的是儿童故事，不是科学书。对比一句错误的话和一句正确的话：</p>
			{/snippet}
		</Lang>
		{#if rows.length}
			<div class="pairs">
				{#each rows as row (row.text)}
					<div class="pr">
						<span class="tag {row.label.en}">{t(row.label)}</span>
						<span class="sent">
							{#each row.r.ids as id, i (i)}
								<span
									class="tk"
									style:background={i === 0 ? null : bitsScale.css(-Math.log2(row.r.p[i]))}
									style:color={i === 0 ? null : bitsScale.ink(-Math.log2(row.r.p[i]))}
									title={i === 0 ? '' : `${fmt3(-Math.log2(row.r.p[i]))} bits`}>{S.disp(id)}</span
								>
							{/each}
						</span>
						<span class="nums small">
							<span><Num value={row.r.meanNats} /> {t({ en: 'nats/token', zh: '奈特/词元' })}</span>
							<span class="muted">{t({ en: 'total', zh: '合计' })} <Num value={total(row.r)} /> {t({ en: 'bits', zh: '比特' })}</span>
						</span>
					</div>
				{/each}
				<p class="small muted src">
					{t({ en: 'TinyGPT, final weights, computed in your browser. Darker = more surprising (bits).', zh: 'TinyGPT 最终权重，在你的浏览器中计算。颜色越深 = 越意外（比特）。' })}
					<ProvenanceBadge mode="live" compact />
				</p>
			</div>
			<Lang>
				{#snippet en()}
					<p>
						The false sentence wins on both counts, and not because TinyGPT “believes” the moon is cheese: the tokens
						“ che” and “ rock” cost about the same (7.4 and 7.3 bits). The difference is the ending. After “ che”,
						finishing the word and the full stop cost about 2 bits in total. After “ rock”, TinyGPT puts 94% of its
						probability on “s”, because “made of rocks” is the usual phrase in its stories, so the full stop costs about
						9 bits. Nothing in the loss asks whether a sentence is true. Big models
						trained on the web are the same kind of thing at a larger scale: they learn what text <em>usually</em> says,
						which is often, but not always, what is true. That is one reason chat models get further training after
						pre-training.
					</p>
				{/snippet}
				{#snippet zh()}
					<p>
						错误的那句在两项上都赢了，但这并不是因为 TinyGPT“相信”月亮是奶酪做的：“ che”和“ rock”这两个词元的代价差不多（分别约 7.4 和 7.3 比特）。差别在结尾。“ che”之后，把单词补完再加上句号，一共只花大约 2 比特；而“ rock”之后，TinyGPT 把 94% 的概率押在“s”上，因为在它读过的故事里，常见的说法是“made of rocks”，所以句号要花大约 9 比特。损失里没有任何一项在问一句话是不是真的。在网络文本上训练的大模型本质相同，只是规模更大：它们学到的是文本<em>通常</em>怎么说，这往往是真的，但并不总是。这也是聊天模型在预训练之后还要接着训练的原因之一。
					</p>
				{/snippet}
			</Lang>
		{:else}
			<p class="muted small">{t({ en: 'Scoring the two sentences…', zh: '正在给两个句子打分…' })}</p>
		{/if}
	</section>

	<section class="more">
		<h3>{t({ en: 'Three more traps', zh: '另外三个陷阱' })}</h3>
		<Lang>
			{#snippet en()}
				<ul>
					<li>
						<strong>“Perplexity 6.81 means it is right one time in seven.”</strong> No. <Term id="perplexity" /> is e to the
						average −ln p, a geometric mean. How often the top guess is right is a different number: on your text it was
						{top1Share === null ? '…' : fmtPct(top1Share, 0)} of tokens.
					</li>
					<li>
						<strong>“A loss can't be worse than guessing.”</strong> It can. Guessing evenly costs ln 1024 = 6.93 nats, but a
						trained model is confident, and confidently wrong is expensive: TinyGPT scores 我爱自然语言处理 at about 14.9 nats
						per token.
					</li>
					<li>
						<strong>“Model A has lower perplexity than model B, so it is better.”</strong> Only if both were measured on the
						same text with the same tokenizer. Change the tokenizer and the number of tokens changes, and so does the average.
					</li>
				</ul>
			{/snippet}
			{#snippet zh()}
				<ul>
					<li>
						<strong>“困惑度 6.81 意味着它七次里猜对一次。”</strong>不是。<Term id="perplexity" />是 e 的“平均 −ln p”次方，是一种几何平均。第一猜测有多常猜中，是另一个数：在你的文本上，这个比例是 {top1Share === null ? '…' : fmtPct(top1Share, 0)}。
					</li>
					<li>
						<strong>“损失不可能比瞎猜还差。”</strong>可能。平均瞎猜的代价是 ln 1024 = 6.93 奈特，但训练过的模型很自信，而自信地猜错代价很高：TinyGPT 在“我爱自然语言处理”上的损失约为每个词元 14.9 奈特。
					</li>
					<li>
						<strong>“模型 A 的困惑度比 B 低，所以 A 更好。”</strong>只有在同一份文本、同一个分词器上测出来才能这么比。换了分词器，词元个数就变了，平均值也随之改变。
					</li>
				</ul>
			{/snippet}
		</Lang>
	</section>
</div>

<style>
	.pit {
		display: grid;
		gap: var(--s5);
		max-width: none;
	}
	.myth,
	.more {
		display: grid;
		gap: 10px;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-lg);
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 10px;
	}
	.badge-m {
		font-family: var(--font-body);
		font-size: var(--fs-xs);
		font-weight: 650;
		padding: 1px 8px;
		border-radius: var(--r-pill);
		background: var(--illus-soft);
		color: var(--illus);
		border: 1px solid currentColor;
	}
	p {
		margin: 0;
	}
	ul {
		margin: 0;
		padding-left: 1.2em;
	}
	li + li {
		margin-top: 8px;
	}
	.evidence {
		display: grid;
		gap: 8px;
		padding: 12px 14px;
		border-radius: var(--r3);
		background: var(--surface-2);
	}
	.evidence p {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.spreadbar {
		display: flex;
		gap: 2px;
		height: 16px;
		border-radius: 4px;
		overflow: hidden;
	}
	.seg {
		display: block;
		flex-basis: 0;
		min-width: 3px;
	}
	.seg.a,
	.sw.a {
		background: var(--real);
	}
	.seg.tp,
	.sw.tp {
		background: var(--seq-hi);
	}
	.seg.r,
	.sw.r {
		background: var(--surface-3);
	}
	.keys {
		list-style: none;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	.keys li + li {
		margin-top: 0;
	}
	.sw {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 2px;
		margin-right: 6px;
	}
	.tokc {
		font-family: var(--font-mono);
		background: var(--surface);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
	}
	.pairs {
		display: grid;
		gap: 8px;
	}
	.pr {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		gap: 8px 12px;
		align-items: center;
		padding: 8px 10px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	@media (max-width: 560px) {
		.pr {
			grid-template-columns: auto minmax(0, 1fr);
		}
		.nums {
			grid-column: 1 / -1;
		}
	}
	.tag {
		font-size: var(--fs-xs);
		font-weight: 650;
		padding: 1px 8px;
		border-radius: var(--r-pill);
		border: 1px solid var(--rule-strong);
	}
	.sent {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		line-height: 1.9;
	}
	.tk {
		border-radius: 3px;
		padding: 1px 0;
	}
	.nums {
		display: grid;
		justify-items: end;
		gap: 2px;
	}
	@media (max-width: 560px) {
		.nums {
			justify-items: start;
		}
	}
	.src {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.small {
		font-size: var(--fs-xs);
	}
</style>
