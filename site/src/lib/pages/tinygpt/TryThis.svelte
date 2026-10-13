<!--
  "Try this": guided experiments that set the page state and point at the view to watch, each with a
  predict-first prompt; a contrast pair (temperature 0 vs 1.5, generated live in the worker); and one
  misconception check backed by the live logit lens.
-->
<script lang="ts">
	import Lang from '$lib/kit/Lang.svelte';
	import Term from '$lib/kit/Term.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmtPct } from '$lib/kit/format';
	import type { TGState } from './state.svelte';

	interface Props {
		S: TGState;
		load: (text: string) => Promise<void>;
		goto: (id: string) => void;
		setMetric: (m: 'prev' | 'first' | 'self') => void;
		compare: (n: number, hot: number, seed: number) => Promise<{ greedy: string; hot: string; ms: number } | null>;
		seed: number;
		promptText: string;
	}
	let { S, load, goto, setMetric, compare, seed, promptText }: Props = $props();

	type Exp = { id: string; title: Text; predict: Text; look: Text; button: Text; run: () => Promise<void> };
	const LONG = 'Tom and his dog went to the park. They played with a red ball. Tom threw the ball and the dog ran after it. The dog was';
	const LENS = 'Once upon a time, there was a little girl named Lily. She loved to play outside with her';
	const exps: Exp[] = [
		{
			id: 'cat',
			title: { en: '“The cat sat on the”', zh: '“The cat sat on the”' },
			predict: { en: 'Guess the most likely next token before you look. Will it be a whole word?', zh: '先猜一猜：最可能的下一个词元是什么？它会是一个完整的词吗？' },
			look: { en: 'In Output, read the top bars. A 1,024-token vocabulary splits rarer words, so the top guess can be a word piece (look for a leading ␣ and an incomplete word).', zh: '在“输出”里看最上面的几条。只有 1024 个词元的词表会把不常见的词拆开，所以第一名可能只是一个词片（注意开头的 ␣ 和不完整的词）。' },
			button: { en: 'Load and show the output', zh: '载入并查看输出' },
			run: async () => {
				await load('The cat sat on the');
				goto('output');
			}
		},
		{
			id: 'prev',
			title: { en: 'Find a head that looks at the previous token', zh: '找一个“看前一个词元”的头' },
			predict: { en: 'Which block do you expect to hold a head that mostly attends one token back: an early one or a late one?', zh: '你觉得哪个块里会有一个主要关注前一个词元的头：靠前的还是靠后的？' },
			look: { en: 'The number under each map is the average weight on the previous token; ★ marks the highest. Open that head and hover the text: does each token light up the one just before it? (In a small model the pattern is partial, not perfect.)', zh: '每张图下方的数字是落在前一个词元上的平均权重，★ 标出最高者。打开那个头，在文本上悬停：每个词元是否点亮了它前面的那个？（小模型里这种模式只是部分出现，并不完美。）' },
			button: { en: 'Load a story and open the heads', zh: '载入故事并打开各个头' },
			run: async () => {
				await load(LONG);
				setMetric('prev');
				goto('attention');
			}
		},
		{
			id: 'lens',
			title: { en: 'Watch the logit lens decide', zh: '看 logit lens 何时做出决定' },
			predict: { en: 'Will the final answer already be the top guess after block 1, or only near the end?', zh: '最终答案在第 1 块之后就已经是第一猜测了，还是要到快结束时才出现？' },
			look: { en: 'The ladder shows the top guess after every step and the probability of the final answer. Click other columns of the table: easy continuations (“ a” after “upon”, “ with” after “play”) are already the top guess after block 1 and stay on top; harder ones (“ girl” after “little”) only at the very end.', zh: '阶梯列表显示每一步之后的第一猜测以及最终答案的概率。点击表格的其他列：容易的接续（“upon”之后的“ a”、“play”之后的“ with”）在第 1 块之后就已是第一猜测，并一直保持；难的（“little”之后的“ girl”）要到最后一步才定。' },
			button: { en: 'Load and open the logit lens', zh: '载入并打开 logit lens' },
			run: async () => {
				await load(LENS);
				goto('lens');
			}
		},
		{
			id: 'zh',
			title: { en: 'Type Chinese', zh: '输入中文' },
			predict: { en: 'How many tokens will the 8 characters 我爱自然语言处理 become?', zh: '“我爱自然语言处理”这 8 个字会变成多少个词元？' },
			look: { en: 'Each character is 3 UTF-8 bytes and the tokenizer has no merges for them: 24 byte tokens. The model never saw these bytes in training, so the predictions are meaningless — but every number is still computed for real.', zh: '每个汉字是 3 个 UTF-8 字节，而分词器没有为它们学过合并规则：结果是 24 个字节词元。模型在训练时从没见过这些字节，所以预测没有意义——但每个数字仍然是真实计算出来的。' },
			button: { en: 'Load 我爱自然语言处理', zh: '载入“我爱自然语言处理”' },
			run: async () => {
				await load('我爱自然语言处理');
				goto('input');
			}
		}
	];

	// contrast pair: temperature 0 vs 1.5, same prompt, same seed
	let cmp = $state<{ greedy: string; hot: string; ms: number; prompt: string } | null>(null);
	let cmpBusy = $state(false);
	async function runCompare() {
		cmpBusy = true;
		try {
			const r = await compare(24, 1.5, seed);
			if (r) cmp = { ...r, prompt: promptText };
		} finally {
			cmpBusy = false;
		}
	}

	// misconception check
	let answer = $state<number | null>(null);
	const lens = $derived(S.run?.lens);
	const deltas = $derived.by(() => {
		if (!lens) return null;
		const T = lens.T;
		const pos = S.pos;
		const P = lens.points.length;
		const out: { id: string; d: number; kind: 'attn' | 'mlp' }[] = [];
		for (let p = 1; p < P; p++) {
			out.push({ id: lens.points[p], d: lens.pFinal[p * T + pos] - lens.pFinal[(p - 1) * T + pos], kind: lens.points[p].endsWith('resid1') ? 'attn' : 'mlp' });
		}
		const sum = (k: 'attn' | 'mlp') => out.filter((o) => o.kind === k).reduce((a, o) => a + o.d, 0);
		const maxAbs = Math.max(1e-6, ...out.map((o) => Math.abs(o.d)));
		return { out, attn: sum('attn'), mlp: sum('mlp'), maxAbs, p0: lens.pFinal[pos], pEnd: lens.pFinal[(P - 1) * T + pos], final: lens.finalTop[pos] };
	});
	const stepName = (id: string) => {
		const m = /^L(\d)\.resid(\d)$/.exec(id)!;
		const b = Number(m[1]) + 1;
		return m[2] === '1' ? t({ en: `B${b} attn`, zh: `第${b}块 注意力` }) : t({ en: `B${b} MLP`, zh: `第${b}块 MLP` });
	};
</script>

<div class="tt">
	<div class="cards">
		{#each exps as e (e.id)}
			<article class="card">
				<h3>{t(e.title)}</h3>
				<p class="pf"><span class="tag">{t({ en: 'Predict first', zh: '先预测' })}</span> {t(e.predict)}</p>
				<button type="button" class="btn primary" onclick={() => void e.run()}>{t(e.button)}</button>
				<details>
					<summary>{t({ en: 'What to look for', zh: '要观察什么' })}</summary>
					<p>{t(e.look)}</p>
				</details>
			</article>
		{/each}

		<article class="card wide">
			<h3>{t({ en: 'Temperature 0 vs 1.5 (a contrast pair)', zh: '温度 0 与 1.5 对比' })} <ProvenanceBadge mode="live" compact /></h3>
			<p class="pf">
				<span class="tag">{t({ en: 'Predict first', zh: '先预测' })}</span>
				{t({ en: 'Same prompt, same model. Which continuation will be safer and more repetitive, and which will invent odd words?', zh: '同样的提示词、同样的模型。哪一个续写会更保守、更爱重复，哪一个会编出奇怪的词？' })}
			</p>
			<button type="button" class="btn primary" onclick={runCompare} disabled={cmpBusy || !S.run}>
				{cmpBusy ? t({ en: 'Generating…', zh: '生成中…' }) : t({ en: 'Generate 24 tokens both ways', zh: '两种方式各生成 24 个词元' })}
			</button>
			{#if cmp}
				<div class="pair">
					<div>
						<p class="h">{t({ en: 'temperature 0 (greedy)', zh: '温度 0（贪心）' })}</p>
						<p class="story"><span class="pr">{cmp.prompt}</span><span class="gw">{cmp.greedy}</span></p>
					</div>
					<div>
						<p class="h">{t({ en: `temperature 1.5, seed ${seed}`, zh: `温度 1.5，种子 ${seed}` })}</p>
						<p class="story"><span class="pr">{cmp.prompt}</span><span class="gw">{cmp.hot}</span></p>
					</div>
				</div>
				<p class="muted small">{t({ en: `Both computed just now in your browser (${Math.round(cmp.ms)} ms). Greedy always gives the same text; at 1.5 a different seed gives a different text.`, zh: `两段都是刚才在你的浏览器里算出来的（${Math.round(cmp.ms)} 毫秒）。贪心总是给出同样的文本；温度 1.5 时换一个种子就会得到不同的文本。` })}</p>
			{/if}
		</article>
	</div>

	<article class="miscon" aria-labelledby="miscon-h">
		<h3 id="miscon-h">{t({ en: 'Misconception check', zh: '误区检查' })}</h3>
		<p class="claim">“{t({ en: 'The attention weights show what the model understands.', zh: '注意力权重显示了模型“理解”了什么。' })}”</p>
		<div class="opts" role="radiogroup" aria-label={t({ en: 'Is this right?', zh: '这句话对吗？' })}>
			<label class:on={answer === 0}><input type="radio" name="miscon" value={0} bind:group={answer} /> {t({ en: 'Yes, attention is the explanation', zh: '对，注意力就是解释' })}</label>
			<label class:on={answer === 1}><input type="radio" name="miscon" value={1} bind:group={answer} /> {t({ en: 'Only partly', zh: '只说对了一部分' })}</label>
		</div>
		{#if answer !== null}
			<div class="reveal">
				<p>
					<strong>{answer === 1 ? t({ en: 'Right.', zh: '没错。' }) : t({ en: 'Not quite.', zh: '不完全对。' })}</strong>
					<Lang>
						{#snippet en()}
							Attention weights only say <em>where</em> each head copies information from. <em>What</em> is copied depends on the
							values and the output projection, and half of each block is the <Term id="mlp" />, which never looks at other tokens
							but does a lot of the work. The <Term id="residual-connection" /> also carries the token's own embedding straight to
							the end. High attention on a token is not proof that the token mattered for the prediction.
						{/snippet}
						{#snippet zh()}
							注意力权重只说明每个头从<em>哪里</em>复制信息；复制<em>什么</em>取决于值向量和输出投影。而每个块的另一半是
							<Term id="mlp" />，它从不看其他词元，却承担了大量工作。<Term id="residual-connection" />还把词元自己的嵌入一路直接送到最后。某个词元得到很高的注意力，并不能证明它对预测很重要。
						{/snippet}
					</Lang>
				</p>
				{#if deltas}
					<p class="small">
						{t({ en: 'Live evidence for your text: how the probability of the final answer', zh: '基于你的文本的实时证据：最终答案' })}
						<span class="tok">{S.idLabel(deltas.final)}</span>
						{t({ en: `(after ${S.label(S.pos)}) changed at each step, from ${fmtPct(deltas.p0)} after the embedding to ${fmtPct(deltas.pEnd)} at the output (logit lens):`, zh: `（接在 ${S.label(S.pos)} 之后）的概率在每一步如何变化：从嵌入后的 ${fmtPct(deltas.p0)} 到输出时的 ${fmtPct(deltas.pEnd)}（logit lens）：` })}
					</p>
					<div class="dbars" role="img" aria-label={t({ en: 'Change of the final answer’s probability per step', zh: '每一步最终答案概率的变化' })}>
						{#each deltas.out as o (o.id)}
							<div class="db">
								<span class="dl">{stepName(o.id)}</span>
								<span class="track">
									<span class="mid"></span>
									<span class="fill" class:neg={o.d < 0} style="{o.d < 0 ? 'right' : 'left'}: 50%; width: {(Math.abs(o.d) / deltas.maxAbs) * 50}%"></span>
								</span>
								<span class="num dv">{o.d >= 0 ? '+' : '−'}{fmtPct(Math.abs(o.d))}</span>
							</div>
						{/each}
					</div>
					<p class="small">
						{t({ en: `Sum of the attention steps: ${deltas.attn >= 0 ? '+' : '−'}${fmtPct(Math.abs(deltas.attn))}; sum of the MLP steps: ${deltas.mlp >= 0 ? '+' : '−'}${fmtPct(Math.abs(deltas.mlp))}.`, zh: `注意力步骤合计：${deltas.attn >= 0 ? '+' : '−'}${fmtPct(Math.abs(deltas.attn))}；MLP 步骤合计：${deltas.mlp >= 0 ? '+' : '−'}${fmtPct(Math.abs(deltas.mlp))}。` })}
						<span class="muted">{t({ en: 'Try other tokens and texts: the MLP steps often move the prediction as much as attention, or more.', zh: '换其他词元和文本试试：MLP 步骤对预测的推动常常不亚于注意力，甚至更大。' })}</span>
					</p>
				{/if}
			</div>
		{/if}
	</article>
</div>

<style>
	.tt {
		display: grid;
		gap: 18px;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 270px), 1fr));
		gap: 12px;
	}
	.card {
		display: grid;
		align-content: start;
		gap: 8px;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.card.wide {
		grid-column: 1 / -1;
	}
	.card h3 {
		font-size: var(--fs-md);
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
	.card .btn {
		justify-self: start;
	}
	.pf {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.tag {
		display: inline-block;
		font-size: var(--t-xs);
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--accent-strong);
		background: var(--accent-soft);
		border-radius: var(--r1);
		padding: 0 5px;
		margin-right: 4px;
	}
	details p {
		margin: 6px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	summary {
		cursor: pointer;
		font-size: var(--fs-sm);
		color: var(--accent);
	}
	.pair {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
		gap: 12px;
	}
	.h {
		margin: 0 0 4px;
		font-size: var(--fs-xs);
		font-weight: 700;
		color: var(--ink-2);
	}
	.story {
		margin: 0;
		padding: 8px 10px;
		border-radius: var(--r2);
		background: var(--surface-2);
		font-family: var(--font-mono);
		font-size: 0.88rem;
		white-space: pre-wrap;
		word-break: break-word;
	}
	.pr {
		color: var(--ink-2);
	}
	.gw {
		color: var(--generated-text);
		font-weight: 600;
	}
	.small {
		font-size: var(--fs-sm);
		margin: 0;
	}
	.miscon {
		padding: 16px 18px;
		border-radius: var(--r3);
		border: 1.5px dashed var(--illus);
		background: var(--surface);
		display: grid;
		gap: 10px;
	}
	.miscon h3 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.claim {
		margin: 0;
		font-family: var(--font-display);
		font-size: var(--fs-lg);
	}
	.opts {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.opts label {
		display: flex;
		gap: 8px;
		align-items: center;
		padding: 6px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		cursor: pointer;
	}
	.opts label.on {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.opts input {
		accent-color: var(--accent);
	}
	.reveal {
		display: grid;
		gap: 8px;
	}
	.reveal p {
		margin: 0;
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 4px;
		white-space: pre;
	}
	.dbars {
		display: grid;
		gap: 3px;
		max-width: 560px;
	}
	.db {
		display: grid;
		grid-template-columns: 7em minmax(0, 1fr) 4.8em;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-xs);
	}
	.track {
		position: relative;
		height: 12px;
		background: var(--surface-2);
		border-radius: 3px;
	}
	.mid {
		position: absolute;
		left: 50%;
		top: -2px;
		bottom: -2px;
		border-left: 1px solid var(--ink-2);
	}
	.fill {
		position: absolute;
		top: 0;
		bottom: 0;
		background: var(--div-pos);
		border-radius: 2px;
	}
	.fill.neg {
		background: var(--div-neg);
	}
	.dv {
		text-align: right;
	}
</style>
