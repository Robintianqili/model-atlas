<!--
  Checkpoint questions (answered by looking at the widgets above), plus credits.
-->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	type Q = { id: string; q: Text; options: Text[]; answer: number; explain: Text; where?: { href: string; label: Text } };
	const QS: Q[] = [
		{
			id: 'dhead',
			q: { en: 'Find it in the block view: how many numbers does one head’s query vector have for one token?', zh: '在块视图里找答案：一个头的查询向量，对一个词元来说有多少个数？' },
			options: [
				{ en: '76', zh: '76' },
				{ en: '19', zh: '19' },
				{ en: '304', zh: '304' },
				{ en: '64', zh: '64' }
			],
			answer: 1,
			explain: { en: 'Q is [T×76], split into 4 heads of 19 columns each (76 / 4 = 19). That is why the scores are divided by √19.', zh: 'Q 是 [T×76]，分成 4 个头，每个头 19 列（76 / 4 = 19）。所以分数要除以 √19。' },
			where: { href: '#unit', label: { en: 'Inside a block, step 2', zh: '块的内部，第 2 步' } }
		},
		{
			id: 'mask',
			q: { en: 'Why is the upper-right triangle of every attention map hatched?', zh: '为什么每张注意力图的右上三角都画了阴影？' },
			options: [
				{ en: 'Those weights happen to be very small', zh: '那些权重恰好很小' },
				{ en: 'The causal mask: a token may not look at tokens after it', zh: '因果掩码：词元不能看它后面的词元' },
				{ en: 'The text is longer than the context window', zh: '文本超出了上下文窗口' }
			],
			answer: 1,
			explain: { en: 'Their scores are set to −∞ before the softmax, so their weights are exactly 0. Training asks the model to predict each next token, so it must not see it.', zh: '这些分数在 softmax 之前被设为 −∞，所以权重恰好为 0。训练要求模型预测每一个下一个词元，因此它不能先看到答案。' }
		},
		{
			id: 'seed',
			q: { en: 'Two people type the same prompt, use temperature 1 and the same seed, and generate 20 tokens. Their stories are…', zh: '两个人输入同样的提示词，用温度 1 和同一个种子各生成 20 个词元。他们的故事会……' },
			options: [
				{ en: 'identical', zh: '完全相同' },
				{ en: 'different, because sampling is random', zh: '不同，因为采样是随机的' }
			],
			answer: 0,
			explain: { en: 'The “random” numbers come from a seeded generator (mulberry32), so the same seed gives the same draws, and the model is deterministic. Change the seed (or the temperature) to get a different story.', zh: '“随机”数来自带种子的生成器（mulberry32），同一个种子给出同样的随机数，而模型本身是确定的。换一个种子（或温度）才会得到不同的故事。' }
		}
	];
	let picks = $state<Record<string, number>>({});
</script>

<div class="cpq">
	{#each QS as q (q.id)}
		<div class="q">
			<p class="qt" id="cpq-{q.id}">{t(q.q)}</p>
			<div class="opts" role="radiogroup" aria-labelledby="cpq-{q.id}">
				{#each q.options as o, i (i)}
					<label class:right={picks[q.id] !== undefined && i === q.answer} class:wrong={picks[q.id] === i && i !== q.answer}>
						<input type="radio" name="cp-{q.id}" value={i} bind:group={picks[q.id]} />
						{t(o)}
					</label>
				{/each}
			</div>
			{#if picks[q.id] !== undefined}
				<p class="ex" aria-live="polite">
					<strong>{picks[q.id] === q.answer ? t({ en: 'Correct.', zh: '正确。' }) : t({ en: 'Not quite.', zh: '不太对。' })}</strong>
					{t(q.explain)}
					{#if q.where}<a href={q.where.href}>{t(q.where.label)}</a>{/if}
				</p>
			{/if}
		</div>
	{/each}
</div>

<style>
	.cpq {
		display: grid;
		gap: 12px;
	}
	.q {
		margin: 0;
		padding: 12px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--surface);
	}
	.qt {
		font-weight: 650;
		margin: 0;
	}
	.opts {
		display: grid;
		gap: 4px;
		margin-top: 6px;
	}
	label {
		display: flex;
		gap: 8px;
		align-items: center;
		padding: 4px 8px;
		border-radius: var(--r1);
		border: 1px solid transparent;
		cursor: pointer;
	}
	label.right {
		border-color: var(--live);
		background: var(--live-soft);
	}
	label.wrong {
		border-color: var(--illus);
	}
	input {
		accent-color: var(--accent);
	}
	.ex {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
</style>
