<!--
  Checkpoint questions at the end of the page: one decision question and one "find the value
  in the widget" question (answered from Part 1 / Part 2).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';

	const q1 = {
		q: {
			en: 'A friend has a pre-trained image model and 20 labeled photos of each of two kinds of leaf. What should they try first?',
			zh: '朋友有一个预训练好的图像模型，以及两种树叶各 20 张带标签的照片。他们应该先试什么？'
		},
		options: [
			{ en: 'Fine-tune every layer with a large learning rate, so the model adapts fast.', zh: '用较大的学习率微调所有层，让模型快速适应。' },
			{ en: 'Freeze the pre-trained layers and train a new small head; fine-tune more only if a validation split says it helps.', zh: '冻结预训练层，训练一个新的小输出头；只有验证集表明有用时，才进一步微调更多层。' },
			{ en: 'Train a new network from scratch, so old knowledge cannot get in the way.', zh: '从零训练一个新网络，免得旧知识碍事。' }
		],
		answer: 1,
		explain: {
			en: 'With 20 examples per class, the frozen head has the fewest parameters to fit and starts from useful features. On TinyCNN with 16 per class it scored 97.9% vs 97.6% (full fine-tune) and 95.2% (from scratch); with fewer examples the gap is larger. A large learning rate makes small-data fine-tuning even more fragile (the BERT-Large runs on RTE).',
			zh: '每类 20 个样本时，冻结主体 + 新输出头要拟合的参数最少，而且起点就是有用的特征。在 TinyCNN 上，每类 16 张时它得到 97.9%，全量微调 97.6%，从零训练 95.2%；样本更少时差距更大。较大的学习率会让小数据微调更加脆弱（见 BERT-Large 在 RTE 上的实验）。'
		}
	};
	let c1 = $state<number | null>(null);
	let v2 = $state('');
	let checked2 = $state(false);
	const ok2 = $derived(Number(v2.replace(/[,\s]/g, '')) === 514);
</script>

<div class="cp">
	<div class="q">
		<p class="qt" id="cp-q1">1. {t(q1.q)}</p>
		<div class="opts" role="radiogroup" aria-labelledby="cp-q1">
			{#each q1.options as o, i (i)}
				<label class="opt" class:on={c1 === i}>
					<input type="radio" name="cp-q1" value={i} bind:group={c1} />
					<span>{t(o)}</span>
				</label>
			{/each}
		</div>
		{#if c1 !== null}
			<p class="ans" class:ok={c1 === q1.answer} role="status">
				<strong>{c1 === q1.answer ? t({ en: 'Right.', zh: '答对了。' }) : t({ en: 'Not quite.', zh: '不太对。' })}</strong>
				{t(q1.explain)}
			</p>
		{/if}
	</div>
	<div class="q">
		<p class="qt"><label for="cp-q2">2. {t({ en: 'Find it on the page: how many parameters did your new head train?', zh: '在页面上找答案：你的新输出头训练了多少个参数？' })}</label></p>
		<div class="row">
			<input id="cp-q2" type="text" inputmode="numeric" maxlength="8" bind:value={v2} onkeydown={(e) => e.key === 'Enter' && (checked2 = true)} />
			<button type="button" class="btn" onclick={() => (checked2 = true)}>{t({ en: 'Check', zh: '检查' })}</button>
		</div>
		{#if checked2}
			<p class="ans" class:ok={ok2} role="status">
				{#if ok2}
					<strong>{t({ en: 'Right: 514.', zh: '答对了：514。' })}</strong>
					{t({ en: '256 features × 2 classes + 2 biases. The other 4,152 parameters stayed frozen.', zh: '256 个特征 × 2 个类别 + 2 个偏置。其余 4,152 个参数一直是冻结的。' })}
				{:else}
					<strong>{t({ en: 'Not yet.', zh: '还不对。' })}</strong>
					{t({ en: 'Look at the “New head” block in Part 1, or the button next to Train.', zh: '看看第 1 部分里的“新输出头”这一块，或者“训练”按钮旁边的说明。' })}
				{/if}
			</p>
		{/if}
	</div>
</div>

<style>
	.cp {
		display: grid;
		gap: var(--s4);
	}
	.q {
		display: grid;
		gap: 8px;
		padding: 14px 16px;
		border-radius: var(--r3);
		border: 1px solid var(--rule);
		background: var(--surface);
	}
	.qt {
		margin: 0;
		font-weight: 650;
	}
	.opts {
		display: grid;
		gap: 6px;
	}
	.opt {
		display: flex;
		gap: 10px;
		align-items: flex-start;
		padding: 8px 12px;
		border-radius: var(--r2);
		border: 1px solid var(--rule);
		cursor: pointer;
	}
	.opt.on {
		border-color: var(--accent);
	}
	.opt input {
		margin-top: 0.3em;
		accent-color: var(--accent);
	}
	.row {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
	}
	.row input {
		width: 8em;
		min-height: 40px;
		padding: 4px 10px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		background: var(--surface);
		color: var(--ink);
		font: inherit;
		font-family: var(--font-mono);
	}
	.ans {
		margin: 0;
		padding: 8px 12px;
		border-left: 3px solid var(--illus);
		background: var(--illus-soft);
		border-radius: var(--r1);
		max-width: var(--measure);
	}
	.ans.ok {
		border-left-color: var(--live);
		background: var(--live-soft);
	}
</style>
