<!--
  train-step: one training step, phase by phase (forward → loss → backward → update) for a fine-detail train source.
  Every number is read from the recorded frame; the update shows the rule that actually ran.
-->
<script lang="ts">
	import type { StageProps } from '../../../types/view.ts';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { Tensor } from '../../../types/core.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtSetting, paren } from '../../../core/format.ts';
	import Value from '../../train/Value.svelte';
	import MiniGrid from '../../train/MiniGrid.svelte';
	import meta, { PHASES, phaseName, stepData, updateDigits } from './meta.ts';

	// props one by one (no rest/spread props: they would grow the Svelte runtime chunk every page preloads)
	let { rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props }: StageProps = $props();
	const p: StageProps = $derived({ rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props });
	const d = $derived(stepData(p));
	const nd = $derived(updateDigits(d, p.props?.digits));
	const rows = $derived(Math.max(1, Math.min(8, Math.floor(Number(p.props?.rows ?? 4)) || 4)));

	const cols = (x: Tensor | undefined) => (x && x.shape.length === 2 ? x.shape[1] : 1);
	const row = (x: Tensor | undefined, r: number) => {
		if (!x) return [];
		const c = cols(x);
		return Array.from({ length: c }, (_, k) => x.data[r * c + k]);
	};
	const batchKeys = $derived(Object.keys(d.batch));
	const B = $derived(batchKeys.length ? d.batch[batchKeys[0]].shape[0] : 0);
	const focusGrad = $derived(d.key ? d.grads[d.key] : undefined);
	const where = $derived(d.key && d.index ? `${d.key}[${d.index.join(', ')}]` : '');
	const description = $derived(t(meta.describe(p)));
</script>

<div class="ts" style:min-height="{meta.minHeight}px">
	<ol class="phases" aria-label={t({ en: 'The four parts of one training step', zh: '一步训练的四个部分' })}>
		{#each PHASES as ph, i (ph)}
			<li class:on={d.phase === ph} aria-current={d.phase === ph ? 'step' : undefined}><span class="n">{i + 1}</span>{t(phaseName(ph))}</li>
		{/each}
	</ol>

	{#if !d.phase}
		<p class="muted">{description}</p>
	{:else}
		<p class="head small muted">{t({ en: `Training step ${d.step}`, zh: `第 ${d.step} 步训练` })} · {t({ en: 'learning rate', zh: '学习率' })} <Value value={d.lr} /> · {d.optimizer || '—'}</p>

		{#if d.phase === 'forward'}
			<p>{t({ en: `A batch of ${B} examples goes through the network. The first ${Math.min(rows, B)}:`, zh: `一批 ${B} 个样例通过网络。前 ${Math.min(rows, B)} 个：` })}</p>
			<div class="scroll" use:keyScroll>
				<table class="num-table">
					<thead>
						<tr>
							<th>#</th>
							{#each batchKeys as k (k)}<th>{k}</th>{/each}
							{#if d.output}<th>{t({ en: 'network output', zh: '网络输出' })}</th>{/if}
						</tr>
					</thead>
					<tbody>
						{#each Array.from({ length: Math.min(rows, B) }, (_, r) => r) as r (r)}
							<tr>
								<td class="muted">{r}</td>
								{#each batchKeys as k (k)}<td class="num">{row(d.batch[k], r).map((v) => fmt3(v)).join(', ')}</td>{/each}
								{#if d.output}<td class="num">{row(d.output, r).map((v) => fmt3(v)).join(', ')}</td>{/if}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else if d.phase === 'loss'}
			<p class="big"><span class="lbl">{t({ en: 'loss', zh: '损失' })}</span> <Value value={d.loss} digits={4} /></p>
			<p class="small muted">{t({ en: 'One number for the whole batch: the average of how wrong each example is.', zh: '整批只有一个数：每个样例错了多少的平均值。' })}</p>
		{:else if d.phase === 'backward'}
			<p>{t({ en: 'Backpropagation gives every weight a gradient. For the weight in focus:', zh: '反向传播给每个权重一个梯度。对焦点中的权重：' })}</p>
			<p class="big"><span class="lbl">∂loss/∂{where}</span> <Value value={d.g} digits={4} signed /></p>
			{#if focusGrad && focusGrad.shape.length === 2}
				<MiniGrid
					data={focusGrad.data}
					rows={focusGrad.shape[0]}
					cols={focusGrad.shape[1]}
					label={t({ en: `gradient of ${d.key}`, zh: `${d.key} 的梯度` })}
					mark={d.index && d.index.length === 2 ? [d.index[0], d.index[1]] : null}
				/>
			{/if}
			<p class="small muted">{t({ en: 'Size of all gradients together', zh: '全部梯度合起来的大小' })}: <Value value={d.gradNorm} /></p>
		{:else}
			<p>{t({ en: `The weight ${where}:`, zh: `权重 ${where}：` })}</p>
			{#if d.sgd}
				<p class="eq num">
					<span title={t({ en: 'old value', zh: '原来的值' })}>{fmt3(d.old, nd)}</span> − <span title={t({ en: 'learning rate', zh: '学习率' })}>{fmtSetting(d.lr)}</span> × <span title={t({ en: 'gradient', zh: '梯度' })}>{paren(d.g)}</span> = <strong>{fmt3(d.now, nd)}</strong>
				</p>
				<p class="small muted">{t({ en: 'old − learning rate × gradient = new (plain gradient descent).', zh: '旧值 − 学习率 × 梯度 = 新值（普通梯度下降）。' })}</p>
			{:else}
				<dl class="kv">
					<div><dt>{t({ en: 'before', zh: '之前' })}</dt><dd><Value value={d.old} digits={4} /></dd></div>
					<div><dt>{t({ en: 'after', zh: '之后' })}</dt><dd><Value value={d.now} digits={4} /></dd></div>
					<div><dt>{t({ en: 'change', zh: '变化' })}</dt><dd><Value value={d.now - d.old} digits={3} signed /></dd></div>
					<div><dt>{t({ en: 'gradient', zh: '梯度' })}</dt><dd><Value value={d.g} signed /></dd></div>
					{#if Number.isFinite(d.m)}<div><dt>m</dt><dd><Value value={d.m} signed /></dd></div>{/if}
					{#if Number.isFinite(d.v)}<div><dt>v</dt><dd><Value value={d.v} /></dd></div>{/if}
				</dl>
				<p class="small muted">
					{d.optimizer === 'adam' || d.optimizer === 'adamw'
						? t({ en: 'Adam does not move by learning rate × gradient: it divides a running average of the gradient (m) by the running size of the gradient (√v), so every weight moves by roughly the learning rate.', zh: 'Adam 并不按“学习率 × 梯度”移动：它用梯度的滑动平均（m）除以梯度的滑动大小（√v），所以每个权重大约移动一个学习率那么多。' })
						: t({ en: 'This optimizer does not use the plain rule old − learning rate × gradient.', zh: '这个优化器用的不是“旧值 − 学习率 × 梯度”的简单规则。' })}
				</p>
			{/if}
		{/if}
	{/if}
	<p class="visually-hidden" aria-live="polite">{description}</p>
</div>

<style>
	.ts {
		display: grid;
		gap: var(--s2);
		align-content: start;
		min-width: 0;
	}
	.phases {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.phases li {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 3px 10px 3px 4px;
		border-radius: var(--r-pill);
		border: 1px solid var(--rule);
		color: var(--muted);
		font-size: 13px;
	}
	.phases li.on {
		border-color: var(--accent);
		background: var(--accent-soft);
		color: var(--ink);
		font-weight: 600;
	}
	.n {
		display: inline-grid;
		place-items: center;
		width: 20px;
		height: 20px;
		border-radius: 50%;
		background: var(--surface-2);
		font: 12px var(--font-mono);
	}
	.head {
		margin: 0;
	}
	p {
		margin: 0;
	}
	.big {
		font-size: 1.4rem;
	}
	.lbl {
		color: var(--muted);
		font-size: 0.95rem;
		margin-right: 6px;
	}
	.eq {
		font-size: 1.25rem;
	}
	.scroll {
		overflow-x: auto;
		max-width: 100%;
	}
	.num-table {
		border-collapse: collapse;
		font-size: 13px;
	}
	.num-table th,
	.num-table td {
		padding: 3px 10px 3px 0;
		text-align: left;
		white-space: nowrap;
	}
	.num-table th {
		color: var(--muted);
		font-weight: 500;
	}
	.kv {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s2) var(--s4);
		margin: 0;
	}
	.kv div {
		display: grid;
		gap: 2px;
	}
	dt {
		font-size: 12px;
		color: var(--muted);
	}
	dd {
		margin: 0;
	}
</style>
