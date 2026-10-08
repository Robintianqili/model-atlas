<!--
  Logit lens ("when does it decide?"): the final LayerNorm and the output head applied to the
  residual stream after the embedding and after every attention / MLP sub-layer, for every
  position. Computed live in the worker (lens.ts); the last row is exactly the model's output.
-->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { sym } from '$lib/kit/tex';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmtPct } from '$lib/kit/format';
	import type { TGState } from './state.svelte';

	type Part = 'intro' | 'ladder' | 'table';
	interface Props {
		S: TGState;
		/** which pieces to show (the lesson shows the ladder only); default all */
		parts?: Part[];
	}
	let { S, parts = ['intro', 'ladder', 'table'] }: Props = $props();
	const has = (p: Part) => parts.includes(p);

	let fine = $state(true); // show after-attention rows too
	const L = $derived(S.run?.lens);
	const T = $derived(L?.T ?? 0);
	const pos = $derived(S.pos);
	const rows = $derived(L ? L.points.map((id, p) => ({ id, p })).filter((r) => fine || r.id === 'embed' || r.id.endsWith('resid2')) : []);
	const sc = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	let heads: HTMLButtonElement[] = $state([]);

	function pointLabel(id: string): Text {
		if (id === 'embed') return { en: 'Embedding (before block 1)', zh: '嵌入（第 1 块之前）' };
		const m = /^L(\d)\.resid(\d)$/.exec(id)!;
		const b = Number(m[1]) + 1;
		return m[2] === '1'
			? { en: `Block ${b} · after attention`, zh: `第 ${b} 块 · 注意力之后` }
			: { en: `Block ${b} · after MLP`, zh: `第 ${b} 块 · MLP 之后` };
	}
	const cellTop = (p: number, t: number, k = 0) => {
		const o = (p * L!.T + t) * L!.K + k;
		return { id: L!.topIds[o], p: L!.topP[o] };
	};
	const decided = $derived.by(() => {
		if (!L) return -1;
		const P = L.points.length;
		let first = P - 1;
		for (let p = P - 1; p >= 0; p--) {
			if (cellTop(p, pos).id === L.finalTop[pos]) first = p;
			else break;
		}
		return first;
	});
	function onkeydown(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight') j = Math.min(T - 1, i + 1);
		else if (e.key === 'ArrowLeft') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = T - 1;
		else return;
		e.preventDefault();
		S.selectPos(j);
		heads[j]?.focus();
	}
	const short = (s: string) => ([...s].length > 7 ? [...s].slice(0, 6).join('') + '…' : s);
</script>

{#if L}
	<div class="ll" class:one={!has('intro') || !has('ladder')}>
		{#if has('intro')}
		<div class="intro">
			<FormulaCard
				tex={`\\text{lens}_\\ell(t) = \\operatorname{softmax}\\big(\\mathrm{LN}_f(${sym('act', 'x^{(\\ell)}_t')})\\; ${sym('param', 'E')}^{\\top}\\big)`}
				caption={{ en: 'Read the residual stream early', zh: '提前读出残差流' }}
				where={[
					{ tex: sym('act', 'x^{(\\ell)}_t'), text: { en: 'residual stream of token t after step ℓ', zh: '第 ℓ 步之后词元 t 的残差流' } },
					{ tex: `\\mathrm{LN}_f,\\ ${sym('param', 'E')}`, text: { en: 'the model’s own final LayerNorm and output head', zh: '模型自己的最终层归一化和输出头' } }
				]}
				provenance="live"
			/>
			<p class="expl">
				{t({
					en: 'The model only reads out its prediction at the very end. The logit lens asks: what if we read it out earlier? We apply the same final LayerNorm and the same output head to the residual stream after each step. The last row is exactly the real output; earlier rows show the guess forming.',
					zh: '模型只在最后一步读出预测。logit lens 的问题是：如果更早读出会怎样？我们把同样的最终层归一化和输出头作用在每一步之后的残差流上。最后一行就是真实输出；前面几行展示了猜测是如何逐步成形的。'
				})}
			</p>
			<label class="tg">
				<input type="checkbox" bind:checked={fine} />
				{t({ en: 'also show the stream after each attention step (9 rows instead of 5)', zh: '同时显示每个注意力步骤之后的残差流（9 行而不是 5 行）' })}
			</label>
		</div>
		{/if}

		{#if has('ladder')}
		<div class="ladder" aria-live={has('table') ? 'polite' : undefined}>
			<h4>
				{t({ en: 'After', zh: '在' })} <span class="tok">{S.label(pos)}</span>{t({ en: ', the guess at each step', zh: ' 之后，每一步的猜测' })}
			</h4>
			<ol>
				{#each rows as r (r.id)}
					{@const top = cellTop(r.p, pos)}
					{@const pf = L.pFinal[r.p * T + pos]}
					<li class:decided={r.p >= decided} class:final={r.p === L.points.length - 1}>
						<span class="pl">{t(pointLabel(r.id))}</span>
						<span class="tok big" style="background: {sc.css(top.p)}; color: {sc.ink(top.p)}">{S.idLabel(top.id)}</span>
						<span class="num pp">{fmtPct(top.p)}</span>
						<span class="others">
							{#each [1, 2] as k (k)}
								{@const o = cellTop(r.p, pos, k)}
								<span class="tok sm">{S.idLabel(o.id)}</span>
							{/each}
						</span>
						<span class="final-p" title={t({ en: 'probability of the final answer at this step', zh: '最终答案在这一步的概率' })}>
							<span class="bar"><span style="width: {pf * 100}%"></span></span>
							<span class="num">{fmtPct(pf)} · #{L.rankFinal[r.p * T + pos]}</span>
						</span>
					</li>
				{/each}
			</ol>
			<p class="verdict">
				{t({ en: 'Final answer', zh: '最终答案' })} <span class="tok">{S.idLabel(L.finalTop[pos])}</span>:
				{#if decided === 0}
					{t({ en: 'already the top guess right after the embedding (no block needed).', zh: '嵌入之后就已经是第一猜测（不需要任何块）。' })}
				{:else}
					{t({ en: 'it becomes the top guess, and stays there, at', zh: '从这一步开始成为第一猜测并保持不变：' })} <b>{t(pointLabel(L.points[decided]))}</b>{t({ en: '.', zh: '。' })}
				{/if}
				<span class="muted">{t({ en: 'Bars: probability of the final answer at each step (# = its rank).', zh: '条形：最终答案在每一步的概率（# 为排名）。' })}</span>
				<span class="muted">{t({ en: 'The first row usually “predicts” the input token itself: the output head is the embedding table, so a token’s own embedding matches its own row best. The blocks have to move the vector away from that.', zh: '第一行通常“预测”出输入词元本身：输出头就是嵌入表，词元自己的嵌入与自己那一行最相近。各个块的工作就是把向量从那里推开。' })}</span>
			</p>
		</div>
		{/if}

		{#if has('table')}
		<div class="tablewrap">
			<table>
				<caption class="visually-hidden">{t({ en: 'Logit lens: top guess for every position (columns) after every step (rows)', zh: 'Logit lens：每个位置（列）在每一步（行）之后的第一猜测' })}</caption>
				<thead>
					<tr>
						<th scope="col" class="corner">{t({ en: 'input →', zh: '输入 →' })}</th>
						{#each S.labels as lab, i (i)}
							<th scope="col" class:cur={i === pos}>
								<button
									bind:this={heads[i]}
									type="button"
									class="colbtn"
									tabindex={i === pos ? 0 : -1}
									onclick={() => S.selectPos(i)}
									onkeydown={(e) => onkeydown(e, i)}
									aria-pressed={i === pos}>{short(lab)}</button
								>
							</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each rows as r (r.id)}
						<tr class:final={r.p === L.points.length - 1}>
							<th scope="row" class="rl">{t(pointLabel(r.id))}</th>
							{#each S.labels as _lab, i (i)}
								{@const c = cellTop(r.p, i)}
								<td
									class:cur={i === pos}
									class:match={c.id === L.finalTop[i]}
									style="background: {sc.css(c.p)}; color: {sc.ink(c.p)}"
									title="{S.idLabel(c.id)} {fmtPct(c.p)}">{short(S.idLabel(c.id))}</td
								>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="legend muted">
			{t({
				en: 'Each cell: the top guess for the next token after that column’s token, shaded by its probability (darker = more confident). Underlined cells already agree with the final answer. Click a column header (or use ←/→) to follow that token.',
				zh: '每格：在该列词元之后，对下一个词元的第一猜测，颜色越深表示越有把握。带下划线的格子已与最终答案一致。点击列标题（或用 ←/→）可跟踪那个词元。'
			})}
		</p>
		{/if}
	</div>
{/if}

<style>
	.ll {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 16px 24px;
		align-items: start;
	}
	.ll.one {
		grid-template-columns: minmax(0, 1fr);
	}
	.intro {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.expl {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.tg {
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.tg input {
		accent-color: var(--accent);
		width: 18px;
		height: 18px;
	}
	.ladder {
		min-width: 0;
	}
	h4 {
		font-family: var(--font-body);
		font-size: var(--fs-sm);
		margin: 0 0 6px;
	}
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	li {
		display: grid;
		grid-template-columns: minmax(8em, 1.4fr) auto 3.4em minmax(0, 1fr) 7.5em;
		align-items: center;
		gap: 6px;
		padding: 3px 6px;
		border-radius: var(--r1);
		font-size: var(--fs-xs);
		border-left: 3px solid transparent;
	}
	li.decided {
		border-left-color: var(--accent);
	}
	li.final {
		background: var(--accent-soft);
		font-weight: 650;
	}
	.pl {
		color: var(--ink-2);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 4px;
		white-space: pre;
	}
	.tok.big {
		font-size: var(--fs-sm);
		padding: 1px 6px;
	}
	.tok.sm {
		color: var(--muted);
		font-size: 11px;
	}
	.others {
		display: flex;
		gap: 3px;
		overflow: hidden;
	}
	.pp {
		text-align: right;
	}
	.final-p {
		display: grid;
		gap: 1px;
	}
	.final-p .bar {
		height: 6px;
		background: var(--surface-3);
		border-radius: 3px;
		overflow: hidden;
	}
	.final-p .bar span {
		display: block;
		height: 100%;
		background: var(--accent);
	}
	.final-p .num {
		font-size: 10px;
		color: var(--muted);
	}
	.verdict {
		margin: 8px 0 0;
		font-size: var(--fs-sm);
	}
	.verdict .muted {
		display: block;
		font-size: var(--fs-xs);
	}
	.tablewrap {
		grid-column: 1 / -1;
		overflow-x: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		max-width: 100%;
	}
	table {
		border-collapse: separate;
		border-spacing: 1px;
		font-family: var(--font-mono);
		font-size: 11px;
	}
	th,
	td {
		padding: 2px 4px;
		white-space: pre;
		text-align: center;
	}
	td {
		min-width: 3.4em;
		border-radius: 2px;
	}
	td.match {
		text-decoration: underline;
		text-decoration-thickness: 1.5px;
		text-underline-offset: 2px;
	}
	td.cur {
		outline: 2px solid var(--q);
		outline-offset: -1px;
	}
	.rl,
	.corner {
		position: sticky;
		left: 0;
		background: var(--surface);
		text-align: left;
		font-family: var(--font-body);
		font-weight: 500;
		color: var(--ink-2);
		z-index: 1;
		min-width: 11em;
	}
	tr.final .rl {
		font-weight: 700;
		color: var(--ink);
	}
	.colbtn {
		font-family: var(--font-mono);
		font-size: 11px;
		border: 1px solid var(--rule);
		background: var(--surface);
		border-radius: 3px;
		padding: 1px 4px;
		min-height: 26px;
		white-space: pre;
	}
	th.cur .colbtn {
		border-color: var(--q);
		box-shadow: 0 0 0 1.5px var(--q);
	}
	.legend {
		grid-column: 1 / -1;
		margin: 0;
		font-size: var(--fs-xs);
	}
	@media (max-width: 860px) {
		.ll {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 520px) {
		li {
			grid-template-columns: minmax(0, 1fr) auto 3.4em;
		}
		.others,
		.final-p {
			grid-column: 1 / -1;
		}
		.others {
			display: none;
		}
		.rl,
		.corner {
			min-width: 7em;
			white-space: normal;
		}
	}
</style>
