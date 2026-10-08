<!--
  Lesson step 12: pick the next token, append it, run the model again, repeat. The learner presses
  "Pick the next token" (or "Pick 5, one at a time"); the draw is shown as the running sum of the
  sampling distribution in token-id order with the seeded random number u landing in one token's
  slice (exactly what tinygpt.ts generate() does). Every token is appended to the page's text.
-->
<script lang="ts">
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import type { ProbItem } from '$lib/kit/types';
	import type { TGState } from './state.svelte';
	import { topIndices } from './analysis';
	import type { GenEntry } from './lesson';

	interface Props {
		LS: TGState;
		dist: Float64Array | null;
		temperature: number;
		topK: number;
		seed: number;
		promptText: string;
		genIds: number[];
		history: GenEntry[];
		busy: boolean;
		generating: boolean;
		lastMs: number | null;
		onpick: () => void;
		onpickN: (n: number) => void;
		onstop: () => void;
		onclear: () => void;
		onsettings: () => void;
	}
	let { LS, dist, temperature, topK, seed, promptText, genIds, history, busy, generating, lastMs, onpick, onpickN, onstop, onclear, onsettings }: Props = $props();

	const T = $derived(LS.T);
	const order = $derived(dist ? topIndices(dist, 8) : []);
	// no ids here: in this lesson "#" means a rank (step 10), and the id is not needed to follow the draw
	const items = $derived<ProbItem[]>(dist ? order.map((id) => ({ label: LS.tok?.tokenToDisplay(id) ?? String(id), p: dist[id] })) : []);
	const last = $derived(history.length ? history[history.length - 1] : null);
	const shown = $derived(last?.slices ? last.slices.filter((s) => s.hi - s.lo >= 0.004 || s.id === last.token) : []);
	const settingText = $derived(
		temperature <= 0
			? t({ en: 'temperature 0 (greedy: always the top token)', zh: '温度 0（贪心：总选第一名）' })
			: t({
					en: `temperature ${temperature.toFixed(2)}, top-k ${topK === 0 ? 'off' : topK}, seed ${seed}`,
					zh: `温度 ${temperature.toFixed(2)}，top-k ${topK === 0 ? '关' : topK}，种子 ${seed}`
				})
	);
</script>

<div class="gs">
	<div class="story">
		<p class="cap">{t({ en: 'The text so far', zh: '目前的文本' })} <span class="muted">· {t({ en: 'magenta = written by the model', zh: '品红色 = 模型写的' })}</span></p>
		<p class="txt">
			<span class="prompt">{promptText}</span>{#each genIds as id, i (i)}<span class="g" title="#{i + 1}">{LS.tok?.decode([id]) ?? ''}</span>{/each}{#if busy || generating}<span
					class="cursor"
					aria-hidden="true">▍</span
				>{/if}
		</p>
	</div>

	<div class="buttons">
		<button type="button" class="btn primary" onclick={onpick} disabled={generating || !dist}>{t({ en: 'Pick the next token', zh: '选出下一个词元' })}</button>
		{#if generating}
			<button type="button" class="btn" onclick={onstop}>■ {t({ en: 'Stop', zh: '停止' })}</button>
		{:else}
			<button type="button" class="btn" onclick={() => onpickN(5)} disabled={!dist}>{t({ en: 'Pick 5, one at a time', zh: '一个一个地选 5 个' })}</button>
		{/if}
		<button type="button" class="btn ghost" onclick={onclear} disabled={generating || !genIds.length}>{t({ en: 'Remove generated', zh: '清除生成内容' })}</button>
	</div>
	<p class="set muted">
		{t({ en: 'Settings from step 11:', zh: '第 11 步的设置：' })}
		{settingText}{t({ en: '.', zh: '。' })}
		<button type="button" class="link" onclick={onsettings}>{t({ en: 'Change them', zh: '去修改' })}</button>
	</p>

	{#if last}
		<div class="draw" aria-live="polite">
			<p class="cap">
				{t({ en: `Token #${last.n}: after`, zh: `第 ${last.n} 个词元：在` })}
				<span class="tok">{last.after}</span>{t({ en: ',', zh: ' 之后，' })}
				{#if last.u === null}
					{t({ en: 'temperature 0 takes the most likely token', zh: '温度 0 直接取最可能的词元' })}
					<span class="tok gen">{LS.idLabel(last.token)}</span>
					({fmtPct(last.pModel)}){t({ en: '.', zh: '。' })}
				{:else}
					{t({ en: `draw #${last.draw} of seed ${last.seed} gave u =`, zh: `种子 ${last.seed} 的第 ${last.draw} 次抽取得到 u =` })}
					<span class="num">{last.u.toFixed(4)}</span>
					→ <span class="tok gen">{LS.idLabel(last.token)}</span>
					<span class="muted"
						>{t({
							en: `(its slice ${fmt3(last.lo)}…${fmt3(last.hi)}, p = ${fmtPct(last.p)})`,
							zh: `（它的区间 ${fmt3(last.lo)}…${fmt3(last.hi)}，p = ${fmtPct(last.p)}）`
						})}</span
					>
				{/if}
			</p>
			{#if last.u !== null && last.slices}
				<div
					class="sbar"
					role="img"
					aria-label={t({
						en: `Running sum of the probabilities in token-id order; u = ${last.u.toFixed(3)} falls in the slice of ${LS.idLabel(last.token)}`,
						zh: `按词元 id 顺序累加的概率；u = ${last.u.toFixed(3)} 落在 ${LS.idLabel(last.token)} 的区间里`
					})}
				>
					{#each shown as s (s.id)}
						<span class="sl" class:hit={s.id === last.token} style="left: {s.lo * 100}%; width: {(s.hi - s.lo) * 100}%" title="{LS.idLabel(s.id)} {fmtPct(s.hi - s.lo)}"
							>{#if s.hi - s.lo >= 0.07}<span class="lab">{LS.idLabel(s.id)}</span>{/if}</span
						>
					{/each}
					<span class="u" style="left: {last.u * 100}%"><span>u</span></span>
				</div>
				<p class="ticks num" aria-hidden="true"><span>0</span><span>0.5</span><span>1</span></p>
				<p class="hint muted">
					{t({
						en: 'Each token the settings keep gets a slice as wide as its probability, laid out in token-id order. The seeded random number u picks the slice it lands in, so likely tokens win often but not always.',
						zh: '设置保留下来的每个词元按 id 顺序排开，各占一段，宽度等于它的概率。带种子的随机数 u 落在哪一段，就选哪个词元：可能性大的词元常常被选中，但不是每次都被选中。'
					})}
				</p>
			{/if}
		</div>
	{/if}

	{#if dist}
		<div class="next">
			<ProbBars
				{items}
				sort={false}
				k={8}
				tokens
				label={{ en: `Now: the next token after ${LS.label(T - 1)} (your settings)`, zh: `现在：${LS.label(T - 1)} 之后的下一个词元（按你的设置）` }}
			/>
		</div>
	{/if}

	{#if history.length}
		<div class="hist">
			<div class="tscroll" use:keyScroll>
			<table>
				<caption class="cap">{t({ en: 'Tokens picked in this lesson', zh: '本课程中选出的词元' })}</caption>
				<thead>
					<tr>
						<th scope="col">#</th>
						<th scope="col">{t({ en: 'after', zh: '前一个' })}</th>
						<th scope="col">{t({ en: 'top candidates', zh: '主要候选' })}</th>
						<th scope="col">u</th>
						<th scope="col">{t({ en: 'picked', zh: '选中' })}</th>
					</tr>
				</thead>
				<tbody>
					{#each history.slice(-8) as g (g.n)}
						<tr>
							<td class="num">{g.n}</td>
							<td><span class="tok">{g.after}</span></td>
							<td class="cands">
								{#each g.top as c (c.id)}<span class="cand"><span class="tok sm" class:gen={c.id === g.token}>{LS.idLabel(c.id)}</span> <span class="num muted">{fmtPct(c.p, 0)}</span></span>{/each}
							</td>
							<td class="num">{g.u === null ? '–' : g.u.toFixed(3)}</td>
							<td><span class="tok gen">{LS.idLabel(g.token)}</span> <span class="num muted">{fmtPct(g.p)}</span></td>
						</tr>
					{/each}
				</tbody>
			</table>
			</div>
			{#if lastMs !== null}
				<p class="hint muted">
					{t({
						en: `After each pick the whole model ran again on the longer text (last run: ${Math.round(lastMs)} ms in your browser). TinyGPT keeps no memory between steps; large models cache the keys and values instead of recomputing them.`,
						zh: `每选一次，整个模型都在更长的文本上重新运行一遍（最近一次：在你的浏览器中用时 ${Math.round(lastMs)} 毫秒）。TinyGPT 在各步之间不保留任何记忆；大模型会缓存键和值，而不是重新计算。`
					})}
				</p>
			{/if}
		</div>
	{/if}
</div>

<style>
	.gs {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
	}
	.cap {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		font-weight: 650;
		text-align: left;
	}
	.cap .muted {
		font-weight: 400;
	}
	.story {
		display: grid;
		gap: 4px;
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.txt {
		margin: 0;
		font-family: var(--font-mono);
		font-size: 0.92rem;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 10em;
		overflow-y: auto;
		font-variant-ligatures: none;
	}
	.g {
		color: var(--generated-text);
		font-weight: 650;
		background: color-mix(in srgb, var(--generated) 10%, transparent);
		border-radius: 2px;
	}
	.cursor {
		color: var(--generated-text);
	}
	.buttons {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.set {
		margin: -4px 0 0;
		font-size: var(--fs-xs);
	}
	.link {
		border: 0;
		background: none;
		padding: 0;
		color: var(--accent-strong);
		text-decoration: underline;
		cursor: pointer;
		font: inherit;
	}
	.draw {
		display: grid;
		gap: 4px;
		padding: 10px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	.draw .cap {
		font-weight: 400;
		color: var(--ink);
		font-size: var(--fs-sm);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		border-radius: 3px;
		padding: 0 4px;
		white-space: pre;
		color: var(--ink);
		font-variant-ligatures: none;
	}
	.tok.gen {
		background: color-mix(in srgb, var(--generated) 16%, var(--surface));
		box-shadow: inset 0 0 0 1px var(--generated);
	}
	.tok.sm {
		font-size: 11px;
	}
	.cand {
		display: inline-block;
		margin-right: 6px;
	}
	.sbar {
		position: relative;
		height: 28px;
		margin-top: 18px;
		border-radius: 3px;
		background: var(--surface-3);
	}
	.sl {
		position: absolute;
		top: 0;
		bottom: 0;
		background: var(--surface-2);
		border-right: 1px solid var(--rule-strong);
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.sl.hit {
		background: color-mix(in srgb, var(--generated) 30%, var(--surface));
		box-shadow: inset 0 0 0 2px var(--generated);
		z-index: 1;
	}
	.lab {
		font-family: var(--font-mono);
		font-size: 10px;
		white-space: pre;
		color: var(--ink);
	}
	.u {
		position: absolute;
		top: -16px;
		bottom: -3px;
		width: 0;
		border-left: 2px solid var(--ink);
		z-index: 2;
	}
	.u span {
		position: absolute;
		top: -2px;
		left: 3px;
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 700;
	}
	.ticks {
		display: flex;
		justify-content: space-between;
		margin: 0;
		font-size: 10px;
		color: var(--muted);
	}
	.hint {
		margin: 0;
		font-size: var(--fs-xs);
	}
	.hist {
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.tscroll {
		overflow-x: auto;
	}
	table {
		border-collapse: collapse;
		font-size: 12px;
	}
	th,
	td {
		padding: 3px 8px;
		text-align: left;
		border-bottom: 1px solid var(--surface-2);
		white-space: nowrap;
	}
	thead th {
		font-weight: 650;
		color: var(--ink-2);
	}
	.cands {
		white-space: normal;
		min-width: 12em;
	}
</style>
