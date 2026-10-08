<!--
  Lesson step 3: count every neighbouring pair (inside pre-tokens, weighted by how often each
  pre-token occurs). Left/top: the most frequent pairs before any merge (from the trace). Right/below:
  where the selected pair's count comes from, pre-token by pre-token, summing to the count.
-->
<script lang="ts">
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { fmtInt, visibleToken } from '$lib/kit/format';
	import { bytesToDisplay } from '$lib/models/bpe';
	import Tok from '../Tok.svelte';
	import { segment, type TrainTrace } from '../train';
	import { pairSources } from '../lesson';

	interface Props {
		trace: TrainTrace;
		/** the text the trace was trained on */
		text: string;
		pattern: string;
		/** selected row of the pair table (bindable) */
		sel?: number;
	}
	let { trace, text, pattern, sel = $bindable(0) }: Props = $props();

	const SRC = 6;
	/** bytes drawn per pre-token (a very long one is cut) */
	const IDS = 24;
	const disp = (id: number) => bytesToDisplay(trace.bytes[id]);
	const top = $derived(trace.steps.length ? trace.steps[0].top : trace.finalTop);
	const nPairs = $derived(trace.steps.length ? trace.steps[0].pairs : trace.finalPairs);
	const segs0 = $derived(segment(text, pattern, trace.merges, 0));
	const i = $derived(Math.max(0, Math.min(sel, top.length - 1)));
	const p = $derived(top[i]);
	const src = $derived(p ? pairSources(segs0, p.a, p.b) : []);
	const rest = $derived(src.slice(SRC));
	const restSum = $derived(rest.reduce((s, x) => s + x.per * x.f, 0));
	const maxN = $derived(top[0]?.n ?? 1);
	/** is position j of ids part of an (a, b) pair? */
	const inPair = (ids: number[], j: number, a: number, b: number) => (ids[j] === a && ids[j + 1] === b) || (ids[j - 1] === a && ids[j] === b);

	function key(e: KeyboardEvent) {
		let j = i;
		if (e.key === 'ArrowDown' || e.key === 'ArrowRight') j = Math.min(top.length - 1, i + 1);
		else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = top.length - 1;
		else return;
		e.preventDefault();
		sel = j;
		(e.currentTarget as HTMLElement).closest('ol')?.querySelectorAll<HTMLElement>('.pr')[j]?.focus();
	}
</script>

<div class="ls">
	<h4>{t({ en: 'Pair counts in your text (nothing merged yet)', zh: '你的文本里的相邻对计数（还没有任何合并）' })}</h4>
	{#if top.length && p}
		<p class="flow">
			<span><b class="num">{fmtInt(trace.initialTokenCount)}</b> {t({ en: 'tokens (bytes)', zh: '个词元（字节）' })}</span>
			<span>·</span>
			<span><b class="num">{fmtInt(nPairs)}</b> {t({ en: 'different neighbouring pairs', zh: '种不同的相邻对' })}</span>
		</p>
		<div class="grid">
			<div class="col">
				<p class="sh">{t({ en: `The ${top.length} most frequent`, zh: `出现最多的 ${top.length} 对` })}</p>
				<!-- the radio group wraps the list, so the list keeps its items (axe listitem) -->
				<div role="radiogroup" aria-label={t({ en: 'Most frequent pairs', zh: '出现最多的相邻对' })}>
				<ol>
					{#each top as q, j (j)}
						<li>
							<button
								type="button"
								class="pr"
								class:win={j === 0}
								role="radio"
								aria-checked={j === i}
								tabindex={j === i ? 0 : -1}
								aria-label="{visibleToken(disp(q.a))} + {visibleToken(disp(q.b))}: {q.n}"
								onclick={() => (sel = j)}
								onkeydown={key}
							>
								<span class="pp"><Tok text={disp(q.a)} small mark={j === i ? 'pair' : null} /><Tok text={disp(q.b)} small mark={j === i ? 'pair' : null} /></span>
								<span class="track"><span class="fill" style:width="{(100 * q.n) / maxN}%"></span></span>
								<span class="num n">{q.n}</span>
							</button>
							{#if j === 0}<span class="win-l">{t({ en: '← most frequent', zh: '← 最多' })}</span>{/if}
						</li>
					{/each}
				</ol>
				</div>
			</div>
			<div class="col">
				<p class="sh">
					{#if i18n.lang !== 'zh'}Where{/if} <Tok text={disp(p.a)} small mark="pair" /><Tok text={disp(p.b)} small mark="pair" />
					{t({ en: `= ${p.n} comes from`, zh: `= ${p.n} 是怎么来的` })}
				</p>
				<ul class="src" aria-live="polite">
					{#each src.slice(0, SRC) as s (s.text)}
						<li>
							<span class="ids">{#each s.ids.slice(0, IDS) as id, j (j)}<Tok text={disp(id)} small mark={inPair(s.ids, j, p.a, p.b) ? 'pair' : null} />{/each}{#if s.ids.length > IDS}<span class="muted more">… {t({ en: `${fmtInt(s.ids.length)} bytes`, zh: `共 ${fmtInt(s.ids.length)} 个字节` })}</span>{/if}</span>
							<span class="num mult">{s.per > 1 ? `${s.per} ${t({ en: 'pairs', zh: '对' })} × ${s.f}` : `× ${s.f}`}</span>
							<span class="num add">= {s.per * s.f}</span>
						</li>
					{/each}
					{#if rest.length}
						<li class="restl">
							<span class="muted">{t({ en: `${rest.length} more pre-tokens`, zh: `另外 ${rest.length} 个片段` })}</span>
							<span></span>
							<span class="num add">= {restSum}</span>
						</li>
					{/if}
				</ul>
				<p class="total num">
					{src.slice(0, SRC).map((s) => s.per * s.f).concat(rest.length ? [restSum] : []).join(' + ')} = <b>{p.n}</b>
				</p>
				<p class="hint">{t({ en: 'Each pre-token counts as often as it occurs in the text (×). Pick another pair in the list to see its sources.', zh: '每个片段在文本中出现几次，就计几次（×）。在列表中选另一对，看看它的来源。' })}</p>
			</div>
		</div>
	{:else}
		<p class="muted">{t({ en: 'No pre-token in your text has two bytes in a row, so there is no pair to count. Type a longer text.', zh: '你的文本里没有任何片段包含两个相邻的字节，所以没有相邻对可数。请输入更长的文本。' })}</p>
	{/if}
</div>

<style>
	.ls {
		container: ls / inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 12px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.flow {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		color: var(--ink-2);
		font-size: var(--fs-sm);
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 16px;
	}
	@container ls (min-width: 600px) {
		.grid {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr);
		}
	}
	.col {
		display: grid;
		gap: 6px;
		align-content: start;
		min-width: 0;
	}
	.sh {
		margin: 0;
		font-size: var(--fs-sm);
		font-weight: 650;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px;
	}
	ol,
	ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	ol li {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
	}
	.pr {
		flex: 1;
		min-width: 0;
		display: grid;
		grid-template-columns: minmax(0, auto) minmax(30px, 1fr) 2.6em;
		align-items: center;
		gap: 8px;
		padding: 2px 6px;
		border: 1px solid transparent;
		border-radius: var(--r1);
		background: none;
		color: inherit;
		text-align: left;
		cursor: pointer;
	}
	.pr:hover {
		background: var(--surface-2);
	}
	.pr[aria-checked='true'] {
		border-color: var(--seq-mid);
		background: var(--surface-2);
	}
	.pp {
		display: inline-flex;
		gap: 3px;
		min-width: 0;
		overflow: hidden;
	}
	.track {
		height: 10px;
		border-radius: 3px;
		background: var(--surface-2);
		overflow: hidden;
	}
	.fill {
		display: block;
		height: 100%;
		background: var(--seq-mid);
	}
	.n {
		text-align: right;
		font-size: var(--fs-sm);
	}
	.win-l {
		font-size: var(--fs-xs);
		color: var(--accent);
		white-space: nowrap;
	}
	.src li {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto 3.2em;
		align-items: center;
		gap: 8px;
		padding: 3px 6px;
		border-radius: var(--r1);
		background: var(--surface-2);
	}
	.ids {
		display: flex;
		flex-wrap: wrap;
		gap: 2px;
		min-width: 0;
	}
	.more {
		font-size: var(--fs-xs);
		align-self: center;
	}
	.mult {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.add {
		text-align: right;
		font-weight: 700;
		color: var(--seq-hi);
	}
	.restl {
		background: none !important;
		font-size: var(--fs-sm);
	}
	.total {
		margin: 0;
		font-size: var(--fs-sm);
		overflow-wrap: anywhere;
	}
	.total b {
		font-size: var(--fs-lg);
		color: var(--seq-hi);
	}
	.hint {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
</style>
