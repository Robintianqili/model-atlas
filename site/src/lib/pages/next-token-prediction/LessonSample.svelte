<!--
  Lesson step 7: what the trained model does with the probabilities. The final model's 1,024
  probabilities for the token after the learner's sentence (live, fp32), reshaped by a temperature
  T; a random number u (seeded, reproducible) picks one token from the strip; the token is appended
  and the model runs again. "Write 10 more" runs the same loop in the worker.
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct, visibleToken } from '$lib/kit/format';
	import type { NtpLesson } from './lessonState.svelte';
	import { pickSorted, softmax64, sortedIds } from './lessonMath';

	let { L }: { L: NtpLesson } = $props();
	const S = $derived(L.S);

	let rng = $state.raw<((seed: number) => () => number) | null>(null);
	onMount(() => {
		void import('$lib/models/tinygpt').then((m) => (rng = m.mulberry32));
	});

	let T = $state(1);
	let seed = $state(1);
	let gen = $state<number[]>([]);
	let busy = $state(false);
	let pending = $state<{ u: number; id: number; from: number; to: number; rank: number } | null>(null);

	// a new sentence: start a new continuation
	let lastIds = untrack(() => L.idsKey);
	let run = 0;
	$effect(() => {
		const k = L.idsKey;
		if (k === lastIds) return;
		lastIds = k;
		reset();
	});

	const all = $derived([...S.ids, ...gen]);
	const allKey = $derived(all.join(','));
	let logits = $state.raw<Float32Array | null>(null);
	let logitsKey = $state('');
	const stale = $derived(logitsKey !== allKey);
	$effect(() => {
		const ids = all;
		const key = allKey;
		const c = S.client;
		if (!c || ids.length === 0 || key === untrack(() => logitsKey)) return;
		const h = setTimeout(() => {
			c.next('final', ids, 'lesson-sample', 1).then((r) => {
				if (r) {
					logits = r.logits;
					logitsKey = key;
				}
			});
		}, 80);
		return () => clearTimeout(h);
	});

	const plain = $derived(logits ? softmax64(logits, 1) : null);
	const dist = $derived(logits ? softmax64(logits, T) : null);
	const order = $derived(plain ? sortedIds(plain) : null);
	const ROWS = 8;
	const rows = $derived(order && plain && dist ? Array.from(order.slice(0, ROWS), (id) => ({ id, p1: plain[id], pT: dist[id] })) : []);
	const restT = $derived(dist && order ? 1 - rows.reduce((a, r) => a + r.pT, 0) : 0);
	const maxBar = $derived(Math.max(1e-9, ...rows.map((r) => Math.max(r.p1, r.pT))));

	// changing the temperature changes the strip: a drawn u no longer applies
	$effect(() => {
		void T;
		untrack(() => (pending = null));
	});

	function draw() {
		if (!dist || !order || !rng || stale || busy) return;
		const u = rng(seed)();
		seed++;
		pending = { u, ...pickSorted(dist, order, u) };
	}
	function append() {
		if (!pending) return;
		gen = [...gen, pending.id];
		pending = null;
	}
	async function write10() {
		const c = S.client;
		if (!c || busy || all.length === 0) return;
		pending = null;
		busy = true;
		const my = ++run;
		const r = await c.generate('final', all, { maxNew: 10, temperature: T, seed, topN: 1, stopAtEot: true }, 'lesson-gen', (s) => {
			if (my === run) gen = [...gen, s.token];
		});
		if (my !== run) return;
		if (r) seed++;
		busy = false;
	}
	function reset() {
		run++;
		S.client?.cancel('lesson-gen');
		busy = false;
		gen = [];
		pending = null;
	}

	const promptTail = $derived.by(() => {
		const s = S.decode(S.ids);
		return s.length > 70 ? '…' + s.slice(s.length - 69) : s;
	});
	const presets = [
		{ T: 0.3, label: { en: 'cold 0.3', zh: '低温 0.3' } },
		{ T: 1, label: { en: 'as trained 1', zh: '原样 1' } },
		{ T: 1.8, label: { en: 'hot 1.8', zh: '高温 1.8' } }
	];
	const tokLabel = (id: number) => visibleToken(S.disp(id));
	/** strip labels only where they fit (about 7 px per character of the 11 px mono font) */
	let stripW = $state(320);
	const fits = (p: number, text: string) => p * stripW >= [...text].length * 7 + 10;
</script>

<div class="sm">
	<div class="top">
		<p class="eyebrow">{t({ en: 'The token after your sentence', zh: '你的句子之后的词元' })}</p>
	</div>
	<p class="gen" aria-live="polite">
		<span class="prompt">{promptTail}</span><span class="new">{S.decode(gen)}</span>{#if pending}<span class="pend">{S.disp(pending.id)}</span>{/if}{#if busy}<span class="caret" aria-hidden="true">▍</span>{/if}
	</p>

	{#if S.ids.length === 0}
		<p class="muted">{t({ en: 'Type a sentence first.', zh: '请先输入一句话。' })}</p>
	{:else}
		<div class="temp">
			<label class="row">
				<span class="name">{t({ en: 'temperature T', zh: '温度 T' })} <strong class="num">{fmt3(T)}</strong></span>
				<input type="range" min="0.1" max="2" step="0.05" bind:value={T} />
			</label>
			<div class="chips" role="group" aria-label={t({ en: 'Temperature presets', zh: '温度预设' })}>
				{#each presets as pr (pr.T)}
					<button type="button" class="chip" aria-pressed={Math.abs(T - pr.T) < 1e-9} onclick={() => (T = pr.T)}>{t(pr.label)}</button>
				{/each}
			</div>
		</div>

		{#if rows.length}
			<ol class="bars" class:stale aria-label={t({ en: 'Most likely next tokens: probability as trained (T = 1) and after temperature', zh: '最可能的下一个词元：原始概率（T = 1）与温度调整后的概率' })}>
				{#each rows as r, i (r.id)}
					<li class:picked={pending?.id === r.id}>
						<span class="lab">{tokLabel(r.id)}</span>
						<span class="track" aria-hidden="true">
							<span class="b1" style="width: {(r.p1 / maxBar) * 100}%"></span>
							<span class="bT" style="width: {Math.max(0.4, (r.pT / maxBar) * 100)}%"></span>
						</span>
						<span class="val num">{fmtPct(r.pT)}</span>
						<span class="val1 num muted">{T !== 1 ? `(${fmtPct(r.p1)})` : ''}</span>
						<span class="visually-hidden">{t({ en: `rank ${i + 1}; as trained ${fmtPct(r.p1)}`, zh: `第 ${i + 1} 名；原始 ${fmtPct(r.p1)}` })}</span>
					</li>
				{/each}
			</ol>
			<p class="key small muted">
				<span><i class="k1"></i>{t({ en: 'as trained (T = 1)', zh: '原始概率（T = 1）' })}</span>
				<span><i class="kT"></i>{t({ en: `after temperature T = ${fmt3(T)}`, zh: `温度 T = ${fmt3(T)} 之后` })}</span>
			</p>

			<div class="strip-wrap">
				<p class="small">{t({ en: 'The same probabilities laid end to end on a strip from 0 to 1:', zh: '把同样的概率首尾相接，排在 0 到 1 的长条上：' })}</p>
				<div class="strip" aria-hidden="true" bind:clientWidth={stripW}>
					{#each rows as r, i (r.id)}
						<span class="seg s{i % 2}" class:picked={pending?.id === r.id} style="width: {r.pT * 100}%">{#if fits(r.pT, tokLabel(r.id))}<span class="sl">{tokLabel(r.id)}</span>{/if}</span>
					{/each}
					<span class="seg rest" style="width: {Math.max(0, restT) * 100}%">{#if fits(restT, 'the other 1016')}<span class="sl rl">{t({ en: `the other ${(order?.length ?? 1024) - ROWS}`, zh: `其余 ${(order?.length ?? 1024) - ROWS} 个` })}</span>{/if}</span>
					{#if pending && pending.rank > ROWS}<span class="pick-ov" style="left: {pending.from * 100}%; width: max(2px, {(pending.to - pending.from) * 100}%)"></span>{/if}
					{#if pending}<span class="u" style="left: {pending.u * 100}%"><span class="ul num" style="transform: translateX({pending.u < 0.12 ? '0' : pending.u > 0.88 ? '-100%' : '-50%'})">u = {fmt3(pending.u)}</span></span>{/if}
				</div>
				<p class="small strip-axis num" aria-hidden="true"><span>0</span><span>1</span></p>
			</div>

			<div class="btns">
				{#if pending}
					<button type="button" class="btn primary" onclick={append}>{t({ en: 'Append it and predict again', zh: '接上它，再预测下一个' })}</button>
				{:else}
					<button type="button" class="btn primary" onclick={draw} disabled={!rng || stale || busy}>{t({ en: 'Pick the next token', zh: '挑下一个词元' })}</button>
				{/if}
				<button type="button" class="btn" onclick={write10} disabled={!S.client || busy}>{busy ? t({ en: 'Writing…', zh: '生成中…' }) : t({ en: 'Write 10 more', zh: '再写 10 个' })}</button>
				<button type="button" class="btn ghost" onclick={reset} disabled={gen.length === 0 && !pending && !busy}>{t({ en: 'Reset', zh: '重置' })}</button>
			</div>
			<p class="small" aria-live="polite">
				{#if pending}
					{t({
						en: `The random number u = ${fmt3(pending.u)} falls in the segment of “${S.disp(pending.id)}” (from ${fmt3(pending.from)} to ${fmt3(pending.to)}), so that token is picked${pending.rank > 1 ? `, although ${pending.rank - 1} ${pending.rank === 2 ? 'token was' : 'tokens were'} more likely` : ''}.`,
						zh: `随机数 u = ${fmt3(pending.u)} 落在“${S.disp(pending.id)}”的区段里（从 ${fmt3(pending.from)} 到 ${fmt3(pending.to)}），所以选中了它${pending.rank > 1 ? `，尽管有 ${pending.rank - 1} 个词元比它更可能` : ''}。`
					})}
				{:else}
					<span class="muted">
						{t({
							en: `Each pick draws a random number u between 0 and 1 (seed ${seed}, so every pick can be repeated). A wider segment is more likely to catch it.`,
							zh: `每次挑选都抽一个 0 到 1 之间的随机数 u（种子 ${seed}，所以每次挑选都能复现）。区段越宽，越容易接住它。`
						})}
					</span>
				{/if}
			</p>
		{:else}
			<p class="muted">{t({ en: 'TinyGPT is reading your sentence…', zh: 'TinyGPT 正在读你的句子…' })}</p>
		{/if}
	{/if}
</div>

<style>
	.sm {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
		justify-content: space-between;
	}
	.eyebrow {
		margin: 0;
	}
	.gen {
		margin: 0;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		line-height: 1.55;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		padding: 8px 10px;
		background: var(--surface-2);
		border-radius: var(--r2);
		min-height: 3.2em;
	}
	.prompt {
		color: var(--ink-2);
	}
	.new {
		color: var(--generated-text);
		font-weight: 600;
	}
	.pend {
		color: var(--generated-text);
		font-weight: 600;
		outline: 1.5px dashed var(--generated);
		outline-offset: 1px;
		border-radius: 2px;
		white-space: pre;
	}
	.caret {
		color: var(--generated-text);
		animation: blink 1s steps(2) infinite;
	}
	@keyframes blink {
		50% {
			opacity: 0;
		}
	}
	.temp {
		display: grid;
		gap: 6px;
	}
	.row {
		display: grid;
		gap: 4px;
		font-size: var(--fs-sm);
	}
	.name {
		display: flex;
		justify-content: space-between;
	}
	input[type='range'] {
		width: 100%;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.chips .chip {
		cursor: pointer;
		font: inherit;
		font-size: var(--fs-xs);
	}
	.chips .chip[aria-pressed='true'] {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
	.bars {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 3px;
		transition: opacity var(--dur-2) var(--ease);
	}
	.bars.stale {
		opacity: 0.5;
	}
	.bars li {
		display: grid;
		grid-template-columns: minmax(3.5em, 7em) minmax(0, 1fr) 3.6em 4.2em;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-sm);
		padding: 1px 4px;
		border-radius: var(--r1);
	}
	.bars li.picked {
		background: color-mix(in srgb, var(--generated) 14%, transparent);
	}
	.lab {
		font-family: var(--font-mono);
		white-space: pre;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.track {
		position: relative;
		height: 14px;
	}
	.b1,
	.bT {
		position: absolute;
		left: 0;
		top: 0;
		height: 100%;
		border-radius: 2px;
	}
	.b1 {
		border: 1.5px solid var(--seq-mid);
		background: transparent;
		box-sizing: border-box;
	}
	.bT {
		top: 3px;
		height: calc(100% - 6px);
		background: var(--seq-hi);
		transition: width var(--dur-2) var(--ease);
	}
	.val {
		text-align: right;
		font-weight: 650;
	}
	.val1 {
		font-size: var(--fs-xs);
	}
	.key {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 14px;
		margin: 0;
	}
	.key span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.k1,
	.kT {
		display: inline-block;
		width: 16px;
		height: 9px;
		border-radius: 2px;
	}
	.k1 {
		border: 1.5px solid var(--seq-mid);
	}
	.kT {
		background: var(--seq-hi);
	}
	.strip-wrap p {
		margin: 0 0 4px;
	}
	.strip {
		position: relative;
		display: flex;
		height: 30px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		overflow: visible;
		background: var(--surface);
	}
	.seg {
		display: flex;
		align-items: center;
		justify-content: center;
		height: 100%;
		min-width: 0;
		overflow: hidden;
		border-right: 1px solid var(--surface);
		transition: width var(--dur-2) var(--ease);
	}
	.seg.s0 {
		background: color-mix(in srgb, var(--seq-hi) 70%, var(--surface));
	}
	.seg.s1 {
		background: color-mix(in srgb, var(--seq-mid) 70%, var(--surface));
	}
	.seg.rest {
		background: var(--surface-3);
	}
	.seg.picked {
		background: var(--generated);
	}
	.sl {
		font-family: var(--font-mono);
		font-size: var(--t-xs);
		color: var(--bg);
		white-space: pre;
		overflow: hidden;
		padding: 0 2px;
	}
	.seg.s1 .sl {
		color: var(--ink);
	}
	.sl.rl {
		color: var(--ink-2);
	}
	.seg.picked .sl {
		color: var(--bg);
	}
	.pick-ov {
		position: absolute;
		top: 0;
		bottom: 0;
		background: var(--generated);
	}
	.u {
		position: absolute;
		top: -6px;
		bottom: -6px;
		width: 0;
		border-left: 2.5px solid var(--ink);
	}
	.ul {
		position: absolute;
		top: 100%;
		left: 0;
		margin-top: 2px;
		font-size: var(--t-xs);
		white-space: nowrap;
		background: var(--surface);
		padding: 0 3px;
		border-radius: 3px;
	}
	.strip-axis {
		display: flex;
		justify-content: space-between;
		color: var(--muted);
		margin-top: 18px !important;
	}
	.btns {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.small {
		font-size: var(--fs-xs);
		margin: 0;
	}
</style>
