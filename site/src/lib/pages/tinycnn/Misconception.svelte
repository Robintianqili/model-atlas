<!--
  Misconception check: "a 99% probability means the model is sure it is right". Answer first, then
  see the ladder case and a live run over 200 drawings of the two classes the model never learned.
-->
<script lang="ts">
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Term from '$lib/kit/Term.svelte';
	import Lang from '$lib/kit/Lang.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtBytes, fmtPct } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import { page } from '$app/state';
	import type { CnnStore } from './store.svelte';

	let { S, baseUrl }: { S: CnnStore; baseUrl: string } = $props();

	let answer = $state<boolean | null>(null);
	let running = $state(false);
	let stats = $state.raw<{ n: number; over90: number; over99: number; byClass: number[]; bins: number[] } | null>(null);
	const heldBytes = $derived((page.data as { bytes?: { heldout?: number } }).bytes?.heldout ?? 0);

	function showLadder() {
		S.erase = 'none';
		S.loadPreset('ladder', 2);
	}
	const ladderTop = $derived.by(() => {
		const x = S.presetInput('ladder', 2);
		if (!x || !S.model) return null;
		const r = S.model.predict(x);
		return { k: r.top, p: r.probs[r.top] };
	});

	async function runHeldout() {
		if (!S.model || running) return;
		running = true;
		const h = await S.loadHeldout(baseUrl);
		if (!h || !S.mod) {
			running = false;
			return;
		}
		const all = h.classes.flatMap((c) => c.test);
		const byClass = new Array(10).fill(0);
		const bins = new Array(10).fill(0);
		let over90 = 0;
		let over99 = 0;
		let i = 0;
		// 200 forward passes in small time-boxed chunks (~12 ms each), so no task blocks the page for 50 ms
		await new Promise<void>((done) => {
			const chunk = () => {
				const t0 = performance.now();
				for (; i < all.length && performance.now() - t0 < 12; i++) {
					const r = S.model!.predict(S.mod!.decodePreset(all[i]));
					const p = r.probs[r.top];
					byClass[r.top]++;
					bins[Math.min(9, Math.floor(p * 10))]++;
					if (p >= 0.9) over90++;
					if (p >= 0.99) over99++;
				}
				if (i < all.length) setTimeout(chunk, 0);
				else done();
			};
			chunk();
		});
		stats = { n: all.length, over90, over99, byClass, bins };
		running = false;
	}
	const binMax = $derived(stats ? Math.max(1, ...stats.bins) : 1);
</script>

<article class="card wide mis" id="misconception">
	<header>
		<span class="eyebrow">{t({ en: 'Misconception check', zh: '误区检查' })}</span>
		<h3>{t({ en: '“The model says 99% house, so it is 99% sure it is right.”', zh: '“模型说 99% 是房子，所以它有 99% 的把握自己是对的。”' })}</h3>
	</header>
	<div class="btns" role="group" aria-label={t({ en: 'Your answer', zh: '你的回答' })}>
		<button type="button" class="btn" aria-pressed={answer === true} onclick={() => (answer = true)}>{t({ en: 'True', zh: '对' })}</button>
		<button type="button" class="btn" aria-pressed={answer === false} onclick={() => (answer = false)}>{t({ en: 'False', zh: '错' })}</button>
	</div>
	{#if answer !== null}
		<div class="reveal" role="status">
			<p class="verdict" class:ok={answer === false}>
				{answer === false ? t({ en: 'Right: false.', zh: '答对了：这句话是错的。' }) : t({ en: 'Not quite: it is false.', zh: '不完全对：这句话是错的。' })}
			</p>
			<p>
				<Lang>
					{#snippet en()}<Term id="softmax" /> turns 10 scores into shares that add up to 100%. "99% house" means "much more like a house than like any of the other nine classes". It does not measure whether the drawing is a house at all.{/snippet}
					{#snippet zh()}<Term id="softmax" />把 10 个分数变成加起来等于 100% 的份额。“99% 是房子”的意思是“比其余九个类别都更像房子”，并不衡量这幅画到底是不是房子。{/snippet}
				</Lang>
			</p>
			<div class="ladder">
				<button type="button" class="btn" onclick={showLadder}>{t({ en: 'Load ladder #3', zh: '载入梯子 #3' })}</button>
				{#if ladderTop}
					<span class="num res">
						{t({ en: 'a ladder', zh: '一幅梯子' })} → <span class="sw" style="background: {categorical(ladderTop.k, themeState.resolved)}"></span>{t(S.spec.classes[ladderTop.k])} <strong>{fmtPct(ladderTop.p)}</strong>
					</span>
					<ProvenanceBadge mode="live" compact />
				{/if}
			</div>
			<p>
				{t({ en: 'One ladder could be bad luck. Test the two classes the model never learned:', zh: '一幅梯子可能只是巧合。用模型没学过的两个类别来测一测：' })}
			</p>
			<div class="ladder">
				<button type="button" class="btn primary" onclick={runHeldout} disabled={running || !!stats}>
					{running ? t({ en: 'Running…', zh: '运行中…' }) : t({ en: 'Run 200 ladders and envelopes', zh: '运行 200 幅梯子和信封' })}
				</button>
				{#if heldBytes && !stats}<span class="muted small">{t({ en: `downloads ${fmtBytes(heldBytes)} of Quick, Draw! test drawings`, zh: `需下载 ${fmtBytes(heldBytes)} 的 Quick, Draw! 测试画` })}</span>{/if}
			</div>
			{#if stats}
				<div class="stats">
					<p class="big">
						{t({
							en: `${stats.over90} of ${stats.n} drawings the model never learned got a top probability of 90% or more (${stats.over99} got 99% or more).`,
							zh: `${stats.n} 幅模型从没学过的画中，有 ${stats.over90} 幅的最高概率达到 90% 以上（${stats.over99} 幅达到 99% 以上）。`
						})}
						<ProvenanceBadge mode="live" compact />
					</p>
					<div class="charts">
						<figure>
							<figcaption>{t({ en: 'Top probability (count of drawings)', zh: '最高概率（画的数量）' })}</figcaption>
							<div class="hist">
								{#each stats.bins as c, b (b)}
									<div class="hb">
										<span class="hv num">{c}</span>
										<span class="hbar" style="height: {(c / binMax) * 70}px"></span>
										<span class="hl num">{b * 10}–{b * 10 + 10}%</span>
									</div>
								{/each}
							</div>
						</figure>
						<figure>
							<figcaption>{t({ en: 'Class it picked', zh: '它选的类别' })}</figcaption>
							<ul class="bycls">
								{#each stats.byClass.map((c, k) => ({ c, k })).filter((o) => o.c > 0).sort((a, b) => b.c - a.c) as o (o.k)}
									<li><span class="sw" style="background: {categorical(o.k, themeState.resolved)}"></span>{t(S.spec.classes[o.k])} <span class="num">{o.c}</span></li>
								{/each}
							</ul>
						</figure>
					</div>
				</div>
			{/if}
			<p>
				<Lang>
					{#snippet en()}Every one of these answers is wrong, often with a confident number. A probability is only as meaningful as the data the model was trained on; for inputs unlike that data <span class="nw">(<Term id="out-of-distribution" />),</span> it says nothing about correctness. Even on the 10 known classes, whether 90% really means "right 9 times out of 10" has to be measured <span class="nw">(<Term id="calibration" />).</span>{/snippet}
					{#snippet zh()}这些回答全都是错的，而且常常带着很自信的数字。概率的意义取决于模型的训练数据；对与训练数据不同的输入（<Term id="out-of-distribution" />），它说明不了对错。即使在已知的 10 个类别上，90% 是否真的意味着“十次对九次”，也需要实际测量（<Term id="calibration" />）。{/snippet}
				</Lang>
			</p>
		</div>
	{/if}
</article>

<style>
	.card {
		grid-column: 1 / -1;
		display: grid;
		gap: 10px;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-left: 4px solid var(--illus);
		border-radius: var(--r3);
		background: var(--surface);
		min-width: 0;
	}
	header {
		display: grid;
		gap: 2px;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-lg);
	}
	.btns,
	.ladder {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
	.reveal {
		display: grid;
		gap: 10px;
	}
	.reveal p {
		margin: 0;
		font-size: var(--fs-sm);
		max-width: var(--measure);
	}
	.verdict {
		justify-self: start;
		font-weight: 700;
		padding: 1px 10px;
		border-radius: var(--r-pill);
		background: var(--illus-soft);
		color: var(--illus);
	}
	.verdict.ok {
		background: var(--live-soft);
		color: var(--live);
	}
	.res {
		display: inline-flex;
		gap: 5px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.sw {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 2px;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.nw {
		white-space: nowrap;
	}
	.big {
		font-size: var(--fs-md) !important;
		font-weight: 650;
	}
	.stats {
		display: grid;
		gap: 10px;
	}
	.charts {
		display: flex;
		flex-wrap: wrap;
		gap: 16px 40px;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.hist {
		display: flex;
		align-items: flex-end;
		gap: 3px;
		overflow-x: auto;
		max-width: 100%;
	}
	.hb {
		display: grid;
		justify-items: center;
		gap: 2px;
		width: 34px;
	}
	.hbar {
		display: block;
		width: 22px;
		background: var(--seq-mid);
		border-radius: 2px 2px 0 0;
	}
	.hv,
	.hl {
		font-size: 9.5px;
		color: var(--ink-2);
	}
	.hl {
		color: var(--muted);
		white-space: nowrap;
		font-size: 8.5px;
	}
	.bycls {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
		font-size: var(--fs-sm);
	}
	.bycls li {
		display: flex;
		gap: 6px;
		align-items: center;
	}
</style>
