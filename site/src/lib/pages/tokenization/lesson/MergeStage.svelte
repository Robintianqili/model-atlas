<!--
  Lesson step 4: one merge, animated in three phases (StepPlayer): find every occurrence of the
  winning pair in the text, join each into one new token, rewrite the text (before − saved = after).
  The text is drawn as chips grouped by pre-token. "Next merge" repeats it for merge #2, #3, …
  All numbers come from the live trace of the learner's text.
-->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';
	import { bytesToDisplay } from '$lib/models/bpe';
	import Tok from '../Tok.svelte';
	import { chipHtml } from '../Chips.svelte';
	import type { TrainTrace } from '../train';
	import { SegCursor, mergeUnits, plainToken, tokensAt } from '../lesson';

	interface Props {
		trace: TrainTrace;
		/** the text the trace was trained on */
		text: string;
		pattern: string;
		/** merge number, 1-based (bindable) */
		m?: number;
		autoplay?: boolean;
		onend?: () => void;
	}
	let { trace, text, pattern, m = $bindable(1), autoplay = false, onend }: Props = $props();

	const K = $derived(trace.steps.length);
	const mm = $derived(Math.max(1, Math.min(m, K)));
	const st = $derived(K ? trace.steps[mm - 1] : null);
	const before = $derived(tokensAt(trace, mm - 1));
	const after = $derived(st ? st.tokenCount : before);
	const saved = $derived(before - after);
	const cursor = $derived(new SegCursor(text, pattern, trace.merges));
	const mu = $derived(st ? mergeUnits(cursor.at(mm - 1), st.pair, 700) : null);
	const disp = (id: number) => bytesToDisplay(trace.bytes[id]);
	let phase = $state(0);

	const A = $derived(st ? plainToken(disp(st.pair[0])) : '');
	const B = $derived(st ? plainToken(disp(st.pair[1])) : '');
	const AB = $derived(st ? plainToken(disp(st.newId)) : '');
	const labels = $derived(
		st
			? [
					{ en: `Find every “${A}” followed by “${B}”: ${st.count} times`, zh: `找出每个“${A}”后面紧跟“${B}”的位置：${st.count} 处` },
					{ en: `Join each pair into one new token “${AB}” (ID ${st.newId})`, zh: `把每一对合成一个新词元“${AB}”（ID ${st.newId}）` },
					{ en: `Rewrite the text: ${fmtInt(before)} − ${fmtInt(saved)} = ${fmtInt(after)} tokens`, zh: `改写文本：${fmtInt(before)} − ${fmtInt(saved)} = ${fmtInt(after)} 个词元` }
				]
			: []
	);
	function goMerge(n: number) {
		const j = Math.max(1, Math.min(K, n));
		if (j === mm) return;
		phase = 0;
		m = j;
	}
</script>

<div class="ls">
	{#if st && mu}
		<h4>{t({ en: `Merge #${mm}: the most frequent pair becomes one token`, zh: `第 ${mm} 次合并：出现最多的一对变成一个词元` })}</h4>
		<div class="card">
			<span class="mg">
				<Tok text={disp(st.pair[0])} mark="pair" />
				<span class="op">+</span>
				<Tok text={disp(st.pair[1])} mark="pair" />
				<span class="op">→</span>
				<Tok text={disp(st.newId)} mark="new" />
			</span>
			<span class="meta">
				<span>ID <b class="num">{st.newId}</b></span>
				<span>{t({ en: 'counted', zh: '计数' })} <b class="num pc">{st.count}</b>{t({ en: ' times', zh: ' 次' })}</span>
				<span>{t({ en: 'vocabulary', zh: '词表' })} <span class="num">{255 + mm}</span> → <b class="num">{256 + mm}</b></span>
			</span>
		</div>
		{#key mm}
			<StepPlayer count={3} bind:step={phase} {labels} interval={1700} {autoplay} {onend} speeds={[0.5, 1, 2]}>
				{#snippet children(s)}
					<figure class="mt-wrap">
						<div class="mt p{s}" aria-hidden="true">
							{#each mu.groups as g, gi (gi)}
								<span class="grp">
									{#each g as u (u.i)}
										{#if u.kind === 'occ'}
											<span class="occ">
												{#if s < 2}
													<span class="ch l">{@html chipHtml(disp(u.a))}</span><span class="ch r">{@html chipHtml(disp(u.b))}</span>
												{:else}
													<span class="ch new">{@html chipHtml(disp(st.newId))}</span>
												{/if}
											</span>
										{:else}
											<span class="ch c{u.i % 4}">{@html chipHtml(disp(u.id))}</span>
										{/if}
									{/each}
								</span>
							{/each}
						</div>
						<figcaption class="cap">
							<span class="muted small key">{t({ en: 'Each small box is one token; the gaps separate pre-tokens.', zh: '每个小方块是一个词元；空隙分开不同的预分词片段。' })}</span>
							{#if s < 2}
								<span><span class="lg" aria-hidden="true"></span>{t({ en: `the pair, ${mu.occ} places in your text`, zh: `这一对，在你的文本里有 ${mu.occ} 处` })}{#if mu.occ < st.count}{t({ en: ' (overlapping pairs, as in “aaa”, merge only once)', zh: '（重叠的对，如“aaa”，只合并一次）' })}{/if}</span>
							{:else}
								<span class="eq"><span class="num">{fmtInt(before)}</span> − <span class="num pc">{fmtInt(saved)}</span> = <b class="num">{fmtInt(after)}</b> {t({ en: 'tokens', zh: '个词元' })}</span>
							{/if}
							{#if mu.shown < mu.total}
								<span class="muted small">{t({ en: `(first ${fmtInt(mu.shown)} of ${fmtInt(mu.total)} tokens drawn)`, zh: `（只画出前 ${fmtInt(mu.shown)} 个，共 ${fmtInt(mu.total)} 个词元）` })}</span>
							{/if}
						</figcaption>
					</figure>
				{/snippet}
			</StepPlayer>
		{/key}
		<div class="nav">
			<button type="button" class="btn small" onclick={() => goMerge(mm - 1)} aria-disabled={mm <= 1}>← {t({ en: 'Previous merge', zh: '上一次合并' })}</button>
			<span class="num pos">{t({ en: `Merge #${mm} of ${K}`, zh: `第 ${mm} 次合并（共 ${K} 次）` })}</span>
			<button type="button" class="btn small" onclick={() => goMerge(mm + 1)} aria-disabled={mm >= K}>{t({ en: 'Next merge', zh: '下一次合并' })} →</button>
		</div>
	{:else}
		<h4>{t({ en: 'Nothing to merge', zh: '没有可合并的' })}</h4>
		<p class="muted">{t({ en: 'Every pre-token of your text is already a single byte, so no pair can be merged. Type a longer text.', zh: '你的文本里每个预分词片段都只有一个字节，没有可合并的对。请输入更长的文本。' })}</p>
	{/if}
</div>

<style>
	.ls {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-md);
	}
	.card {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 18px;
		padding: 8px 12px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.mg {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: 1.1rem;
	}
	.op {
		font-family: var(--font-mono);
		color: var(--muted);
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.pc {
		color: var(--seq-hi);
		font-weight: 700;
	}
	.mt-wrap {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.mt {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 7px;
		align-content: flex-start;
		max-height: 250px;
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: 8px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		contain: content;
	}
	.grp,
	.occ {
		display: inline-flex;
		flex-wrap: wrap;
		gap: 2px;
		min-width: 0;
	}
	.occ {
		border-radius: 5px;
		transition:
			gap 350ms ease,
			box-shadow 350ms ease;
	}
	.ch {
		display: inline-block;
		min-width: 1.2em;
		padding: 1px 4px;
		border-radius: 4px;
		font-family: var(--font-mono);
		font-size: 0.86rem;
		line-height: 1.45;
		white-space: pre;
		text-align: center;
		color: var(--ink);
		transition:
			border-radius 350ms ease,
			box-shadow 350ms ease;
	}
	.ch :global(.hex) {
		font-size: max(var(--t-xs), 0.72em);
		color: var(--ink-2);
	}
	.c0 {
		background: color-mix(in srgb, var(--cat-0) 16%, var(--surface));
	}
	.c1 {
		background: color-mix(in srgb, var(--cat-3) 22%, var(--surface));
	}
	.c2 {
		background: color-mix(in srgb, var(--cat-2) 16%, var(--surface));
	}
	.c3 {
		background: color-mix(in srgb, var(--cat-4) 18%, var(--surface));
	}
	/* phase 0: the two tokens of every occurrence ringed in the pair color */
	.occ .l,
	.occ .r {
		background: color-mix(in srgb, var(--seq-mid) 30%, var(--surface));
		box-shadow: 0 0 0 2px var(--seq-mid);
	}
	/* phase 1: the pair closes up into one box */
	.p1 .occ {
		gap: 0;
		box-shadow: 0 0 0 2px var(--seq-mid);
	}
	.p1 .occ .l {
		border-radius: 4px 0 0 4px;
		box-shadow: none;
	}
	.p1 .occ .r {
		border-radius: 0 4px 4px 0;
		box-shadow: none;
	}
	/* phase 2: one new token */
	.ch.new {
		background: var(--accent-soft);
		box-shadow: 0 0 0 2px var(--accent);
		animation: pop 420ms ease-out;
	}
	@keyframes pop {
		0% {
			transform: scale(1.25);
		}
		100% {
			transform: scale(1);
		}
	}
	.cap {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 10px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.eq {
		font-size: var(--fs-lg);
		color: var(--ink);
	}
	.lg {
		display: inline-block;
		width: 14px;
		height: 12px;
		margin-right: 6px;
		vertical-align: -1px;
		border-radius: 3px;
		background: color-mix(in srgb, var(--seq-mid) 30%, var(--surface));
		box-shadow: inset 0 0 0 2px var(--seq-mid);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.key {
		flex-basis: 100%;
	}
	.nav {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 10px;
	}
	.btn.small {
		min-height: 34px;
		padding: 0 12px;
		font-size: var(--fs-sm);
	}
	.btn[aria-disabled='true'] {
		opacity: 0.45;
		cursor: default;
	}
	.pos {
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	@media (prefers-reduced-motion: reduce) {
		.occ,
		.ch {
			transition: none;
		}
		.ch.new {
			animation: none;
		}
	}
</style>
