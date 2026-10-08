<!--
  Lesson step 2: every position is a lesson. A step player slides the (context -> next token)
  window over the learner's tokens; the staircase shows all windows at once (it is the causal
  mask). The window the player stops on is the one the next steps follow.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { visibleToken } from '$lib/kit/format';
	import type { WalkContext } from '$lib/kit/types';
	import type { NtpLesson } from './lessonState.svelte';
	import { ctxStart, tailText, CONTEXT } from './lessonMath';

	let { L, ctx }: { L: NtpLesson; ctx: WalkContext } = $props();
	const S = $derived(L.S);

	const count = $derived(Math.max(1, L.windows));
	// a fresh visit sweeps all windows from the first; a window the learner picked is kept. Windows the
	// sweep passes are not picks: only the learner's own moves (step buttons, keys, a row, pausing) are.
	const sweep = untrack(() => L.picked === null && !ctx.reduced && L.windows > 1);
	let k = $state(untrack(() => (sweep ? 0 : Math.max(0, L.pos - 1))));
	let playing = $state(false);
	$effect(() => {
		const c = count;
		if (untrack(() => k) > c - 1) k = c - 1;
	});
	// pausing the sweep part-way is a pick: the next steps follow the window on screen
	let ended = false;
	let stopped = false;
	let wasPlaying = false;
	$effect(() => {
		const p = playing;
		if (wasPlaying && !p && !ended) {
			stopped = true;
			untrack(() => L.pick(k + 1));
		}
		if (p) ended = stopped = false;
		wasPlaying = p;
	});
	// Play lesson pressed while this step's sweep is not running: the step is done (no need to wait)
	$effect(() => {
		if (ctx.playing && untrack(() => !playing && (!sweep || ended || stopped))) ctx.done();
	});
	// about 9 s for the whole sweep, 0.25-0.7 s per window
	const interval = $derived(Math.max(250, Math.min(700, Math.round(9000 / count))));

	const cur = $derived(Math.max(0, Math.min(k, count - 1)));
	const w = $derived(cur + 1);
	const s0 = $derived(ctxStart(w));
	const disp = (id: number) => visibleToken(S.disp(id));
	const ctxText = $derived(tailText(S.ids.slice(s0, w), (ids) => S.decode(ids), 60, (id) => S.disp(id)));
	function cls(i: number): string {
		if (i < s0) return 'cut';
		if (i < w) return 'ctx';
		if (i === w) return 'tgt';
		return 'fut';
	}

	// staircase geometry
	let boxW = $state(320);
	const n = $derived(S.ids.length);
	const cell = $derived(Math.max(3, Math.min(13, Math.floor((Math.min(boxW, 560) - 2) / Math.max(1, n)))));
	const gridW = $derived(cell * Math.max(1, n));
	const gridH = $derived(cell * Math.max(1, n - 1));
	const rows = $derived(Array.from({ length: Math.max(0, n - 1) }, (_, r) => r));
	function pickRow(e: MouseEvent) {
		const svg = e.currentTarget as SVGSVGElement;
		const r = svg.getBoundingClientRect();
		const row = Math.floor(((e.clientY - r.top) / r.height) * (n - 1));
		if (row >= 0 && row < n - 1) {
			playing = false;
			k = row;
			L.pick(row + 1);
		}
	}
</script>

{#if L.windows < 1}
	<p class="muted">{t({ en: 'Type at least two tokens: the first token has nothing before it, so there is no window yet.', zh: '至少输入两个词元：第一个词元前面什么都没有，还没有窗口。' })}</p>
{:else}
	<div class="wn">
		<StepPlayer
			{count}
			bind:step={k}
			bind:playing
			{interval}
			autoplay={sweep && ctx.shown}
			label={{ en: 'Slide the window over your sentence', zh: '让窗口滑过你的句子' }}
			onstep={(s) => {
				if (!playing) L.pick(s + 1);
			}}
			onend={() => {
				ended = true;
				L.pick(null);
				ctx.done();
			}}
		>
			{#snippet children()}
				<div class="stage">
					<p class="ex num">
						{t({ en: `Window ${w} of ${L.windows}`, zh: `窗口 ${w} / ${L.windows}` })}
					</p>
					<div class="sent" aria-hidden="true">
						{#each S.ids as id, i (i)}<span class="tk {cls(i)}">{disp(id)}</span>{/each}
					</div>
					<div class="pair">
						<div class="side">
							<span class="lab">{t({ en: 'Input: the context', zh: '输入：上下文' })}</span>
							<span class="ctxt">{visibleToken(ctxText)}</span>
							<span class="muted small num">
								{w - s0}
								{t({ en: w - s0 === 1 ? 'token' : 'tokens', zh: '个词元' })}{#if s0 > 0}
									· {t({ en: `starts at token ${s0} (64-token window)`, zh: `从第 ${s0} 个词元开始（64 词元窗口）` })}{/if}
							</span>
						</div>
						<span class="arrow" aria-hidden="true">→</span>
						<div class="side">
							<span class="lab">{t({ en: 'Label: the next token', zh: '标签：下一个词元' })}</span>
							<span class="tk tgt big">{disp(S.ids[w])}</span>
						</div>
					</div>
				</div>
			{/snippet}
		</StepPlayer>

		<figure class="stair" bind:clientWidth={boxW}>
			<figcaption>
				{t({ en: 'All windows at once', zh: '所有窗口一览' })}
				<span class="muted small">{t({ en: '(rows = windows, columns = tokens; click a row)', zh: '（行 = 窗口，列 = 词元；可点击某一行）' })}</span>
			</figcaption>
			<!-- the step player is the keyboard path; a click on a row is a pointer shortcut -->
			<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
			<svg width={gridW} height={gridH} viewBox="0 0 {gridW} {gridH}" onclick={pickRow} role="presentation">
				{#each rows as r (r)}
					{@const a = ctxStart(r + 1)}
					{#if a > 0}<rect class="cutc" x={0} y={r * cell} width={a * cell} height={cell} />{/if}
					<rect class="ctxc" class:cur={r === cur} x={a * cell} y={r * cell} width={(r + 1 - a) * cell} height={cell} />
					<rect class="tgtc" x={(r + 1) * cell} y={r * cell} width={cell} height={cell} />
				{/each}
				<rect class="row" x="0.5" y={cur * cell + 0.5} width={gridW - 1} height={Math.max(1, cell - 1)} />
			</svg>
			<div class="legend small">
				<span><i class="sw ctxc"></i>{t({ en: 'context the model sees', zh: '模型看到的上下文' })}</span>
				<span><i class="sw tgtc"></i>{t({ en: 'label (next token)', zh: '标签（下一个词元）' })}</span>
				{#if n > CONTEXT}<span><i class="sw cutc"></i>{t({ en: 'outside the 64-token window', zh: '在 64 词元窗口之外' })}</span>{/if}
			</div>
		</figure>
	</div>
{/if}

<style>
	.wn {
		display: grid;
		gap: 14px;
		min-width: 0;
	}
	.stage {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.ex {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}
	.sent {
		display: flex;
		flex-wrap: wrap;
		gap: 3px 2px;
	}
	.tk {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		padding: 1px 4px;
		border-radius: 4px;
		border: 1.5px solid transparent;
		white-space: pre;
	}
	.tk.ctx {
		background: var(--accent-soft);
		color: var(--ink);
	}
	.tk.cut {
		background: repeating-linear-gradient(135deg, var(--surface-2) 0 3px, var(--surface) 3px 6px);
		color: var(--muted);
	}
	.tk.tgt {
		border-color: var(--real);
		background: color-mix(in srgb, var(--real) 16%, var(--surface));
		color: var(--ink);
		font-weight: 700;
	}
	.tk.fut {
		color: var(--muted);
		border-style: dashed;
		border-color: var(--rule);
	}
	.tk.big {
		font-size: var(--fs-md);
		padding: 3px 8px;
		justify-self: start;
	}
	.pair {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto auto;
		align-items: center;
		gap: 10px;
		padding: 10px;
		border-radius: var(--r2);
		background: var(--surface-2);
	}
	.side {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.lab {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		font-weight: 650;
	}
	.ctxt {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.arrow {
		font-size: var(--fs-xl);
		color: var(--muted);
	}
	@media (max-width: 480px) {
		.pair {
			grid-template-columns: minmax(0, 1fr);
		}
		.arrow {
			transform: rotate(90deg);
			justify-self: start;
		}
	}
	.small {
		font-size: var(--fs-xs);
	}
	.stair {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.stair figcaption {
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.stair svg {
		display: block;
		max-width: 100%;
		height: auto;
		cursor: pointer;
	}
	.ctxc {
		fill: var(--accent-soft);
		stroke: var(--surface);
		stroke-width: 0.5;
	}
	.ctxc.cur {
		fill: color-mix(in srgb, var(--accent) 45%, var(--accent-soft));
	}
	.tgtc {
		fill: var(--real);
	}
	.cutc {
		fill: var(--surface-3);
	}
	.row {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 1;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	.sw {
		display: inline-block;
		width: 11px;
		height: 11px;
		border-radius: 2px;
		margin-right: 5px;
		vertical-align: -1px;
	}
	.sw.ctxc {
		background: var(--accent-soft);
		border: 1px solid var(--accent);
	}
	.sw.tgtc {
		background: var(--real);
	}
	.sw.cutc {
		background: var(--surface-3);
	}
</style>
