<!--
  Lesson step 6: training lowers the loss. Scrub TinyGPT's 10 saved checkpoints (step 0 = random
  weights … 30,000 = final): the followed window's probabilities at that checkpoint (live, the fp16
  checkpoint file, downloaded only when picked) and this window's loss at every checkpoint loaded so
  far, next to the validation loss logged during training (precomputed).
-->
<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { base } from '$app/paths';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtBytes, fmtCount, fmtInt, visibleToken } from '$lib/kit/format';
	import type { WalkContext } from '$lib/kit/types';
	import LessonWindow from './LessonWindow.svelte';
	import TopBars from './TopBars.svelte';
	import DistChart from './DistChart.svelte';
	import type { NtpLesson } from './lessonState.svelte';
	import { distKey, pct, shortStep } from './lessonMath';
	import { FACTS } from './facts';

	let { L, ctx }: { L: NtpLesson; ctx: WalkContext } = $props();
	const S = $derived(L.S);

	const cks = $derived(L.ckpts ?? []);
	let ci = $state(0);
	const ck = $derived(cks[ci]);
	const step = $derived(ck?.step ?? 0);
	let playing = $state(false);
	let scrubEl: HTMLElement | undefined = $state();

	// KIT rule 11: phones auto-load at most 5 MB of model files. When the lesson's Play starts the
	// training replay on a phone, it plays 4 checkpoints (0, 100, 1k, 30k: 2.9 MB, plus weights.bin
	// 1.46 MB). The "Play training" button (its size is shown) always plays all 10.
	const PHONE_SET = [0, 100, 1000, 30000];
	let only = $state<number[] | null>(null);
	function isPhone(): boolean {
		try {
			return matchMedia('(max-width: 720px), (pointer: coarse)').matches;
		} catch {
			return false;
		}
	}
	function autoPlay() {
		only = isPhone() ? PHONE_SET : null;
		playing = true;
	}
	/** the checkpoint after index i that the replay shows, or -1 at the end */
	function nextIdx(i: number): number {
		for (let j = i + 1; j < cks.length; j++) if (!only || only.includes(cks[j].step)) return j;
		return -1;
	}

	onMount(() => {
		void L.loadIndex(`${base}/models/tinygpt/checkpoints/index.json`);
		// when the whole lesson is playing, play the training too (the learner asked to watch)
		if (untrack(() => ctx.playing && !ctx.reduced)) autoPlay();
	});

	// the selected checkpoint: this window's distribution and the whole sentence's loss
	$effect(() => {
		const st = step;
		void L.ctx;
		void L.target;
		void L.idsKey;
		if (!ck || !S.client) return;
		const h = setTimeout(() => {
			L.request(st);
			L.requestSentence(st);
		}, 60);
		return () => clearTimeout(h);
	});
	// every checkpoint already downloaded: this window's loss for the chart (cheap: one forward each)
	const ready = $derived(cks.filter((c) => S.statusOf(c.step)?.state === 'ready').map((c) => c.step));
	$effect(() => {
		const steps = ready;
		void L.ctx;
		void L.target;
		if (!S.client) return;
		const h = setTimeout(() => steps.forEach((st) => L.request(st)), 200);
		return () => clearTimeout(h);
	});

	const d = $derived(ck ? L.dist(step) : null);
	const status = $derived(ck ? S.statusOf(step) : undefined);
	const failed = $derived(status?.state === 'error');

	// play: show each checkpoint until its numbers are in, then move on
	$effect(() => {
		if (!playing) return;
		if (ctx.reduced) {
			playing = false;
			return;
		}
		const have = !!d || failed;
		if (!have || !cks.length) return;
		const h = setTimeout(() => {
			const j = nextIdx(ci);
			if (j >= 0) ci = j;
			else {
				playing = false;
				only = null;
				ctx.done();
			}
		}, 1400);
		return () => clearTimeout(h);
	});
	// Play lesson pressed while on this step: play the training too, or finish if it was already played
	$effect(() => {
		if (!ctx.playing || ctx.reduced) return;
		untrack(() => {
			if (playing || !cks.length) return;
			if (ci >= cks.length - 1 && (d || failed)) ctx.done();
			else autoPlay();
		});
	});
	function play() {
		if (ctx.reduced) {
			if (ci < cks.length - 1) ci++;
			return;
		}
		if (playing) {
			playing = false;
			return;
		}
		if (ci >= cks.length - 1) ci = 0;
		only = null;
		playing = true;
	}
	function select(i: number) {
		playing = false;
		only = null;
		ci = i;
	}
	function onScrubKey(e: KeyboardEvent, i: number) {
		let j = i;
		if (e.key === 'ArrowRight' || e.key === 'ArrowDown') j = Math.min(cks.length - 1, i + 1);
		else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') j = Math.max(0, i - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = cks.length - 1;
		else return;
		e.preventDefault();
		select(j);
		(scrubEl?.querySelectorAll('button[role=radio]')[j] as HTMLElement | undefined)?.focus();
	}

	const statusText = (s: number) => {
		const st = S.statusOf(s);
		if (!st) return t({ en: 'not downloaded', zh: '未下载' });
		if (st.state === 'ready') return t({ en: 'downloaded', zh: '已下载' });
		if (st.state === 'error') return t({ en: 'failed', zh: '失败' });
		return st.total ? `${Math.round((100 * st.loaded) / st.total)}%` : t({ en: 'downloading', zh: '下载中' });
	};
	const loadedN = $derived(ready.length);
	const left = $derived(Math.max(0, cks.length - loadedN) * FACTS.ckptBytes);

	const sent = $derived(ck ? L.sentenceLoss(step) : null);
	const finalLoss = $derived(S.fresh && S.score && S.score.scored > 0 ? S.score.meanNats : null);

	// ---------------------------------------------------------------- the window's loss at every checkpoint
	let W = $state(320);
	const CH = 178;
	const cm = { l: 30, r: 10, t: 12, b: 34 };
	const pts = $derived(
		cks.map((c, i) => {
			const dd = L.dists[distKey(c.step, L.ctx, L.target)];
			return { i, step: c.step, val: c.valLoss, mine: dd ? -Math.log(dd.p) : null };
		})
	);
	const yTop = $derived(Math.max(7.5, ...pts.map((p) => (p.mine === null ? 0 : Math.min(20, p.mine + 0.5)))));
	const cx = (i: number) => cm.l + (cks.length > 1 ? (i / (cks.length - 1)) * (W - cm.l - cm.r) : 0);
	const cy = (v: number) => cm.t + (1 - Math.min(v, yTop) / yTop) * (CH - cm.t - cm.b);
	const valPath = $derived(pts.length ? 'M' + pts.map((p) => `${cx(p.i).toFixed(1)},${cy(p.val).toFixed(1)}`).join('L') : '');
	const minePath = $derived.by(() => {
		let s = '';
		let pen = false;
		for (const p of pts) {
			if (p.mine === null) {
				pen = false;
				continue;
			}
			s += `${pen ? 'L' : 'M'}${cx(p.i).toFixed(1)},${cy(p.mine).toFixed(1)}`;
			pen = true;
		}
		return s;
	});
	const yTicks = $derived(yTop > 12 ? [0, 5, 10, 15, 20].filter((v) => v <= yTop) : [0, 2, 4, 6].filter((v) => v <= yTop));
</script>

<div class="tr">
	<LessonWindow {L} />
	{#if L.pos < 1}
		<p class="muted">{t({ en: 'Type at least two tokens.', zh: '至少输入两个词元。' })}</p>
	{:else if !cks.length}
		<p class="muted">{L.error ?? t({ en: 'Loading the list of checkpoints…', zh: '正在加载检查点列表…' })}</p>
	{:else}
		<div class="scrub-row">
			<button type="button" class="btn primary play" onclick={play} aria-pressed={ctx.reduced ? undefined : playing}>
				{#if ctx.reduced}
					{t({ en: 'Next checkpoint', zh: '下一个检查点' })}
				{:else if playing}
					<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="3.5" y="3" width="3" height="10" rx="1" fill="currentColor" /><rect x="9.5" y="3" width="3" height="10" rx="1" fill="currentColor" /></svg>
					{t({ en: 'Pause', zh: '暂停' })}
				{:else}
					<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4.5 2.8v10.4L13 8z" fill="currentColor" /></svg>
					{t({ en: 'Play training', zh: '播放训练过程' })}
				{/if}
			</button>
			<div class="scrub" role="radiogroup" aria-label={t({ en: 'Training checkpoint', zh: '训练检查点' })} bind:this={scrubEl}>
				{#each cks as c, i (c.step)}
					{@const st = S.statusOf(c.step)?.state}
					<button
						type="button"
						role="radio"
						aria-checked={i === ci}
						tabindex={i === ci ? 0 : -1}
						class="ckb"
						class:on={i === ci}
						onclick={() => select(i)}
						onkeydown={(e) => onScrubKey(e, i)}
						aria-label="{t({ en: 'step', zh: '第' })} {fmtInt(c.step)}{t({ en: '', zh: ' 步' })}, {statusText(c.step)}"
					>
						<span class="lbl num">{shortStep(c.step)}</span>
						<span class="st {st ?? 'none'}" aria-hidden="true"></span>
					</button>
				{/each}
			</div>
		</div>
		<p class="dl small">
			{t({
				en: `${ck.file} (${fmtBytes(FACTS.ckptBytes)}; fp16: each number stored in 2 bytes instead of 4): ${statusText(step)}. On your device: ${loadedN} of ${cks.length} checkpoint files${left > 0 ? `; ${fmtBytes(left)} more if you play all` : ''}.`,
				zh: `${ck.file}（${fmtBytes(FACTS.ckptBytes)}；fp16：每个数只用 2 个字节存储，而不是 4 个）：${statusText(step)}。本机已有 ${loadedN} / ${cks.length} 个检查点文件${left > 0 ? `；全部播放还需下载 ${fmtBytes(left)}` : ''}。`
			})}
		</p>

		<div class="cols">
			<div class="pane">
				<div class="ph">
					<h4>
						{t({ en: 'Step', zh: '第' })} <span class="num">{fmtInt(step)}</span>{t({ en: '', zh: ' 步' })}
						{#if ck.label}<span class="tag">{t(ck.label)}</span>{/if}
					</h4>
					<ProvenanceBadge
						mode="live"
						compact
						precision="fp16"
						approximate
						note={{
							en: 'Runs the fp16 checkpoint file in your browser. On the same fp16 weights the JS forward pass matches PyTorch within 1.2e-5.',
							zh: '在你的浏览器里运行 fp16 检查点文件。在同样的 fp16 权重上，JS 前向计算与 PyTorch 的差距在 1.2e-5 以内。'
						}}
					/>
				</div>
				<p class="small muted">
					{t({ en: `${fmtCount(ck.tokens)} tokens read so far (${fmt3(ck.tokens / FACTS.trainTokens)} passes over the stories)`, zh: `到此已读 ${fmtCount(ck.tokens)} 个词元（把故事读了 ${fmt3(ck.tokens / FACTS.trainTokens)} 遍）` })}
				</p>
				{#if failed}
					<p class="err small">{t({ en: 'Could not load this checkpoint.', zh: '无法加载这个检查点。' })} {status?.message ?? ''}</p>
				{:else if !d}
					<p class="muted small" aria-live="polite">
						{t({ en: 'Downloading and running', zh: '正在下载并运行' })} <span class="num">{ck.file}</span> … {statusText(step)}
					</p>
				{:else}
					<TopBars dist={d} disp={(id) => S.disp(id)} k={5} label={{ en: 'This window at this checkpoint', zh: '这个窗口在这个检查点上' }} />
					<p class="small">
						p(<span class="tok">{visibleToken(S.disp(d.target))}</span>) = <strong class="num">{pct(d.p)}</strong>
						· {t({ en: 'rank', zh: '排名' })} <span class="num">{d.rank}</span> ·
						−ln p = <strong><Num value={-Math.log(d.p)} /></strong>
						{t({ en: 'nats', zh: '奈特' })}
					</p>
				{/if}
				{#if sent !== null}
					<p class="small muted">
						{t({ en: 'Your whole sentence at this checkpoint:', zh: '你的整句话在这个检查点上：' })}
						<Num value={sent} />
						{t({ en: 'nats/token', zh: '奈特/词元' })}{#if finalLoss !== null && step !== cks[cks.length - 1]?.step}{' '}·
							{t({ en: 'final model:', zh: '最终模型：' })} <Num value={finalLoss} />{/if}
					</p>
				{/if}
			</div>

			<figure class="pane chart" bind:clientWidth={W}>
				<figcaption class="small">
					{t({ en: 'This window’s loss at each checkpoint', zh: '这个窗口在各检查点上的损失' })}
				</figcaption>
				<svg width={W} height={CH} viewBox="0 0 {W} {CH}" role="img" aria-label={t({ en: 'Loss of this window at every downloaded checkpoint, with the validation loss logged during training.', zh: '这个窗口在每个已下载检查点上的损失，以及训练时记录的验证损失。' })}>
					{#each yTicks as v (v)}
						<line class="grid" x1={cm.l} x2={W - cm.r} y1={cy(v)} y2={cy(v)} />
						<text class="tick" x={cm.l - 5} y={cy(v) + 4} text-anchor="end">{v}</text>
					{/each}
					<line class="uni" x1={cm.l} x2={W - cm.r} y1={cy(FACTS.uniformLoss)} y2={cy(FACTS.uniformLoss)} />
					<line class="sel" x1={cx(ci)} x2={cx(ci)} y1={cm.t} y2={CH - cm.b} />
					<path class="val" d={valPath} />
					{#each pts as p (p.step)}
						<circle class="vdot" cx={cx(p.i)} cy={cy(p.val)} r="2.5" />
						<text class="tick" x={cx(p.i)} y={CH - cm.b + 13} text-anchor="middle">{shortStep(p.step)}</text>
					{/each}
					<path class="mine" d={minePath} />
					{#each pts as p (p.step)}
						{#if p.mine !== null}<circle class="mdot" class:on={p.i === ci} cx={cx(p.i)} cy={cy(p.mine)} r={p.i === ci ? 5.5 : 4} />{/if}
					{/each}
					<text class="tick" x={cm.l + (W - cm.l - cm.r) / 2} y={CH - 4} text-anchor="middle">{t({ en: 'training step (not to scale)', zh: '训练步数（不按比例）' })}</text>
				</svg>
				<div class="legend small">
					<span><i class="dt mine-k"></i>{t({ en: 'this window (live)', zh: '这个窗口（实时）' })}</span>
					<span><i class="ln val-k"></i>{t({ en: 'validation loss: average on unseen stories (precomputed)', zh: '验证损失：没见过的故事上的平均值（预计算）' })}</span>
					<span><i class="ln uni-k"></i>{t({ en: 'blind guess 6.93', zh: '盲猜 6.93' })}</span>
				</div>
				{#if d}<div class="all"><DistChart dist={d} disp={(id) => S.disp(id)} height={160} /></div>{/if}
			</figure>
		</div>
	{/if}
</div>

<style>
	.tr {
		container: tr / inline-size;
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.scrub-row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
	.play {
		min-width: 8.5em;
		gap: 6px;
	}
	.scrub {
		display: grid;
		grid-template-columns: repeat(10, minmax(0, 1fr));
		gap: 3px;
		flex: 1 1 360px;
		min-width: 0;
	}
	@container tr (max-width: 520px) {
		.scrub {
			grid-template-columns: repeat(5, minmax(0, 1fr));
			flex-basis: 100%;
		}
	}
	.ckb {
		display: grid;
		justify-items: center;
		gap: 3px;
		padding: 5px 2px 4px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		color: var(--ink);
		cursor: pointer;
		min-height: 42px;
		font: inherit;
	}
	.ckb:hover {
		background: var(--surface-2);
	}
	.ckb.on {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--accent-ink);
	}
	.lbl {
		font-size: var(--fs-sm);
		font-weight: 650;
	}
	.st {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		border: 1.5px solid currentColor;
		opacity: 0.7;
	}
	.st.ready {
		background: currentColor;
		opacity: 1;
	}
	.st.loading {
		border-style: dotted;
		opacity: 1;
	}
	.st.error {
		border-color: var(--k);
	}
	.dl {
		margin: 0;
		color: var(--ink-2);
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 14px;
		align-items: start;
	}
	@container tr (max-width: 600px) {
		.cols {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	.pane {
		display: grid;
		gap: 8px;
		min-width: 0;
		margin: 0;
	}
	.pane p {
		margin: 0;
	}
	.ph {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 10px;
		align-items: center;
		justify-content: space-between;
	}
	h4 {
		margin: 0;
		font-size: var(--fs-lg);
		display: flex;
		gap: 8px;
		align-items: baseline;
	}
	.tag {
		font-size: var(--fs-xs);
		font-weight: 400;
		padding: 1px 8px;
		border-radius: var(--r-pill);
		background: var(--surface-2);
		border: 1px solid var(--rule-strong);
	}
	.small {
		font-size: var(--fs-xs);
	}
	.err {
		color: var(--k-text);
	}
	.tok {
		font-family: var(--font-mono);
		background: var(--surface-2);
		padding: 0 4px;
		border-radius: 3px;
		white-space: pre;
	}
	.chart figcaption {
		font-weight: 650;
		color: var(--ink-2);
	}
	.chart svg {
		display: block;
		width: 100%;
		height: auto;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r2);
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.tick {
		font-family: var(--font-mono);
		font-size: 10px;
		fill: var(--muted);
	}
	.uni {
		stroke: var(--ink-2);
		stroke-dasharray: 4 4;
		opacity: 0.6;
	}
	.sel {
		stroke: var(--accent);
		stroke-width: 2;
		opacity: 0.6;
	}
	.val {
		fill: none;
		stroke: var(--pre);
		stroke-width: 1.5;
	}
	.vdot {
		fill: var(--pre);
	}
	.mine {
		fill: none;
		stroke: var(--real);
		stroke-width: 2;
	}
	.mdot {
		fill: var(--real);
		stroke: var(--surface);
		stroke-width: 2;
	}
	.mdot.on {
		stroke: var(--accent);
		stroke-width: 2.5;
	}
	.all {
		margin-top: 6px;
	}
	@container tr (max-width: 600px) {
		.all {
			display: none;
		}
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 12px;
		color: var(--ink-2);
	}
	.legend span {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.dt {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
	}
	.ln {
		display: inline-block;
		width: 14px;
		height: 0;
		border-top: 2px solid;
	}
	.mine-k {
		background: var(--real);
	}
	.val-k {
		border-color: var(--pre);
	}
	.uni-k {
		border-top-style: dashed;
		border-color: var(--ink-2);
	}
</style>
