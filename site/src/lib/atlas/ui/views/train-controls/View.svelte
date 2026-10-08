<!--
  train-controls: Play / Pause, +1 / +10 steps, Reset and the learning rate for a live train source (TrainSession over
  the source's LiveTape), with step, loss, learning rate, speed, where it runs, and a loss sparkline.
  With props.target, Play stops at that step and the stage reports done.
-->
<script lang="ts">
	import type { StageProps } from '../../../types/view.ts';
	import type { LiveTape } from '../../../types/process.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '../../../core/format.ts';
	import Value from '../../train/Value.svelte';
	import { TrainSession } from '../../train/session.svelte.ts';
	import { downsample, isLive, tapeOf } from '../../train/refs.ts';
	import meta, { controlsInfo, DEFAULT_LRS } from './meta.ts';

	// props one by one (no rest/spread props: they would grow the Svelte runtime chunk every page preloads)
	let { rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props }: StageProps = $props();
	const p: StageProps = $derived({ rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props });

	const session = new TrainSession();
	const tape = $derived(tapeOf(p.rt, p.source));
	const live = $derived(isLive(tape));
	$effect(() => {
		session.bind(live ? (tape as LiveTape) : null);
	});
	$effect(() => () => session.close());

	const lrs = $derived((Array.isArray(p.props?.lrs) ? (p.props.lrs as number[]) : DEFAULT_LRS).filter((v) => v > 0));
	const target = $derived(typeof p.props?.target === 'number' ? (p.props.target as number) : null);
	const showReset = $derived(p.props?.showReset !== false);
	// a fine-detail tape records 4 frames per training step (forward, loss, backward, update): "+1 step" runs to the end
	// of the next update, so the tape always stops on a whole step
	const FINE = ['forward', 'loss', 'backward', 'update'];
	const fine = $derived.by(() => {
		void session.version;
		return !!tape && tape.length > 0 && tape.find({ phase: 'forward' }) >= 0;
	});
	function framesFor(steps: number): number {
		if (!fine || !tape || tape.length === 0) return steps;
		const k = FINE.indexOf(tape.frame(tape.length - 1).phase);
		const toUpdate = k < 0 || k === 3 ? 4 : 3 - k;
		return toUpdate + 4 * (steps - 1);
	}

	// numbers: the live session, or the recorded tape at the cursor
	const info = $derived.by(() => {
		void session.version;
		return controlsInfo(p);
	});
	const lrNow = $derived(info?.lr ?? NaN);

	// the target: pause there and report done once
	let reached = false;
	$effect(() => {
		void session.version;
		if (target === null || reached || !info) return;
		if (info.step >= target) {
			reached = true;
			if (live && !session.paused) session.pause();
			p.ctx?.done();
			p.onend();
		}
	});

	function onLr(e: Event) {
		const v = Number((e.currentTarget as HTMLSelectElement).value);
		if (Number.isFinite(v) && v > 0) session.setLr(v);
	}

	// sparkline of the loss
	let W = $state(320);
	const spark = $derived.by(() => {
		void session.version;
		const tp = tapeOf(p.rt, p.source);
		if (!tp || tp.length < 2) return '';
		const ys = tp.metric('loss');
		const xs = tp.metric('step');
		const d = downsample(xs, ys, 160);
		if (d.x.length < 2) return '';
		let lo = Infinity,
			hi = -Infinity;
		for (const y of d.y) {
			lo = Math.min(lo, y);
			hi = Math.max(hi, y);
		}
		const x0 = d.x[0],
			x1 = d.x[d.x.length - 1] || 1;
		const w = Math.max(80, W);
		return d.x.map((x, i) => `${i ? 'L' : 'M'}${(((x - x0) / (x1 - x0 || 1)) * (w - 4) + 2).toFixed(1)},${(2 + (1 - (d.y[i] - lo) / (hi - lo || 1)) * 30).toFixed(1)}`).join('');
	});
	const lrLabel = (v: number) => (v >= 1 ? String(v) : fmt3(v));
	const description = $derived.by(() => {
		void session.version;
		return t(meta.describe(p));
	});
</script>

<div class="tc" style:min-height="{meta.minHeight}px" bind:clientWidth={W}>
	<div class="row buttons" role="group" aria-label={t({ en: 'Training controls', zh: '训练控制' })}>
		<button class="btn primary" type="button" disabled={!live || session.done} onclick={() => session.toggle()} aria-pressed={live && !session.paused && !session.done}>
			{live && session.done ? t({ en: 'Finished', zh: '已完成' }) : live && !session.paused ? t({ en: 'Pause', zh: '暂停' }) : t({ en: 'Train', zh: '训练' })}
		</button>
		<button class="btn" type="button" disabled={!live || session.done} onclick={() => session.stepBy(framesFor(1))}>{t({ en: '+1 step', zh: '+1 步' })}</button>
		<button class="btn" type="button" disabled={!live || session.done} onclick={() => session.stepBy(framesFor(10))}>{t({ en: '+10', zh: '+10' })}</button>
		{#if showReset}
			<button class="btn ghost" type="button" disabled={!live} onclick={() => session.reset()}>{t({ en: 'Reset', zh: '重置' })}</button>
		{/if}
		<label class="lr">
			<span>{t({ en: 'Learning rate', zh: '学习率' })}</span>
			<select disabled={!live} onchange={onLr} value={lrs.find((v) => Math.abs(v - lrNow) <= 1e-12 * Math.max(1, v)) ?? ''}>
				{#if !lrs.some((v) => Math.abs(v - lrNow) <= 1e-12 * Math.max(1, v))}<option value="" disabled>{fmt3(lrNow)}</option>{/if}
				{#each lrs as v (v)}<option value={v}>{lrLabel(v)}</option>{/each}
			</select>
		</label>
	</div>

	<dl class="row readout" aria-live="off">
		<div><dt>{t({ en: 'Step', zh: '步数' })}</dt><dd class="num">{info ? (Number.isFinite(info.step) ? info.step : '—') : '—'}{#if target !== null}<span class="muted"> / {target}</span>{/if}</dd></div>
		<div><dt>{t({ en: 'Loss', zh: '损失' })}</dt><dd><Value value={info?.loss} /></dd></div>
		<div><dt>{t({ en: 'Learning rate', zh: '学习率' })}</dt><dd><Value value={info?.lr} /></dd></div>
		{#if live}
			<div><dt>{t({ en: 'Speed', zh: '速度' })}</dt><dd class="num">{session.paused || session.done || !session.rate ? '—' : `${Math.round(session.rate)} ${t({ en: 'steps/s', zh: '步/秒' })}`}</dd></div>
			<div><dt>{t({ en: 'Runs', zh: '运行于' })}</dt><dd>{session.where === 'worker' ? t({ en: 'background thread', zh: '后台线程' }) : session.where === 'starting' ? t({ en: 'starting…', zh: '启动中…' }) : t({ en: 'this page', zh: '本页面' })}</dd></div>
		{/if}
	</dl>

	{#if spark}
		<svg class="spark" width={Math.max(80, W)} height="34" viewBox="0 0 {Math.max(80, W)} 34" aria-hidden="true"><path d={spark} /></svg>
	{/if}
	{#if live && session.done && !session.error}
		<p class="note muted small">{t({ en: 'This run is over. Reset to train again from new weights.', zh: '这次运行结束了。点“重置”可以从新的权重重新训练。' })}</p>
	{/if}
	{#if !live}
		<p class="note muted small">{t({ en: 'A recorded run: the buttons work on a live source.', zh: '这是一次记录好的运行：按钮只对实时的数据源有效。' })}</p>
	{/if}
	{#if session.notice}<p class="note small">{t(session.notice)}</p>{/if}
	{#if session.error}<p class="note warn small">{session.error}</p>{/if}
	{#if session.diverged}<p class="note warn small">{t({ en: 'The loss is no longer a finite number: training diverged. Try a smaller learning rate and Reset.', zh: '损失不再是有限的数：训练发散了。试试更小的学习率，然后重置。' })}</p>{/if}
	<p class="visually-hidden" aria-live="polite">{session.paused ? description : ''}</p>
</div>

<style>
	.tc {
		display: grid;
		gap: var(--s2);
		align-content: start;
		min-width: 0;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: var(--s2);
		align-items: center;
	}
	.lr {
		display: inline-flex;
		gap: 6px;
		align-items: center;
		font-size: var(--fs-sm, 14px);
	}
	.lr select {
		font: inherit;
		font-family: var(--font-mono);
		padding: 4px 6px;
		border-radius: var(--r1);
		border: 1px solid var(--rule-strong);
		background: var(--surface);
		color: var(--ink);
	}
	.readout {
		margin: 0;
		gap: var(--s2) var(--s4);
	}
	.readout div {
		display: grid;
		gap: 2px;
		min-width: 64px;
	}
	dt {
		font-size: 12px;
		color: var(--muted);
	}
	dd {
		margin: 0;
		font-size: 15px;
	}
	.spark {
		display: block;
		max-width: 100%;
	}
	.spark path {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.5;
	}
	.note {
		margin: 0;
	}
	.warn {
		color: var(--k-text);
	}
</style>
