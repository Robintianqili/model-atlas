<!--
  StepPlayer: play / pause / step / scrub over `count` steps. Every animation is a pure
  function of the step (or of a continuous t), so play, scrub and screenshots agree.

  Discrete:   <StepPlayer count={9} bind:step labels={stepNames}>{#snippet children(s)}…{/snippet}</StepPlayer>
  Continuous: <StepPlayer count={9} smooth bind:t>{#snippet children(s, t)}{@render view(stateAt(t))}{/snippet}</StepPlayer>

  - `t` runs from 0 to count − 1 (fractional while `smooth` playback is between steps);
    `step` = Math.floor(t).
  - Keyboard (when focus is inside the player): Space play/pause, ←/→ step, Home/End.
  - prefers-reduced-motion: no autoplay and no smooth motion; Play becomes "Next step",
    and the step list (labels) is always shown.
  - `playing` is bindable (read it to, e.g., mask cells that are not computed yet while playing);
    `onend` fires when playback reaches the last step and stops (not when looping, not on manual steps).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { motion } from './theme.svelte';
	import { ui, t as tr, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		count: number;
		step?: number;
		t?: number;
		/** ms per step at 1× */
		interval?: number;
		loop?: boolean;
		autoplay?: boolean;
		smooth?: boolean;
		labels?: Text[];
		/** show the step list even without reduced motion */
		showList?: boolean;
		speeds?: number[];
		label?: Text;
		children?: Snippet<[number, number]>;
		onstep?: (step: number) => void;
		/** bindable: true while playing */
		playing?: boolean;
		/** called when playback reaches the last step and stops */
		onend?: () => void;
		/** text for "Step i of n" (i is 1-based), e.g. "Position i of n" when a lesson around it has steps too */
		position?: (i: number, n: number) => string;
		/**
		 * position text with {i} and {n} placeholders (default ui('stepOf'): "Step {i} of {n}"); e.g. a player
		 * inside a lesson step can say "Part {i} of {n}" so it is not mistaken for the lesson's own steps
		 */
		counter?: Text;
	}
	let {
		count,
		step = $bindable(0),
		t = $bindable(0),
		interval = 800,
		loop = false,
		autoplay = false,
		smooth = false,
		labels,
		showList = false,
		speeds = [0.5, 1, 2],
		label,
		children,
		onstep,
		playing = $bindable(false),
		onend,
		position,
		counter
	}: Props = $props();
	const posText = (i: number, n: number) =>
		position
			? position(i, n)
			: counter
				? tr(counter).replaceAll('{i}', String(i)).replaceAll('{n}', String(n))
				: ui('stepOf', { i, n });

	let speed = $state(1);
	const last = $derived(Math.max(0, count - 1));
	const reduced = $derived(motion.reduced);
	const gid = $props.id();

	function setT(v: number) {
		const clamped = Math.max(0, Math.min(last, v));
		t = clamped;
		const s = Math.floor(clamped + 1e-9);
		if (s !== step) {
			step = s;
			onstep?.(s);
		}
	}
	function go(s: number) {
		setT(Math.round(s));
	}
	function next() {
		if (step >= last) {
			if (loop) go(0);
			else playing = false;
		} else go(step + 1);
	}
	function prev() {
		go(step - 1);
	}
	function toggle() {
		if (reduced) {
			next();
			return;
		}
		if (!playing && step >= last && !loop) go(0);
		playing = !playing;
	}

	// keep t in sync when the parent sets `step`
	$effect(() => {
		const s = step;
		if (Math.floor(t + 1e-9) !== s) t = s;
	});

	$effect(() => {
		if (autoplay && !reduced) playing = true;
	});

	// playback loop
	$effect(() => {
		if (!playing || reduced) return;
		let raf = 0;
		let prevTime = performance.now();
		let acc = 0;
		const tick = (now: number) => {
			const dt = now - prevTime;
			prevTime = now;
			if (smooth) {
				let nt = t + (dt * speed) / interval;
				if (nt >= last) {
					if (loop) nt = 0;
					else {
						setT(last);
						playing = false;
						onend?.();
						return;
					}
				}
				setT(nt);
			} else {
				acc += dt * speed;
				if (acc >= interval) {
					acc = 0;
					next();
					if (!playing) {
						onend?.();
						return;
					}
				}
			}
			raf = requestAnimationFrame(tick);
		};
		raf = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(raf);
	});

	function onkeydown(e: KeyboardEvent) {
		// A view inside the stage (TensorGrid, TokenChips, ModelOverview, …) that handled the key
		// already called preventDefault: do not also step. Browser shortcuts (Alt+←, Ctrl+→) and
		// typing in a text field inside the stage are left alone.
		if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
		const el = e.target as HTMLElement;
		const tag = el.tagName;
		if (el.isContentEditable || tag === 'TEXTAREA' || tag === 'SELECT') return;
		if (tag === 'INPUT' && (el as HTMLInputElement).type !== 'range') return;
		if (tag === 'INPUT' && e.key.startsWith('Arrow')) return;
		if (e.key === ' ' || e.key === 'k') {
			if (tag === 'BUTTON' && e.key === ' ') return; // the focused button handles Space itself
			e.preventDefault();
			toggle();
		} else if (e.key === 'ArrowRight') {
			e.preventDefault();
			playing = false;
			next();
		} else if (e.key === 'ArrowLeft') {
			e.preventDefault();
			playing = false;
			prev();
		} else if (e.key === 'Home') {
			e.preventDefault();
			go(0);
		} else if (e.key === 'End') {
			e.preventDefault();
			go(last);
		}
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions (keyboard shortcuts for the controls inside) -->
<section class="player" aria-labelledby={label ? gid : undefined} {onkeydown}>
	{#if label}<h4 id={gid} class="ttl">{tr(label)}</h4>{/if}
	{#if children}<div class="stage">{@render children(step, t)}</div>{/if}
	<div class="controls">
		<!-- aria-disabled (not disabled): a focused button that becomes disabled drops keyboard focus to <body> -->
		<button type="button" class="btn icon" onclick={() => step !== 0 && go(0)} aria-label={ui('restart')} title={ui('restart')} aria-disabled={step === 0}>
			<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M3 3v10M13 3 6 8l7 5z" fill="currentColor" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round" /></svg>
		</button>
		<button type="button" class="btn icon" onclick={() => { playing = false; if (step !== 0) prev(); }} aria-label={ui('stepBack')} title={ui('stepBack')} aria-disabled={step === 0}>
			<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M10.5 3 5 8l5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
		</button>
		<button type="button" class="btn primary play" onclick={toggle} aria-label={reduced ? ui('stepForward') : playing ? ui('pause') : ui('play')}>
			{#if reduced}
				{ui('stepForward')}
			{:else if playing}
				<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><rect x="3.5" y="3" width="3" height="10" rx="1" fill="currentColor" /><rect x="9.5" y="3" width="3" height="10" rx="1" fill="currentColor" /></svg>
				<span>{ui('pause')}</span>
			{:else}
				<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M4.5 2.8v10.4L13 8z" fill="currentColor" /></svg>
				<span>{ui('play')}</span>
			{/if}
		</button>
		<button type="button" class="btn icon" onclick={() => { playing = false; if (step < last || loop) next(); }} aria-label={ui('stepForward')} title={ui('stepForward')} aria-disabled={step >= last && !loop}>
			<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M5.5 3 11 8l-5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
		</button>
		<input
			class="scrub"
			type="range"
			min="0"
			max={last}
			step={smooth && !reduced ? 0.01 : 1}
			value={t}
			aria-label={ui('scrub')}
			aria-valuetext={posText(step + 1, count)}
			oninput={(e) => {
				playing = false;
				setT(Number((e.currentTarget as HTMLInputElement).value));
			}}
		/>
		<span class="pos num" aria-live={playing ? "off" : "polite"}>{posText(step + 1, count)}</span>
		{#if !reduced && speeds.length > 1}
			<span class="segmented speeds" role="group" aria-label={ui('speed')}>
				{#each speeds as s (s)}
					<button type="button" aria-pressed={speed === s} onclick={() => (speed = s)}>{s}×</button>
				{/each}
			</span>
		{/if}
	</div>
	{#if labels && labels[step]}<p class="cap">{tr(labels[step])}</p>{/if}
	{#if labels && (reduced || showList)}
		<ol class="list">
			{#each labels as l, i (i)}
				<li class:cur={i === step}>
					<button type="button" onclick={() => go(i)} aria-current={i === step ? 'step' : undefined}>{tr(l)}</button>
				</li>
			{/each}
		</ol>
	{/if}
	{#if reduced}<p class="note">{ui('reducedMotionNote')}</p>{/if}
</section>

<style>
	.player {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 10px;
		min-width: 0;
	}
	.ttl {
		font-size: var(--fs-md);
		margin: 0;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.play {
		min-width: 96px;
	}
	.scrub {
		flex: 1 1 140px;
		min-width: 120px;
	}
	.pos {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		min-width: 6.5em;
	}
	.speeds button {
		min-height: 28px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.cap {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.list {
		margin: 0;
		padding-left: 1.6em;
		font-size: var(--fs-sm);
	}
	.list button {
		border: 0;
		background: none;
		vertical-align: top; /* keep the list number on the first line when a caption wraps */
		padding: 2px 4px;
		text-align: left;
		color: var(--ink-2);
		border-radius: 4px;
	}
	.list li.cur button {
		color: var(--ink);
		font-weight: 700;
		background: var(--accent-soft);
	}
	.note {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
</style>
