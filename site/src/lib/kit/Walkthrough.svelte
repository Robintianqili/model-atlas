<!--
  Walkthrough: a guided, step-by-step lesson. One focused visualization (the stage) at a time,
  with narration, Back / Next / Play, "Step k of N", a clickable step list and deep links.

  <Walkthrough steps={steps} bind:index ready={!!trace} label={{ en: 'TinyCNN lesson', zh: '…' }}>
    {#snippet stage(step, ctx)}{#if step.id === 'pixels'}…{:else if …}…{/if}{/snippet}
    {#snippet aside(ctx)}…your input, small…{/snippet}
  </Walkthrough>

  - steps: WalkStep[] (types.ts): { id, title, narration, look?, stage?, onEnter?, onLeave?, duration?,
    waitForDone?, linger? }. A step's own `stage` snippet wins over the shared `stage(step, ctx)` snippet.
    The stage is re-created on every step ({#key}), so components inside start fresh (autoplay on enter).
  - ctx (WalkContext): index, step, playing, visible, shown (latched "the stage has been in view since this
    step began": start animations on it; on a phone that is when the learner scrolls past the narration
    to the stage), reduced, done() (the step's animation finished).
  - Hooks fire only while `ready` is true (e.g. once the model has run); when `ready` turns true the current
    step's onEnter fires.
  - Layout: container query at 860 px. Wide: stage left (60 %), narration + step list right. Narrow:
    narration first (title, text, "Look at"), then the stage, then the aside and the step list, with
    progress dots; the control bar is sticky under the site header. On wide layouts the stage keeps the
    tallest height seen so far (reset when the width changes), so the page does not jump between steps.
  - Play-through: advances after each step's duration (default from the narration length, 6–16 s), or
    after the stage calls ctx.done() (+ linger) for waitForDone steps. Any pointer or key interaction
    inside the lesson (other than the Play button) pauses it; so does scrolling it off screen. Exception: a step with
    keepPlay that Play is waiting for (a question to answer, a task) takes presses in its stage without pausing, and
    its waitText replaces the "Playing this step's animation…" line. A touch
    that scrolls the page (pointercancel) does not pause; a tap or a drawing stroke does.
    keepPlay (opt-in, K1): until the step is done, presses and typing in the stage AND in the aside (where a task's input
    often lives) do not pause Play, also while a narrow layout still gives the narration its reading time; a waitText that
    returns '' falls back to the default line.
    prefers-reduced-motion: no auto-advance; the Play button is not shown (Next is the primary action).
  - Keyboard (focus inside the lesson, not typing, key not already handled by a view in the stage):
    ←/→ previous/next, Home/End first/last. Focus stays on the control that was used.
  - Deep link: #<id>-<step id> on load (and on hashchange) opens that step and scrolls the lesson into
    view; with `syncHash` the hash follows the current step (replaceState, no history entries). The plain
    #<id> anchor (e.g. the home page's lesson link) is re-aligned once `ready`, because content above
    the lesson can grow when the model loads.
  - Accessibility: the step title + narration are announced once per step in a polite live region (K1: a step with
    `pending` is announced once pending() is false, at most 10 s after entering, so the narration has its numbers);
    the step list is a list of buttons with aria-current="step".
-->
<script lang="ts">
	import { onMount, tick, untrack, type Snippet } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { motion } from './theme.svelte';
	import { i18n, ui, t, colon, type Text } from '$lib/i18n/i18n.svelte';
	import type { WalkContext, WalkStep } from './types';

	interface Props {
		steps: WalkStep[];
		/** current step (bindable) */
		index?: number;
		/** anchor prefix for deep links: #<id>-<step id> */
		id?: string;
		/** accessible name of the lesson */
		label?: Text;
		/** hooks (onEnter/onLeave) fire only when true, e.g. once the model has produced a trace */
		ready?: boolean;
		/** shared stage renderer (a step's own `stage` wins) */
		stage?: Snippet<[WalkStep, WalkContext]>;
		/** extra panel in the side column (e.g. a small copy of the input) */
		aside?: Snippet<[WalkContext]>;
		/** shown under the narration on the last step (e.g. "what next") */
		end?: Snippet;
		/** smallest stage height in px */
		stageMinHeight?: number;
		/** keep the URL hash on the current step (default true) */
		syncHash?: boolean;
		/** deprecated, ignored: the narration now always comes first on narrow layouts */
		narrationFirst?: boolean;
		onstep?: (index: number, step: WalkStep) => void;
	}
	let {
		steps,
		index = $bindable(0),
		id = 'lesson',
		label,
		ready = true,
		stage,
		aside,
		end,
		stageMinHeight = 300,
		syncHash = true,
		onstep
	}: Props = $props();

	const gid = $props.id();
	const n = $derived(steps.length);
	const last = $derived(Math.max(0, n - 1));
	const at = $derived(Math.max(0, Math.min(last, index)));
	const cur = $derived(steps[at]);
	const reduced = $derived(motion.reduced);

	let root: HTMLElement | undefined = $state();
	let stageIn: HTMLElement | undefined = $state();
	let asideIn: HTMLElement | undefined = $state();
	let playBtn: HTMLButtonElement | undefined = $state();
	let playing = $state(false);
	let visible = $state(false);
	/** the stage's top part is in view (its top is above the lower 30 % of the viewport) */
	let stageVis = $state(false);
	let shown = $state(false);
	let doneFor = $state(-1);
	/** 0..1: time used before Play moves on */
	let progress = $state(0);
	let waiting = $state(false);
	let announce = $state('');
	let wide = $state(true);
	let lockH = $state(0);
	let deepLinked: '' | 'step' | 'top' = '';

	type Val = Text | (() => Text) | undefined;
	const val = (v: Val): Text | undefined => (typeof v === 'function' ? v() : v);
	const say = (v: Val) => t(val(v));
	/** reading time from the English text: about 55 ms per character, 6–16 s */
	function readMs(st: WalkStep): number {
		const v = val(st.narration);
		const s = typeof v === 'string' ? v : (v?.en ?? '');
		return Math.max(6000, Math.min(16000, s.length * 55));
	}

	function markDone() {
		doneFor = untrack(() => at);
	}
	const ctx = $derived<WalkContext>({ index: at, step: cur, playing, visible, shown, reduced, done: markDone });

	// ------------------------------------------------------------ enter / leave hooks + announcement
	let entered = -1;
	/** the step whose announcement waits for its narration's numbers ("…" while a source computes; K1) */
	let announceFor = $state(-1);
	let announceBy = 0;
	const announceText = (i: number) => `${ui('stepOf', { i: i + 1, n })}${colon()} ${t(steps[i].title)}. ${say(steps[i].narration)}`;
	$effect(() => {
		const i = at;
		if (!ready) return;
		untrack(() => {
			if (i === entered) return;
			const first = entered < 0;
			if (!first) steps[entered]?.onLeave?.();
			entered = i;
			doneFor = -1;
			progress = 0;
			shown = stageVis;
			steps[i]?.onEnter?.();
			if (!first) {
				announceFor = i;
				announceBy = performance.now() + 10000;
			}
			onstep?.(i, steps[i]);
		});
	});
	$effect(() => {
		const i = announceFor;
		if (i < 0 || i !== at) return;
		const busy = steps[i]?.pending?.() ?? false; // tracked: the step's numbers arrive as its source computes
		const late = performance.now() >= announceBy;
		if (busy && !late) {
			const id = setTimeout(
				() => {
					if (announceFor !== i) return;
					announce = announceText(i);
					announceFor = -1;
				},
				Math.max(0, announceBy - performance.now())
			);
			return () => clearTimeout(id);
		}
		untrack(() => {
			announce = announceText(i);
			announceFor = -1;
		});
	});
	/** a step that waits for the learner (keepPlay) and is not done yet: presses and typing there are the learner's turn */
	const awaitingLearner = () => !!cur?.keepPlay && !!cur.waitForDone && doneFor !== at;
	const inLearnerArea = (n: Node | null) => !!n && ((!!stageIn && stageIn.contains(n)) || (!!asideIn && asideIn.contains(n)));
	$effect(() => {
		if (stageVis) shown = true;
	});

	// ------------------------------------------------------------ navigation
	function writeHash() {
		if (!syncHash) return;
		try {
			const u = new URL(location.href);
			u.hash = `${id}-${steps[at].id}`;
			replaceState(u.href, {});
		} catch {
			/* router not ready (or not a SvelteKit page): leave the URL alone */
		}
	}
	function bringIntoView() {
		if (!root) return;
		const top = root.getBoundingClientRect().top;
		const head = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 0;
		if (top < head - 2) root.scrollIntoView({ block: 'start', behavior: reduced ? 'auto' : 'smooth' });
	}
	function go(i: number) {
		const j = Math.max(0, Math.min(last, i));
		if (j === at) return;
		const focusInside = !!root && root.contains(document.activeElement);
		index = j;
		writeHash();
		bringIntoView();
		// if the focused element lived in the old stage, keep focus inside the lesson
		if (focusInside)
			tick().then(() => {
				const a = document.activeElement;
				if (root && (!a || a === document.body || !root.contains(a))) root.focus({ preventScroll: true });
			});
	}
	function prev() {
		playing = false;
		go(at - 1);
	}
	function next() {
		playing = false;
		go(at + 1);
	}
	function restart() {
		playing = false;
		go(0);
		root?.scrollIntoView({ block: 'start', behavior: reduced ? 'auto' : 'smooth' });
	}
	function togglePlay() {
		if (reduced) {
			go(at + 1);
			return;
		}
		if (playing) {
			playing = false;
			return;
		}
		if (at >= last) go(0);
		playing = true;
	}

	// ------------------------------------------------------------ play-through
	$effect(() => {
		if (!playing) {
			progress = 0;
			waiting = false;
			return;
		}
		if (reduced || !visible) {
			playing = false;
			return;
		}
		const i = at;
		const st = cur;
		const finished = doneFor === i;
		// narrow layouts show the narration above the stage: give it its reading time, then bring the
		// stage into view (its animation starts there) and only then count the step's own time
		const needStage = !wide && !shown;
		const wait = !needStage && !!st.waitForDone && !finished;
		const ms = needStage ? readMs(st) : wait ? (st.duration ?? 45000) : st.waitForDone ? (st.linger ?? 2500) : (st.duration ?? readMs(st));
		waiting = wait;
		progress = 0;
		const t0 = performance.now();
		let raf = requestAnimationFrame(function frame(now) {
			const p = Math.min(1, (now - t0) / ms);
			if (!wait) progress = p;
			if (p >= 1) {
				if (needStage) {
					showStage();
					return;
				}
				if (i >= last) playing = false;
				else go(i + 1);
				return;
			}
			raf = requestAnimationFrame(frame);
		});
		return () => cancelAnimationFrame(raf);
	});
	/** Play on a narrow layout: scroll so the stage sits just under the sticky control bar */
	function showStage() {
		const bar = root?.querySelector('.bar');
		if (stageIn && bar) {
			const dy = stageIn.getBoundingClientRect().top - bar.getBoundingClientRect().bottom - 8;
			if (dy > 0) window.scrollBy({ top: dy, behavior: 'smooth' });
		}
		shown = true;
	}

	/**
	 * Any interaction inside the lesson (except the Play button) takes control back from Play. A touch
	 * counts when it ends as a tap or a drag the page handles (e.g. drawing); a touch that the browser
	 * turns into page scrolling (pointercancel) does not, so a phone can scroll a tall stage while
	 * the lesson plays.
	 */
	let touchId: number | null = null;
	function onpointerdowncapture(e: PointerEvent) {
		if (!playing || (playBtn && playBtn.contains(e.target as Node))) return;
		// a step that waits for the learner (keepPlay): a press in the stage or the aside is the answer or the task, so Play
		// goes on (K1: the aside too, and also before Play started waiting, during a narrow layout's reading time)
		if (awaitingLearner() && inLearnerArea(e.target as Node)) return;
		if (e.pointerType === 'touch') touchId = e.pointerId;
		else playing = false;
	}
	function onpointerupcapture(e: PointerEvent) {
		if (touchId !== e.pointerId) return;
		touchId = null;
		playing = false;
	}
	function onpointercancelcapture(e: PointerEvent) {
		if (touchId === e.pointerId) touchId = null;
	}

	function onkeydown(e: KeyboardEvent) {
		const onPlay = !!playBtn && playBtn.contains(e.target as Node);
		const el = e.target as HTMLElement;
		const tag = el.tagName;
		const typing = el.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
		// Enter or Space on a control, or typing into a field, in the stage or the aside of a step that waits for the learner
		// (keepPlay) is the answer or the task: Play goes on (K1: the aside and typing too)
		const answering = awaitingLearner() && inLearnerArea(e.target as Node) && (typing || e.key === 'Enter' || e.key === ' ');
		if (playing && !onPlay && !answering && e.key !== 'Tab' && e.key !== 'Shift') playing = false;
		// a view in the stage that handled the key called preventDefault; browser shortcuts and typing are left alone
		if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
		if (el.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
		let j: number;
		if (e.key === 'ArrowRight') j = at + 1;
		else if (e.key === 'ArrowLeft') j = at - 1;
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = last;
		else return;
		e.preventDefault();
		playing = false;
		go(j);
	}

	// ------------------------------------------------------------ deep links, visibility, sizes
	/** 'step': #<id>-<step id> opened a step; 'top': the plain #<id> anchor (e.g. the home page's lesson link) */
	function fromHash(): '' | 'step' | 'top' {
		let h = '';
		try {
			h = decodeURIComponent(location.hash.slice(1));
		} catch {
			return '';
		}
		if (h === id) return 'top';
		if (!h.startsWith(`${id}-`)) return '';
		const i = steps.findIndex((s) => s.id === h.slice(id.length + 1));
		if (i < 0) return '';
		index = i;
		return 'step';
	}
	onMount(() => {
		const r = fromHash();
		if (r) {
			deepLinked = r;
			if (r === 'step') requestAnimationFrame(() => root?.scrollIntoView({ block: 'start' }));
		}
		const onhash = () => {
			if (fromHash() === 'step') root?.scrollIntoView({ block: 'start', behavior: motion.reduced ? 'auto' : 'smooth' });
		};
		addEventListener('hashchange', onhash);
		return () => removeEventListener('hashchange', onhash);
	});
	// content above the lesson can change height once the model is ready: re-align a deep link (a step
	// link aligns the lesson box; the plain #<id> anchor aligns the element with that id, e.g. the section)
	$effect(() => {
		if (ready && deepLinked) {
			const r = deepLinked;
			deepLinked = '';
			setTimeout(() => (r === 'top' ? (document.getElementById(id) ?? root) : root)?.scrollIntoView({ block: 'start' }), 80);
		}
	});
	$effect(() => {
		if (!root) return;
		const io = new IntersectionObserver((es) => (visible = es[es.length - 1].isIntersecting));
		io.observe(root);
		let lastW = 0;
		const ro = new ResizeObserver((es) => {
			const w = es[0].contentRect.width;
			wide = w >= 860;
			if (Math.abs(w - lastW) > 40) {
				lastW = w;
				lockH = 0;
			}
		});
		ro.observe(root);
		return () => {
			io.disconnect();
			ro.disconnect();
		};
	});
	$effect(() => {
		if (!stageIn) return;
		const ro = new ResizeObserver((es) => {
			const h = Math.ceil(es[0].contentRect.height);
			if (h > untrack(() => lockH)) lockH = h;
		});
		ro.observe(stageIn);
		// animations start once the stage itself is in view (narrow layouts put the narration above it)
		const io = new IntersectionObserver((es) => (stageVis = es[es.length - 1].isIntersecting), { rootMargin: '0px 0px -30% 0px' });
		io.observe(stageIn);
		return () => {
			ro.disconnect();
			io.disconnect();
		};
	});
	const minH = $derived(wide ? Math.max(stageMinHeight, lockH) : stageMinHeight);
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions (arrow-key shortcuts for the lesson controls inside) -->
<section
	class="wt"
	bind:this={root}
	tabindex="-1"
	aria-label={label ? t(label) : ui('secLesson')}
	aria-roledescription={t({ en: 'lesson', zh: '课程' })}
	{onkeydown}
	{onpointerdowncapture}
	{onpointerupcapture}
	{onpointercancelcapture}
>
	<div class="bar">
		<div class="ctl">
			<!-- aria-disabled (not disabled) keeps keyboard focus on the button at the ends -->
			<button type="button" class="btn" onclick={prev} aria-disabled={at === 0}>
				<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M10.5 3 5 8l5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
				{ui('lessonPrev')}
			</button>
			{#if !reduced}
				<button type="button" class="btn play" bind:this={playBtn} onclick={togglePlay} aria-pressed={playing}>
					{#if playing}
						<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><rect x="3.5" y="3" width="3" height="10" rx="1" fill="currentColor" /><rect x="9.5" y="3" width="3" height="10" rx="1" fill="currentColor" /></svg>
						{ui('lessonPause')}
					{:else}
						<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M4.5 2.8v10.4L13 8z" fill="currentColor" /></svg>
						{ui('lessonPlay')}
					{/if}
				</button>
			{/if}
			<button type="button" class="btn primary" onclick={next} aria-disabled={at >= last}>
				{ui('lessonNext')}
				<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M5.5 3 11 8l-5.5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
			</button>
		</div>
		<span class="pos num">{ui('stepOf', { i: at + 1, n })}</span>
		<span class="meter" aria-hidden="true">
			{#each steps as s, i (s.id)}<span class:on={i <= at}></span>{/each}
		</span>
		<button type="button" class="btn ghost restart" onclick={restart}>
			<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 8a5 5 0 1 0 1.6-3.7M3 2.5v3h3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg>
			{ui('lessonRestart')}
		</button>
		<ol class="dots" aria-label={ui('lessonProgress')}>
			{#each steps as s, i (s.id)}
				<li>
					<button
						type="button"
						class="dot"
						class:on={i === at}
						class:past={i < at}
						aria-current={i === at ? 'step' : undefined}
						aria-label={ui('lessonGoTo', { i: i + 1, title: t(s.title) })}
						onclick={() => {
							playing = false;
							go(i);
						}}><span></span></button
					>
				</li>
			{/each}
		</ol>
		{#if playing}
			<span class="timer" class:waiting style="--p: {progress}" aria-hidden="true"></span>
		{/if}
	</div>

	<!-- DOM order = reading order: what this step is (narration), the stage, then the rest. Narrow layouts
	     show them in that order, so a phone reads the explanation before a tall stage; wide layouts put the
	     stage in the left column next to the narration. -->
	<div class="body">
		<div class="narr">
			<p class="kicker num">{ui('stepOf', { i: at + 1, n })}</p>
			<h3 id="{gid}-t">{t(cur?.title)}</h3>
			<p class="say">{say(cur?.narration)}</p>
			{#if cur?.look}
				<p class="look"><span class="lk">{ui('lessonLook')}{colon()}</span>{i18n.lang === 'zh' ? '' : ' '}{say(cur.look)}</p>
			{/if}
			{#if playing && waiting}<p class="wait">{(cur?.waitText ? say(cur.waitText) : '') || ui('lessonWaiting')}</p>{/if}
		</div>

		<div class="stage" style="min-height: {minH}px">
			<div class="stage-in" bind:this={stageIn}>
				{#key at}
					<div class="fade">
						{#if cur?.stage}{@render cur.stage(ctx)}{:else if stage && cur}{@render stage(cur, ctx)}{/if}
					</div>
				{/key}
			</div>
		</div>

		{#if at === last && end}<div class="end">{@render end()}</div>{/if}

		<div class="side">
			{#if aside}<div class="aside" bind:this={asideIn}>{@render aside(ctx)}</div>{/if}
			<nav class="list-wide" aria-label={ui('lessonAllSteps')}>
				<ol class="list">
					{#each steps as s, i (s.id)}
						<li class:cur={i === at}>
							<button
								type="button"
								aria-current={i === at ? 'step' : undefined}
								onclick={() => {
									playing = false;
									go(i);
								}}><span class="i num">{i + 1}</span><span>{t(s.title)}</span></button
							>
						</li>
					{/each}
				</ol>
			</nav>
			<details class="list-narrow">
				<summary>{ui('lessonAllSteps')}</summary>
				<ol class="list">
					{#each steps as s, i (s.id)}
						<li class:cur={i === at}>
							<button
								type="button"
								aria-current={i === at ? 'step' : undefined}
								onclick={() => {
									playing = false;
									go(i);
								}}><span class="i num">{i + 1}</span><span>{t(s.title)}</span></button
							>
						</li>
					{/each}
				</ol>
			</details>
			<p class="keys">{reduced ? ui('lessonReduced') : ui('lessonKeys')}</p>
		</div>
	</div>
	<div class="visually-hidden" aria-live="polite" aria-atomic="true">{announce}</div>
</section>

<style>
	.wt {
		container: wt / inline-size;
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		border: 1px solid var(--rule-strong);
		border-radius: var(--r3);
		background: var(--surface);
		box-shadow: var(--shadow-1);
		outline: none;
		scroll-margin-top: calc(var(--header-h) + 8px);
	}
	.wt:focus-visible {
		outline: 3px solid var(--accent);
		outline-offset: 2px;
	}
	/* ------------------------------------------------ control bar (sticky under the site header) */
	.bar {
		position: sticky;
		top: var(--header-h);
		z-index: 6;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 12px;
		padding: 8px 10px;
		background: color-mix(in srgb, var(--surface) 94%, transparent);
		backdrop-filter: blur(8px);
		-webkit-backdrop-filter: blur(8px);
		border-bottom: 1px solid var(--rule);
		border-radius: var(--r3) var(--r3) 0 0;
	}
	.ctl {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.ctl .btn,
	.restart {
		gap: 6px;
	}
	.btn[aria-disabled='true'] {
		opacity: 0.45;
		cursor: default;
	}
	.play[aria-pressed='true'] {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
	.pos {
		font-size: var(--fs-sm);
		color: var(--ink-2);
		white-space: nowrap;
	}
	.restart {
		margin-left: auto;
		font-size: var(--fs-sm);
	}
	.meter {
		display: none;
		flex: 1 1 120px;
		max-width: 260px;
		gap: 3px;
	}
	.meter span {
		flex: 1;
		height: 5px;
		border-radius: 3px;
		background: var(--surface-3);
	}
	.meter span.on {
		background: var(--accent);
	}
	.dots {
		display: flex;
		flex-wrap: wrap;
		list-style: none;
		margin: 0 -4px;
		padding: 0;
		flex-basis: 100%;
	}
	.dot {
		display: grid;
		place-items: center;
		width: 28px;
		height: 26px;
		padding: 0;
		border: 0;
		background: none;
		border-radius: 50%;
		cursor: pointer;
	}
	.dot span {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--surface-3);
		border: 1.5px solid var(--rule-strong);
	}
	.dot.past span {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
	.dot.on span {
		width: 14px;
		height: 14px;
		background: var(--accent);
		border-color: var(--accent);
	}
	.timer {
		position: absolute;
		left: 0;
		bottom: -1px;
		height: 3px;
		width: calc(var(--p) * 100%);
		background: var(--accent);
		border-radius: 0 2px 2px 0;
	}
	.timer.waiting {
		width: 100%;
		background: repeating-linear-gradient(90deg, var(--accent) 0 10px, transparent 10px 20px);
		opacity: 0.6;
	}
	/* ------------------------------------------------ body */
	.body {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: 16px;
		padding: 14px 12px 16px;
	}
	.stage,
	.stage-in {
		min-width: 0;
	}
	.fade {
		animation: wt-in 200ms ease-out;
	}
	@keyframes wt-in {
		from {
			opacity: 0;
			transform: translateY(4px);
		}
	}
	.side {
		display: grid;
		gap: 14px;
		align-content: start;
		min-width: 0;
	}
	.narr {
		display: grid;
		gap: 6px;
	}
	.kicker {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--accent);
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}
	h3 {
		margin: 0;
		font-size: var(--fs-xl);
	}
	.say {
		margin: 0;
		color: var(--ink);
		max-width: var(--measure);
	}
	.look {
		margin: 0;
		padding: 6px 10px;
		border-left: 3px solid var(--accent);
		background: var(--accent-soft);
		border-radius: 0 var(--r1) var(--r1) 0;
		font-size: var(--fs-sm);
	}
	.lk {
		font-weight: 700;
	}
	.wait {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.end {
		margin-top: -4px;
	}
	.aside {
		min-width: 0;
	}
	.list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 1px;
		font-size: var(--fs-sm);
	}
	.list button {
		display: flex;
		gap: 8px;
		align-items: baseline;
		width: 100%;
		padding: 4px 8px;
		border: 0;
		border-radius: var(--r1);
		background: none;
		color: var(--ink-2);
		text-align: left;
		cursor: pointer;
	}
	.list button:hover {
		background: var(--surface-2);
	}
	.list li.cur button {
		background: var(--accent-soft);
		color: var(--ink);
		font-weight: 700;
	}
	.list .i {
		flex: 0 0 1.6em;
		color: var(--muted);
		font-size: var(--fs-xs);
	}
	.list li.cur .i {
		color: var(--accent);
	}
	.list-wide {
		display: none;
	}
	.list-narrow summary {
		cursor: pointer;
		font-size: var(--fs-sm);
		color: var(--ink-2);
		padding: 4px 0;
	}
	.keys {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	/* ------------------------------------------------ wide: stage left (60 %), narration right */
	@container wt (min-width: 860px) {
		.bar {
			padding: 10px 16px;
		}
		.dots {
			display: none;
		}
		.meter {
			display: flex;
		}
		.body {
			grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
			grid-template-rows: auto auto 1fr;
			column-gap: 24px;
			row-gap: 0;
			padding: 18px 16px 20px;
		}
		.stage {
			grid-column: 1;
			grid-row: 1 / span 3;
		}
		.narr,
		.end,
		.side {
			grid-column: 2;
			border-left: 1px solid var(--rule);
			padding-left: 20px;
		}
		.narr {
			grid-row: 1;
			padding-bottom: 14px;
		}
		.end {
			grid-row: 2;
			margin-top: 0;
			padding-bottom: 14px;
		}
		.side {
			grid-row: 3;
		}
		.list-wide {
			display: block;
		}
		.list-narrow {
			display: none;
		}
	}
	@container wt (max-width: 420px) {
		.ctl .btn,
		.restart {
			padding-inline: 10px;
		}
	}
</style>
