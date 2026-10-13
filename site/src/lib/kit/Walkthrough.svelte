<!--
  Walkthrough: a guided, step-by-step lesson, laid out as the lesson app (redesign spec §5): one step fills one screen.
  The stage (the step's one visualization, with the input chip and the provenance label in its header) takes the width;
  the narration sits beside it (eyebrow, title, say, Look line, the learner's beat, Why); the action bar holds Back, Play,
  the chapter rail, the step list and the two-line Next.

  <Walkthrough steps={steps} bind:index ready={!!trace} label={{ en: 'TinyCNN lesson', zh: '…' }}>
    {#snippet stage(step, ctx)}{#if step.id === 'pixels'}…{:else if …}…{/if}{/snippet}
  </Walkthrough>

  - steps: WalkStep[] (types.ts). A step's own `stage` snippet wins over the shared `stage(step, ctx)` snippet. The stage is
    re-created on every step ({#key}), so components inside start fresh. R1: chapter, why, prov (the stage header's
    provenance label), predict (answered in the narration; the stage is gated until then), task (a "Your turn" beat).
  - Inside a lesson page shell (ModelPageShell / ConceptPageShell set the lesson-app context, kit/lesson/app.ts) it is the
    page: the top bar with the view tabs, the cover (step 0: answer first; first visit only, then the saved step with a
    "Resumed" toast), app mode at 960 x 600 and up (one screen, the stage and narration panes scroll on their own;
    data-lesson-mode="app") and flow mode below (the page scrolls, the bar sticks to the bottom; phones put the picture
    first and fix the bar). Without the context it is a card in the page (the DevKit tutorial, the kit gallery).
  - ctx (WalkContext): index, step, playing, visible, shown (latched "the stage has been in view since this step began"),
    reduced, done(), answered (this step's predict), gated, density (roomy | regular | compact, from the stage pane).
  - Hooks fire only while `ready` is true; when `ready` turns true the current step's onEnter fires.
  - Play-through: advances after each step's duration (default from the narration length, 6–16 s), or after the stage
    calls ctx.done() (+ linger) for waitForDone steps. Any pointer or key interaction inside the lesson (other than Play)
    pauses it, except the learner's answer or task on a keepPlay step. Reduced motion: no autoplay, no Play button.
  - Keys (focus not in a field, the key not handled by a view): ← → steps, Home / End, Enter the primary (Next, Start,
    Continue), 1-4 an option, T "Show as text"; Esc closes the step list and the sheets (native dialogs).
  - Deep links: #<id>-<step id> opens that step; with `syncHash` the hash follows the step (replaceState, no history).
  - Accessibility: the step title + narration are announced once per step in a polite live region (K1 `pending` rule), and
    the predict feedback too; the rail is a nav of buttons with aria-current="step".
-->
<script lang="ts">
	import { onMount, tick, untrack, type Snippet } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { motion } from './theme.svelte';
	import { lui, t, type Text } from '$lib/i18n/i18n.svelte';
	import Icon from './Icon.svelte';
	import { eye, left, list as listIcon, pause, play as playIcon } from './icons';
	import ProvenanceBadge from './ProvenanceBadge.svelte';
	import { keyScroll } from './keyscroll';
	import { getLessonApp } from './lesson/app';
	import { chapterGroups, densityOf } from './lesson/views';
	import { readSaved, writeSaved, clearSaved } from './lesson/resume';
	import LessonTop from './lesson/LessonTop.svelte';
	import NextButton from './lesson/NextButton.svelte';
	import Beat from './lesson/Beat.svelte';
	import FeedbackBar from './lesson/FeedbackBar.svelte';
	import Cover from './lesson/Cover.svelte';
	import ChipText from './lesson/ChipText.svelte';
	import type { Density, StageProv, WalkContext, WalkPredict, WalkStep } from './types';

	interface Props {
		steps: WalkStep[];
		/** current step (bindable) */
		index?: number;
		/** anchor prefix for deep links: #<id>-<step id> */
		id?: string;
		/** accessible name of the lesson (and the cover's title) */
		label?: Text;
		/** hooks (onEnter/onLeave) fire only when true, e.g. once the model has produced a trace */
		ready?: boolean;
		/** shared stage renderer (a step's own `stage` wins) */
		stage?: Snippet<[WalkStep, WalkContext]>;
		/** R1: rendered in the stage header (the input chip's place) */
		aside?: Snippet<[WalkContext]>;
		/** shown in the end card (after Finish on the last step) */
		end?: Snippet;
		/** smallest stage height in px (the card layout only; ignored in app mode) */
		stageMinHeight?: number;
		/** keep the URL hash on the current step (default true) */
		syncHash?: boolean;
		/** deprecated, ignored */
		narrationFirst?: boolean;
		onstep?: (index: number, step: WalkStep) => void;
		/** R1: chapter groups of the steps, in order */
		chapters?: { title: Text; count: number }[];
		/** R1: the cover's stage (step 0, answer first); without it the cover shows the first step's stage */
		cover?: Snippet<[WalkContext]>;
		/** R1: the title on the cover's Start button (default: step 1's title) */
		coverNext?: Text;
		/** R1: the end card's recap (default: the chapter titles) */
		recap?: Snippet;
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
		onstep,
		chapters,
		cover,
		coverNext,
		recap
	}: Props = $props();

	const gid = $props.id();
	const app = getLessonApp();
	/** the lesson is the page (a lesson route's shell), not a card inside one */
	const page = !!app && !app.embedded;
	const n = $derived(steps.length);
	const last = $derived(Math.max(0, n - 1));
	const at = $derived(Math.max(0, Math.min(last, index)));
	const cur = $derived(steps[at]);
	const reduced = $derived(motion.reduced);
	const active = $derived(!app || app.nav.view === 'lesson');

	let root: HTMLElement | undefined = $state();
	let stageEl: HTMLElement | undefined = $state();
	let narrEl: HTMLElement | undefined = $state();
	let stageIn: HTMLElement | undefined = $state();
	let asideIn: HTMLElement | undefined = $state();
	let beatEl: HTMLElement | undefined = $state();
	let whyEl: HTMLDetailsElement | undefined = $state();
	let playBtn: HTMLButtonElement | undefined = $state();
	let listBtn: HTMLButtonElement | undefined = $state();
	let listBtnPh: HTMLButtonElement | undefined = $state();
	let listAnchor: HTMLElement | null = $state(null);
	let listOpen = $state(false);
	/** the step list and the end card load when first needed (they are not part of the first screen) */
	let listUsed = $state(false);
	const loadList = () => import('./lesson/StepList.svelte');
	const loadEnd = () => import('./lesson/EndCard.svelte');
	let playing = $state(false);
	let visible = $state(false);
	let stageVis = $state(false);
	let shown = $state(false);
	let doneFor = $state(-1);
	let progress = $state(0);
	let waiting = $state(false);
	let announce = $state('');
	let mode = $state<'app' | 'flow' | undefined>(undefined);
	let density = $state<Density>('regular');
	/** the cover (step 0) shows; decided on mount (a deep link or a saved step skips it) */
	const coverStage = $derived(cover ?? app?.cover);
	const hasCover = $derived(!!app && (!!coverStage || app.goals.length > 0 || !!app.prequestion || !!app.summary));
	let covered = $state(false);
	let coverDecided = false;
	let finished = $state(false);
	/** per step: the chosen predict option (-1: revealed with Show me) */
	let answers = $state<Record<number, number>>({});
	/** the step whose feedback bar is up */
	let fbFor = $state(-1);
	let toast = $state<{ k: number } | null>(null);
	let resumeAt = $state<number | undefined>(undefined);

	type Val = Text | (() => Text) | undefined;
	const val = (v: Val): Text | undefined => (typeof v === 'function' ? v() : v);
	const say = (v: Val) => t(val(v));
	/** reading time from the English text: about 55 ms per character, 6–16 s */
	function readMs(st: WalkStep): number {
		const v = val(st.narration);
		const s = typeof v === 'string' ? v : (v?.en ?? '');
		return Math.max(6000, Math.min(16000, s.length * 55));
	}
	const predictOf = (st: WalkStep | undefined): WalkPredict | undefined => (typeof st?.predict === 'function' ? st.predict() : st?.predict);
	const provOf = (st: WalkStep | undefined): StageProv | undefined => (typeof st?.prov === 'function' ? st.prov() : st?.prov);

	const pq = $derived(covered || finished ? undefined : predictOf(cur));
	const answered = $derived(answers[at] ?? null);
	const gated = $derived(!!pq && pq.gate !== false && answered === null);
	const prov = $derived.by(() => {
		const p = provOf(covered ? steps[0] : cur);
		return p === undefined ? undefined : typeof p === 'string' ? { mode: p } : p;
	});
	const taskDone = $derived.by(() => {
		if (!cur?.task || covered) return false;
		try {
			return cur.task.done();
		} catch {
			return false;
		}
	});

	function markDone() {
		doneFor = untrack(() => at);
	}
	const ctx = $derived<WalkContext>({ index: at, step: cur, playing, visible, shown, reduced, done: markDone, answered, gated, density });

	// ------------------------------------------------------------ chapters
	const groups = $derived(chapterGroups<Text>(n, { chapters, stepChapter: (i) => steps[i]?.chapter }));
	const groupOf = (i: number) => groups.find((g) => i >= g.from && i <= g.to);
	const chapterAt = $derived(groupOf(at)?.title);
	const titles = $derived(steps.map((s) => s.title));
	const animated = $derived(steps.map((s) => !!s.animated));

	// tell the shell where the learner is (its top bar and the "Back to the lesson" pill in the other views)
	$effect(() => {
		if (!app) return;
		const p = { at, n, cover: covered, title: cur?.title, chapter: chapterAt, id };
		untrack(() => Object.assign(app.pos, p));
	});

	// ------------------------------------------------------------ enter / leave hooks + announcement
	let entered = -1;
	let announceFor = $state(-1);
	let announceBy = 0;
	const announceText = (i: number) => `${lui('stepOf', { k: i + 1, n })}: ${t(steps[i].title)}. ${say(steps[i].narration)}`;
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
			if (!first && !covered) {
				announceFor = i;
				announceBy = performance.now() + 10000;
			}
			onstep?.(i, steps[i]);
		});
	});
	$effect(() => {
		const i = announceFor;
		if (i < 0 || i !== at) return;
		const busy = steps[i]?.pending?.() ?? false;
		const late = performance.now() >= announceBy;
		if (busy && !late) {
			const tid = setTimeout(
				() => {
					if (announceFor !== i) return;
					announce = announceText(i);
					announceFor = -1;
				},
				Math.max(0, announceBy - performance.now())
			);
			return () => clearTimeout(tid);
		}
		untrack(() => {
			announce = announceText(i);
			announceFor = -1;
		});
	});
	const awaitingLearner = () => !!cur?.keepPlay && !!cur.waitForDone && doneFor !== at;
	const inLearnerArea = (x: Node | null) => !!x && [stageIn, asideIn, beatEl].some((el) => !!el && el.contains(x));
	$effect(() => {
		if (stageVis) shown = true;
	});

	// ------------------------------------------------------------ navigation
	function writeHash() {
		if (!syncHash || covered || !active) return;
		try {
			const u = new URL(location.href);
			u.hash = `${id}-${steps[at].id}`;
			if (u.href !== location.href) replaceState(u.href, {});
		} catch {
			/* router not ready (or not a SvelteKit page): leave the URL alone */
		}
	}
	function save() {
		if (app && !covered && steps[at]) writeSaved(app.slug, steps[at].id);
	}
	function bringIntoView() {
		if (mode === 'app') {
			stageEl?.scrollTo({ top: 0 });
			narrEl?.scrollTo({ top: 0 });
			return;
		}
		if (!root) return;
		const top = root.getBoundingClientRect().top;
		const head = page ? 0 : parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 0;
		if (top < head - 2) root.scrollIntoView({ block: 'start', behavior: reduced ? 'auto' : 'smooth' });
	}
	function go(i: number) {
		const j = Math.max(0, Math.min(last, i));
		const wasCover = covered || finished;
		covered = false;
		finished = false;
		listOpen = false;
		if (j === at && !wasCover) return;
		const focusInside = !!root && root.contains(document.activeElement);
		index = j;
		tick().then(() => {
			writeHash();
			save();
		});
		bringIntoView();
		if (focusInside)
			tick().then(() => {
				const a = document.activeElement;
				if (root && (!a || a === document.body || !root.contains(a))) root.focus({ preventScroll: true });
			});
	}
	function prev() {
		playing = false;
		if (finished) {
			finished = false;
			return;
		}
		if (at === 0 && hasCover && !covered) {
			covered = true;
			bringIntoView();
			return;
		}
		go(at - 1);
	}
	function next() {
		playing = false;
		go(at + 1);
	}
	function start() {
		playing = false;
		toast = null;
		go(resumeAt ?? 0);
		resumeAt = undefined;
	}
	function finish() {
		playing = false;
		finished = true;
		// app mode: the panes start at the top; flow (phones): the end card is in the narration, under the stage
		if (mode === 'app') bringIntoView();
		else
			loadEnd()
				.then(() => tick())
				.then(() => requestAnimationFrame(() => narrEl?.scrollIntoView({ block: 'start' })));
		announce = t({ en: 'You finished the lesson.', zh: '你学完了这节课。' });
	}
	function restart() {
		playing = false;
		answers = {};
		fbFor = -1;
		toast = null;
		resumeAt = undefined;
		if (app) clearSaved(app.slug);
		go(0);
		bringIntoView();
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
		if (covered) start();
		else if (at >= last) go(0);
		playing = true;
	}
	function answer(i: number) {
		if (!pq || answers[at] !== undefined) return;
		answers = { ...answers, [at]: i };
		fbFor = at;
		announce = `${i === pq.answer ? lui('right') : lui('notQuite')} ${t(pq.explain)}`;
	}
	function showMe() {
		if (!pq) return;
		answers = { ...answers, [at]: -1 };
	}
	function openWhy() {
		if (!whyEl) return;
		whyEl.open = true;
		whyEl.scrollIntoView({ block: 'nearest', behavior: reduced ? 'auto' : 'smooth' });
	}
	function openList(phone: boolean) {
		listAnchor = phone ? (listBtnPh ?? null) : (listBtn ?? null);
		listUsed = true;
		listOpen = !listOpen;
	}

	/** the primary action: what Next does right now */
	const primary = $derived.by((): { kind: 'next' | 'start' | 'continue' | 'skip' | 'finish' | 'resume'; k: number; title: string; run: () => void } => {
		if (covered)
			return resumeAt !== undefined
				? { kind: 'resume', k: resumeAt + 1, title: t(steps[resumeAt]?.title), run: start }
				: { kind: 'start', k: 1, title: t(coverNext ?? steps[0]?.title), run: start };
		if (finished) return { kind: 'start', k: 1, title: t(steps[0]?.title), run: restart };
		const nextTitle = at < last ? t(steps[at + 1]?.title) : t({ en: 'See what you learned', zh: '看看你学到了什么' });
		if (gated) return at < last ? { kind: 'skip', k: at + 2, title: nextTitle, run: next } : { kind: 'skip', k: at + 1, title: nextTitle, run: finish };
		if (at >= last) return { kind: 'finish', k: at + 1, title: nextTitle, run: finish };
		if (fbFor === at) return { kind: 'continue', k: at + 2, title: nextTitle, run: next };
		return { kind: 'next', k: at + 2, title: nextTitle, run: next };
	});
	const feedback = $derived(!covered && !finished && fbFor === at && pq && answered !== null && answered >= 0 ? { right: answered === pq.answer, explain: pq.explain } : null);

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
		const finishedStep = doneFor === i;
		const wait = !!st.waitForDone && !finishedStep;
		const ms = wait ? (st.duration ?? 45000) : st.waitForDone ? (st.linger ?? 2500) : (st.duration ?? readMs(st));
		waiting = wait;
		progress = 0;
		const t0 = performance.now();
		let raf = requestAnimationFrame(function frame(now) {
			const p = Math.min(1, (now - t0) / ms);
			if (!wait) progress = p;
			if (p >= 1) {
				if (i >= last) playing = false;
				else go(i + 1);
				return;
			}
			raf = requestAnimationFrame(frame);
		});
		return () => cancelAnimationFrame(raf);
	});

	let touchId: number | null = null;
	function onpointerdowncapture(e: PointerEvent) {
		if (!playing || (playBtn && playBtn.contains(e.target as Node))) return;
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

	const typingIn = (el: HTMLElement) => el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName);
	function onkey(e: KeyboardEvent) {
		if (!active) return;
		const el = e.target as HTMLElement;
		// keys inside an open dialog (the step list, a sheet) belong to it
		if (el?.closest?.('dialog') || document.querySelector('dialog[open]')) return;
		const onPlay = !!playBtn && playBtn.contains(el);
		const answering = awaitingLearner() && inLearnerArea(el) && (typingIn(el) || e.key === 'Enter' || e.key === ' ');
		if (playing && !onPlay && !answering && e.key !== 'Tab' && e.key !== 'Shift') playing = false;
		if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
		if (typingIn(el)) return;
		const onControl = !!el?.closest?.('button, a[href], summary, [role="slider"], [role="radio"], [tabindex]:not([tabindex="-1"])');
		let j: number;
		if (e.key === 'ArrowRight') j = covered ? 0 : at + 1;
		else if (e.key === 'ArrowLeft') {
			e.preventDefault();
			prev();
			return;
		} else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = last;
		else if (e.key === 'Enter' && !onControl && primary.kind !== 'skip') {
			e.preventDefault();
			primary.run();
			return;
		} else if (/^[1-4]$/.test(e.key)) {
			const k = Number(e.key) - 1;
			if (pq && answered === null && k < pq.options.length) {
				e.preventDefault();
				answer(k);
			}
			return;
		} else if ((e.key === 't' || e.key === 'T') && !onControl) {
			const d = stageIn?.querySelector<HTMLDetailsElement>('details.as-text');
			if (d) {
				e.preventDefault();
				d.open = !d.open;
			}
			return;
		} else return;
		e.preventDefault();
		playing = false;
		go(j);
	}

	// ------------------------------------------------------------ deep links, resume, visibility, sizes
	function fromHash(): '' | 'step' | 'top' | 'cover' | 'other' {
		let h = '';
		try {
			h = decodeURIComponent(location.hash.slice(1));
		} catch {
			return '';
		}
		if (!h) return '';
		if (h === id) return 'top';
		if (h === 'goals') return 'cover';
		if (!h.startsWith(`${id}-`)) return 'other';
		const i = steps.findIndex((s) => s.id === h.slice(id.length + 1));
		if (i < 0) return 'other';
		index = i;
		return 'step';
	}
	onMount(() => {
		const r = fromHash();
		if (app && !coverDecided) {
			coverDecided = true;
			const saved = readSaved(app.slug);
			const si = saved ? steps.findIndex((s) => s.id === saved.step) : -1;
			if (r === 'step') covered = false;
			else if (r === 'cover') {
				covered = hasCover;
				if (si > 0) resumeAt = si;
			} else if (si >= 0 && saved?.started) {
				index = si;
				covered = false;
				if (r === '' && si > 0) toast = { k: si + 1 };
			} else covered = hasCover && (r === '' || r === 'other');
		}
		if (r === 'step' && !page) requestAnimationFrame(() => root?.scrollIntoView({ block: 'start' }));
		const onhash = () => {
			const k = fromHash();
			if (k === 'step') {
				covered = false;
				finished = false;
				if (!page) root?.scrollIntoView({ block: 'start', behavior: motion.reduced ? 'auto' : 'smooth' });
			} else if (k === 'cover' && hasCover) covered = true;
		};
		addEventListener('hashchange', onhash);
		return () => removeEventListener('hashchange', onhash);
	});
	// back in the Lesson view: the hash names the step again
	$effect(() => {
		if (active && app) untrack(() => tick().then(writeHash));
	});
	$effect(() => {
		if (!toast) return;
		const tid = setTimeout(() => (toast = null), 4000);
		return () => clearTimeout(tid);
	});
	$effect(() => {
		if (!page) {
			mode = undefined;
			return;
		}
		const mq = matchMedia('(min-width: 960px) and (min-height: 600px)');
		const f = () => (mode = mq.matches ? 'app' : 'flow');
		f();
		mq.addEventListener('change', f);
		return () => mq.removeEventListener('change', f);
	});
	$effect(() => {
		if (!root) return;
		const io = new IntersectionObserver((es) => (visible = es[es.length - 1].isIntersecting));
		io.observe(root);
		return () => io.disconnect();
	});
	$effect(() => {
		const el = stageEl;
		if (!el) return;
		const ro = new ResizeObserver((es) => {
			const r = es[0].contentRect;
			const phone = innerWidth < 760;
			density = densityOf(r.width, untrack(() => mode) === 'app' ? r.height : innerHeight, phone);
		});
		ro.observe(el);
		return () => ro.disconnect();
	});
	$effect(() => {
		if (!stageIn) return;
		const io = new IntersectionObserver((es) => (stageVis = es[es.length - 1].isIntersecting), { rootMargin: '0px 0px -20% 0px' });
		io.observe(stageIn);
		return () => io.disconnect();
	});
	// a new step: the feedback bar of the old one goes
	$effect(() => {
		void at;
		untrack(() => {
			if (fbFor !== at) fbFor = -1;
		});
	});
	const eyebrow = $derived(chapterAt ? lui('stepOfIn', { chapter: t(chapterAt), k: at + 1, n }) : lui('stepOf', { k: at + 1, n }));
	const segW = $derived(`clamp(14px, calc(min(476px, 36vw) / ${Math.max(1, n)} - 4px), 28px)`);
	const right = $derived(Object.entries(answers).filter(([i, v]) => v >= 0 && v === predictOf(steps[Number(i)])?.answer).length);
	const asked = $derived(Object.values(answers).filter((v) => v >= 0).length);
</script>

<svelte:window onkeydown={page ? onkey : undefined} />

<!-- svelte-ignore a11y_no_noninteractive_element_interactions (arrow-key shortcuts for the lesson controls inside) -->
<section
	class="wt"
	class:page
	class:card={!page}
	class:covered
	bind:this={root}
	tabindex="-1"
	hidden={!active || undefined}
	aria-label={page ? undefined : label ? t(label) : lui('viewLesson')}
	data-lesson-mode={page && active ? mode : undefined}
	data-density={density}
	onkeydown={page ? undefined : onkey}
	{onpointerdowncapture}
	{onpointerupcapture}
	{onpointercancelcapture}
>
	<!-- where the learner is, first in the lesson (atlas e2e reads the first .pos; the visible one is the phone's steps button) -->
	<span class="pos" hidden aria-hidden="true">{covered ? n : `${at + 1}/${n}`}</span>
	{#if app}<LessonTop {app} current="lesson" {groups} />{/if}
	<svelte:element this={page ? 'main' : 'div'} class="main" id={page ? 'main' : undefined} tabindex={page ? -1 : undefined}>
		<section class="stage" bind:this={stageEl} use:keyScroll aria-label={lui('stage')} style:min-height={page ? undefined : `${stageMinHeight}px`}>
			<div class="stage-h">
				<div class="chip-slot" bind:this={asideIn}>
					{#if app?.inputChip}{@render app.inputChip(() => (app.input.open = true))}{:else if aside}{@render aside(ctx)}{/if}
				</div>
				{#if prov}
					<span class="prov">
						<ProvenanceBadge mode={prov.mode} precision={prov.precision} tolerance={prov.tolerance} against={prov.against} note={prov.note} compact />
						{#if prov.short}<span class="pl short">{t(prov.short)}</span>{:else}<span class="pl">· {prov.mode === 'live' ? t({ en: 'computed on your input', zh: '用你的输入现场计算' }) : prov.mode === 'precomputed' ? t({ en: 'recorded from a real run', zh: '来自一次真实运行的记录' }) : t({ en: 'a drawing, not model output', zh: '示意图，不是模型输出' })}</span>{/if}
					</span>
				{/if}
			</div>
			<div class="stage-b" class:gated>
				<div class="gate" bind:this={stageIn} inert={gated || undefined}>
					{#if covered}
						{#if coverStage}{@render coverStage(ctx)}{:else if cur?.stage}{@render cur.stage(ctx)}{:else if stage && cur}{@render stage(cur, ctx)}{/if}
					{:else}
						{#key at}
							<div class="fade">
								{#if cur?.stage}{@render cur.stage(ctx)}{:else if stage && cur}{@render stage(cur, ctx)}{/if}
							</div>
						{/key}
					{/if}
				</div>
				{#if gated}
					<div class="veil">
						<div class="veil-c">
							<p><b>{lui('answerFirst')}</b> {lui('answerFirstBody')}</p>
							<button type="button" class="btn" onclick={showMe}><Icon d={eye} size={16} />{lui('showMe')}</button>
						</div>
					</div>
				{/if}
			</div>
		</section>

		<section class="narr" bind:this={narrEl} use:keyScroll aria-labelledby="{gid}-t">
			{#if covered}
				<span id="{gid}-t" class="visually-hidden">{t(label ?? { en: 'Lesson', zh: '课程' })}</span>
				<Cover title={label ?? cur?.title ?? ''} summary={app?.summary} goals={app?.goals} prequestion={app?.prequestion} {n} minutes={app?.minutes} {resumeAt} />
			{:else if finished}
				<h2 class="ttl" id="{gid}-t">{t({ en: 'What you learned', zh: '你学到了什么' })}</h2>
				{#await loadEnd() then m}
					<m.default {recap} lines={groups.filter((g) => g.title).map((g) => g.title!)} {right} {asked} prequestion={app?.prequestion} upNext={app?.upNext} tabs={app?.tabs} {end} />
				{/await}
			{:else}
				<p class="eyebrow ew">{eyebrow}</p>
				<svelte:element this={app ? 'h1' : 'h3'} class="ttl" id="{gid}-t">{t(cur?.title)}</svelte:element>
				<p class="say">{#if cur?.chips?.narration}<ChipText text={say(cur.chips.narration)} onref={cur.onref} />{:else}{say(cur?.narration)}{/if}</p>
				{#if cur?.look && !gated}
					<p class="look"><Icon d={eye} size={18} class="li" /><span><span class="visually-hidden">{`${lui('look')}: `}</span>{#if cur.chips?.look}<ChipText text={say(cur.chips.look)} onref={cur.onref} />{:else}{say(cur.look)}{/if}</span></p>
				{/if}
				{#if pq || cur?.task}
					<div bind:this={beatEl}>
						<Beat predict={pq} task={pq ? undefined : cur?.task} {answered} onanswer={answer} inline={answered !== null && !feedback} {taskDone} />
					</div>
				{/if}
				{#if cur?.why}
					<details class="why" bind:this={whyEl}>
						<summary>{t(cur.why.q)}</summary>
						<p>{say(cur.why.a)}</p>
					</details>
				{/if}
				{#if playing && waiting}<p class="wait">{(cur?.waitText ? say(cur.waitText) : '') || t({ en: 'Playing this step…', zh: '正在播放这一步……' })}</p>{/if}
				{#if !app && at === last && end}<div class="end">{@render end()}</div>{/if}
			{/if}
		</section>
	</svelte:element>

	{#snippet primaryBtn()}
		<NextButton kind={primary.kind} k={primary.k} title={primary.title} onclick={primary.run} />
	{/snippet}
	<svelte:element this={page ? 'footer' : 'div'} class="bar" class:has-fb={!!feedback} aria-label={lui('controls')} role={page ? undefined : 'group'}>
		{#if feedback}
			<div class="fbw">
				<FeedbackBar right={feedback.right} explain={feedback.explain} onwhy={cur?.why ? openWhy : undefined} next={primaryBtn} />
			</div>
		{/if}
		<div class="abar">
			<div class="l">
				<button type="button" class="btn back" onclick={prev} aria-disabled={(covered || (at === 0 && !hasCover)) && !finished ? true : undefined} aria-label={lui('back')}>
					<Icon d={left} size={18} /><span class="dt">{lui('back')}</span>
				</button>
				{#if !reduced}
					<button type="button" class="btn btn-quiet play dt" bind:this={playBtn} onclick={togglePlay} aria-pressed={playing} aria-label={playing ? lui('pause') : lui('play')}>
						<Icon d={playing ? pause : playIcon} size={16} /><span class="play-lbl">{playing ? lui('pause') : lui('play')}</span>
					</button>
				{/if}
			</div>
			<div class="railx">
				<nav class="rail" aria-label={lui('steps')}>
					<ol class="dots" style:--seg-w={segW}>
						{#each groups as g, gi (gi)}
							<li class="ch" class:cur={!covered && at >= g.from && at <= g.to}>
								{#if g.title}<span class="ch-n" aria-hidden="true">{t(g.title)}</span>{/if}
								<span class="segs">
									{#each { length: g.to - g.from + 1 } as _, j (j)}
										{@const k = g.from + j}
										<button
											type="button"
											class="dot"
											class:done={!covered && k < at}
											class:now={!covered && k === at}
											aria-current={!covered && k === at ? 'step' : undefined}
											aria-label={lui('goToStep', { k: k + 1, title: t(steps[k].title) })}
											onclick={() => {
												playing = false;
												go(k);
											}}
										></button>
									{/each}
								</span>
							</li>
						{/each}
					</ol>
				</nav>
				<button type="button" class="icon-btn list-btn" bind:this={listBtn} aria-label={t({ en: 'All steps', zh: '全部步骤' })} aria-haspopup="dialog" aria-expanded={listOpen} onclick={() => openList(false)}><Icon d={listIcon} /></button>
				<button type="button" class="steps-btn" bind:this={listBtnPh} aria-haspopup="dialog" aria-expanded={listOpen} aria-label={t({ en: 'All steps', zh: '全部步骤' })} onclick={() => openList(true)}>
					<Icon d={listIcon} size={18} /><span class="pos num">{covered ? n : `${at + 1}/${n}`}</span>
				</button>
			</div>
			<div class="r">{@render primaryBtn()}</div>
		</div>
		{#if playing}<span class="timer" class:waiting style="--p: {progress}" aria-hidden="true"></span>{/if}
	</svelte:element>

	{#if listUsed}
		{#await loadList() then m}
			<m.default bind:open={listOpen} anchor={listAnchor} name={t(app?.name ?? label ?? { en: 'Lesson', zh: '课程' })} {titles} {groups} {at} cover={covered} {animated} onpick={(k) => go(k)} onrestart={restart} />
		{/await}
	{/if}

	{#if toast}
		<div class="toast" role="status">
			<span>{lui('resumed', { k: toast.k })}</span>
			<button type="button" class="btn btn-quiet" onclick={restart}>{lui('startOver')}</button>
		</div>
	{/if}
	<div class="visually-hidden" aria-live="polite" aria-atomic="true">{announce}</div>
</section>

<style>
	/* ------------------------------------------------ the frame (both layouts) */
	.wt {
		display: flex;
		flex-direction: column;
		background: var(--panel);
		outline: none;
	}
	.wt[hidden] {
		display: none;
	}
	.main {
		display: flex;
		flex-direction: column;
		min-width: 0;
		outline: none;
	}
	.stage {
		position: relative;
		display: flex;
		flex-direction: column;
		min-width: 0;
		padding: 16px 28px;
		background: var(--stage);
		container: stage / inline-size;
	}
	.stage-h {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 8px 12px;
		min-height: 46px;
	}
	.chip-slot {
		min-width: 0;
		max-width: 100%;
	}
	/* top-aligned in the header's first row: beside a tall concept control it sat half-way down the stage (QA) */
	.prov {
		display: inline-flex;
		align-self: flex-start;
		align-items: center;
		gap: 6px;
		min-height: 46px;
		margin-left: auto;
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.stage-b {
		position: relative;
		display: flex;
		flex: 1;
		flex-direction: column;
		min-height: 0;
		padding: 12px 0 4px;
	}
	/* centred while it fits; a taller view starts at the top and the pane scrolls (never overflows upwards) */
	.gate {
		margin-block: auto;
		min-width: 0;
		transition: opacity var(--dur-2) var(--ease);
	}
	.gated .gate {
		opacity: 0.18;
		filter: saturate(0.4);
	}
	/* K4: an element of the stage with class="atlas-answer" stays hidden while a predict gates the stage */
	.gate[inert] :global(.atlas-answer) {
		visibility: hidden;
	}
	.veil {
		position: absolute;
		inset: 0;
		z-index: 2;
		display: grid;
		place-items: center;
		padding: 12px;
	}
	.veil-c {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 12px;
		max-width: 340px;
		padding: 18px 22px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		box-shadow: var(--shadow-2);
		text-align: center;
	}
	.veil-c p {
		margin: 0;
		color: var(--ink-2);
	}
	.veil-c b {
		color: var(--ink);
	}
	.fade {
		min-width: 0;
		animation: wt-in var(--dur-2) var(--ease);
	}
	@keyframes wt-in {
		from {
			opacity: 0;
		}
	}
	/* ------------------------------------------------ narration */
	.narr {
		display: flex;
		flex-direction: column;
		gap: 16px;
		min-width: 0;
		padding: 28px 36px;
		background: var(--panel);
	}
	.narr > :global(*) {
		margin: 0;
	}
	.ew {
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
	}
	.ttl {
		margin: -8px 0 0;
		font-size: var(--t-xl);
		line-height: var(--lh-xl);
		font-weight: var(--w-title);
		letter-spacing: -0.012em;
	}
	.say {
		color: var(--ink);
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
		text-wrap: pretty;
	}
	.look {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 10px;
		align-items: start;
		padding: 12px 14px;
		border-radius: var(--r3);
		background: var(--mark-soft);
		box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--mark-line) 35%, transparent);
		color: var(--ink-2);
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.look :global(.li) {
		margin-top: 3px;
		color: var(--mark-ink);
	}
	:global(html[lang='zh-CN']) .say {
		line-height: 1.8;
	}
	:global(html[lang='zh-CN']) .look {
		line-height: 1.75;
	}
	:global(html[lang='zh-CN']) .ttl {
		letter-spacing: 0;
		line-height: 1.4;
	}
	.why {
		padding-top: 14px;
		border-top: 1px solid var(--rule);
	}
	.why summary {
		display: flex;
		align-items: center;
		gap: 8px;
		min-height: 28px;
		color: var(--ink-2);
		font-weight: var(--w-ui);
		cursor: pointer;
		list-style: none;
	}
	.why summary::-webkit-details-marker {
		display: none;
	}
	.why summary::before {
		content: '';
		width: 7px;
		height: 7px;
		margin: 0 4px 0 2px;
		border-right: 2px solid currentColor;
		border-bottom: 2px solid currentColor;
		transform: rotate(-45deg);
		transition: transform var(--dur-1) var(--ease);
	}
	.why[open] summary::before {
		transform: rotate(45deg);
	}
	.why p {
		margin: 8px 0 0;
		color: var(--ink-2);
	}
	.wait {
		color: var(--muted);
		font-size: var(--t-sm);
	}
	/* ------------------------------------------------ action bar */
	.bar {
		position: relative;
		z-index: 5;
		border-top: 1px solid var(--rule);
		background: var(--panel);
	}
	.abar {
		display: grid;
		grid-template-columns: 1fr auto 1fr;
		align-items: center;
		gap: 20px;
		min-height: var(--abar-h);
		padding: 10px 20px;
	}
	.has-fb .abar {
		display: none;
	}
	.l {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.r {
		display: flex;
		justify-content: flex-end;
		min-width: 0;
	}
	.back[aria-disabled='true'] {
		opacity: 0.45;
	}
	.play[aria-pressed='true'] {
		background: var(--here-bg);
	}
	.railx {
		display: flex;
		align-items: center;
		gap: 14px;
		min-width: 0;
	}
	.dots {
		display: flex;
		align-items: flex-end;
		gap: 14px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.ch {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 0;
	}
	.ch-n {
		overflow: hidden;
		max-width: 160px;
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-ui);
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.ch.cur .ch-n {
		color: var(--ink);
	}
	.segs {
		display: flex;
		gap: 4px;
	}
	.dot {
		position: relative;
		width: var(--seg-w);
		height: 24px;
		padding: 0;
		border: 0;
		border-radius: 4px;
		background: none;
		cursor: pointer;
	}
	.dot::after {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		top: 9px;
		height: 6px;
		border-radius: var(--r-pill);
		background: var(--rule-strong);
		transition: background var(--dur-3) var(--ease);
	}
	.dot.done::after {
		background: var(--ink-2);
	}
	.dot.now::after {
		top: 8px;
		height: 8px;
		background: var(--ink);
		box-shadow: 0 0 0 3px var(--mark);
	}
	.dot:hover::after {
		background: var(--muted);
	}
	.list-btn {
		color: var(--ink-2);
	}
	.steps-btn {
		display: none;
		align-items: center;
		gap: 8px;
		min-height: 48px;
		padding: 0 14px;
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		background: var(--panel);
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.timer {
		position: absolute;
		left: 0;
		top: -1px;
		height: 2px;
		width: calc(var(--p) * 100%);
		background: var(--ink-2);
	}
	.timer.waiting {
		width: 100%;
		background: repeating-linear-gradient(90deg, var(--ink-2) 0 10px, transparent 10px 20px);
	}
	.toast {
		position: fixed;
		left: 50%;
		bottom: calc(var(--abar-h) + 16px + env(safe-area-inset-bottom, 0px));
		z-index: 20;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 4px 4px 4px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--raised);
		box-shadow: var(--shadow-2);
		transform: translateX(-50%);
		font-size: var(--t-sm);
		white-space: nowrap;
	}
	.toast .btn {
		min-height: 36px;
		padding: 0 12px;
		font-size: var(--t-sm);
	}

	/* ------------------------------------------------ the page: app mode (>= 960 x 600) */
	.page {
		min-height: 100vh;
		min-height: 100dvh;
	}
	.page .main {
		flex: 1;
	}
	@media (min-width: 960px) and (min-height: 600px) {
		.page {
			height: 100vh;
			height: 100dvh;
			min-height: 0;
		}
		.page .main {
			display: grid;
			grid-template-columns: minmax(0, 1fr) var(--narr-w);
			min-height: 0;
		}
		.page .stage,
		.page .narr {
			overflow: auto;
			scrollbar-gutter: stable;
		}
		.page .stage {
			container: stage / size;
		}
		.page .narr {
			border-left: 1px solid var(--rule);
		}
	}
	/* flow, wide: shorter than 600 or 760-959 wide (portrait tablets): the page scrolls, the bar sticks */
	@media (min-width: 760px) and (max-height: 599px), (min-width: 760px) and (max-width: 959px) {
		.page .bar {
			position: sticky;
			bottom: 0;
		}
		.page .narr {
			border-top: 1px solid var(--rule);
			padding: 24px max(32px, calc((100% - 640px) / 2)) 28px;
		}
	}
	@media (max-width: 1099px) {
		.narr {
			padding: 24px;
		}
		.ch-n,
		.play-lbl {
			display: none;
		}
		.dots {
			gap: 8px;
		}
	}
	/* ------------------------------------------------ phones (< 760): picture first, the page scrolls, the bar is fixed */
	@media (max-width: 759px) {
		.page .stage {
			padding: 10px 16px 14px;
		}
		.stage-h {
			flex-wrap: nowrap;
			min-height: 44px;
		}
		.stage-b {
			padding: 10px 0 0;
		}
		/* a gated stage folds to a strip, so the question and its options are in the first screen */
		.stage-b.gated {
			max-height: 150px;
			overflow: hidden;
		}
		.veil-c {
			gap: 8px;
			max-width: 300px;
			padding: 12px 14px;
			box-shadow: var(--shadow-1);
		}
		.veil-c p {
			font-size: var(--t-sm);
			line-height: var(--lh-sm);
		}
		.pl {
			display: none;
		}
		.page .narr {
			gap: 14px;
			padding: 18px 16px calc(var(--abar-h) + 24px + env(safe-area-inset-bottom, 0px));
			border-top: 1px solid var(--rule);
		}
		.ttl {
			margin-top: -6px;
		}
		.look {
			padding: 10px 12px;
			font-size: var(--t-sm);
			line-height: var(--lh-sm);
		}
		:global(html[lang='zh-CN']) .look {
			line-height: 1.7;
		}
		.page .bar {
			position: fixed;
			left: 0;
			right: 0;
			bottom: 0;
		}
		.abar {
			grid-template-columns: auto auto minmax(0, 1fr);
			gap: 8px;
			min-height: 0;
			padding: 8px 12px calc(8px + env(safe-area-inset-bottom, 0px));
		}
		.abar .btn.back {
			min-height: 48px;
			padding: 0 14px;
		}
		.dt,
		.rail,
		.list-btn {
			display: none;
		}
		.steps-btn {
			display: inline-flex;
		}
		.r {
			justify-content: stretch;
		}
		.fbw {
			display: none;
		}
		.has-fb .abar {
			display: grid;
		}
	}

	/* ------------------------------------------------ a card in a page (no lesson-app context, or a page with the site chrome) */
	.card {
		container: wt / inline-size;
		overflow: clip;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r4);
		box-shadow: var(--shadow-1);
	}
	.card .stage {
		padding: 12px 16px;
	}
	.card .narr {
		padding: 18px 20px;
	}
	.card .bar {
		position: sticky;
		bottom: 0;
	}
	.card .abar {
		grid-template-columns: auto minmax(0, 1fr) auto;
		gap: 12px;
		min-height: 0;
		padding: 8px 12px;
	}
	.card .railx {
		justify-content: center;
	}
	.card .ch-n,
	.card .play-lbl {
		display: none;
	}
	.card .dots {
		flex-wrap: wrap;
		gap: 8px;
	}
	@container wt (min-width: 860px) {
		.card .main {
			display: grid;
			grid-template-columns: minmax(0, 1fr) 360px;
		}
		.card .narr {
			border-left: 1px solid var(--rule);
		}
	}
	@container wt (max-width: 520px) {
		.card .rail,
		.card .list-btn {
			display: none;
		}
		.card .steps-btn {
			display: inline-flex;
			min-height: 44px;
		}
	}
</style>
