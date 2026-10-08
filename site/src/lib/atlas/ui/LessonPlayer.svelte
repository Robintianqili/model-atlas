<!--
  LessonPlayer: a compiled lesson (lesson/compile.ts) played by the kit Walkthrough, unchanged (KIT rule 15; design §2.1).
  Compiled steps become WalkSteps (title; narration and look as closures over the live runtime) and one shared `stage`
  snippet renders StageHost. Entering a step runs stepAt() (runtime/sources.ts): it replaces the LESSON's own selection
  (rt.sel, never the explore views'), resolves and clamps the cursor and runs the source as far as needed.
  - stepAt() calls are serialized, so the selection always belongs to the step that was entered last.
  - On a new input every cheap step (trace, scan and precomputed tape sources) is resolved again right away, so the
    narration a screen reader hears on entering a step already has its numbers; live training / process steps are
    resolved when entered and show "Computing… k / n" meanwhile.
  - Deep links (#<lesson id>-<step id>), keyboard, reduced motion and the narration-first phone layout are the
    Walkthrough's own.
  - Play and the learner (K0): a step with a predict question or a task waits for the learner (keepPlay: answering or
    doing the task in the stage does not pause Play), says so ("Waiting for your answer…"), and after the answer stays
    long enough to read the explanation it revealed (linger from its length, unless the step sets linger).
  - K1: once answered, the Play line stops saying "Waiting for your answer…" while the step's own range plays (K-R5); a
    task lingers by the length of its narration as it reads once done (K-R2: it was 4 s); the next step's live source is
    prefetched while the learner reads this one; the end card waits until the last step has its numbers.
  - K2: a step the empty input skipped (its source did not run) shows no "Look:" line (the stage holds only the input's
    note), and its narration leaves out the sentences that need numbers (lesson/narrate.ts), never "…".
-->
<script lang="ts">
	import { untrack, type Snippet } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import type { WalkContext, WalkStep } from '$lib/kit/types';
	import type { Text } from '$lib/i18n/i18n.svelte';
	import type { At } from '../types/lesson.ts';
	import type { CompiledLesson } from '../lesson/compile.ts';
	import { stepAt, type StepPosition } from '../runtime/sources.ts';
	import type { AtlasStore } from './runtime.svelte.ts';
	import StageHost from './StageHost.svelte';
	import { STR } from './strings.ts';

	interface Props {
		store: AtlasStore;
		lesson: CompiledLesson;
		label?: Text;
		/** current step (bindable) */
		index?: number;
		stageMinHeight?: number;
		lab?: boolean;
		aside?: Snippet<[WalkContext]>;
		end?: Snippet;
		onstep?: (i: number, id: string) => void;
	}
	let { store, lesson, label, index = $bindable(0), stageMinHeight = 320, lab = false, aside, end, onstep }: Props = $props();

	let positions = $state.raw<(StepPosition | undefined)[]>([]);
	/** per step: the predict is answered and the task done (StageHost reports it; K1) */
	let learner = $state<Record<number, boolean>>({});
	let gen = 0;
	let queue: Promise<unknown> = Promise.resolve();
	/** stepAt() replaces rt.sel: run one at a time, so the last one entered owns the selection */
	function serial<T>(f: () => Promise<T>): Promise<T> {
		const p = queue.then(f, f);
		queue = p.catch(() => undefined);
		return p;
	}
	function put(i: number, p: StepPosition) {
		const next = positions.slice();
		next[i] = p;
		positions = next;
	}
	async function enterStep(i: number) {
		const g = gen;
		const s = lesson.steps[i];
		if (!s) return;
		const p = await serial(() => stepAt(store.rt, s));
		store.touch();
		if (g === gen) put(i, p);
		prefetch(i + 1);
	}
	/** K1: start the next step's live source (train, process, live-model, tape) while the learner reads this step, so
	 *  its narration does not wait on "…" when entered. Runs in the worker when the package has one. */
	function prefetch(i: number) {
		const s = lesson.steps[i];
		if (!s || cheap(i) || positions[i]) return;
		let c;
		try {
			c = s.at(store.rt);
		} catch {
			c = 'first' as const;
		}
		void store.rt.ensure(s.source, c).catch(() => -1);
	}
	const cheap = (i: number): boolean => {
		const k = store.rt.sources[lesson.steps[i].source]?.kind ?? (lesson.steps[i].source.startsWith('scan:') ? 'scan' : undefined);
		return k === 'trace' || k === 'scan' || k === 'tape';
	};
	async function resolveAll() {
		const g = ++gen;
		// cheap steps keep their old position until the new one replaces it (no flicker while drawing); live ones restart
		positions = positions.map((p, i) => (cheap(i) ? p : undefined));
		for (let i = 0; i < lesson.steps.length; i++) {
			if (!cheap(i)) continue;
			const p = await serial(() => stepAt(store.rt, lesson.steps[i]));
			if (g !== gen) return;
			put(i, p);
		}
		if (g !== gen) return;
		await enterStep(index); // last, so the lesson selection is the current step's again
	}
	// every new input (and the first one): re-resolve the steps' cursors (design §4.7 setInput)
	$effect(() => {
		void store.inputGen;
		void lesson;
		untrack(() => void resolveAll());
	});

	function atOf(i: number): At {
		return positions[i]?.at ?? { index: -1 };
	}
	const steps = $derived<WalkStep[]>(
		lesson.steps.map((s, i) => {
			const w: WalkStep = {
				id: s.id,
				title: s.title,
				narration: () => {
					void store.version;
					return s.narration(store.rt, atOf(i));
				},
				onEnter: () => void enterStep(i),
				waitForDone: s.waitForDone,
				// K1 (K-R7): the screen-reader announcement waits until the step's numbers are computed
				pending: () => !positions[i]
			};
			if (s.look) {
				const look = () => {
					void store.version;
					return s.look!(store.rt, atOf(i)) ?? '';
				};
				// K2: no "Look:" line on a step the empty input skipped (a getter, so the Walkthrough re-reads it)
				Object.defineProperty(w, 'look', { enumerable: true, get: () => (positions[i]?.skipped ? undefined : look) });
			}
			if (s.predict || s.task) {
				w.keepPlay = true;
				// K1: only while the learner has not answered; then the default line ("Playing this step's animation…")
				w.waitText = () => (learner[i] ? '' : s.predict ? STR.waitAnswer : STR.waitTask);
				// after the answer: time to read what it revealed (about 55 ms per character, 5-14 s); a task: its narration
				// as it reads once done (K1: 4 s was too short when the answer is in the updated narration)
				if (s.linger === undefined) {
					if (s.predict) w.linger = Math.max(5000, Math.min(14000, (s.predict.explain.en ?? '').length * 55));
					else
						Object.defineProperty(w, 'linger', {
							enumerable: true,
							get: () => {
								const n = s.narration(store.rt, atOf(i));
								return Math.max(5000, Math.min(14000, (n.en ?? '').length * 55));
							}
						});
				}
			}
			if (s.linger !== undefined) w.linger = s.linger;
			if (s.duration !== undefined) w.duration = s.duration;
			return w;
		})
	);
</script>

<!-- K1: the end card ("That is the whole lesson") waits until the last step has its numbers -->
{#snippet endGate()}
	{#if end && positions[lesson.steps.length - 1]}{@render end()}{/if}
{/snippet}

<Walkthrough {steps} bind:index id={lesson.id} {label} ready={true} {stageMinHeight} {aside} end={end ? endGate : undefined} onstep={(i, s) => onstep?.(i, s.id)}>
	{#snippet stage(_ws, ctx)}
		{@const s = lesson.steps[ctx.index]}
		{#if s}
			<StageHost {store} step={s} pos={positions[ctx.index]} {ctx} {lab} onlearner={(done) => (learner[ctx.index] !== done ? (learner = { ...learner, [ctx.index]: done }) : undefined)} />
		{/if}
	{/snippet}
</Walkthrough>
