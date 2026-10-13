<!--
  LessonPlayer: a compiled lesson (lesson/compile.ts) played by the kit Walkthrough (KIT rule 15; design §2.1), laid out
  as the lesson app (redesign spec §5). Compiled steps become WalkSteps (title; narration and look as closures over the
  live runtime; R1: the stage header's provenance label, the predict question and the task as beats in the narration,
  the chapter, the Why) and one shared `stage` snippet renders StageHost. Entering a step runs stepAt()
  (runtime/sources.ts): it replaces the LESSON's own selection (rt.sel, never the explore views'), resolves and clamps
  the cursor and runs the source as far as needed.
  - stepAt() calls are serialized, so the selection always belongs to the step that was entered last.
  - On a new input every cheap step (trace, scan and precomputed tape sources) is resolved again right away, so the
    narration a screen reader hears on entering a step already has its numbers; live training / process steps are
    resolved when entered and show "Computing… k / n" meanwhile.
  - Deep links (#<lesson id>-<step id>), keyboard, reduced motion, the cover and the end card are the Walkthrough's own.
  - Play and the learner (K0): a step with a predict question or a task waits for the learner (keepPlay: answering in
    the narration or doing the task in the stage does not pause Play), says so ("Waiting for your answer…"), and after
    the answer stays long enough to read the explanation it revealed (linger from its length, unless the step sets
    linger).
  - K1: once answered, the Play line stops saying "Waiting for your answer…" while the step's own range plays (K-R5); a
    task lingers by the length of its narration as it reads once done (K-R2); the next step's live source is prefetched
    while the learner reads this one; the end card waits until the last step has its numbers.
  - K2: a step the empty input skipped (its source did not run) shows no "Look:" line (the stage holds only the input's
    note), and its narration leaves out the sentences that need numbers (lesson/narrate.ts), never "…".
  - K4: the predict question's live parts (StepDef.predictAt) are read at the step's own position.
-->
<script lang="ts">
	import { untrack, type Snippet } from 'svelte';
	import Walkthrough from '$lib/kit/Walkthrough.svelte';
	import type { StageProv, WalkContext, WalkPredict, WalkStep } from '$lib/kit/types';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { At } from '../types/lesson.ts';
	import type { CompiledLesson } from '../lesson/compile.ts';
	import { effectiveProvenance, stepAt, type StepPosition } from '../runtime/sources.ts';
	import type { AtlasStore } from './runtime.svelte.ts';
	import { parityBadge, sourceBadge } from './kitspec.ts';
	import StageHost from './StageHost.svelte';
	import { focusBus } from './focus.svelte.ts';
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
		/** R1: the cover's stage (step 0) */
		cover?: Snippet<[WalkContext]>;
		onstep?: (i: number, id: string) => void;
	}
	let { store, lesson, label, index = $bindable(0), stageMinHeight = 320, lab = false, aside, end, cover, onstep }: Props = $props();

	let positions = $state.raw<(StepPosition | undefined)[]>([]);
	/** per step: the predict is answered and the task done (StageHost reports it; K1) */
	let learner = $state<Record<number, boolean>>({});
	/** per step: the task is done (StageHost reads it at the playing frame) */
	let tasks = $state<Record<number, boolean>>({});
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
	/** K1: start the next step's live source while the learner reads this step */
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
		positions = positions.map((p, i) => (cheap(i) ? p : undefined));
		for (let i = 0; i < lesson.steps.length; i++) {
			if (!cheap(i)) continue;
			const p = await serial(() => stepAt(store.rt, lesson.steps[i]));
			if (g !== gen) return;
			put(i, p);
		}
		if (g !== gen) return;
		await enterStep(index);
	}
	$effect(() => {
		void store.inputGen;
		void lesson;
		untrack(() => void resolveAll());
	});

	function atOf(i: number): At {
		return positions[i]?.at ?? { index: -1 };
	}
	function safe<T>(f: () => T, fallback: T): T {
		try {
			return f();
		} catch {
			return fallback;
		}
	}
	/** the stage header's provenance label: a live step on a precomputed fallback says so; the parity detail of kind A,
	 *  or of a train or process source's twin or reference (K0, K1) */
	function provOf(i: number): StageProv {
		void store.version;
		const s = lesson.steps[i];
		const rt = store.rt;
		const p = effectiveProvenance(rt, s);
		const src = rt.sources[s.source];
		let b: { tolerance?: number; against?: string; precision?: string; note?: Text; short?: Text } = { precision: rt.spec?.precision };
		if (src && p.mode !== 'illustrative') {
			if (src.kind === 'trace' || src.kind === 'scan') b = parityBadge(rt.spec);
			else if ((src.kind === 'train' || src.kind === 'process') && p.mode === 'live') b = { ...b, ...sourceBadge(rt.pkg.parity?.sources[s.source]) };
		}
		return { mode: p.mode, precision: b.precision, tolerance: b.tolerance, against: b.against, note: p.note ?? b.note, short: b.short };
	}
	function predictOf(i: number): WalkPredict | undefined {
		const s = lesson.steps[i];
		if (!s.predict) return undefined;
		void store.version;
		const pos = positions[i];
		const a = pos && !pos.error && !pos.skipped && pos.at.index >= 0 ? pos.at : null;
		const live = a && s.predictAt ? safe(() => s.predictAt!(store.rt, a), undefined) : undefined;
		const q = { ...s.predict, ...(live ?? {}) };
		return { question: q.question, options: q.options, answer: q.answer, explain: q.explain };
	}
	const chapterOf = $derived.by(() => {
		const m = new Map<string, Text>();
		for (const c of lesson.chapters ?? []) for (const id of c.steps) m.set(id, c.title);
		return m;
	});

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
				pending: () => !positions[i],
				prov: () => provOf(i),
				animated: !!s.play
			};
			const ch = chapterOf.get(s.id);
			if (ch) w.chapter = ch;
			if (s.why) {
				const why = s.why;
				w.why = {
					q: why.q,
					a: () => {
						void store.version;
						return why.a(store.rt, atOf(i));
					}
				};
			}
			if (s.look) {
				const look = () => {
					void store.version;
					return s.look!(store.rt, atOf(i)) ?? '';
				};
				// K2: no "Look:" line on a step the empty input skipped (a getter, so the Walkthrough re-reads it)
				Object.defineProperty(w, 'look', { enumerable: true, get: () => (positions[i]?.skipped ? undefined : look) });
			}
			if (s.predict) w.predict = () => predictOf(i);
			// R1 number chips: the narration with its {name@ref} values marked; a chip moves the stage's ring to its ref
			if (s.chips) {
				const ch = s.chips;
				w.chips = {};
				if (ch.narration)
					w.chips.narration = () => {
						void store.version;
						return ch.narration!(store.rt, atOf(i));
					};
				if (ch.look)
					w.chips.look = () => {
						void store.version;
						return ch.look!(store.rt, atOf(i));
					};
			}
			if (s.refs) {
				const refs = s.refs;
				w.onref = (ref) => {
					const f = ref ? safe(() => refs(store.rt, atOf(i))[ref], undefined) : undefined;
					focusBus(store.rt, 'lesson').setHover(f ?? null);
				};
			}
			if (s.task) {
				const task = s.task;
				w.task = { prompt: task.prompt, hint: task.hint, waiting: task.waiting, doneText: task.doneText, done: () => !!tasks[i] };
			}
			if (s.predict || s.task) {
				w.keepPlay = true;
				// K1: only while the learner has not answered; then the default line ("Playing this step's animation…")
				w.waitText = () => (learner[i] ? '' : s.predict ? STR.waitAnswer : STR.waitTask);
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
	const lastAt = $derived(atOf(lesson.steps.length - 1));
	const recapText = $derived.by(() => {
		void store.version;
		return (lesson.recap ?? []).map((r) => safe(() => r(store.rt, lastAt), { en: '' }));
	});
</script>

<!-- K1: the end card waits until the last step has its numbers -->
{#snippet endGate()}
	{#if end && positions[lesson.steps.length - 1]}{@render end()}{/if}
{/snippet}

{#snippet recapList()}
	<ul class="recap">{#each recapText as r, i (i)}<li>{t(r)}</li>{/each}</ul>
{/snippet}

<Walkthrough
	{steps}
	bind:index
	id={lesson.id}
	{label}
	ready={true}
	{stageMinHeight}
	{aside}
	{cover}
	end={end ? endGate : undefined}
	recap={lesson.recap?.length ? recapList : undefined}
	onstep={(i, s) => onstep?.(i, s.id)}
>
	{#snippet stage(_ws, ctx)}
		{@const s = lesson.steps[ctx.index]}
		{#if s}
			<StageHost
				{store}
				step={s}
				pos={positions[ctx.index]}
				{ctx}
				{lab}
				badge={false}
				beats={false}
				onlearner={(done) => (learner[ctx.index] !== done ? (learner = { ...learner, [ctx.index]: done }) : undefined)}
				ontask={(done) => (tasks[ctx.index] !== done ? (tasks = { ...tasks, [ctx.index]: done }) : undefined)}
			/>
		{/if}
	{/snippet}
</Walkthrough>

<style>
	.recap {
		margin: 0;
		padding-left: 1.2em;
	}
	.recap li + li {
		margin-top: 4px;
	}
</style>
