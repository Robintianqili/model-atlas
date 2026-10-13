<!--
  StageHost: the stage of one lesson step (design §2.2 "what the kit adds to every step"):
    - the provenance badge (effectiveProvenance: a live step on a precomputed fallback says so);
    - "Computing… k / n" while a live source catches up with the step's cursor ("Training the network first… k / n"
      while a live-model step waits for its train source, e.g. after a deep link straight to it);
    - the clamped cursor (a note when the step's moment did not happen for this input; never a crash);
    - the adapter's empty note when the input is empty;
    - the view (lazy, from the registry; FallbackView when the key is not registered) with its describe() text;
    - a TapeScrubber for `play: { from, to, phases }`; `play: 'view'` lets the view animate and report onend;
    - predict gating (the view is dimmed and inert until the learner answers) and task gating (task.done(rt, at)), with
      the task's own status line (task.waiting);
    - a step's `compare`: both sources at the step's moment (the same cursor, or the playing frame's phase and step;
      K0: they were aligned by frame index);
    - setInput for the view (StageProps.setInput: a view that edits the learner's input goes through the page, K0);
    - glossary terms: the site glossary's, else the package's own (manifest.glossary), so a package term works before
      promote adds it to the site glossary;
    - ctx.done() once everything the step waits for has happened (Play then moves on after `linger`).
  The Walkthrough re-creates the stage on every step ({#key}), so all state here is per step.
  Redesign W2: in the lesson app the provenance label sits in the Walkthrough's stage header and the predict question and
  the task are beats in its narration: LessonPlayer passes badge={false} and beats={false}, the gate comes from
  ctx.gated, and the task's state goes back through ontask. A free-standing panel (ConceptPanel) keeps all of it here.
  StageProps.density (R1) passes the stage pane's density on to the view.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Term from '$lib/kit/Term.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { WalkContext } from '$lib/kit/types';
	import type { At } from '../types/lesson.ts';
	import type { Cursor, Focus, LiveTape } from '../types/process.ts';
	import type { StageProps } from '../types/view.ts';
	import type { AtlasCompiledStep } from '../lesson/compile.ts';
	import { alignedIndex, atFor, effectiveProvenance, type StepPosition } from '../runtime/sources.ts';
	import Tooltip from '$lib/kit/Tooltip.svelte';
	import { sourceDef } from '../lesson/define.ts';
	import type { AtlasStore } from './runtime.svelte.ts';
	import { focusBus } from './focus.svelte.ts';
	import { parityBadge, sourceBadge } from './kitspec.ts';
	import type { I18n } from '../types/core.ts';
	import ViewSlot from './ViewSlot.svelte';
	import TapeScrubber from './TapeScrubber.svelte';
	import { STR, say } from './strings.ts';

	interface Props {
		store: AtlasStore;
		step: AtlasCompiledStep;
		/** the step's resolved position (LessonPlayer: stepAt); undefined while it is being computed */
		pos: StepPosition | undefined;
		ctx: WalkContext;
		/** Lesson Lab: show registry notes */
		lab?: boolean;
		/** K1: true once the predict is answered and the task done (LessonPlayer: Play stops saying "Waiting for your answer…") */
		onlearner?: (done: boolean) => void;
		/** W2: show the provenance badge in the stage (false: the Walkthrough's stage header shows it) */
		badge?: boolean;
		/** W2: ask the predict question and show the task here (false: the Walkthrough's narration does; the gate is ctx.gated) */
		beats?: boolean;
		/** W2: the task's state, read at the shown frame */
		ontask?: (done: boolean) => void;
	}
	let { store, step, pos, ctx, lab = false, onlearner, badge: showBadge = true, beats = true, ontask }: Props = $props();

	const rt = $derived(store.r);
	const src = $derived.by(() => {
		void store.version;
		return rt.sources[step.source];
	});
	const prov = $derived.by(() => {
		void store.version;
		return effectiveProvenance(rt, step);
	});
	// kind A: the spec's parity; train and process sources: the package's parity record of their twin or reference (K0).
	// K1: a train or process step with a passing record shows its check next to the compact badge ("= PyTorch fp64").
	const badge = $derived.by((): { tolerance?: number; against?: string; precision?: string; note?: I18n; short?: I18n } => {
		if (!src || prov.mode === 'illustrative') return { precision: rt.spec?.precision };
		if (src.kind === 'trace' || src.kind === 'scan') return parityBadge(rt.spec);
		if ((src.kind === 'train' || src.kind === 'process') && prov.mode === 'live') return sourceBadge(rt.pkg.parity?.sources[step.source]);
		return { precision: rt.spec?.precision };
	});
	const play = $derived(typeof step.play === 'object' ? step.play : null);
	const selfPlays = $derived(step.play === 'view');

	// ------------------------------------------------------------------ play range: run the source far enough first
	let playReady = $state(false);
	let playIndex = $state(-1);
	let playing = $state(false);
	let playEnded = $state(false);
	// K1: again after every new input (the process restarts and was recorded only as far as the step's cursor; the
	// scrubber then stopped short of play.to)
	$effect(() => {
		if (!play) return;
		void store.inputGen;
		let alive = true;
		const to = play.to;
		untrack(() =>
			store.rt
				.ensure(step.source, to)
				.catch(() => -1)
				.then(() => {
					if (!alive) return;
					playReady = true;
					store.touch();
				})
		);
		return () => {
			alive = false;
		};
	});
	/** one object per step, shared by the views of this step (the two panels of a compare; K1) */
	const shared = $state<Record<string, unknown>>({});

	// ------------------------------------------------------------------ the At the view shows
	const at = $derived.by((): At | null => {
		void store.version;
		if (!pos || pos.error || pos.skipped) return null;
		if (play && playReady && playIndex >= 0) {
			try {
				return atFor(rt, step.source, playIndex);
			} catch {
				return pos.at;
			}
		}
		return pos.at.index >= 0 ? { ...pos.at } : null;
	});

	function safe<T>(f: () => T, fallback: T): T {
		try {
			return f();
		} catch {
			return fallback;
		}
	}
	const focus = $derived(at ? safe<Focus[]>(() => step.focus(rt, at), []) : []);
	// the 'compare' view (W1-D2) shows both sources itself: it gets the step's compare through props and one stage
	const selfCompares = $derived(!!step.compare && step.view === 'compare');
	const viewProps = $derived.by((): Record<string, unknown> => {
		const p = at ? safe<Record<string, unknown>>(() => step.props(rt, at), {}) : {};
		return selfCompares && p.compare === undefined ? { ...p, compare: step.compare } : p;
	});
	const bus = $derived(focusBus(store.rt, 'lesson'));
	$effect(() => {
		const f = focus;
		untrack(() => bus.setStep(f));
	});

	// ------------------------------------------------------------------ gating: predict, task, play
	// K4: the live parts of the question (StepDef.predictAt: options, answer and explanation from the learner's input),
	// read at the step's own position, so they hold still while the step plays
	const pq = $derived.by(() => {
		if (!step.predict) return null;
		const a = pos && !pos.error && !pos.skipped && pos.at.index >= 0 ? pos.at : at;
		const live = a && step.predictAt ? safe(() => step.predictAt!(rt, a), undefined) : undefined;
		return { ...step.predict, ...(live ?? {}) };
	});
	let answered = $state<number | null>(null);
	const gate = $derived(beats ? !!step.predict && answered === null : !!ctx.gated);
	const taskDone = $derived(step.task && at ? safe(() => !!step.task!.done(rt, at), false) : false);
	const needsPlay = $derived(!!(play || selfPlays) && !ctx.reduced);
	function onPlayEnd() {
		playEnded = true;
	}
	let doneSent = false;
	$effect(() => {
		// everything the step waits for has happened -> tell the Walkthrough (Play moves on after `linger`)
		const ok = !gate && (!step.task || taskDone) && (!needsPlay || playEnded) && !!at && ctx.shown;
		if (ok && !doneSent) {
			doneSent = true;
			untrack(() => ctx.done());
		}
	});

	// ------------------------------------------------------------------ "Computing… k / n"
	function targetOf(c: Cursor | null | undefined): number | null {
		if (c && typeof c === 'object' && typeof c.step === 'number') return c.step;
		const opt = (src?.tape?.opt ?? null) as { steps?: unknown } | null;
		if ((c === 'last' || (c && typeof c === 'object' && c.last)) && typeof opt?.steps === 'number') return opt.steps;
		return null;
	}
	const computing = $derived.by(() => {
		void store.version;
		const busy = !pos || (!!play && !playReady);
		if (!busy) return null;
		// a live-model step waits for its train source (a deep link runs that source to its end first): its progress
		const def = safe(() => sourceDef(rt.pkg.manifest, step.source), undefined);
		if (def?.kind === 'live-model') {
			const tr = rt.sources[def.of]?.tape as LiveTape | null | undefined;
			const n = tr?.length ?? 0;
			const steps = (tr?.opt ?? null) as { steps?: unknown } | null;
			if (n > 0 && typeof steps?.steps === 'number') return say('trainingFirstOf', { k: Math.min(tr!.frame(n - 1).step ?? n, steps.steps), n: steps.steps });
			return say('trainingFirst');
		}
		const tape = src?.tape as LiveTape | null | undefined;
		const n = tape?.length ?? 0;
		const last = n > 0 ? tape!.frame(n - 1) : null;
		const k = last?.step ?? n;
		const target = targetOf(play ? play.to : safe(() => step.at(rt), null));
		if (target !== null && n > 0) return say('computingOf', { k: Math.min(k, target), n: target });
		if (n > 0 && src && (src.status === 'running' || src.status === 'loading')) return say('computingFrames', { k: n });
		// a process that runs after another source (K1): that source's progress
		if (def?.kind === 'process' && def.after && n === 0) {
			const tr = rt.sources[def.after]?.tape as LiveTape | null | undefined;
			const m = tr?.length ?? 0;
			const steps = (tr?.opt ?? null) as { steps?: unknown } | null;
			if (m > 0 && typeof steps?.steps === 'number') return say('trainingFirstOf', { k: Math.min(tr!.frame(m - 1).step ?? m, steps.steps), n: steps.steps });
			if (rt.sources[def.after]?.kind === 'train') return say('trainingFirst');
		}
		return say('computing');
	});

	const stageProps = $derived.by((): StageProps | null => {
		if (!at) return null;
		return {
			rt,
			source: step.source,
			at,
			focus,
			sel: rt.sel,
			ctx: { shown: ctx.shown, reduced: ctx.reduced, playing: ctx.playing || playing, done: onPlayEnd },
			density: ctx.density,
			provenance: prov.mode,
			autoplay: selfPlays && ctx.shown && !ctx.reduced && !gate,
			onend: onPlayEnd,
			props: viewProps,
			setInput: (raw: unknown) => store.setInput(raw),
			shared
		};
	});
	$effect(() => {
		const done = (!step.predict || (beats ? answered !== null : ctx.answered != null)) && (!step.task || taskDone);
		untrack(() => onlearner?.(done));
	});
	$effect(() => {
		const d = !!taskDone;
		untrack(() => ontask?.(d));
	});
	const compareProps = $derived.by((): [StageProps, StageProps] | null => {
		const c = step.compare;
		if (!c || !stageProps || selfCompares) return null;
		const playingFrame = play && playReady && playIndex >= 0 ? at?.frame : undefined;
		const side = (source: string): StageProps => {
			const i = source === step.source ? (at?.index ?? -1) : safe(() => alignedIndex(rt, source, pos?.cursor ?? null, playingFrame), -1);
			return { ...stageProps, source, at: safe(() => atFor(rt, source, i), { index: -1 }) };
		};
		return [side(c.a), side(c.b)];
	});
	const scanCounter = $derived(src?.kind === 'scan' ? 'position' : 'frame');
	/** the package's own glossary terms (manifest.glossary), by id */
	const pkgTerms = $derived(Object.fromEntries((rt.pkg.manifest.glossary ?? []).map((g) => [g.id, g])));
	const notes = $derived.by(() => {
		void store.version;
		return rt.pre?.notes ?? [];
	});
	const emptyInput = $derived.by(() => {
		void store.version;
		return !!rt.pre?.empty;
	});
</script>

<div class="sh" data-step={step.id}>
	{#if showBadge || computing}
		<div class="top" class:bare={!showBadge}>
			{#if computing}
				<span class="busy" role="status"><span class="spin" aria-hidden="true"></span>{computing}</span>
			{/if}
			{#if showBadge}
				<span class="grow"></span>
				<ProvenanceBadge mode={prov.mode} compact precision={badge.precision} tolerance={badge.tolerance} against={badge.against} note={prov.note ?? badge.note} />
				{#if badge.short}<span class="chk small" title={badge.note ? t(badge.note) : undefined}>{t(badge.short)}</span>{/if}
			{/if}
		</div>
	{/if}

	{#if pos?.error}
		<div class="err" role="alert">
			<strong>{say('stageError')}</strong>
			{#if lab}<code>{pos.error}</code>{/if}
		</div>
	{:else if pos?.skipped}
		<div class="empty" role="note">
			{#each notes.length ? notes : [STR.emptyInput] as n, i (i)}<p>{t(n)}</p>{/each}
		</div>
	{:else}
		{#if pos && !pos.found && pos.index >= 0}<p class="clamped" role="note">{say('clamped')}</p>{/if}
		{#if emptyInput}
			<!-- an empty input that still runs (a blank drawing is 784 zeros): real numbers, plus the adapter's note -->
			<div class="empty" role="note">{#each notes.length ? notes : [STR.emptyInput] as n, i (i)}<p>{t(n)}</p>{/each}</div>
		{/if}

		{#if pq && beats}
			<div class="predict" class:done={answered !== null}>
				<p class="q"><span class="tag">{say('predictFirst')}</span> {t(pq.question)}</p>
				<div class="opts" role="group" aria-label={t(pq.question)}>
					{#each pq.options as o, i (i)}
						<button
							type="button"
							class="btn"
							aria-pressed={answered === i}
							class:right={answered !== null && i === pq.answer}
							class:wrong={answered === i && i !== pq.answer}
							onclick={() => (answered = i)}>{t(o)}</button
						>
					{/each}
				</div>
				{#if answered === null}
					<p class="muted small">{say('predictHint')}</p>
				{:else}
					<p class="small" aria-live="polite"><strong>{answered === pq.answer ? say('right') : say('notQuite')}</strong> {t(pq.explain)}</p>
				{/if}
			</div>
		{/if}

		{#if step.task && beats}
			<div class="task" class:done={taskDone} aria-live="polite">
				<p><span class="tag">{say('yourTurn')}</span> {t(step.task.prompt)}</p>
				<p class="small">{taskDone ? (step.task.doneText ? t(step.task.doneText) : say('taskDone')) : step.task.waiting ? t(step.task.waiting) : say('taskWaiting')}</p>
				{#if step.task.hint && !taskDone}
					<details class="small"><summary>{say('hint')}</summary>{t(step.task.hint)}</details>
				{/if}
			</div>
		{/if}

		<div class="view" class:dim={gate && beats} inert={(gate && beats) || undefined}>
			{#if compareProps && step.compare}
				<div class="cmp">
					{#each compareProps as sp, i (i)}
						<div>
							<p class="cmp-l">{t(step.compare.label[i])}</p>
							<ViewSlot view={step.view} stage={sp} {lab} />
						</div>
					{/each}
				</div>
			{:else if stageProps}
				<ViewSlot view={step.view} stage={stageProps} {lab} />
			{/if}
		</div>

		{#if play && playReady && src?.tape}
			<TapeScrubber
				tape={src.tape}
				from={play.from}
				to={play.to}
				phases={play.phases}
				interval={play.interval ?? 120}
				autoplay={ctx.shown && !ctx.reduced && !gate}
				start={pos?.at.index}
				counter={scanCounter}
				unit={play.unit}
				by={play.by}
				version={store.version}
				bind:index={playIndex}
				bind:playing
				onend={onPlayEnd}
			/>
		{/if}
		{#if step.terms.length}
			<p class="terms small">
				<span class="muted">{say('termsLabel')}:</span>
				<!-- K2: a separator between terms (two multi-word terms read as one phrase) -->
				{#each step.terms as id, i (id)}
					{@const own = pkgTerms[id]}
					{#if i > 0}<span class="muted" aria-hidden="true">{' · '}</span>{/if}
					{#if own}
						<Tooltip width={300}>
							{#snippet content()}<strong>{t(own.term)}</strong> — {t(own.def)}{/snippet}
							<span class="term">{t(own.term)}</span>
						</Tooltip>
					{:else}<Term {id} />{/if}
				{/each}
			</p>
		{/if}
	{/if}
</div>

<style>
	/* K4 (rule of two: two-tower, sasrec): an element of the stage with class="atlas-answer" stays hidden while a predict
	   question keeps the stage inert, so the view cannot show the answer before the learner commits */
	.view[inert] :global(.atlas-answer) {
		visibility: hidden;
	}
	.sh {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.top.bare {
		min-height: 0;
	}
	.top {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 8px;
		min-height: 24px;
	}
	.chk {
		color: var(--ink-2);
		white-space: nowrap;
	}
	.grow {
		flex: 1;
	}
	.busy {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--t-xs);
		color: var(--ink-2);
	}
	.spin {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		border: 2px solid var(--rule-strong);
		border-top-color: var(--ink);
		animation: spin 0.9s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
	.err,
	.empty {
		padding: 10px 12px;
		border-radius: var(--r2);
		font-size: var(--t-sm);
	}
	.err {
		border: 1px solid var(--div-neg);
		background: var(--well);
	}
	.err code {
		display: block;
		margin-top: 4px;
		font-size: var(--t-xs);
		white-space: pre-wrap;
		word-break: break-word;
	}
	.empty {
		border: 1.5px dashed var(--rule-strong);
		color: var(--ink-2);
	}
	.empty p {
		margin: 0;
	}
	.clamped {
		margin: 0;
		font-size: var(--t-xs);
		color: var(--ink-2);
		border-left: 3px solid var(--pre);
		padding-left: 8px;
	}
	.predict,
	.task {
		padding: 10px 12px;
		border-radius: var(--r2);
		background: var(--well);
		border: 1px solid var(--rule);
		display: grid;
		gap: 6px;
	}
	.predict p,
	.task p {
		margin: 0;
	}
	.task.done {
		border-color: var(--live);
		background: var(--live-soft);
	}
	.tag {
		font-size: var(--t-sm);
		font-weight: var(--w-title);
		color: var(--muted);
		margin-right: 4px;
	}
	.opts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.opts .btn {
		min-height: 34px;
		font-size: var(--t-sm);
	}
	.opts .btn.right {
		border-color: var(--live);
		background: var(--live-soft);
	}
	.opts .btn.wrong {
		border-color: var(--div-neg);
	}
	.small {
		font-size: var(--t-xs);
	}
	.view {
		min-width: 0;
		transition: opacity 0.2s;
	}
	.view.dim {
		opacity: 0.25;
		filter: saturate(0.4);
	}
	.terms {
		margin: 0;
	}
	.term {
		text-decoration: underline dotted;
		text-decoration-color: var(--ink);
		text-underline-offset: 3px;
		text-decoration-thickness: 1.5px;
	}
	.cmp {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
		gap: 12px;
	}
	.cmp-l {
		margin: 0 0 4px;
		font-size: var(--t-sm);
		font-weight: 600;
	}
</style>
