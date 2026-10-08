<!--
  TapeScrubber: play / step / scrub over a range of a tape's frames (a scan's positions, a process run, a training run),
  built on the kit StepPlayer with its own counter ("Position i of n" / "Frame i of n"), so it is never mistaken for the
  lesson's steps. `index` (bindable) is the tape index shown. Coarse training tapes alternate 'step' and 'snapshot' frames:
  pass `phases` to keep only some of them; jump marks merge short alternating segments (scrub.ts).
  Reduced motion: no autoplay; short ranges get the StepPlayer's clickable step list, long ones the jump marks.
  K1: `start` is the tape index the step's narration describes (its cursor). Until Play animates the range, the scrubber
  rests there (it used to show the range's first frame, so a narration about the end of a run sat next to frame 0, always
  under reduced motion); when autoplay begins it plays the range from its start. A scan with `unit` counts in that unit
  ("Patch 3 of 16", marks "patch 2").
-->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Cursor, Tape } from '../types/process.ts';
	import { frameLabel, playFrames, positionLabel, scrubMarks, unitText } from './scrub.ts';
	import type { I18n } from '../types/core.ts';
	import { STR, say } from './strings.ts';

	interface Props {
		tape: Tape;
		from?: Cursor;
		to?: Cursor;
		phases?: string[];
		/** the tape index shown (bindable) */
		index?: number;
		/** ms per frame at 1× */
		interval?: number;
		autoplay?: boolean;
		/** 'position' for scans (cells of one layer), 'frame' for process and training tapes */
		counter?: 'position' | 'frame';
		label?: string;
		/** the version of the runtime (a live tape grew) */
		version?: number;
		onend?: () => void;
		playing?: boolean;
		/** what the marks count (StepDef.play.unit / by): "episode 12" instead of "step 12" */
		unit?: I18n;
		by?: 'step' | 'sub';
		/** the tape index to rest on until autoplay starts (the step's cursor; K1) */
		start?: number;
	}
	let {
		tape,
		from = 'first',
		to = 'last',
		phases,
		index = $bindable(0),
		interval = 120,
		autoplay = false,
		counter = 'frame',
		label,
		version = 0,
		onend,
		playing = $bindable(false),
		unit,
		by,
		start
	}: Props = $props();

	const frames = $derived.by(() => {
		void version;
		return playFrames(tape, from, to, phases);
	});
	let step = $state(0);
	// a growing live tape keeps the position; a shorter one clamps it
	$effect(() => {
		const n = frames.length;
		if (step > n - 1) step = Math.max(0, n - 1);
	});
	// K1: rest on the step's own frame until autoplay; autoplay (re)starts the range from its first frame
	let placed = false;
	let wasAuto = false;
	$effect(() => {
		const a = autoplay;
		const n = frames.length;
		if (a && !wasAuto) step = 0;
		wasAuto = a;
		if (placed || a || n === 0 || start === undefined || start < 0) return;
		let k = 0;
		for (let j = 0; j < n; j++) if (frames[j] <= start) k = j;
		step = k;
		placed = true;
	});
	$effect(() => {
		const fi = frames[step];
		if (fi !== undefined && fi !== index) index = fi;
	});
	const count = $derived({ unit, by });
	// a scan with a unit word counts its places in that unit (K1, K-R4): "patch 2" (from 0), "Patch 3 of 16"
	const unitPos = (k: number): I18n | null => (counter === 'position' && unit ? unitText(unit, k) : null);
	// short ranges list every frame: a scan names its cells ("feature 12"), a run its frames ("Update · episode 3")
	const labels = $derived(
		frames.length > 0 && frames.length <= 40 ? frames.map((fi, k) => t(unitPos(k) ?? (counter === 'position' ? positionLabel(tape, fi) : null) ?? frameLabel(tape, fi, count))) : undefined
	);
	const marks = $derived(frames.length > 40 ? scrubMarks(tape, frames, { unit: counter, count }).map((mk) => ({ ...mk, label: unitPos(mk.at) ?? mk.label })) : []);
	const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
	const counterText = $derived(
		counter === 'position' && unit
			? // K2: a unit template ('第 {n} 秒') places the counter itself: "第 {i} 秒 / {n}"
				{ en: unit.en.includes('{n}') ? `${cap(unit.en.replace('{n}', '{i}'))} of {n}` : STR.unitOf.en.replace('{unit}', cap(unit.en)), zh: (unit.zh ?? unit.en).includes('{n}') ? `${(unit.zh ?? unit.en).replace('{n}', '{i}')} / {n}` : STR.unitOf.zh.replace('{unit}', unit.zh ?? unit.en) }
			: counter === 'position'
				? STR.positionOf
				: STR.frameOf
	);
</script>

{#if frames.length > 1}
	<div class="scrub">
		<StepPlayer count={frames.length} bind:step bind:playing {interval} {autoplay} {labels} counter={counterText} {label} {onend} />
		{#if marks.length > 1}
			<div class="marks" role="group" aria-label={say('jumpTo')}>
				<span class="muted">{say('jumpTo')}:</span>
				{#each marks as mk (mk.at)}
					<button type="button" class="chip" aria-pressed={step >= mk.at && (marks.find((x) => x.at > mk.at)?.at ?? Infinity) > step} onclick={() => (step = mk.at)}>{t(mk.label)}</button>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<style>
	.scrub {
		display: grid;
		gap: 6px;
		min-width: 0;
	}
	.marks {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 6px;
		align-items: center;
		font-size: var(--fs-xs);
	}
	.marks .chip {
		cursor: pointer;
		font-size: var(--fs-xs);
		padding: 2px 8px;
	}
	.marks .chip[aria-pressed='true'] {
		background: var(--accent-soft);
		border-color: var(--accent);
	}
</style>
