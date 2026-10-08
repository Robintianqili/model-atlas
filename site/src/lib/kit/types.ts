/**
 * Shared kit types. Model pages import these to talk to kit components.
 */
import type { Snippet } from 'svelte';
import type { I18n, Text } from '../i18n/types.ts';
import type { ColorScope, Domain, ScaleKind } from './color.ts';

export type { I18n, Text, ColorScope, Domain, ScaleKind };

/** Where the numbers on a panel come from (CONTRACTS.md rule 2). */
export type Provenance = 'live' | 'precomputed' | 'illustrative';

/** Tensor roles and training status markers (brief §4.7). */
export type Role = 'param' | 'grad' | 'activation' | 'cache' | 'optimizer' | 'frozen' | 'trainable' | 'lora';

/** One traced tensor, as returned in `trace.t[id]` by src/lib/models/<slug>.ts (CONTRACTS.md). */
export interface TensorLike {
	shape: number[];
	data: Float32Array | ArrayLike<number>;
}

/** A cell address in a grid. */
export interface Cell {
	row: number;
	col: number;
}

/** A cell plus its value (hover / focus callbacks). */
export interface CellValue extends Cell {
	value: number;
}

/** Rectangle drawn on top of a TensorGrid (e.g. the kernel window). Rows/cols in cells. */
export interface Highlight {
	row: number;
	col: number;
	/** height in cells (default 1) */
	h?: number;
	/** width in cells (default 1) */
	w?: number;
	style?: 'solid' | 'dashed';
	/** CSS color; defaults to the theme's highlight ink */
	color?: string;
	label?: string;
}

/** A layer node for ModelOverview / Inspector (a subset of spec.json `layers`). */
export interface LayerNode {
	id: string;
	label: Text;
	/** op name, e.g. "conv2d", "relu", "softmax" */
	op?: string;
	/** output shape, shown as [8×26×26] */
	outShape?: (number | string)[];
	/** training status marker */
	status?: 'frozen' | 'trainable';
}

/** A probability item for ProbBars. */
export interface ProbItem {
	label: string;
	p: number;
	/** token id / class index, shown as a sub-label */
	id?: string | number;
	/** categorical color for class outputs */
	color?: string;
}

/** A token chip for TokenChips. */
export interface TokenItem {
	text: string;
	id?: number;
	/** raw bytes (byte-level BPE); shown as hex in the tooltip */
	bytes?: number[];
	special?: boolean;
	/** optional tone: 'generated' (sampled by the model, magenta) or 'muted' (e.g. outside the context window) */
	tone?: 'generated' | 'muted';
}

/** A "predict first" question (shown at the top of a page and revisited at the checkpoint). */
export interface PredictQuestion {
	/** unique per page, used as a storage key */
	id: string;
	question: I18n;
	options: I18n[];
	/** index of the correct option */
	answer: number;
	explain: I18n;
}

/** What a Walkthrough passes to each step's stage snippet. */
export interface WalkContext {
	/** index of the current step (0-based) */
	index: number;
	step: WalkStep;
	/** the lesson is playing through (auto-advance on) */
	playing: boolean;
	/** the lesson is on screen right now */
	visible: boolean;
	/** the stage has been in view at least once since this step was entered (latched): start animations on this */
	shown: boolean;
	/** prefers-reduced-motion: never autoplay */
	reduced: boolean;
	/** call when this step's animation has finished (Play then moves on after `linger` ms) */
	done: () => void;
}

/** One step of a guided lesson (kit Walkthrough). */
export interface WalkStep {
	/** stable id, used in the deep link #<lessonId>-<step id> */
	id: string;
	title: Text;
	/** 2–4 short sentences: what you see, what changed, why it matters. A function may read live values. */
	narration: Text | (() => Text);
	/** optional "Look at …" hint */
	look?: Text | (() => Text);
	/** the one focused visualization for this step (falls back to the Walkthrough's `stage` snippet) */
	stage?: Snippet<[WalkContext]>;
	/** called when the step becomes current (after the Walkthrough is `ready`), e.g. select a layer */
	onEnter?: () => void;
	onLeave?: () => void;
	/** Play: advance after this many ms (default from the narration length, 6–16 s) */
	duration?: number;
	/** Play: wait for the stage to call ctx.done() (at most `duration` ms, default 45 s), then `linger` ms */
	waitForDone?: boolean;
	/** ms to stay on the finished animation before moving on (default 2500) */
	linger?: number;
	/** Play: what the step waits for, shown while Play waits (default: "Playing this step's animation…"); e.g. the
	 *  learner's answer to a question or a task (DevKit K0) */
	waitText?: Text | (() => Text);
	/** Play: while Play waits for this step (waitForDone), presses inside the stage are the learner doing what the step
	 *  asks (answering, doing the task) and do not pause Play (DevKit K0; default false: any press pauses Play) */
	keepPlay?: boolean;
	/** true while the step's numbers are still being computed: its screen-reader announcement waits until it is false
	 *  (at most 10 s; DevKit K1). Default: announced on entering. */
	pending?: () => boolean;
}
