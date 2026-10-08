/**
 * State shared between the sections of the transfer-learning page (module-level runes).
 *   lab.liveRun   the learner's latest head-only run in Part 1 (shown on the Part 2 chart)
 *   lab.part2N    the training-set size selected on the Part 2 chart (Try-this and misconception links set it)
 *   lab.part2Seeds whether the Part 2 chart shows every seed as a dot (the second misconception's link turns it on)
 */
export interface LiveRun {
	/** examples per class (preset drawings + own drawings of the preset pair) */
	nA: number;
	nB: number;
	/** accuracy on the held-out test drawings (0..1), or null for the learner's own pair */
	testAcc: number | null;
	testN: number;
	/** true when the classes are the preset pair (ladder vs envelope) */
	preset: boolean;
}

/** What the Part 2 chart shows; the guided lesson keeps its own copy, so it never moves the explore view. */
export interface CurveSel {
	part2N: number;
	part2Zoom: boolean;
	part2Seeds: boolean;
}

export const lab = $state<{ liveRun: LiveRun | null } & CurveSel>({
	liveRun: null,
	part2N: 1,
	part2Zoom: false,
	part2Seeds: false
});

/** Scroll to an element id (respects reduced motion) and move focus there for keyboard users. */
export function goTo(id: string) {
	if (typeof document === 'undefined') return;
	const el = document.getElementById(id);
	if (!el) return;
	const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
	el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
	if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '-1');
	el.focus({ preventScroll: true });
}
