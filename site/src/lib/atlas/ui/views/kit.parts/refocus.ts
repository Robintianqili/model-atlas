// K4, promoted by the rule of two (kalman-filter, gaussian-process, hmm and ecg-autoencoder each carried the same
// snippet): a control that edits the learner's input (setInput) re-creates the lesson stage, so the slider being moved
// lost the keyboard focus and held arrow keys dropped presses. Before setInput, remember which control moved; the new
// stage gives that control the focus back.
//
//   import { refocus, rememberFocus } from '$lib/atlas/ui/views/kit.parts/refocus.ts';
//   <div use:refocus>
//     <input type="range" data-refocus="q" oninput={(e) => { rememberFocus('q'); setInput({ ...raw, q: +e.currentTarget.value }); }} />
//   </div>
//
// Browser-only (an action); pure apart from the DOM it is given. The memory is module-level on purpose: it outlives the
// stage that is torn down. It expires after `ms` (3 s), so a later visit to the step never steals the focus.

let pending: { key: string; at: number } | null = null;

/** Remember that the control with data-refocus="<key>" is changing the input (call it right before setInput). */
export function rememberFocus(key: string): void {
	pending = { key, at: Date.now() };
}

/** Give the remembered control inside `root` the focus back (once; within `ms` of rememberFocus). True when it did. */
export function restoreFocus(root: ParentNode | null | undefined, ms = 3000): boolean {
	const p = pending;
	if (!root || !p) return false;
	pending = null;
	if (Date.now() - p.at > ms) return false;
	const el = root.querySelector<HTMLElement>(`[data-refocus="${p.key.replace(/["\\]/g, '\\$&')}"]`);
	if (!el) return false;
	el.focus({ preventScroll: true });
	return true;
}

/** Svelte action: on mount, focus the remembered control inside this element (see the header). */
export function refocus(node: HTMLElement): { destroy(): void } {
	const raf = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (f: () => void) => setTimeout(f, 0);
	raf(() => restoreFocus(node));
	return { destroy() {} };
}
