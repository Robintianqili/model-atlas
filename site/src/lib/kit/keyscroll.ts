/**
 * Keyboard access for scroll containers (WCAG 2.1.1; the axe rule scrollable-region-focusable).
 *
 * A box with `overflow: auto | scroll` whose content is wider (or taller) than the box can only be scrolled with a
 * mouse or a finger unless the keyboard can focus it. `keyScroll` gives such a box `tabindex="0"` while it actually
 * overflows and has nothing focusable inside, and takes the tab stop away again when the content fits (a wide formula
 * on a phone scrolls; the same formula on a desktop does not, and adds no tab stop there).
 *
 *   <pre use:keyScroll>…</pre>                                  the element itself
 *   <div use:keyScroll={'.katex-display'}>{@html html}</div>    the element and every match inside it
 *
 * Content changes ({@html}, a new step) and size changes are picked up (MutationObserver, ResizeObserver). All the
 * boxes of the page are checked together, at most once per frame: every size is read first (one layout), then the
 * tab stops are written, so a lesson step that mounts many views never forces a layout per view (AK603). A `tabindex`
 * the author wrote is never touched. Browser-only: the action runs after hydration.
 */

const MARK = 'data-keyscroll';
const FOCUSABLE = 'a[href], button, input, select, textarea, summary, iframe, [contenteditable="true"], [tabindex]:not([tabindex="-1"])';
const SCROLLS = /^(auto|scroll)$/;

/** Does `el` scroll (overflow auto/scroll and content larger than the box)? */
export function overflows(el: HTMLElement): boolean {
	const cs = getComputedStyle(el);
	// no tolerance, as axe: a formula 1 px wider than its box is already a scroll region
	return (SCROLLS.test(cs.overflowX) && el.scrollWidth > el.clientWidth) || (SCROLLS.test(cs.overflowY) && el.scrollHeight > el.clientHeight);
}

/** Add or remove the keyScroll tab stop on one element, given whether it overflows (no layout read). */
function apply(el: HTMLElement, over: boolean): void {
	const ours = el.hasAttribute(MARK);
	if (!ours && el.hasAttribute('tabindex')) return;
	const need = over && !el.querySelector(FOCUSABLE);
	if (need && !ours) {
		el.setAttribute('tabindex', '0');
		el.setAttribute(MARK, '');
	} else if (!need && ours) {
		el.removeAttribute('tabindex');
		el.removeAttribute(MARK);
	}
}

/** Add or remove the keyScroll tab stop on one element, now. */
export function syncKeyScroll(el: HTMLElement): void {
	apply(el, overflows(el));
}

// one frame for every keyScroll on the page: collect the targets, read all sizes, then write
const due = new Set<() => HTMLElement[]>();
let frame = 0;
function flush(): void {
	frame = 0;
	const seen = new Set<HTMLElement>();
	for (const collect of due) for (const el of collect()) if (el.isConnected) seen.add(el);
	due.clear();
	// document order, then innermost first: an outer box that holds an inner tab stop needs none of its own
	const els = [...seen].sort((a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
	const over = els.map(overflows);
	for (let i = els.length - 1; i >= 0; i--) apply(els[i], over[i]);
}
function request(collect: () => HTMLElement[]): void {
	due.add(collect);
	if (!frame) frame = requestAnimationFrame(flush);
}

/** Svelte action: see the file comment. `selector` also covers matching descendants. */
export function keyScroll(node: HTMLElement, selector?: string) {
	let sel = selector;
	const watched = new Set<Element>();
	const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => request(collect));
	function collect(): HTMLElement[] {
		const targets: HTMLElement[] = [node, ...(sel ? Array.from(node.querySelectorAll<HTMLElement>(sel)) : [])];
		for (const el of watched)
			if (!el.isConnected) {
				ro?.unobserve(el);
				watched.delete(el);
			}
		for (const el of targets)
			if (!watched.has(el)) {
				ro?.observe(el);
				watched.add(el);
			}
		return targets;
	}
	const schedule = () => request(collect);
	const mo = new MutationObserver(schedule);
	mo.observe(node, { childList: true, subtree: true, characterData: true });
	// a web font that arrives late changes how wide the content is, not the box
	const fonts = document.fonts as FontFaceSet | undefined;
	fonts?.addEventListener?.('loadingdone', schedule);
	schedule();
	return {
		update(s?: string) {
			sel = s;
			schedule();
		},
		destroy() {
			due.delete(collect);
			ro?.disconnect();
			mo.disconnect();
			fonts?.removeEventListener?.('loadingdone', schedule);
		}
	};
}
