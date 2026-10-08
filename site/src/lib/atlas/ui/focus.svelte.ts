// FocusBus (design §2.1): linked highlights between the views of one stage. The stage sets the step's focus list; a view
// that the learner hovers or taps publishes `hover`, and every other view of the same stage can highlight the same cells.
//   const bus = focusBus(rt, 'lesson');   // StageHost (the lesson) / AutoModelPage ('explore'): one bus per runtime and scope
//   const bus = focusBus(rt, (props.focusScope as string) ?? 'lesson');   // a view (explore stages pass focusScope 'explore')
//   bus.hover, bus.setHover({ field: 'out', index: [2, 5, 7] }), bus.of('out')
// Buses are keyed by the runtime object (every StageProps carries `rt`), not by Svelte context: views load lazily, and
// avoiding getContext/setContext keeps Svelte's context helpers out of the chunk every page shares (page-shell budget).
import type { Focus } from '../types/process.ts';

export type FocusScope = 'lesson' | 'explore' | (string & {});

export class FocusBus {
	/** the step's own focus list (from StepDef.focus) */
	step = $state.raw<Focus[]>([]);
	/** what the learner points at right now, in any view of this stage */
	hover = $state.raw<Focus | null>(null);

	setStep(f: Focus[]): void {
		this.step = f;
	}

	setHover(f: Focus | null): void {
		this.hover = f;
	}

	/** every focus of a field (step focus first, then the hover) */
	of(field: string): Focus[] {
		const out = this.step.filter((f) => f.field === field);
		if (this.hover && this.hover.field === field) out.push(this.hover);
		return out;
	}
}

const buses = new WeakMap<object, Map<string, FocusBus>>();

/** The bus of a runtime and scope (created on first use). */
export function focusBus(rt: object, scope: FocusScope = 'lesson'): FocusBus {
	let m = buses.get(rt);
	if (!m) buses.set(rt, (m = new Map()));
	let b = m.get(scope);
	if (!b) m.set(scope, (b = new FocusBus()));
	return b;
}
