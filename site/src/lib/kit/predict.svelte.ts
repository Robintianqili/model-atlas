/**
 * Shared state for "predict first" questions: the answer chosen at the top of a page is shown
 * again at the checkpoint. Kept in memory and in sessionStorage (wrapped in try/catch).
 */
const PREFIX = 'atlas.predict.';

export const predictions = $state<Record<string, number>>({});

export function savePrediction(id: string, choice: number) {
	predictions[id] = choice;
	try {
		sessionStorage.setItem(PREFIX + id, String(choice));
	} catch {
		/* storage blocked: memory only */
	}
}

export function clearPrediction(id: string) {
	delete predictions[id];
	try {
		sessionStorage.removeItem(PREFIX + id);
	} catch {
		/* ignore */
	}
}

/** Restore a saved choice (call from onMount / $effect, never during SSR). */
export function restorePrediction(id: string) {
	if (id in predictions) return;
	try {
		const v = sessionStorage.getItem(PREFIX + id);
		if (v !== null && !Number.isNaN(Number(v))) predictions[id] = Number(v);
	} catch {
		/* ignore */
	}
}
