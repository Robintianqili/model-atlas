/**
 * Page-wide state for the tokenization page: the Web Worker client and the loading status of the
 * real tokenizers. The page calls startClient() in onMount and stopClient() when it unmounts.
 */
import { base } from '$app/paths';
import { TokClient } from './client';
import { REAL, REAL_BY_ID, type RealId } from './tokenizers';

export type LoadStatus = 'idle' | 'loading' | 'ready' | 'error';

export const real = $state<{
	status: Record<RealId, LoadStatus>;
	ms: Partial<Record<RealId, number>>;
	error: Partial<Record<RealId, string>>;
	/** bumps whenever a tokenizer becomes ready */
	version: number;
}>({ status: { gpt2: 'idle', bert: 'idle', cl100k: 'idle', qwen: 'idle' }, ms: {}, error: {}, version: 0 });

export const conn = $state<{ client: TokClient | null }>({ client: null });

export function startClient(): TokClient {
	if (!conn.client) conn.client = new TokClient();
	return conn.client;
}

export function stopClient() {
	conn.client?.destroy();
	conn.client = null;
	for (const r of REAL) if (real.status[r.id] !== 'error') real.status[r.id] = 'idle';
}

export function fileUrl(id: RealId): string {
	return new URL(`${base}/pages/tokenization/tok/${REAL_BY_ID[id].file}`, location.href).href;
}

export async function loadReal(id: RealId): Promise<void> {
	const c = conn.client;
	if (!c || real.status[id] === 'loading' || real.status[id] === 'ready') return;
	real.status[id] = 'loading';
	real.error[id] = undefined;
	try {
		const r = await c.load(id, fileUrl(id));
		if (conn.client !== c) return;
		real.ms[id] = r.ms;
		real.status[id] = 'ready';
		real.version++;
	} catch (e) {
		if (conn.client !== c) return; // the page was left while loading (stopClient reset the status)
		real.status[id] = 'error';
		real.error[id] = e instanceof Error ? e.message : String(e);
	}
}

export function loadAll() {
	for (const r of REAL) void loadReal(r.id);
}

export function readyIds(): RealId[] {
	return REAL.filter((r) => real.status[r.id] === 'ready').map((r) => r.id);
}
