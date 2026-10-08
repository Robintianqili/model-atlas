/**
 * Types for the model asset spec (CONTRACTS.md "Model asset format") and small helpers that
 * turn it into kit inputs. The model streams own the loaders (src/lib/models/<slug>.ts);
 * this file only describes the shape the kit reads.
 */
import type { I18n } from '../i18n/types.ts';
import type { LayerNode } from './types.ts';

export interface SpecLayer {
	id: string;
	op: string;
	label: I18n;
	in?: string[];
	outShape?: number[];
	params?: Record<string, unknown>;
}

export interface ModelSpec {
	slug: string;
	name?: I18n;
	version?: string;
	params?: number;
	precision?: string;
	input?: { kind: string; shape: number[]; notes?: string };
	tensors?: { name: string; shape: number[]; offset: number; length: number }[];
	layers?: SpecLayer[];
	training?: { dataset?: string; steps?: number; metrics?: Record<string, number | string> };
	parity?: { against: string; max_abs_diff: number; checked: string };
	[extra: string]: unknown;
}

/** spec.layers -> LayerNode[] for ModelOverview and Inspector. */
export function specToLayers(spec: Pick<ModelSpec, 'layers'>, input?: LayerNode): LayerNode[] {
	const nodes: LayerNode[] = (spec.layers ?? []).map((l) => ({ id: l.id, label: l.label, op: l.op, outShape: l.outShape }));
	return input ? [input, ...nodes] : nodes;
}

/** Fetch and parse a model's spec.json. `baseUrl` is e.g. `${base}/models/tinycnn`. */
export async function fetchSpec(fetchFn: typeof fetch, baseUrl: string): Promise<ModelSpec> {
	const res = await fetchFn(`${baseUrl}/spec.json`);
	if (!res.ok) throw new Error(`spec.json: HTTP ${res.status}`);
	return (await res.json()) as ModelSpec;
}
