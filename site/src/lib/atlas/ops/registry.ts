// Op registry (design §4.3): createRegistry(...ops) -> OpRegistry.
//   get('dense') -> { def: linear, params: {} }   alias params are returned so the executor merges them UNDER layer params
//   Registration order matters only for clashes: a later op with the same name (or alias) replaces an earlier one
//   (a package's ops.ts lists [...CORE_OPS, ...local], so local ops may shadow core ones on purpose).
import type { OpDef, OpRegistry, ResolvedOp } from '../types/index.ts';

export function createRegistry(...ops: (OpDef | OpDef[])[]): OpRegistry {
	const byName = new Map<string, ResolvedOp>();
	for (const def of ops.flat() as OpDef[]) {
		if (!def || typeof def.op !== 'string') throw new Error('createRegistry: every entry must be an OpDef with an `op` name');
		byName.set(def.op, { def, params: {} });
		for (const [alias, params] of Object.entries(def.aliases ?? {})) byName.set(alias, { def, params: { ...(params as Record<string, unknown>) } });
	}
	return {
		get(op: string): ResolvedOp {
			const r = byName.get(op);
			if (!r) throw new Error(`op '${op}' is not registered (known: ${[...byName.keys()].sort().join(', ')})`);
			return { def: r.def, params: { ...r.params } };
		},
		has: (op: string) => byName.has(op),
		/** every registered name, aliases included, sorted */
		list: () => [...byName.keys()].sort()
	};
}

/** Accept an OpRegistry or a plain list of OpDefs (PackageModules.ops). */
export function toRegistry(ops: OpRegistry | OpDef[]): OpRegistry {
	return Array.isArray(ops) ? createRegistry(...ops) : ops;
}

/** The parameters an op runs with: declared attr defaults, then alias params, then the layer's params. */
export function mergeParams(def: OpDef, aliasParams: Record<string, unknown>, layerParams: Record<string, unknown> | undefined): Record<string, unknown> {
	const p: Record<string, unknown> = {};
	for (const [k, a] of Object.entries(def.attrs)) if (a.default !== undefined) p[k] = a.default;
	Object.assign(p, aliasParams);
	if (layerParams) for (const [k, v] of Object.entries(layerParams)) if (v !== undefined) p[k] = v;
	return p;
}
