// fakeRuntime(partial) (design §4.7): an ExplainerRuntime from hand-made data, for view demos (ViewMeta.demo), the
// gallery and tests. Everything it serves is illustrative: sources made from `tapes` and `trace` carry provenance
// 'illustrative', and nothing is fetched or run.
import type { Trace } from '../types/core.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { Cursor, Tape } from '../types/process.ts';
import type { ExplainerRuntime, SourceState } from '../types/runtime.ts';
import { builtinFact, type FactValue } from '../lesson/facts.ts';
import { traceToTape } from '../process/adapters.ts';
import { resolveCursor } from './selection.ts';

export interface FakeInit extends Partial<ExplainerRuntime> {
	/** values for rt.fact(path), checked before the built-in paths */
	facts?: Record<string, FactValue>;
	/** process-like sources: { name: tape } (kind 'process', provenance 'illustrative') */
	tapes?: Record<string, Tape>;
	/** a trace for the source named `traceSource` (default 'fwd'); needs `spec` for the tape */
	trace?: Trace;
	traceSource?: string;
}

export function fakeRuntime(p: FakeInit = {}): ExplainerRuntime {
	const sources: Record<string, SourceState> = { ...(p.sources ?? {}) };
	for (const [name, tape] of Object.entries(p.tapes ?? {})) sources[name] = { kind: 'process', status: 'ready', provenance: 'illustrative', tape };
	if (p.trace) {
		const spec = p.spec ?? ({ slug: p.slug ?? 'fake', version: 'demo', layers: [] } as unknown as AtlasSpec);
		sources[p.traceSource ?? 'fwd'] = { kind: 'trace', status: 'ready', provenance: 'illustrative', trace: p.trace, tape: traceToTape(spec, p.trace, { provenance: 'illustrative' }), model: p.model ?? null };
	}
	const rt: ExplainerRuntime = {
		slug: p.slug ?? 'fake',
		spec: p.spec ?? null,
		model: p.model ?? null,
		input: p.input ?? null,
		pre: p.pre ?? null,
		sources,
		sel: { ...(p.sel ?? {}) },
		fact(path: string) {
			if (p.facts && path in p.facts) return p.facts[path];
			if (p.fact) return p.fact(path);
			return builtinFact({ spec: rt.spec, model: rt.model, sources: rt.sources, sel: rt.sel }, path);
		},
		num(path: string) {
			const v = rt.fact(path);
			return typeof v === 'number' ? v : NaN;
		},
		async ensure(source: string, c: Cursor) {
			if (p.ensure) return p.ensure(source, c);
			const st = rt.sources[source];
			if (!st) throw new Error(`fakeRuntime: unknown source '${source}'`);
			return resolveCursor(st.tape, c).index;
		},
		setInput(raw: unknown) {
			if (p.setInput) return p.setInput(raw);
			rt.input = raw;
		}
	};
	return rt;
}
