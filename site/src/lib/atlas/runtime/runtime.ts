// createRuntime(pkg, { fetch, base, input?, runner? }) (design §4.7): the pure-TS runtime one explainer page runs on.
// The browser wraps the same object in $state.raw (ui/runtime.svelte.ts, W1-D1): this module never mutates a published
// value in place. rt.sources, rt.sel, rt.pre and every SourceState are replaced by new objects, and onChange() fires.
//
// Sources (manifest.sources, default { fwd: { kind: 'trace' } }; 'scan:<layer>' is implicit):
//   trace       computed on every setInput (synchronous); skipped when the input is empty and has no numbers
//   scan        built on first ensure() from its `of` trace; rebuilt when the trace or the selection-derived unit changes
//   train       process 'train:<trainer>' opened through the runner on first ensure(). setInput restarts it on the new
//               input (K0): { type: 'input' } then { type: 'reset' } on the same tape, so the run starts again from fresh
//               weights (same seed) and keeps the options the learner set ('set' controls); a finished run used to keep
//               its old weights. To swap the data without restarting, send { type: 'input' } to the tape yourself.
//               K1: not run while the adapter says the input is empty (unless runOnEmpty): its steps show the empty note.
//   process     opened through the runner on first ensure(); setInput closes it (reopened on the next ensure), unless
//               its input(rt) gives the same input as before (K1: a process that ignores the learner's input keeps its
//               run). run: 'main' records on the main thread (K1; the engine runner used to send every process to the
//               worker); after: '<source>' opens it once that source has run to its end (K1), so input(rt) can read it.
//   precomputed a train or process source with `precomputed` (phones) uses its tape file for the first input only; after
//               setInput it runs live (the file was recorded on the start input)
//   tape        a precomputed tape file (<stem>.tape.json + .tape.bin), loaded on first ensure()
//   live-model  a kind-A trace on a train source's latest weights (K1: at: 'first', its starting weights); needs the train ProcessDef to carry
//               liveSpec(state, opt) -> { spec, weights } (train/trainer.ts trainingProcess adds it since integration 1a;
//               it calls the model's toSpec()). A train source that has not run yet is first run to 'last' (a deep link
//               straight to a live-model step works and shows the same numbers as stepping in).
import type { AnyTensor, FetchLike, I18n, Tensor } from '../types/core.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { GraphModel, UnitExplanation } from '../types/op.ts';
import type { Cursor, LiveTape, ProcessDef, Tape, TapeHeader } from '../types/process.ts';
import type { ExplainerRuntime, PackageModules, Preprocessed, SelectionState, SourceDef, SourceRunner, SourceState } from '../types/runtime.ts';
import type { InputAdapter } from '../types/explainer.ts';
import { fromParts, loadGraph, normalizeSpec } from './seams.ts';
import { localRunner } from './local.ts';
import { sourceDef, sourcesOf, SCAN_PREFIX } from '../lesson/define.ts';
import { builtinFact, outputRow, type FactCtx, type FactsFile } from '../lesson/facts.ts';
import { validateFacts } from '../schema/facts.ts';
import { errorsOf } from '../schema/spec.ts';
import { scanTape, traceToTape } from '../process/adapters.ts';
import { loadTape } from '../process/io.ts';
import { TapeStore } from '../process/tape.ts';
import { resolveCursor } from './selection.ts';

export interface RuntimeOpts {
	fetch: FetchLike;
	/** URL (browser) or path prefix (Node fsFetch) of the static folder; '' for the site root */
	base: string;
	input?: unknown;
	runner?: SourceRunner;
	/** use a source's `precomputed` tape instead of running it (phones) */
	preferPrecomputed?: boolean;
}

export interface AtlasRuntime extends ExplainerRuntime {
	readonly pkg: PackageModules;
	readonly dir: string;
	readonly facts: FactsFile | null;
	/** JSON files from manifest.assets.auto (besides spec, weights and facts) and the presets file */
	readonly assets: Record<string, unknown>;
	readonly presets: { id: string; label: I18n; raw: unknown }[];
	readonly adapter: InputAdapter | null;
	/** problems met while loading (missing files, invalid facts.json); the page shows them in dev */
	readonly warnings: string[];
	/** memo of unit explanations for the current input (replaced by setInput) */
	explainMemo: WeakMap<object, Map<string, UnitExplanation | null>>;
	onChange(fn: () => void): () => void;
	close(): void;
}

const join = (...parts: string[]) => parts.filter((p) => p !== '').map((p, i) => (i === 0 ? p.replace(/\/+$/, '') : p.replace(/^\/+|\/+$/g, ''))).join('/');

async function getJson(fetch: FetchLike, url: string): Promise<unknown> {
	const r = await fetch(url);
	if (r.ok === false) throw new Error(`could not load ${url} (${r.status ?? 'error'})`);
	return r.json();
}

function numel(t: AnyTensor): number {
	return t.data.length;
}

export async function createRuntime(pkg: PackageModules, o: RuntimeOpts): Promise<AtlasRuntime> {
	const m = pkg.manifest;
	const defs = sourcesOf(m);
	const dir = join(o.base, m.assets?.dir ?? `models/${m.slug}`);
	const runner = o.runner ?? localRunner;
	const warnings: string[] = [];
	const auto = m.assets?.auto ?? [];
	const needsModel = m.kinds.includes('trace') && Object.values(defs).some((d) => d.kind === 'trace' && !d.assets);

	let model: GraphModel | null = null;
	let spec: AtlasSpec | null = null;
	if (needsModel) {
		try {
			model = await loadGraph(o.fetch, dir, pkg.ops);
			spec = model.spec;
		} catch (e) {
			warnings.push(e instanceof Error ? e.message : String(e));
		}
	} else if (auto.includes('spec.json')) {
		try {
			spec = normalizeSpec(await getJson(o.fetch, join(dir, 'spec.json')));
		} catch (e) {
			warnings.push(e instanceof Error ? e.message : String(e));
		}
	}
	const extraModels: Record<string, GraphModel> = {};
	for (const [name, d] of Object.entries(defs)) {
		if (d.kind !== 'trace' || !d.assets) continue;
		try {
			extraModels[name] = await loadGraph(o.fetch, join(o.base, d.assets), pkg.ops);
		} catch (e) {
			warnings.push(`${name}: ${e instanceof Error ? e.message : String(e)}`);
		}
	}
	let facts: FactsFile | null = null;
	const assets: Record<string, unknown> = {};
	for (const f of auto) {
		if (f === 'spec.json' || !f.endsWith('.json')) continue;
		try {
			const v = await getJson(o.fetch, join(dir, f));
			if (f === 'facts.json') {
				const bad = errorsOf(validateFacts(v)); // the one facts validator (W1-A schema/facts.ts) since integration 1a
				if (bad.length) warnings.push(`facts.json: ${bad.map((i) => `${i.path}: ${i.message}`).join('; ')}`);
				else facts = v as FactsFile;
			} else assets[f] = v;
		} catch (e) {
			warnings.push(e instanceof Error ? e.message : String(e));
		}
	}
	const adapter = (m.input?.adapter ? pkg.adapters[m.input.adapter] : undefined) ?? null;
	if (m.input?.adapter && !adapter) warnings.push(`adapter '${m.input.adapter}' is not registered`);
	let presets: { id: string; label: I18n; raw: unknown }[] = [];
	if (m.input?.presets) {
		try {
			const v = assets[m.input.presets] ?? (await getJson(o.fetch, join(dir, m.input.presets)));
			assets[m.input.presets] = v;
			presets = adapter?.presets?.({ [m.input.presets]: v, presets: v }) ?? [];
		} catch (e) {
			warnings.push(e instanceof Error ? e.message : String(e));
		}
	} else if (m.input?.presetsFromAdapter) {
		// K2: presets made in TS by the adapter (no file)
		try {
			presets = adapter?.presets?.({}) ?? [];
		} catch (e) {
			warnings.push(e instanceof Error ? e.message : String(e));
		}
	}

	const listeners = new Set<() => void>();
	let notifyQueued = false;
	// one notification per burst of changes (a setInput replaces many source states at once)
	const changed = () => {
		if (notifyQueued || listeners.size === 0) return;
		notifyQueued = true;
		queueMicrotask(() => {
			notifyQueued = false;
			for (const fn of listeners) fn();
		});
	};
	const live: Record<string, LiveTape> = {};
	/** the input each open process source was opened with (K1: a new learner input that leaves it unchanged keeps the run) */
	const openedWith: Record<string, unknown> = {};
	/** train sources that did not run because the input was empty: they hold an empty, finished tape (K1) */
	const skipped = new Set<string>();
	const scanKeys: Record<string, string> = {};
	const loading: Record<string, Promise<void> | undefined> = {};
	let generation = 0;

	const rt: AtlasRuntime = {
		slug: m.slug,
		spec,
		model,
		input: null,
		pre: null,
		sources: {},
		sel: {},
		pkg,
		dir,
		facts,
		assets,
		presets,
		adapter,
		warnings,
		explainMemo: new WeakMap(),
		fact(path: string) {
			// K2: top.* read the output at the package's answer row (manifest output.row)
			const ctx: FactCtx = { spec: rt.spec, model: rt.model, sources: rt.sources, defs, sel: rt.sel, facts, explainMemo: rt.explainMemo, pre: rt.pre, outputRow: (t) => outputRow(rt, t) };
			return builtinFact(ctx, path);
		},
		num(path: string) {
			const v = rt.fact(path);
			return typeof v === 'number' ? v : NaN;
		},
		async ensure(source: string, c: Cursor): Promise<number> {
			await ready(source);
			const st = rt.sources[source];
			const tape = st?.tape;
			if (!tape) return -1;
			const lt = tape as LiveTape;
			if (typeof lt.ensure === 'function' && !lt.done) {
				await lt.ensure(c);
				put(source, { ...rt.sources[source], status: lt.done ? 'ready' : 'running', progress: { at: lt.length, target: lt.length } });
			}
			return resolveCursor(tape, c).index;
		},
		setInput(raw: unknown) {
			setInput(raw);
			changed();
		},
		onChange(fn) {
			listeners.add(fn);
			return () => listeners.delete(fn);
		},
		close() {
			for (const t of Object.values(live)) t.close();
			listeners.clear();
		}
	};

	function put(name: string, st: SourceState) {
		rt.sources = { ...rt.sources, [name]: st };
		changed();
	}

	function preprocess(raw: unknown): Preprocessed | null {
		if (m.inputs?.length) {
			const parts: Preprocessed[] = [];
			const tensors: Record<string, AnyTensor> = {};
			for (const inp of m.inputs) {
				const a = pkg.adapters[inp.adapter];
				if (!a) throw new Error(`adapter '${inp.adapter}' is not registered`);
				const p = a.preprocess((raw as Record<string, unknown> | null)?.[inp.name], spec, inp.options ?? {});
				parts.push(p);
				const ks = Object.keys(p.tensors);
				if (ks.length === 1) tensors[inp.name] = p.tensors[ks[0]];
				else Object.assign(tensors, p.tensors);
			}
			return { tensors, steps: parts.flatMap((p) => p.steps), empty: parts.some((p) => p.empty), notes: parts.flatMap((p) => p.notes ?? []) };
		}
		if (!adapter) return null;
		const p = adapter.preprocess(raw, spec, m.input?.options ?? {});
		const declared = spec?.inputs?.map((i) => i.name) ?? [];
		const ks = Object.keys(p.tensors);
		if (declared.length === 1 && ks.length === 1 && ks[0] !== declared[0]) return { ...p, tensors: { [declared[0]]: p.tensors[ks[0]] } };
		return p;
	}

	function runTrace(name: string, d: Extract<SourceDef, { kind: 'trace' }>) {
		const mdl = d.assets ? extraModels[name] : model;
		if (!mdl) return put(name, { kind: 'trace', status: 'error', provenance: 'live', trace: null, tape: null, model: null, error: warnings.join('; ') || 'model not loaded' });
		const pre = rt.pre;
		const ins = pre?.tensors ?? {};
		const fed = d.inputs ? Object.fromEntries(d.inputs.filter((k) => ins[k]).map((k) => [k, ins[k]])) : ins;
		if (!pre || Object.keys(fed).length === 0 || Object.values(fed).some((t) => numel(t) === 0)) {
			return put(name, { kind: 'trace', status: 'idle', provenance: 'live', trace: null, tape: null, model: mdl });
		}
		try {
			const trace = mdl.forward(fed);
			put(name, { kind: 'trace', status: 'ready', provenance: 'live', trace, tape: traceToTape(mdl.spec, trace), model: mdl });
		} catch (e) {
			put(name, { kind: 'trace', status: 'error', provenance: 'live', trace: null, tape: null, model: mdl, error: e instanceof Error ? e.message : String(e) });
		}
	}

	function setInput(raw: unknown) {
		generation++;
		rt.input = raw;
		try {
			rt.pre = preprocess(raw);
		} catch (e) {
			rt.pre = null;
			warnings.push(e instanceof Error ? e.message : String(e));
		}
		rt.explainMemo = new WeakMap();
		const keepIfSame: [string, Extract<SourceDef, { kind: 'process' }>][] = [];
		for (const [name, d] of Object.entries(defs)) {
			if (d.kind === 'trace') runTrace(name, d);
			else if (d.kind === 'scan' || d.kind === 'live-model') put(name, { kind: d.kind, status: 'idle', provenance: 'live', tape: null });
			else if (d.kind === 'process' && live[name]) {
				// K1: decided after every trace has run again (input(rt) may read them): close it unless its input is unchanged
				if (!d.after && d.input) keepIfSame.push([name, d]);
				else closeProcess(name);
			} else if (d.kind === 'train' && skipped.has(name)) {
				// it did not run on the empty input: run it on this one when a step needs it
				skipped.delete(name);
				put(name, { kind: 'train', status: 'idle', provenance: 'live', tape: null });
			} else if (d.kind === 'train' && live[name] && rt.pre?.empty && !d.runOnEmpty) {
				// an empty input stops the run (K1): its steps show the adapter's empty note; the next input starts a fresh one
				live[name].close();
				delete live[name];
				put(name, { kind: 'train', status: 'idle', provenance: 'live', tape: null });
			} else if (d.kind === 'train' && live[name]) {
				// a new input restarts the run on it (fresh weights, same seed, the learner's options kept); the tape object stays
				// the same, so a TrainSession bound to it keeps working
				const t = live[name];
				t.control({ type: 'input', input: raw });
				t.control({ type: 'reset' });
				put(name, { kind: 'train', status: 'running', provenance: 'live', tape: t, progress: { at: 0, target: 0 } });
			} else if ((d.kind === 'train' || d.kind === 'process') && rt.sources[name]?.provenance === 'precomputed' && pkg.processes[d.kind === 'train' ? `train:${d.trainer}` : d.process]) {
				// a precomputed tape belongs to the start input: run live from now on
				put(name, { kind: d.kind, status: 'idle', provenance: 'live', tape: null });
			} else if (!rt.sources[name]) put(name, { kind: d.kind, status: 'idle', provenance: d.kind === 'tape' ? 'precomputed' : 'live', tape: null });
		}
		for (const k of Object.keys(rt.sources)) if (k.startsWith(SCAN_PREFIX) && !defs[k]) put(k, { kind: 'scan', status: 'idle', provenance: 'live', tape: null });
		for (const [name, d] of keepIfSame) if (!sameInput(openedWith[name], inputOf(d))) closeProcess(name);
	}

	function closeProcess(name: string) {
		live[name].close();
		delete live[name];
		delete openedWith[name];
		put(name, { kind: 'process', status: 'idle', provenance: 'live', tape: null });
	}

	/** a process source's input for the current runtime state (its input(rt), else the learner's raw input); undefined
	 *  when input(rt) throws */
	function inputOf(d: Extract<SourceDef, { kind: 'process' }>): unknown {
		try {
			return d.input ? d.input(rt) : rt.input;
		} catch {
			return undefined;
		}
	}

	async function loadTapeFile(stem: string, p?: ProcessDef<any, any, any>): Promise<Tape> {
		const base = stem.replace(/\.tape\.(json|bin)$/, '');
		const header = (await getJson(o.fetch, join(dir, `${base}.tape.json`))) as TapeHeader;
		const r = await o.fetch(join(dir, `${base}.tape.bin`));
		const blob = r.ok === false ? new ArrayBuffer(0) : await r.arrayBuffer();
		return loadTape(header, blob, p, { provenance: 'precomputed' });
	}

	function scanTemplate(layer: string, def: Extract<SourceDef, { kind: 'scan' }>, mdl: GraphModel): number[] | undefined {
		if (def.unit) return def.unit(rt);
		const sel: SelectionState = rt.sel;
		const out = rt.sources[def.of]?.trace?.t[layer];
		if (!out) return undefined;
		let mode: string | undefined;
		try {
			mode = mdl.ops.get(mdl.layer(layer).op).def.unit?.scan;
		} catch {
			mode = undefined;
		}
		const unit = sel.layer === layer ? sel.unit : undefined;
		const r = out.shape.length;
		if (mode === 'time') return [-1, ...(unit?.slice(1) ?? [])];
		const keep = mode === 'rows' ? r - 1 : r - 2;
		if (keep <= 0) return [];
		const ch = unit?.slice(0, keep) ?? (sel.ch !== undefined && keep === 1 ? [sel.ch] : undefined);
		return ch ?? Array.from({ length: keep }, () => 0);
	}

	async function ready(name: string): Promise<void> {
		const d = sourceDef(m, name);
		if (!d) throw new Error(`${m.slug}: unknown source '${name}' (declared: ${Object.keys(defs).join(', ') || 'none'})`);
		const cur = rt.sources[name];
		if (d.kind === 'trace') return;
		if (d.kind === 'scan') {
			const of = rt.sources[d.of];
			const mdl = (of?.model as GraphModel | null | undefined) ?? model;
			if (!of?.trace || !mdl) return put(name, { kind: 'scan', status: rt.pre?.empty ? 'idle' : 'error', provenance: 'live', tape: null, error: of?.trace ? 'no model' : `source '${d.of}' has no trace` });
			const layer = typeof d.layer === 'function' ? d.layer(rt) : d.layer;
			const tpl = scanTemplate(layer, d, mdl);
			const key = `${generation}|${layer}|${JSON.stringify(tpl ?? null)}|${d.order ?? ''}|${d.stride ?? 1}`;
			if (cur?.tape && scanKeys[name] === key) return;
			scanKeys[name] = key;
			try {
				put(name, { kind: 'scan', status: 'ready', provenance: of.provenance, tape: scanTape(mdl, of.trace, layer, tpl, { order: d.order, stride: d.stride }), trace: of.trace, model: mdl });
			} catch (e) {
				put(name, { kind: 'scan', status: 'error', provenance: 'live', tape: null, error: e instanceof Error ? e.message : String(e) });
			}
			return;
		}
		if (d.kind === 'live-model') {
			const srcDef = defs[d.of];
			if (srcDef?.kind !== 'train') return put(name, { kind: 'live-model', status: 'error', provenance: 'live', tape: null, error: `source '${d.of}' is not a train source (a live-model source reads a train source's weights)` });
			await ready(d.of);
			let tape = rt.sources[d.of]?.tape;
			const first = d.at === 'first';
			// Entered before its train source has run (a deep link #lesson-<id> straight to this step, or a lesson whose
			// live step comes first): run the train source to its end first, exactly as a step `at: 'last'` on it would, so
			// the numbers equal the ones a learner gets by stepping in. Meanwhile the stage shows "Training first… k / n"
			// (StageHost reads the train source's progress). A train source that already has frames is used as it is: its
			// latest weights, as far as it has run (on a page where the learner trains, that may be mid-run).
			const lt = tape as LiveTape | null | undefined;
			if (lt && lt.length === 0 && typeof lt.ensure === 'function' && !lt.done) {
				put(name, { kind: 'live-model', status: 'loading', provenance: 'live', tape: null });
				await rt.ensure(d.of, first ? 'first' : 'last');
				tape = rt.sources[d.of]?.tape;
			}
			const src = rt.sources[d.of];
			if (!tape || tape.length === 0) {
				// a train source that does not run on an empty input (K1): the step shows the adapter's empty note
				if (rt.pre?.empty && !srcDef.runOnEmpty) return put(name, { kind: 'live-model', status: 'idle', provenance: 'live', tape: null });
				return put(name, { kind: 'live-model', status: 'error', provenance: 'live', tape: null, error: `source '${d.of}' has no frames${src?.error ? ` (${src.error})` : ''}` });
			}
			const p = pkg.processes[`train:${srcDef.trainer}`] as ProcessDef<any, any, any> | undefined;
			if (!p?.liveSpec) return put(name, { kind: 'live-model', status: 'error', provenance: 'live', tape: null, error: `process 'train:${srcDef.trainer}' has no liveSpec(state)` });
			try {
				const { spec: s2, weights } = p.liveSpec(tape.state(first ? 0 : tape.length - 1), srcDef.opt);
				const mdl = fromParts(s2, weights, pkg.ops);
				const trace = mdl.forward(rt.pre?.tensors ?? {});
				put(name, { kind: 'live-model', status: 'ready', provenance: src?.provenance ?? 'live', trace, tape: traceToTape(s2, trace), model: mdl });
			} catch (e) {
				put(name, { kind: 'live-model', status: 'error', provenance: 'live', tape: null, error: e instanceof Error ? e.message : String(e) });
			}
			return;
		}
		if (cur?.tape) return;
		if (loading[name]) return loading[name];
		const job = (async () => {
			await null; // run after `loading[name] = job` below, so the finally clause clears the right entry
			put(name, { kind: d.kind, status: 'loading', provenance: d.kind === 'tape' ? 'precomputed' : 'live', tape: null });
			try {
				if (d.kind === 'tape') {
					put(name, { kind: 'tape', status: 'ready', provenance: 'precomputed', tape: await loadTapeFile(d.file) });
					return;
				}
				const procName = d.kind === 'train' ? `train:${d.trainer}` : d.process;
				const p = pkg.processes[procName];
				if (d.precomputed && (!p || (o.preferPrecomputed && generation === 1))) {
					put(name, { kind: d.kind, status: 'ready', provenance: 'precomputed', tape: await loadTapeFile(d.precomputed, p) });
					return;
				}
				if (!p) throw new Error(`${m.slug}: no process '${procName}'`);
				// K1: a train source does not run on an empty input (no 'init' frame of an untrained network on nothing): it
				// holds an empty, finished tape, so its steps show the adapter's empty note (stepAt: no frames + empty input)
				if (d.kind === 'train' && rt.pre?.empty && !d.runOnEmpty) {
					skipped.add(name);
					put(name, { kind: 'train', status: 'ready', provenance: 'live', tape: emptyTape(p, d.opt) });
					return;
				}
				// K1: after: '<source>' waits for that source to finish, so input(rt) can read its frames
				if (d.kind === 'process' && d.after) {
					const g = generation;
					if (d.after === name || !defs[d.after]) throw new Error(`source '${name}': after '${d.after}' is not another declared source`);
					await rt.ensure(d.after, 'last');
					if (g !== generation) {
						put(name, { kind: 'process', status: 'idle', provenance: 'live', tape: null });
						return;
					}
					const before = rt.sources[d.after];
					if (!before?.tape || before.tape.length === 0) {
						if (rt.pre?.empty) return put(name, { kind: 'process', status: 'idle', provenance: 'live', tape: null });
						throw new Error(`source '${name}' runs after '${d.after}', which has no frames${before?.error ? ` (${before.error})` : ''}`);
					}
				}
				const opt = { ...(d.opt ?? {}) } as Record<string, unknown>;
				const recordDraws = opt.recordDraws === true || d.recordDraws === true || p.recordDraws === true;
				const seed = d.kind === 'process' ? (d.seed ?? 1) : typeof opt.seed === 'number' ? opt.seed : 1;
				const input = d.kind === 'process' && d.input ? d.input(rt) : rt.input;
				const run = d.kind === 'process' ? d.run : undefined;
				// K1: Node runs every source in-process, the browser sends it to the package worker unless run: 'main'. Catch an
				// input a worker cannot receive (a model object, a function) here, so lint fails as the page would.
				if (d.kind === 'process' && run !== 'main' && typeof (globalThis as { document?: unknown }).document === 'undefined' && typeof structuredClone === 'function') {
					try {
						structuredClone(input);
					} catch (e) {
						throw new Error(`source '${name}' runs in the package worker, but its input cannot be sent to a worker (${e instanceof Error ? e.message.split('\n')[0] : String(e)}): set run: 'main' on the source, or give input(rt) plain data`);
					}
				}
				const t = runner.open({ slug: m.slug, source: name, process: procName, input, opt, seed, detail: d.detail ?? 'coarse', recordDraws, ...(run ? { run } : {}) }, pkg);
				live[name] = t;
				if (d.kind === 'process') openedWith[name] = snapshot(input);
				put(name, { kind: d.kind, status: 'running', provenance: 'live', tape: t, progress: { at: 0, target: 0 } });
				// new frames replace the source state (progress, status), so a $state.raw wrapper sees them
				t.subscribe((n) => {
					if (live[name] !== t) return;
					put(name, { ...rt.sources[name], status: t.done ? 'ready' : 'running', progress: { at: n, target: n } });
				});
			} catch (e) {
				put(name, { kind: d.kind, status: 'error', provenance: 'live', tape: null, error: e instanceof Error ? e.message : String(e) });
			} finally {
				loading[name] = undefined;
			}
		})();
		loading[name] = job;
		return job;
	}

	/** A finished tape without frames (a train source on an empty input; K1). */
	function emptyTape(p: ProcessDef<any, any, any>, opt?: Record<string, unknown>): LiveTape {
		const store = new TapeStore({ process: p.id, seed: 1, opt: { ...(p.defaults as object), ...(opt ?? {}) }, provenance: 'live', schema: p.schema, phases: p.phases });
		return {
			process: store.process,
			seed: store.seed,
			opt: store.opt,
			provenance: 'live',
			version: store.version,
			phases: store.phases,
			length: 0,
			done: true,
			frame: (i) => store.frame(i),
			state: (i) => store.state(i),
			find: () => -1,
			segments: () => [],
			metric: (n) => store.metric(n),
			ensure: async () => -1,
			control: () => {},
			subscribe: () => () => {},
			close: () => {}
		};
	}

	/** A copy of a process input for sameInput(): plain data is copied (an edit made in place shows as a change), other
	 *  objects (a model) are kept by reference. */
	function snapshot(v: unknown, depth = 0): unknown {
		if (!v || typeof v !== 'object' || depth > 8) return v;
		if (ArrayBuffer.isView(v)) return (v as unknown as { slice(): unknown }).slice();
		if (Array.isArray(v)) return v.map((x) => snapshot(x, depth + 1));
		const pr = Object.getPrototypeOf(v);
		if (pr !== Object.prototype && pr !== null) return v;
		return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, snapshot(x, depth + 1)]));
	}

	/** Structural equality of two process inputs: numbers, strings, arrays, typed arrays and plain objects; other objects
	 *  (a model, a class instance) only by identity. */
	function sameInput(a: unknown, b: unknown, depth = 0): boolean {
		if (a === b) return true;
		if (depth > 8 || !a || !b || typeof a !== 'object' || typeof b !== 'object') return typeof a === 'number' && typeof b === 'number' && Number.isNaN(a) && Number.isNaN(b);
		if (ArrayBuffer.isView(a) || ArrayBuffer.isView(b)) {
			if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b) || a.constructor !== b.constructor) return false;
			const x = a as unknown as ArrayLike<number>;
			const y = b as unknown as ArrayLike<number>;
			if (x.length !== y.length) return false;
			for (let i = 0; i < x.length; i++) if (x[i] !== y[i] && !(Number.isNaN(x[i]) && Number.isNaN(y[i]))) return false;
			return true;
		}
		if (Array.isArray(a) || Array.isArray(b)) {
			if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
			for (let i = 0; i < a.length; i++) if (!sameInput(a[i], b[i], depth + 1)) return false;
			return true;
		}
		const plain = (v: object) => {
			const pr = Object.getPrototypeOf(v);
			return pr === Object.prototype || pr === null;
		};
		if (!plain(a) || !plain(b)) return false;
		const ka = Object.keys(a);
		const kb = Object.keys(b);
		if (ka.length !== kb.length) return false;
		for (const k of ka) if (!Object.prototype.hasOwnProperty.call(b, k) || !sameInput((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], depth + 1)) return false;
		return true;
	}

	// the first input: explicit, the manifest's start, the first preset, else the adapter's first edge fixture
	const first = o.input !== undefined ? o.input : (m.input?.start ?? presets[0]?.raw ?? adapter?.edgeFixtures()[0]?.raw ?? null);
	setInput(first);
	return rt;
}
