// atlas parity <slug> [--write] [--json] [--dir <folder under static/>] [--steps n] [--fixtures]   (design §4.12, §6.2 item 8)
//
//   kind A  every traced id and sub-step of every reference case (static/<assets.dir>/reference.json) against the
//           kit executor, bar = spec.parity.bar (1e-4); explain() = trace on 2,000 sampled units (<= 1e-5, AK202);
//           a warning when the cases miss a symbolic dim's min or max.
//   kind B  per train source with a twin (pipeline/<slug>/twin.py): the TS trainer in fine detail writes
//           .atlas/<slug>/train_fixture.<trainer>.<optimizer>.json (init params + batches); python -m
//           atlas_kit.train_reference runs the twin in float64; losses, step-1 gradients and final params must agree:
//           plain SGD for 50 steps <= 1e-10 relative, the page's own optimizer for 50 steps <= 1e-7 relative.
//   kind C  per process source with pipeline/<slug>/reference.py: the TS tape (draws recorded) against the Python
//           reference tape, frame by frame: phase/step/sub exact, every stored field within its `tol`
//           ('exact' or a number, relative: |a - b| <= tol * max(1, |b|); default 1e-12 for floats, exact otherwise),
//           metrics within the process's metricTol (K1; default 1e-12). A source with reference: false is skipped (K1).
//           The input reaches reference.py as --input <json>, or, when the JSON is long (K1: the Windows command line
//           holds about 32K characters), as --input @<file> (read it with ak.json_arg). Values that are not plain data
//           (a model object, a function) are left out of it; typed arrays become lists.
//           The TS tape, the draws and the reference tape go to .atlas/<slug>/parity/ (K1: they used to overwrite the
//           tape `atlas run --save` keeps in .atlas/<slug>/).
//   kind B  (K1) one twin run per distinct trainer and options (sources that differ in lambda or c each get their own
//           record); TrainerDef.twin.sgd sets the options of the plain-SGD run (a stiff loss needs a smaller lr there).
//   legacy  tinycnn / tinygpt (no package): the legacy scripts pipeline/parity/<slug>.parity.ts with --no-write, their
//           measured number against the shipped spec.parity, plus the kit executor on the shipped reference. spec.json
//           is hashed before and after. --write is refused for them except in the main tree.
// Writes .atlas/<slug>/parity.json. --write also writes spec.parity (kind-A record with perLayer; else the B or C
// record) into static/<assets.dir>/spec.json for kit packages, and (K0) the per-source record of every twin and reference
// into src/lib/explainers/<slug>/parity.json (format atlas-parity/1): the page's badge reads it for train and process
// steps ("matches PyTorch fp64 …", "same steps as an independent Python run"), also in packages without a spec.json.
// --fixtures (K0) also runs every process reference on every fixture input (lesson fixtures, adapter edge fixtures,
// presets), not only on the default input. `export` runs this with --write.
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { isAbsolute, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { AnyTensor, AtlasSpec, Frame, GraphModel, ParityFile, ParityRecord, ProcessDef, SchemaJson, SourceDef, SourceParity, Tensor } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import { python, runNode, runPython } from '../lib/child.ts';
import { fileSafe, posix, readJson, writeBytes, writeJson, writeText } from '../lib/io.ts';
import type { Kit } from '../lib/kit.ts';
import { Issues, isLegacy, openPackage, sig, today, type Opened } from '../lib/pkg.ts';
import { roots } from '../root.ts';

/** --dir: an absolute folder, a folder under the site starting with .atlas/, or a folder under static/. */
export function resolveDir(site: string, d: string): string {
	if (isAbsolute(d)) return d;
	const parts = d.split(/[\\/]+/).filter(Boolean);
	if (parts[0] === '.atlas') return join(site, ...parts);
	return join(site, 'static', ...parts);
}

const sha = (f: string) => (existsSync(f) ? createHash('sha256').update(readFileSync(f)).digest('hex') : null);
const relErr = (a: number, b: number) => Math.abs(a - b) / Math.max(1, Math.abs(b));

// ------------------------------------------------------------------ kind A
export interface ForwardResult {
	dir: string;
	bar: number;
	cases: number;
	edgeCases: string[];
	max: number;
	worst: { id: string; case: string; at: number; js: number; ref: number } | null;
	perLayer: Record<string, number>;
	missing: string[];
	explain: { units: number; max: number; worst: unknown };
	dims: Record<string, { min: number; max: number; seen: number[] }>;
	pass: boolean;
	ms: number;
}

export async function forwardParity(k: Kit, dir: string, ops: unknown[], is: Issues, o: { sample?: number } = {}): Promise<ForwardResult> {
	const t0 = performance.now();
	if (!existsSync(join(dir, 'spec.json')))
		throw new AtlasError('AK301', `no spec.json in ${posix(dir)}`, { fix: 'export the model first: npm run atlas -- export <slug>' });
	const model: GraphModel = await k.executor.loadGraph(k.io.fsFetch, dir, ops as never);
	const spec = model.spec;
	const refFile = (spec.files as Record<string, string> | undefined)?.reference ?? 'reference.json';
	if (!existsSync(join(dir, refFile))) throw new AtlasError('AK301', `no ${refFile} in ${posix(dir)}`, { fix: 'export writes reference.json (atlas-ref-1)' });
	const cases = k.refs.parseReference(readJson(join(dir, refFile)));
	const bar = Number(spec.parity?.bar ?? spec.parity?.tolerance ?? 1e-4);
	const perLayer: Record<string, number> = {};
	const missing = new Set<string>();
	let max = 0;
	let worst: ForwardResult['worst'] = null;
	let exMax = 0;
	let exUnits = 0;
	let exWorst: unknown = null;
	const dims: ForwardResult['dims'] = {};
	for (const [d, decl] of Object.entries(spec.dims ?? {})) dims[d] = { min: decl.min, max: decl.max, seen: [] };
	for (const c of cases) {
		for (const decl of spec.inputs ?? []) {
			const x = c.inputs[decl.name];
			if (!x) continue;
			decl.shape.forEach((d, ax) => {
				if (typeof d === 'string' && dims[d] && !dims[d].seen.includes(x.shape[ax])) dims[d].seen.push(x.shape[ax]);
			});
		}
		let trace;
		try {
			trace = model.forward(c.inputs as Record<string, AnyTensor>, { record: 'all' });
		} catch (e) {
			is.error('AK201', `parity: case "${c.id}" does not run: ${(e as Error).message}`, { fix: 'check the reference inputs against spec.inputs (names, shapes, dims)' });
			continue;
		}
		for (const id of c.order) {
			const ref = c.tensors[id];
			if (!ref) continue;
			const mine = trace.t[id];
			if (!mine) {
				missing.add(id);
				continue;
			}
			const d = k.compare.maxDiff(mine.data, ref.data);
			const v = mine.data.length === ref.data.length ? d.max : Infinity;
			if (!(v <= (perLayer[id] ?? 0))) perLayer[id] = v;
			if (!(v <= max)) {
				max = v;
				worst = { id, case: c.id, at: d.at, js: d.at >= 0 ? mine.data[d.at] : NaN, ref: d.at >= 0 ? ref.data[d.at] : NaN };
			}
		}
		const ex = k.explain.checkExplain(model, trace, { sample: Math.max(1, Math.ceil((o.sample ?? 2000) / cases.length)), seed: 7 });
		exUnits += ex.units;
		if (!(ex.max <= exMax)) {
			exMax = ex.max;
			exWorst = ex.worst;
		}
	}
	for (const id of missing) is.error('AK201', `parity: the kit trace has no "${id}", which the reference has`, { likely: 'a layer id or sub-step the TS op does not emit', fix: 'record the same ids as the reference (layer.sub, ctx.emit)' });
	if (!(max <= bar) && worst) {
		const shape = cases.find((c) => c.id === worst!.case)?.tensors[worst.id]?.shape ?? [];
		is.error('AK201', `parity: layer "${worst.id}" max |diff| ${sig(max)} > ${sig(bar)}`, {
			where: `case "${worst.case}", flat index ${worst.at} of [${shape.join('×')}]: js ${sig(worst.js)}, torch ${sig(worst.ref)}`,
			likely: 'the TS op and the PyTorch module compute different things (a missing attribute, a mask, an eps, the wrong layout)',
			fix: 'compare the layer params in spec.json with the module; teach the op the missing attribute or describe() it differently'
		});
	}
	if (!(exMax <= 1e-5))
		is.error('AK202', `explain() differs from the trace by ${sig(exMax)} (> 1e-5)`, { where: JSON.stringify(exWorst), fix: 'explain() must recompute the unit with the same arithmetic as forward()' });
	for (const [d, v] of Object.entries(dims)) {
		const lacks = [v.min, v.max].filter((x) => !v.seen.includes(x));
		if (lacks.length) is.warn('AK201', `parity: no reference case at ${d} = ${lacks.join(' and ')} (cases cover ${v.seen.sort((a, b) => a - b).join(', ')})`, { fix: `add reference cases at the dim extremes (${d} = ${v.min} and ${d} = ${v.max}) in export.py` });
	}
	return {
		dir: posix(dir),
		bar,
		cases: cases.length,
		edgeCases: cases.filter((c) => c.edge).map((c) => c.id),
		max,
		worst,
		perLayer,
		missing: [...missing],
		explain: { units: exUnits, max: exMax, worst: exWorst },
		dims,
		pass: max <= bar && missing.size === 0 && exMax <= 1e-5,
		ms: performance.now() - t0
	};
}

// ------------------------------------------------------------------ legacy
export interface LegacyResult {
	script: string;
	exit: number;
	parity: Record<string, unknown> | null;
	shipped: number | null;
	measured: number | null;
	specSha: { before: string | null; after: string | null; unchanged: boolean };
	laneSpecSha: { before: string | null; after: string | null; unchanged: boolean } | null;
	kit: ForwardResult | null;
	tail: string[];
}

export async function legacyParity(k: Kit, slug: string, write: boolean, is: Issues, echo: boolean): Promise<LegacyResult> {
	const r = roots();
	const script = join(r.root, 'pipeline', 'parity', `${slug}.parity.ts`);
	if (!existsSync(script)) throw new AtlasError('AK301', `no legacy parity script ${posix(script)}`);
	if (write && r.lane) throw new AtlasError('AK702', `--write on legacy model '${slug}' is refused in lane '${r.lane}'`, { fix: 'only the main session runs legacy parity without --no-write, from the main tree' });
	const mainSpec = join(r.root, 'site', 'static', 'models', slug, 'spec.json');
	const laneSpec = r.site !== join(r.root, 'site') ? join(r.site, 'static', 'models', slug, 'spec.json') : null;
	const before = sha(mainSpec);
	const laneBefore = laneSpec ? sha(laneSpec) : null;
	const run = await runNode([script, ...(write ? [] : ['--no-write'])], { cwd: r.root, echo: echo ? `  [${slug}.parity] ` : false, timeoutMs: 540_000 });
	const after = sha(mainSpec);
	const laneAfter = laneSpec ? sha(laneSpec) : null;
	const line = run.stdout.split(/\r?\n/).find((l) => l.startsWith('PARITY_JSON '));
	const parity = line ? (JSON.parse(line.slice(12)) as Record<string, unknown>) : null;
	const shippedSpec = readJson<AtlasSpec>(laneSpec ?? mainSpec);
	const shipped = typeof shippedSpec.parity?.max_abs_diff === 'number' ? shippedSpec.parity.max_abs_diff : null;
	const measured = parity && typeof parity.max_abs_diff === 'number' ? (parity.max_abs_diff as number) : null;
	if (run.code !== 0) is.error('AK201', `legacy parity script failed (exit ${run.code})`, { where: posix(script), likely: run.stderr.trim().split('\n').slice(-3).join(' | ') });
	if (!write && before !== after) is.error('AK702', `${posix(mainSpec)} changed during a --no-write run`, { fix: 'restore it with git checkout; the legacy scripts must leave it untouched with --no-write' });
	if (!write && !parity) is.error('AK201', 'the legacy script printed no PARITY_JSON line (is --no-write supported?)', { where: posix(script) });
	if (measured !== null && shipped !== null && Number(measured.toPrecision(3)) !== Number(shipped.toPrecision(3)))
		is.warn('AK201', `measured max |diff| ${sig(measured)} differs from the shipped spec.parity ${sig(shipped)}`, { fix: 'expected after retraining or a runtime change; the main session re-runs the legacy script without --no-write' });
	// the kit executor on the same shipped assets (read through legacy.ts; informational for legacy models)
	let kitRes: ForwardResult | null = null;
	try {
		const kis = new Issues();
		kitRes = await forwardParity(k, join(r.site, 'static', 'models', slug), k.ops.CORE_OPS, kis);
		for (const i of kis.list) is.list.push({ ...i, level: 'warn', message: `kit executor: ${i.message}` });
	} catch (e) {
		is.warn('AK201', `kit executor on the shipped ${slug} assets: ${(e as Error).message}`);
	}
	return {
		script: posix(script),
		exit: run.code,
		parity,
		shipped,
		measured,
		specSha: { before, after, unchanged: before === after },
		laneSpecSha: laneSpec ? { before: laneBefore, after: laneAfter, unchanged: laneBefore === laneAfter } : null,
		kit: kitRes,
		tail: run.stdout.trim().split(/\r?\n/).slice(-6)
	};
}

// ------------------------------------------------------------------ kind B
export interface TrainRun {
	optimizer: string;
	/** steps compared (K1: up to the blow-up of a diverged plain-SGD run) */
	steps: number;
	/** K1: the step at which the plain-SGD run blew up */
	diverged?: number;
	bar: number;
	losses: number;
	grads0: number;
	final: number;
	pass: boolean;
	fixture: string;
}
export interface TrainResult {
	source: string;
	trainer: string;
	twin: string | null;
	runs: TrainRun[];
	pass: boolean;
	note?: string;
	/** K1: every train source this run stands for (same trainer and options) */
	sources?: string[];
}

/** Options that change what a twin run computes: the source's options without steps (parity sets its own) and
 *  snapshotEvery (fine detail records every step anyway). K1. */
export function twinKey(trainer: string, opt: Record<string, unknown>): string {
	const o: Record<string, unknown> = {};
	for (const k of Object.keys(opt).sort()) if (k !== 'steps' && k !== 'snapshotEvery') o[k] = opt[k];
	return `${trainer} ${JSON.stringify(o)}`;
}

/** JSON of a process input for reference.py: typed arrays become lists, values that are not plain data (a model
 *  object, a function, a class instance without toJSON) are left out. K1. */
export function inputJson(input: unknown): string {
	return JSON.stringify(input ?? null, function (this: unknown, _k: string, v: unknown) {
		if (v === null || typeof v !== 'object') return typeof v === 'function' || typeof v === 'symbol' || typeof v === 'bigint' ? undefined : v;
		if (ArrayBuffer.isView(v)) return Array.from(v as unknown as ArrayLike<number>);
		if (Array.isArray(v)) return v;
		const pr = Object.getPrototypeOf(v);
		return pr === Object.prototype || pr === null ? v : undefined;
	});
}

const plainT = (t: Tensor | AnyTensor) => ({ shape: t.shape.slice(), data: Array.from(t.data as ArrayLike<number>) });
const plainRec = (r: Record<string, Tensor> | undefined) => Object.fromEntries(Object.entries(r ?? {}).map(([k, v]) => [k, plainT(v)]));

function compareRec(a: Record<string, Tensor>, b: Record<string, { shape: number[]; data: number[] }>, what: string, is: Issues, bar: number, label: string): number {
	let m = 0;
	for (const [k, v] of Object.entries(a)) {
		const w = b[k];
		if (!w) {
			is.error('AK203', `${label}: the twin has no ${what} '${k}'`, { fix: 'the twin state_dict names must equal the TS Module.named() names' });
			return Infinity;
		}
		if (w.data.length !== v.data.length) {
			is.error('AK203', `${label}: ${what} '${k}' has ${w.data.length} values in the twin, ${v.data.length} in TS`);
			return Infinity;
		}
		for (let i = 0; i < v.data.length; i++) {
			const d = relErr(v.data[i], w.data[i]);
			if (!(d <= m)) m = d;
		}
	}
	return m;
}

/** K2: what a twin run's answer depends on: the fixture (TS init, batches, options), the steps, every .py file of
 *  pipeline/<slug>/ and pipeline/atlas_kit/, and the configured Python. */
function twinCacheKey(root: string, slug: string, fixture: string, steps: number): string {
	const h = createHash('sha256').update(readFileSync(fixture)).update(`|${steps}|${python()}|`);
	const py = (d: string): string[] =>
		existsSync(d)
			? readdirSync(d, { withFileTypes: true })
					.sort((a, b) => (a.name < b.name ? -1 : 1))
					.flatMap((e) => (e.isDirectory() ? (e.name === '__pycache__' || e.name === 'tests' ? [] : py(join(d, e.name))) : e.name.endsWith('.py') ? [join(d, e.name)] : []))
			: [];
	for (const file of [...py(join(root, 'pipeline', slug)), ...py(join(root, 'pipeline', 'atlas_kit'))]) h.update(file.slice(root.length)).update(readFileSync(file));
	return h.digest('hex').slice(0, 32);
}

export async function trainParity(o: Opened, source: string, trainer: string, srcOpt: Record<string, unknown>, is: Issues, steps: number, echo: boolean): Promise<TrainResult> {
	const r = roots();
	const twin = join(r.root, 'pipeline', o.slug, 'twin.py');
	const p = o.l.pkg.processes[`train:${trainer}`];
	if (!p) throw new AtlasError('AK301', `source '${source}' runs trainer '${trainer}', which is not registered`, { fix: `registered: ${Object.keys(o.l.pkg.processes).join(', ')}` });
	if (!existsSync(twin))
		return { source, trainer, twin: null, runs: [], pass: true, note: 'no pipeline/<slug>/twin.py: the badge says "core ops match PyTorch fp64 (op tests)"' };
	const rt = await o.runtime();
	const input = rt.input;
	rt.close?.();
	const base = { ...(p.defaults as Record<string, unknown>), ...srcOpt };
	const page = String(base.optimizer ?? 'sgd') + (base.decoupled === true ? 'w' : '');
	// K1: TrainerDef.twin.sgd overrides the plain-SGD run's options (the page's lr diverges for a stiff loss)
	const sgdOpt = ((p as { twin?: { sgd?: Record<string, unknown> } }).twin?.sgd ?? {}) as Record<string, unknown>;
	const plans: { name: string; opt: Record<string, unknown>; bar: number }[] = [{ name: 'sgd', opt: { optimizer: 'sgd', decoupled: false, weightDecay: 0, ...sgdOpt }, bar: 1e-10 }];
	if (page !== 'sgd') plans.push({ name: page, opt: {}, bar: base.optimizer === 'momentum' ? 1e-10 : 1e-7 });
	const runs: TrainRun[] = [];
	for (const plan of plans) {
		const opt = { ...base, ...plan.opt, steps, stopOnDiverge: true };
		// the runtime's rule (runtime/runtime.ts): a train source runs with seed = source opt.seed, else 1
		const seed = typeof srcOpt.seed === 'number' ? (srcOpt.seed as number) : 1;
		const tape = o.kit.record.record(p, input, opt, { seed, detail: 'fine', maxFrames: 4 * steps + 8 });
		const batches: { step: number; batch: unknown }[] = [];
		const losses: number[] = [];
		let init: Record<string, Tensor> | null = null;
		let grads0: Record<string, Tensor> | null = null;
		let final: Record<string, Tensor> | null = null;
		for (let i = 0; i < tape.length; i++) {
			const f = tape.frame(i) as Frame<Record<string, unknown>>;
			const s = f.state as Record<string, unknown>;
			if (f.phase === 'init' && !init) init = s.params as Record<string, Tensor>;
			if (f.phase === 'forward') batches.push({ step: f.step!, batch: plainRec(s.batch as Record<string, Tensor>) });
			if (f.phase === 'loss') losses.push(f.metrics!.loss);
			if (f.phase === 'backward' && f.step === 0) grads0 = s.grads as Record<string, Tensor>;
			if (f.phase === 'update') final = s.params as Record<string, Tensor>;
		}
		if (!init || !grads0 || !final || losses.length < steps) {
			is.error('AK203', `${source} (${plan.name}): the TS trainer tape is incomplete (${losses.length} of ${steps} steps)`, {
				likely: 'the loss diverged or the trainer stopped early',
				fix: plan.name === 'sgd' ? `if plain SGD diverges at the page's lr, give the trainer twin: { sgd: { lr: <smaller> } } (trainer.ts)` : undefined
			});
			continue;
		}
		// K1: a plain-SGD run that blows up at the page's lr (a stiff loss) amplifies float rounding without bound: it is
		// compared up to the step before the blow-up (and on its step-1 gradients), with a warning, not failed
		const blowAt = losses.findIndex((l, i) => !Number.isFinite(l) || (i > 0 && Math.abs(l) > 1e6 * Math.max(1, Math.abs(losses[0]))));
		const diverged = blowAt >= 0 && plan.name === 'sgd';
		if (diverged)
			is.warn('AK203', `${source} (sgd): the plain-SGD run diverges at step ${blowAt} (loss ${sig(losses[0])} -> ${sig(Math.max(...losses.filter(Number.isFinite)))}); it is compared up to step ${blowAt - 1} and on its step-1 gradients`, {
				fix: `give the trainer twin: { sgd: { lr: <smaller> } } (trainer.ts) for a full 50-step comparison; the page's own optimizer run is unchanged`
			});
		const dir = join(r.site, '.atlas', o.slug);
		const fixture = join(dir, `train_fixture.${fileSafe(trainer)}.${plan.name}.json`);
		const outFile = join(dir, `train_reference.${fileSafe(trainer)}.${plan.name}.json`);
		writeJson(fixture, {
			format: 'atlas-train-fixture/1',
			slug: o.slug,
			source,
			trainer,
			seed,
			opt,
			steps,
			init: plainRec(init),
			batches,
			ts: { losses, grads0: plainRec(grads0), final: plainRec(final) }
		});
		// K2: the twin's answer for the same fixture, Python files and Python is reused (.atlas/<slug>/twin-cache/): a
		// package with many option sets (the MLP playground's wrapped sources) pays the Python runs once, not every check
		const cached = join(dir, 'twin-cache', `${twinCacheKey(r.root, o.slug, fixture, steps)}.json`);
		const py = existsSync(cached)
			? (writeBytes(outFile, readFileSync(cached)), { code: 0, stdout: '', stderr: '' })
			: await runPython(['-m', 'atlas_kit.train_reference', '--fixture', fixture, '--twin', twin, '--out', outFile, '--steps', String(steps)], { echo: echo ? '  [twin] ' : false, timeoutMs: 540_000 });
		if (py.code === 0 && existsSync(outFile) && !existsSync(cached)) writeBytes(cached, readFileSync(outFile));
		if (py.code !== 0 || !existsSync(outFile)) {
			is.error('AK203', `${source} (${plan.name}): the twin failed (exit ${py.code})`, { where: posix(twin), likely: py.stderr.trim().split('\n').slice(-4).join(' | ') });
			continue;
		}
		const ref = readJson<{ losses: number[]; grads0: Record<string, { shape: number[]; data: number[] }>; final: Record<string, { shape: number[]; data: number[] }> }>(outFile);
		const label = `${source} (${plan.name}, ${steps} steps)`;
		let lm = 0;
		let lat = -1;
		for (let i = 0; i < (diverged ? blowAt : steps); i++) {
			const d = relErr(losses[i], ref.losses[i]);
			if (!(d <= lm)) {
				lm = d;
				lat = i;
			}
		}
		const g0 = compareRec(grads0, ref.grads0, 'gradient', is, 1e-10, label);
		// a diverged run's final params say nothing about the twin: its losses up to the blow-up stand for them
		const fm = diverged ? lm : compareRec(final, ref.final, 'parameter', is, plan.bar, label);
		const pass = lm <= plan.bar && g0 <= 1e-10 && fm <= plan.bar;
		if (!pass)
			is.error('AK203', `twin mismatch: ${label}: losses ${sig(lm)}, step-1 grads ${sig(g0)}, final params ${sig(fm)} (bars: ${sig(plan.bar)} relative, grads 1e-10)`, {
				where: lat >= 0 ? `step ${lat}: TS loss ${losses[lat]}, twin ${ref.losses[lat]}` : undefined,
				likely: 'the twin differs from the TS trainer (architecture, activation, loss reduction, optimizer settings, label dtype)',
				fix: `make pipeline/${o.slug}/twin.py build(cfg) and loss() mirror trainer.ts exactly; check opt (betas, eps, weightDecay, decoupled)`
			});
		runs.push({ optimizer: plan.name, steps: diverged ? blowAt : steps, bar: plan.bar, losses: lm, grads0: g0, final: fm, pass, fixture: posix(fixture), ...(diverged ? { diverged: blowAt } : {}) });
	}
	return { source, trainer, twin: posix(twin), runs, pass: runs.length === plans.length && runs.every((x) => x.pass) };
}

// ------------------------------------------------------------------ kind C
export interface ProcessResult {
	source: string;
	process: string;
	reference: string | null;
	frames: number;
	refFrames: number;
	firstDivergence: { frame: number; phase: string; field: string; js: unknown; py: unknown } | null;
	maxFloat: number;
	pass: boolean;
	note?: string;
	/** --fixtures: the fixture input this run used */
	fixture?: string;
}

function eqVal(a: unknown, b: unknown, tol: number | 'exact'): { ok: boolean; d: number; at?: number } {
	const num = (x: unknown) => typeof x === 'number';
	const isT = (x: unknown): x is { shape: number[]; data: ArrayLike<number> } => !!x && typeof x === 'object' && 'data' in (x as object) && 'shape' in (x as object);
	if (num(a) && num(b)) {
		const x = a as number;
		const y = b as number;
		if (x === y || (Number.isNaN(x) && Number.isNaN(y))) return { ok: true, d: 0 };
		if (tol === 'exact') return { ok: false, d: Infinity };
		const d = relErr(x, y);
		return { ok: d <= tol, d };
	}
	if (isT(a) && isT(b)) {
		if (a.shape.join() !== b.shape.join() || a.data.length !== b.data.length) return { ok: false, d: Infinity };
		let m = 0;
		for (let i = 0; i < a.data.length; i++) {
			const r = eqVal(a.data[i], b.data[i], tol);
			if (!r.ok) return { ok: false, d: r.d, at: i };
			m = Math.max(m, r.d);
		}
		return { ok: true, d: m };
	}
	if (a && b && typeof a === 'object' && typeof b === 'object') {
		const ka = Object.keys(a as object).sort();
		const kb = Object.keys(b as object).sort();
		if (ka.join() !== kb.join()) return { ok: false, d: Infinity };
		let m = 0;
		for (const kk of ka) {
			const r = eqVal((a as Record<string, unknown>)[kk], (b as Record<string, unknown>)[kk], tol);
			if (!r.ok) return r;
			m = Math.max(m, r.d);
		}
		return { ok: true, d: m };
	}
	return { ok: JSON.stringify(a) === JSON.stringify(b), d: 0 };
}

export async function processParity(
	o: Opened,
	source: string,
	procId: string,
	d: { opt?: Record<string, unknown>; seed?: number; input?: unknown; after?: string; reference?: boolean },
	is: Issues,
	echo: boolean,
	fx?: { id: string; raw: unknown }
): Promise<ProcessResult> {
	const r = roots();
	const refPy = join(r.root, 'pipeline', o.slug, 'reference.py');
	const p = o.l.pkg.processes[procId] as ProcessDef<any, any, any> | undefined;
	if (!p) throw new AtlasError('AK301', `source '${source}' runs process '${procId}', which is not registered`);
	const base = { source, process: procId, reference: null, frames: 0, refFrames: 0, firstDivergence: null, maxFloat: 0, ...(fx ? { fixture: fx.id } : {}) };
	if (d.reference === false) return { ...base, pass: true, note: 'reference: false on the source: no Python reference (the badge says "Live")' };
	if (!existsSync(refPy)) return { ...base, pass: true, note: 'no pipeline/<slug>/reference.py: the badge cannot say "same steps as an independent Python run"' };
	const rt = await o.runtime(fx ? fx.raw : undefined);
	// K1: a source with after: '<source>' reads that source's frames in input(rt)
	if (d.after) await rt.ensure(d.after, 'last');
	const input = typeof d.input === 'function' ? (d.input as (x: unknown) => unknown)(rt) : rt.input;
	rt.close?.();
	const opt = { ...(p.defaults as Record<string, unknown>), ...(d.opt ?? {}) };
	const seed = d.seed ?? 1;
	const tape = o.kit.record.record(p, input, opt, { seed, recordDraws: true });
	// K1: parity's own folder, so `atlas run --save` tapes in .atlas/<slug>/ are never overwritten
	const dir = join(r.site, '.atlas', o.slug, 'parity');
	const stem = join(dir, fileSafe(fx ? `${source}@${fx.id}` : source));
	const saved = o.kit.procIo.saveTape(tape, { binName: `${fileSafe(fx ? `${source}@${fx.id}` : source)}.tape.bin` });
	writeBytes(`${stem}.tape.bin`, new Uint8Array(saved.blob));
	writeText(`${stem}.tape.json`, JSON.stringify(saved.header) + '\n');
	const drawsFile = join(dir, `${fileSafe(procId)}.draws.json`);
	writeJson(drawsFile, { format: 'atlas-draws/1', process: procId, source, seed, opt, frames: Array.from({ length: tape.length }, (_, i) => tape.frame(i).draws ?? []) });
	const refStem = join(dir, `${fileSafe(procId)}.ref`);
	// K1: a long input goes in a file (--input @<file>; ak.json_arg reads both forms)
	let inputArg = inputJson(input);
	if (inputArg.length > 8000) {
		const f = join(dir, `${fileSafe(fx ? `${source}@${fx.id}` : source)}.input.json`);
		writeText(f, inputArg + '\n');
		inputArg = `@${f}`;
	}
	const py = await runPython([refPy, '--process', procId, '--seed', String(seed), '--opt', JSON.stringify(opt), '--draws', drawsFile, '--input', inputArg, '--out', refStem], {
		echo: echo ? '  [reference] ' : false,
		timeoutMs: 540_000
	});
	if (py.code !== 0 || !existsSync(`${refStem}.tape.json`)) {
		is.error('AK204', `${source}: the Python reference failed (exit ${py.code})`, { where: posix(refPy), likely: py.stderr.trim().split('\n').slice(-4).join(' | '), fix: 'reference.py must accept --process --seed --opt --draws --input --out (see the template)' });
		// K2: the summary must not read "identical" next to this failure
		return { ...base, reference: posix(refPy), pass: false, note: `the Python reference failed (exit ${py.code}): nothing was compared` };
	}
	const h = readJson<{ bin: { file: string }; schema: SchemaJson } & Record<string, unknown>>(`${refStem}.tape.json`);
	const blob = readFileSync(join(dir, h.bin.file));
	const ref = o.kit.procIo.loadTape(h as never, blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength) as ArrayBuffer);
	const res: ProcessResult = { ...base, reference: posix(refPy), frames: tape.length, refFrames: ref.length, pass: true };
	const fields = Object.entries(p.schema).filter(([, f]) => !f.derive);
	const tolOf = (k: string): number | 'exact' => {
		const f = p.schema[k];
		if (f?.tol !== undefined) return f.tol;
		return f && (f.kind === 'scalar' || f.kind === 'tensor') ? 1e-12 : 'exact';
	};
	// K1: ProcessDef.metricTol, one bar for every metric or one per name (default 1e-12 relative)
	const metricTolOf = (k: string): number | 'exact' => {
		const mt = p.metricTol;
		if (mt === undefined) return 1e-12;
		if (typeof mt === 'number' || mt === 'exact') return mt;
		return mt[k] ?? 1e-12;
	};
	const n = Math.min(tape.length, ref.length);
	outer: for (let i = 0; i < n; i++) {
		const a = tape.frame(i);
		const b = ref.frame(i);
		for (const key of ['phase', 'step', 'sub'] as const)
			if ((a[key] ?? null) !== (b[key] ?? null)) {
				res.firstDivergence = { frame: i, phase: a.phase, field: key, js: a[key], py: b[key] };
				break outer;
			}
		const sa = tape.rawState(i) as Record<string, unknown>;
		const sb = ref.rawState(i) as Record<string, unknown>;
		for (const [k] of fields) {
			if (!(k in sa) && !(k in sb)) continue;
			const c = eqVal(sa[k], sb[k], tolOf(k));
			if (!c.ok) {
				res.firstDivergence = { frame: i, phase: a.phase, field: c.at !== undefined ? `${k}[${c.at}]` : k, js: sa[k], py: sb[k] };
				break outer;
			}
			if (c.d > res.maxFloat) res.maxFloat = c.d;
		}
		for (const [mk, mv] of Object.entries(a.metrics ?? {})) {
			const pv = b.metrics?.[mk];
			if (pv === undefined) continue;
			const c = eqVal(mv, pv, metricTolOf(mk));
			if (!c.ok) {
				res.firstDivergence = { frame: i, phase: a.phase, field: `metrics.${mk}`, js: mv, py: pv };
				break outer;
			}
		}
	}
	if (!res.firstDivergence && tape.length !== ref.length) res.firstDivergence = { frame: n, phase: n < tape.length ? tape.frame(n).phase : ref.frame(n).phase, field: 'length', js: tape.length, py: ref.length };
	if (res.firstDivergence) {
		res.pass = false;
		const fd = res.firstDivergence;
		const show = (v: unknown) => (v && typeof v === 'object' && 'data' in (v as object) ? `[${Array.from(((v as { data: ArrayLike<number> }).data) as ArrayLike<number>).slice(0, 6).join(', ')}…]` : JSON.stringify(v));
		is.error('AK204', `reference mismatch: ${source}${fx ? ` on fixture '${fx.id}'` : ''} first diverges at frame ${fd.frame} (${fd.phase}), ${fd.field}`, {
			where: `TS ${show(fd.js)}, Python ${show(fd.py)}`,
			likely: 'the reference makes a different choice (ties break to the smallest index), draws differently (replay the recorded draws) or orders frames differently',
			fix: `step both with npm run atlas -- run ${o.slug} ${source} --frames ${Math.max(0, fd.frame - 3)}..${fd.frame + 3}`
		});
	}
	return res;
}

// ------------------------------------------------------------------ the command
export interface ParityReport {
	slug: string;
	checked: string;
	legacy?: LegacyResult;
	forward?: ForwardResult;
	train: TrainResult[];
	process: ProcessResult[];
	pass: boolean;
	wrote: string | null;
	/** --write: src/lib/explainers/<slug>/parity.json (the per-source record of twins and references) */
	wroteSources?: string;
	issues: { code: string; level: string; message: string }[];
}

/** The per-source parity record (atlas-parity/1) of a report's twins and references, or null when there is none. Every
 *  train source of a checked trainer gets its record; a process source gets its own (over its fixtures with --fixtures). */
export function sourceParityFile(slug: string, report: ParityReport, sources: Record<string, SourceDef>): ParityFile | null {
	const out: ParityFile = { format: 'atlas-parity/1', slug, checked: report.checked, sources: {} };
	for (const t of report.train) {
		if (!t.runs.length) continue;
		const r: SourceParity = {
			scope: 'train',
			against: 'pytorch-fp64-cpu',
			pass: t.pass && t.runs.every((x) => x.pass),
			max: Number(Math.max(...t.runs.map((x) => Math.max(x.losses, x.final))).toPrecision(3)),
			bar: Math.max(...t.runs.map((x) => x.bar)),
			relative: true,
			steps: Math.max(...t.runs.map((x) => x.steps)),
			checked: report.checked
		};
		// K1: the sources this run stands for (the same trainer and options); older reports: every source of the trainer
		// K2: also the process sources that wrap the trainer (listed in t.sources by wrapTwin)
		for (const [name, d] of Object.entries(sources)) if (t.sources ? t.sources.includes(name) : d.kind === 'train' && d.trainer === t.trainer) out.sources[name] = r;
	}
	const bySource = new Map<string, ProcessResult[]>();
	for (const pr of report.process) if (pr.reference) bySource.set(pr.source, [...(bySource.get(pr.source) ?? []), pr]);
	for (const [name, rs] of bySource)
		out.sources[name] = {
			scope: 'process',
			against: 'python-reference',
			pass: rs.every((x) => x.pass),
			max: Number(Math.max(...rs.map((x) => x.maxFloat)).toPrecision(3)),
			frames: rs[0].frames,
			fixtures: rs.filter((x) => x.fixture).length,
			checked: report.checked
		};
	return Object.keys(out.sources).length ? out : null;
}

/** K2: a process source whose ProcessDef wraps a trainer (train/wraps.ts) and makes exactly its frames gets the trainer's
 *  twin run with the source's options (shared with a train source of the same trainer and options). Its twin failures
 *  are warnings: the source then keeps the plain "Live" badge. False when the source is not such a wrapper. */
async function wrapTwin(o: Opened, name: string, s: Extract<SourceDef, { kind: 'process' }>, twinRuns: Map<string, TrainResult>, report: ParityReport, is: Issues, po: ParityOpts): Promise<boolean> {
	const proc = o.l.pkg.processes[s.process] as ProcessDef<any, any, any> | undefined;
	const { wrappedTrainer, sameFramesAsTrainer } = await import(pathToFileURL(join(o.kit.site, 'src', 'lib', 'atlas', 'train', 'wraps.ts')).href);
	const tr: string | null = wrappedTrainer(proc);
	const trainer = tr ? (o.l.pkg.processes[`train:${tr}`] as ProcessDef<any, any, any> | undefined) : undefined;
	if (!proc || !tr || !trainer || s.input || s.after || !existsSync(join(roots().root, 'pipeline', o.slug, 'twin.py'))) return false;
	const rt = await o.runtime();
	const input = rt.input;
	rt.close?.();
	const seed = s.seed ?? 1;
	const opt = { ...((s.opt ?? {}) as Record<string, unknown>), seed };
	const chk = sameFramesAsTrainer(proc, trainer, input, opt, seed, { detail: s.detail });
	if (!chk.same) return false;
	const key = twinKey(tr, opt);
	const seen = twinRuns.get(key);
	if (seen) {
		seen.sources!.push(name);
		return true;
	}
	const sub = new Issues();
	const res = await trainParity(o, name, tr, opt, sub, po.steps ?? 50, !!po.echo);
	for (const i of sub.list) is.list.push({ ...i, level: 'warn', message: `${i.message} (process source '${name}' wraps trainer '${tr}'; it keeps the plain "Live" badge)` });
	res.sources = [name];
	res.note = `process '${s.process}' makes the frames of trainer '${tr}' (${chk.frames} frames compared, ${chk.detail})`;
	twinRuns.set(key, res);
	report.train.push(res);
	return true;
}

export interface ParityOpts {
	write?: boolean;
	dir?: string;
	steps?: number;
	echo?: boolean;
	/** also run every process reference on every fixture input */
	fixtures?: boolean;
}

export async function parity(slug: string, po: ParityOpts = {}): Promise<{ report: ParityReport; issues: Issues }> {
	const r = roots();
	const is = new Issues();
	const report: ParityReport = { slug, checked: today(), train: [], process: [], pass: true, wrote: null, issues: [] };
	const hasPkg = existsSync(join(r.site, 'src', 'lib', 'explainers', slug, 'explainer.ts'));
	if (!hasPkg && isLegacy(slug)) {
		const { kit } = await import('../lib/kit.ts');
		const k = await kit(r.site);
		report.legacy = await legacyParity(k, slug, !!po.write, is, !!po.echo);
	} else if (!hasPkg && po.dir) {
		const { kit } = await import('../lib/kit.ts');
		const k = await kit(r.site);
		report.forward = await forwardParity(k, resolveDir(r.site, po.dir), k.ops.CORE_OPS, is);
	} else {
		const o = await openPackage(slug, r.site);
		const m = o.l.pkg.manifest;
		const dir = po.dir ? resolveDir(r.site, po.dir) : o.dir;
		const traceSrc = Object.values(o.sources).some((s) => s.kind === 'trace' && !s.assets);
		if (m.kinds.includes('trace') && traceSrc) report.forward = await forwardParity(o.kit, dir, o.l.pkg.ops, is);
		// K1: one twin run per distinct trainer and options; every source with the same pair shares its record
		const twinRuns = new Map<string, TrainResult>();
		// K2: train sources first, so a twin a train source shares with a wrapping process source keeps its strict gate
		const ordered = Object.entries(o.sources).sort(([, a], [, b]) => Number(b.kind === 'train') - Number(a.kind === 'train'));
		for (const [name, s] of ordered) {
			if (s.kind === 'train') {
				const key = twinKey(s.trainer, (s.opt ?? {}) as Record<string, unknown>);
				const seen = twinRuns.get(key);
				if (seen) {
					seen.sources!.push(name);
					continue;
				}
				const res = await trainParity(o, name, s.trainer, (s.opt ?? {}) as Record<string, unknown>, is, po.steps ?? 50, !!po.echo);
				res.sources = [name];
				twinRuns.set(key, res);
				report.train.push(res);
			} else if (s.kind === 'process' && !s.precomputed) {
				// K2: a process that makes its trainer's frames (play, fit) gets that trainer's twin and badge
				if (await wrapTwin(o, name, s, twinRuns, report, is, po)) continue;
				const d = { opt: s.opt as Record<string, unknown>, seed: s.seed, input: s.input, after: s.after, reference: s.reference };
				const first = await processParity(o, name, s.process, d, is, !!po.echo);
				report.process.push(first);
				if (po.fixtures && first.reference) for (const fx of await o.fixtures()) report.process.push(await processParity(o, name, s.process, d, is, !!po.echo, fx));
			}
		}
		if (po.write) {
			const rec = sourceParityFile(slug, report, o.sources);
			if (rec) {
				const f = join(o.l.dir, 'parity.json');
				writeJson(f, rec, 1);
				report.wroteSources = posix(f);
			}
		}
		if (po.write) {
			const specFile = join(dir, 'spec.json');
			if (!existsSync(specFile)) {
				if (report.forward) throw new AtlasError('AK301', `no ${posix(specFile)} to write`);
			} else {
				const spec = readJson<AtlasSpec>(specFile);
				let rec: ParityRecord | null = null;
				if (report.forward) {
					const f = report.forward;
					rec = {
						against: (spec.parity?.against as string) ?? 'pytorch-fp32-cpu',
						max_abs_diff: Number(f.max.toPrecision(3)),
						checked: report.checked,
						bar: f.bar,
						pass: f.pass,
						scope: 'forward',
						script: 'atlas parity',
						cases: f.cases,
						edgeCases: f.edgeCases,
						perLayer: Object.fromEntries(Object.entries(f.perLayer).map(([k, v]) => [k, Number(v.toPrecision(3))])),
						details_max_abs_diff: Number(f.explain.max.toPrecision(3))
					};
				} else if (report.train.some((t) => t.runs.length)) {
					const runs = report.train.flatMap((t) => t.runs);
					const worst = Math.max(...runs.map((x) => Math.max(x.losses, x.final)));
					rec = { against: 'pytorch-fp64-cpu', max_abs_diff: Number(worst.toPrecision(3)), checked: report.checked, bar: Math.max(...runs.map((x) => x.bar)), relative: true, pass: runs.every((x) => x.pass), scope: 'train', script: 'atlas parity', steps: Math.max(...runs.map((x) => x.steps)) };
				} else if (report.process.some((p) => p.reference)) {
					const ps = report.process.filter((p) => p.reference);
					rec = { against: 'python-reference', max_abs_diff: Number(Math.max(...ps.map((p) => p.maxFloat)).toPrecision(3)), checked: report.checked, pass: ps.every((p) => p.pass), scope: 'process', script: 'atlas parity', cases: ps.length };
				}
				if (rec) {
					spec.parity = rec;
					writeText(specFile, JSON.stringify(spec, null, 1) + '\n');
					report.wrote = posix(specFile);
				}
			}
		}
	}
	report.pass = is.errors.length === 0;
	report.issues = is.list.map((i) => ({ code: i.code, level: i.level, message: i.message }));
	writeJson(join(r.site, '.atlas', slug, 'parity.json'), report, 1);
	return { report, issues: is };
}

function parse(args: string[]): { slug: string; o: ParityOpts & { json: boolean } } {
	const pos: string[] = [];
	const o: ParityOpts & { json: boolean } = { json: false };
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--write') o.write = true;
		else if (a === '--json') o.json = true;
		else if (a === '--dir') o.dir = args[++i];
		else if (a === '--steps') o.steps = Number(args[++i]);
		else if (a === '--fixtures') o.fixtures = true;
		else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`, { fix: 'atlas parity <slug> [--write] [--json] [--dir <folder under static/>] [--steps n] [--fixtures]' });
		else pos.push(a);
	}
	if (pos.length !== 1) throw new AtlasError('AK301', 'usage: atlas parity <slug> [--write] [--json] [--dir <folder under static/>] [--steps n] [--fixtures]');
	return { slug: pos[0], o };
}

export function printReport(rep: ParityReport): void {
	console.log(`atlas parity ${rep.slug}: ${rep.pass ? 'PASS' : 'FAIL'}`);
	if (rep.legacy) {
		const L = rep.legacy;
		console.log(`  legacy script  ${L.script} --no-write: exit ${L.exit}; measured max |diff| ${sig(L.measured)} (shipped spec.parity ${sig(L.shipped)})`);
		console.log(`  spec.json      sha256 ${L.specSha.unchanged ? 'unchanged' : 'CHANGED'} (${(L.specSha.after ?? '').slice(0, 12)})${L.laneSpecSha ? `; lane copy ${L.laneSpecSha.unchanged ? 'unchanged' : 'CHANGED'}` : ''}`);
		if (L.kit) console.log(`  kit executor   ${L.kit.cases} cases, max |diff| ${sig(L.kit.max)} (bar ${sig(L.kit.bar)}), explain = trace ${sig(L.kit.explain.max)} over ${L.kit.explain.units} units`);
	}
	if (rep.forward) {
		const f = rep.forward;
		console.log(`  kind A         ${f.cases} cases (${f.edgeCases.length} edge), max |diff| ${sig(f.max)} (bar ${sig(f.bar)})${f.worst ? ` at ${f.worst.id} (case ${f.worst.case})` : ''}; explain = trace ${sig(f.explain.max)} over ${f.explain.units} units; ${f.ms.toFixed(0)} ms`);
		const top = Object.entries(f.perLayer).sort((a, b) => b[1] - a[1]).slice(0, 5);
		if (top.length) console.log(`  worst layers   ${top.map(([k, v]) => `${k} ${sig(v)}`).join(', ')}`);
	}
	for (const t of rep.train) {
		if (!t.runs.length) console.log(`  kind B         ${t.source} (${t.trainer}): ${t.note ?? 'no runs'}`);
		for (const x of t.runs) console.log(`  kind B         ${t.source} ${x.optimizer} ${x.steps} steps: losses ${sig(x.losses)}, step-1 grads ${sig(x.grads0)}, final params ${sig(x.final)} (relative; bar ${sig(x.bar)}) ${x.pass ? 'ok' : 'FAIL'}`);
	}
	for (const p of rep.process) {
		if (!p.reference) console.log(`  kind C         ${p.source} (${p.process}): ${p.note}`);
		else if (!p.pass && !p.firstDivergence && p.note) console.log(`  kind C         ${p.source}${p.fixture ? ` @${p.fixture}` : ''} (${p.process}): ${p.note} FAIL`);
		else console.log(`  kind C         ${p.source}${p.fixture ? ` @${p.fixture}` : ''} (${p.process}): ${p.frames} TS frames vs ${p.refFrames} Python frames, max float difference ${sig(p.maxFloat)}${p.firstDivergence ? `, first divergence at frame ${p.firstDivergence.frame}` : ', identical'} ${p.pass ? 'ok' : 'FAIL'}`);
	}
	if (rep.wrote) console.log(`  wrote spec.parity -> ${rep.wrote}`);
	if (rep.wroteSources) console.log(`  wrote the per-source record -> ${rep.wroteSources} (the badge of train and process steps reads it)`);
	console.log(`  report         .atlas/${rep.slug}/parity.json`);
}

export async function main(args: string[]): Promise<number> {
	const { slug, o } = parse(args);
	const { report, issues } = await parity(slug, { ...o, echo: !o.json });
	if (o.json) console.log(JSON.stringify(report));
	else {
		printReport(report);
		if (issues.list.length) console.error(issues.format());
	}
	return report.pass ? 0 : 1;
}
