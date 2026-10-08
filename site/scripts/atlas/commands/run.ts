// atlas run <slug> <source> [--fixture id] [--frames a..b] [--seed n] [--detail coarse|fine] [--opt '{"k":1}']
//           [--draws] [--max n] [--save] [--precompute] [--json]
// Records a process or trainer source in Node and prints its tape as a table, before any UI exists (design §4.12).
//   --save        .atlas/<slug>/<source>.tape.json|.bin (+ <process>.draws.json with the recorded draws, for Python
//                 (file names use fileSafe(): 'train:fit' -> 'train-fit')
//                 references that replay them); for a train source also .atlas/<slug>/train_fixture.json (init params,
//                 batches and losses of a fine-detail run: the input of python -m atlas_kit.train_reference)
//   --precompute  static/models/<slug>/tapes/<source>.tape.json|.bin (provenance 'precomputed'; phone tapes <= 2 MB)
// The recorder is W1-C's record() (src/lib/atlas/process/record.ts) when this tree has it, else the fallback in
// lib/fallback.ts (same semantics; the run header says which one ran).
// K1: the input comes from a real runtime on the fixture (the page's own: spec, model, presets, adapter), so a process
// whose input(rt) reads rt.model or rt.pre (a sampler over the exported model) records what the page records; a source
// with after: '<source>' first runs that source to its end. --fixture takes lesson fixture ids, edge:<id> (or the bare
// edge id) and preset:<id>; without it, the runtime's first input (input.start, else the first preset).
import { existsSync } from 'node:fs';
import { basename, join } from 'node:path';
import type { ExplainerRuntime, Frame, ProcessDef, RecordOpts, SourceDef, Tape, TapeHeader } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import { recordFallback, type MiniTape, isTensorLike } from '../lib/fallback.ts';
import { fileSafe, posix, writeBytes, writeJson, writeText } from '../lib/io.ts';
import { fmtNum, renderTable } from '../lib/table.ts';
import { readTapeFiles, writeTapeFiles } from '../lib/tapefile.ts';
import { importTs, loadPackage, type LoadedPackage } from '../load.ts';
import { openPackage } from '../lib/pkg.ts';
import { roots } from '../root.ts';

interface Flags {
	fixture?: string;
	frames?: [number, number];
	seed?: number;
	detail?: 'coarse' | 'fine';
	opt?: Record<string, unknown>;
	draws: boolean;
	max?: number;
	save: boolean;
	precompute: boolean;
	json: boolean;
}

export function parseArgs(args: string[]): { slug: string; source: string; flags: Flags } {
	const pos: string[] = [];
	const flags: Flags = { draws: false, save: false, precompute: false, json: false };
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		const val = () => {
			const v = args[++i];
			if (v === undefined) throw new AtlasError('AK301', `${a} needs a value`);
			return v;
		};
		if (a === '--fixture') flags.fixture = val();
		else if (a === '--frames') {
			const m = /^(\d+)\.\.(\d+)$/.exec(val());
			if (!m) throw new AtlasError('AK301', '--frames takes a..b (e.g. 0..40)');
			flags.frames = [Number(m[1]), Number(m[2])];
		} else if (a === '--seed') flags.seed = Number(val());
		else if (a === '--detail') {
			const d = val();
			if (d !== 'coarse' && d !== 'fine') throw new AtlasError('AK301', '--detail is coarse or fine');
			flags.detail = d;
		} else if (a === '--opt') {
			try {
				flags.opt = JSON.parse(val());
			} catch {
				throw new AtlasError('AK301', `--opt takes JSON, e.g. --opt '{"lr":0.1}'`);
			}
		} else if (a === '--max') flags.max = Number(val());
		else if (a === '--draws') flags.draws = true;
		else if (a === '--save') flags.save = true;
		else if (a === '--precompute') flags.precompute = true;
		else if (a === '--json') flags.json = true;
		else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`);
		else pos.push(a);
	}
	if (pos.length !== 2) throw new AtlasError('AK301', 'usage: atlas run <slug> <source> [--fixture id] [--frames a..b] [--seed n] [--detail coarse|fine] [--save] [--precompute]');
	return { slug: pos[0], source: pos[1], flags };
}

function sourcesOf(l: LoadedPackage): Record<string, SourceDef> {
	const m = l.pkg.manifest;
	return m.sources ?? (m.kinds.includes('trace') ? { fwd: { kind: 'trace' } } : {});
}

function stubRuntime(l: LoadedPackage, input: unknown, pre: ExplainerRuntime['pre']): ExplainerRuntime {
	return {
		slug: l.pkg.slug,
		spec: null,
		model: null,
		input,
		pre,
		sources: {},
		sel: {},
		fact: () => undefined,
		num: () => NaN,
		ensure: async () => 0,
		setInput: () => {}
	};
}

function pickInput(l: LoadedPackage, fixture: string | undefined): { raw: unknown; label: string } {
	const m = l.pkg.manifest;
	if (fixture) {
		const lf = l.pkg.lesson?.fixtures?.find((f) => f.id === fixture);
		if (lf) return { raw: lf.raw, label: `lesson fixture '${fixture}'` };
		const ad = m.input ? l.pkg.adapters[m.input.adapter] : undefined;
		const ef = ad?.edgeFixtures().find((f) => f.id === fixture || `edge:${f.id}` === fixture);
		if (ef) return { raw: ef.raw, label: `adapter edge fixture '${fixture}'` };
		// presets (preset:<id>) are found by the runtime (K1)
		if (fixture.startsWith('preset:')) return { raw: m.input?.start ?? null, label: `preset '${fixture}'` };
		const ids = [...(l.pkg.lesson?.fixtures ?? []).map((f) => f.id), ...(ad?.edgeFixtures().map((f) => `edge:${f.id}`) ?? []), 'preset:<id>'];
		throw new AtlasError('AK404', `fixture '${fixture}' not found`, { fix: `fixtures: ${ids.join(', ') || '(none: add LessonDef.fixtures)'}` });
	}
	return { raw: m.input?.start ?? null, label: m.input?.start !== undefined ? 'manifest.input.start' : 'null (no input.start)' };
}

async function recorder(site: string): Promise<{ name: string; record: (p: ProcessDef<any, any, any>, input: unknown, opt: Record<string, unknown>, ro: RecordOpts) => Tape }> {
	const f = join(site, 'src', 'lib', 'atlas', 'process', 'record.ts');
	if (existsSync(f)) {
		const m = (await importTs(f, 'atlas/process/record.ts')) as { record?: (p: ProcessDef<any, any, any>, input: unknown, opt?: Record<string, unknown>, ro?: RecordOpts) => Tape };
		if (typeof m.record === 'function') return { name: 'atlas/process/record.ts', record: (p, i, o, ro) => m.record!(p, i, o, ro) };
	}
	return { name: 'fallback recorder (scripts/atlas/lib/fallback.ts; W1-C record.ts not in this tree)', record: recordFallback };
}

/** Save a tape: W1-C's saveTape when present, else the fallback writer (the same layout). */
async function saveTape(site: string, tape: Tape, p: ProcessDef<any, any, any>, stem: string, provenance: TapeHeader['provenance']): Promise<{ json: string; bin: string; bytes: number }> {
	const io = join(site, 'src', 'lib', 'atlas', 'process', 'io.ts');
	if ((tape as MiniTape).frames && (tape as MiniTape).schema && !existsSync(io)) return writeTapeFiles(stem, tape as MiniTape, p.phases, provenance);
	if (!existsSync(io)) throw new AtlasError('AK301', 'cannot save: atlas/process/io.ts (W1-C) is missing and the tape is not a fallback tape');
	const m = (await importTs(io, 'atlas/process/io.ts')) as { saveTape: (t: Tape, o?: { binName?: string }) => { header: TapeHeader; blob: ArrayBuffer } };
	(tape as { provenance: string }).provenance = provenance;
	const bin = `${stem}.tape.bin`;
	const json = `${stem}.tape.json`;
	// the header's bin.file must name the file written next to it (saveTape defaults to <process>.tape.bin)
	const { header, blob } = m.saveTape(tape, { binName: basename(bin) });
	writeBytes(bin, new Uint8Array(blob));
	const n = writeText(json, JSON.stringify(header) + '\n');
	return { json, bin, bytes: n + blob.byteLength };
}

const plain = (v: unknown): unknown => (isTensorLike(v) ? { shape: v.shape, data: Array.from(v.data as ArrayLike<number>) } : v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, plain(x)])) : v);

/** train_fixture.json from a fine-detail trainer tape: init params, the batch of every step, losses. */
export function trainFixture(tape: Tape, meta: Record<string, unknown>): Record<string, unknown> {
	let init: unknown = null;
	const batches: { step: number; batch: unknown }[] = [];
	const losses: { step: number; loss: number }[] = [];
	for (let i = 0; i < tape.length; i++) {
		const f = tape.frame(i) as Frame<Record<string, unknown>>;
		if (init === null && f.state.params) init = plain(f.state.params);
		if (f.state.batch && f.step !== undefined && !batches.some((b) => b.step === f.step)) batches.push({ step: f.step, batch: plain(f.state.batch) });
		const loss = f.metrics?.loss ?? (typeof f.state.loss === 'number' ? (f.state.loss as number) : undefined);
		if (loss !== undefined && f.step !== undefined && !losses.some((x) => x.step === f.step)) losses.push({ step: f.step, loss });
	}
	if (!init) throw new AtlasError('AK203', 'the trainer tape has no params frame (expected an init frame with state.params)');
	if (!batches.length) throw new AtlasError('AK203', 'the trainer tape has no batches (fine detail should carry state.batch)', { likely: 'trainingProcess did not record batches at fine detail' });
	return { format: 'atlas-train-fixture/1', ...meta, init, batches, losses };
}

export async function main(args: string[]): Promise<number> {
	const { slug, source, flags } = parseArgs(args);
	const r = roots();
	const l = await loadPackage(slug, r.site);
	const sources = sourcesOf(l);
	const src = sources[source];
	if (!src) throw new AtlasError('AK301', `package '${slug}' has no source '${source}'`, { fix: `sources: ${Object.keys(sources).join(', ') || '(none)'}` });

	if (src.kind === 'tape') {
		const dir = join(r.site, 'static', l.pkg.manifest.assets?.dir ?? `models/${slug}`);
		const stem = join(dir, src.file).replace(/\.tape\.json$/, '');
		const { header, frames } = readTapeFiles(stem);
		const t = { length: frames.length, frame: (i: number) => frames[i] } as unknown as Tape;
		const [a, b] = flags.frames ?? [0, Math.min(frames.length - 1, 40)];
		console.log(`atlas run ${slug} ${source}: precomputed tape ${posix(stem)}.tape.json (${header.process}, ${frames.length} frames, seed ${header.seed})`);
		console.log(renderTable(t, a, b));
		return 0;
	}
	if (src.kind !== 'process' && src.kind !== 'train')
		throw new AtlasError('AK301', `atlas run prints process and train sources; '${source}' is a ${src.kind} source`, {
			fix: 'trace, scan and live-model sources are checked by `atlas parity` and shown by `atlas lesson print`'
		});
	const key = src.kind === 'process' ? src.process : `train:${src.trainer}`;
	const p = l.pkg.processes[key];
	if (!p)
		throw new AtlasError('AK301', `source '${source}' runs process '${key}', which the package does not register`, {
			likely: l.notes.join('; ') || undefined,
			fix: `registered: ${Object.keys(l.pkg.processes).join(', ') || '(none)'}`
		});

	let { raw, label } = pickInput(l, flags.fixture);
	let input: unknown = raw;
	try {
		const o = await openPackage(slug, r.site);
		if (flags.fixture) {
			const fx = (await o.fixtures()).find((f) => f.id === flags.fixture || f.id === `edge:${flags.fixture}`);
			if (fx) {
				raw = fx.raw;
				label = `fixture '${fx.id}'`;
			}
		}
		const rt = await o.runtime(flags.fixture ? raw : undefined);
		if (!flags.fixture) label = "the runtime's first input (input.start, else the first preset)";
		if (src.kind === 'process' && src.after) await rt.ensure(src.after, 'last');
		input = src.kind === 'process' && src.input ? src.input(rt) : rt.input;
		rt.close?.();
	} catch (e) {
		// a tree without the kit runtime (very old lanes): the stub runtime, as before K1
		if (src.kind === 'process' && src.input) {
			const ad = l.pkg.manifest.input ? l.pkg.adapters[l.pkg.manifest.input.adapter] : undefined;
			let pre: ExplainerRuntime['pre'] = null;
			if (ad) pre = ad.preprocess(raw, null, l.pkg.manifest.input?.options ?? {});
			input = src.input(stubRuntime(l, raw, pre));
		}
		label += ` (stub runtime: ${(e as Error).message.split('\n')[0]})`;
	}
	const opt: Record<string, unknown> = { ...(p.defaults as Record<string, unknown>), ...(src.opt ?? {}), ...(flags.opt ?? {}) };
	const seed = flags.seed ?? (src.kind === 'process' ? src.seed : undefined) ?? (typeof opt.seed === 'number' ? (opt.seed as number) : 1);
	// K2: a process source's own detail too (word2vec's 'one' printed 'detail coarse' for a fine source)
	const detail = flags.save && src.kind === 'train' ? 'fine' : (flags.detail ?? src.detail ?? 'coarse');
	const ro: RecordOpts = { seed, detail, recordDraws: flags.draws || flags.save, ...(flags.max ? { maxFrames: flags.max } : {}) };
	const rec = await recorder(r.site);
	const t0 = performance.now();
	const tape = rec.record(p, input, opt, ro);
	const ms = performance.now() - t0;

	const notes = [...l.notes, ...(((tape as MiniTape).notes as string[] | undefined) ?? [])];
	if (flags.json) {
		const last = tape.length - 1;
		console.log(JSON.stringify({ slug, source, process: key, recorder: rec.name, seed, opt, detail, frames: tape.length, segments: tape.segments(), lastFrame: last >= 0 ? plain(tape.frame(last)) : null, notes }));
	} else {
		console.log(`atlas run ${slug} ${source}: process '${key}' (${p.kind}), ${tape.length} frames in ${ms.toFixed(1)} ms`);
		console.log(`  input   ${label}`);
		console.log(`  seed    ${seed}   detail ${detail}   opt ${JSON.stringify(opt)}`);
		const segs = tape.segments();
		const count: Record<string, number> = {};
		for (let i = 0; i < tape.length; i++) count[tape.frame(i).phase] = (count[tape.frame(i).phase] ?? 0) + 1;
		console.log(`  phases  ${Object.entries(count).map(([ph, n]) => `${ph} ×${n}`).join(', ')} (${segs.length} segment${segs.length === 1 ? '' : 's'})`);
		console.log(`  using   ${rec.name}`);
		for (const n of notes) console.log(`  note    ${n}`);
		const [a, b] = flags.frames ?? [0, Math.min(tape.length - 1, 40)];
		console.log('');
		console.log(renderTable(tape, a, b));
		if (!flags.frames && tape.length > 41) console.log(`… ${tape.length - 41} more frames (--frames a..b)`);
		const names = new Set<string>();
		for (let i = 0; i < tape.length; i++) for (const k of Object.keys(tape.frame(i).metrics ?? {})) names.add(k);
		if (names.size) {
			console.log('');
			for (const n of names) {
				const m = Array.from(tape.metric(n)).filter((x) => !Number.isNaN(x));
				if (m.length) console.log(`  metric ${n}: first ${fmtNum(m[0])}, last ${fmtNum(m[m.length - 1])}, min ${fmtNum(Math.min(...m))}, max ${fmtNum(Math.max(...m))} (${m.length} frames)`);
			}
		}
	}

	if (flags.save) {
		const dir = join(r.site, '.atlas', slug);
		const saved = await saveTape(r.site, tape, p, join(dir, fileSafe(source)), 'live');
		console.log(`saved   ${posix(saved.json)} (+ .tape.bin), ${(saved.bytes / 1024).toFixed(1)} KB`);
		const draws = Array.from({ length: tape.length }, (_, i) => tape.frame(i).draws ?? []);
		if (draws.some((d) => d.length)) {
			const f = join(dir, `${fileSafe(p.id)}.draws.json`);
			writeJson(f, { format: 'atlas-draws/1', process: p.id, source, seed, opt, frames: draws });
			console.log(`saved   ${posix(f)} (${draws.reduce((s, d) => s + d.length, 0)} draws over ${tape.length} frames)`);
		}
		if (src.kind === 'train') {
			const f = join(dir, 'train_fixture.json');
			writeJson(f, trainFixture(tape, { slug, source, trainer: src.trainer, seed, opt, detail }));
			console.log(`saved   ${posix(f)}`);
		}
	}
	if (flags.precompute) {
		const dir = join(r.site, 'static', l.pkg.manifest.assets?.dir ?? `models/${slug}`, 'tapes');
		const saved = await saveTape(r.site, tape, p, join(dir, fileSafe(source)), 'precomputed');
		console.log(`precomputed ${posix(saved.json)} (+ .tape.bin), ${(saved.bytes / 1024).toFixed(1)} KB`);
		if (saved.bytes > 2e6)
			throw new AtlasError('AK303', `the precomputed tape is ${(saved.bytes / 1e6).toFixed(2)} MB (> 2 MB for phones)`, {
				where: posix(saved.json),
				fix: 'record coarser (snapshotEvery), keep fewer frames (RecordOpts.keep) or a smaller model'
			});
	}
	return 0;
}
