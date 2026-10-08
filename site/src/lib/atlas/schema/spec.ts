// Hand-written validator for spec.json (spec 1.1, and v1 specs read through legacy.ts). Design §6.3 Schema gate:
//   structure and types; slug; unique ids without '#'; every `in` names an earlier layer or a declared input;
//   ops registered; attributes match OpDef.attrs (type, required, unknown); weights resolve to spec.tensors;
//   inferShape == outShape at the extremes of every symbolic dim (AK302).
//   validateSpec(raw, { ops?, normalize?, weightsBytes? }) -> SchemaIssue[]   (errors and warnings; never throws)
//   assertSpec(raw, o) -> AtlasSpec                                             (throws on any error)
// Also exports the shared issue helpers used by schema/ref.ts and schema/facts.ts.
import type { AtlasLayer, AtlasSpec, Axis, Dim, I18n, OpDef, OpRegistry } from '../types/index.ts';
import { normalizeSpec } from '../graph/legacy.ts';
import { mergeParams, toRegistry } from '../ops/registry.ts';
import { weightName } from '../ops/common.ts';

// ---------------------------------------------------------------- shared helpers

export interface SchemaIssue {
	/** AK301 schema, AK302 inferShape ≠ outShape at a dim extreme, AK303 size budget */
	code: 'AK301' | 'AK302' | 'AK303';
	level: 'error' | 'warn';
	/** path of the offending value, e.g. "layers[3].params.kernel" */
	path: string;
	message: string;
	fix?: string;
}

export class Issues {
	list: SchemaIssue[] = [];
	err(path: string, message: string, fix?: string, code: SchemaIssue['code'] = 'AK301'): void {
		this.list.push(fix ? { code, level: 'error', path, message, fix } : { code, level: 'error', path, message });
	}
	warn(path: string, message: string, fix?: string, code: SchemaIssue['code'] = 'AK301'): void {
		this.list.push(fix ? { code, level: 'warn', path, message, fix } : { code, level: 'warn', path, message });
	}
}

export const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null && !Array.isArray(x);
export const isInt = (x: unknown): x is number => typeof x === 'number' && Number.isInteger(x);
export const isStr = (x: unknown): x is string => typeof x === 'string';

/** An I18n needs a non-empty `en`; a missing `zh` is a warning (lesson lint AK406 makes it an error for lesson titles). */
export function checkI18n(is: Issues, path: string, x: unknown, required = true): x is I18n {
	if (x === undefined) {
		if (required) is.err(path, 'missing bilingual text { en, zh }');
		return false;
	}
	if (!isObj(x) || !isStr(x.en) || !x.en.trim()) {
		is.err(path, 'must be { en: non-empty string, zh?: string }');
		return false;
	}
	if (x.zh === undefined || x.zh === '') is.warn(`${path}.zh`, 'Chinese text is missing');
	else if (!isStr(x.zh)) is.err(`${path}.zh`, 'must be a string');
	return true;
}

export function errorsOf(list: SchemaIssue[]): SchemaIssue[] {
	return list.filter((i) => i.level === 'error');
}

/** One line per issue, for test output and CLI reports. */
export function formatIssues(list: SchemaIssue[]): string {
	return list.map((i) => `${i.code} ${i.level.padEnd(5)} ${i.path}: ${i.message}${i.fix ? `  (fix: ${i.fix})` : ''}`).join('\n');
}

/** Decoded size of a base64 payload without decoding it, or −1 when it is not canonical base64. */
export function b64Bytes(b64: string): number {
	if (b64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(b64)) return -1;
	const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
	return (b64.length / 4) * 3 - pad;
}

// ---------------------------------------------------------------- spec

const AXES = new Set<Axis>(['c', 'y', 'x', 't', 'd', 'h', 'tq', 'tk', 'n', 'f', 'class', 'vocab', 'p', 'xy', 'k', 'nk', 'b']);
const PRECISION = new Set(['fp32', 'f64', 'fp16', 'mixed']);
const KINDS = new Set(['trace', 'train', 'process']);
const SLUG = /^[a-z0-9][a-z0-9-]*$/;

function checkAttrType(type: string, v: unknown): boolean {
	switch (type) {
		case 'int':
			return isInt(v);
		case 'float':
			return typeof v === 'number' && Number.isFinite(v);
		case 'bool':
			return typeof v === 'boolean';
		case 'string':
			return typeof v === 'string';
		case 'int[]':
			return Array.isArray(v) && v.every(isInt);
		case 'float[]':
			return Array.isArray(v) && v.every((x) => typeof x === 'number' && Number.isFinite(x));
		default:
			return false;
	}
}

function checkShape(is: Issues, path: string, shape: unknown, symbols: Set<string>): shape is Dim[] {
	if (!Array.isArray(shape)) {
		is.err(path, 'must be an array of dims (non-negative integers or declared symbols)');
		return false;
	}
	let ok = true;
	shape.forEach((d, i) => {
		if (isInt(d) && d >= 0) return;
		if (isStr(d) && symbols.has(d)) return;
		ok = false;
		is.err(`${path}[${i}]`, isStr(d) ? `symbolic dim '${d}' is not declared in spec.dims` : `invalid dim ${JSON.stringify(d)}`, isStr(d) ? `add spec.dims.${d} = { min, max, label }` : undefined);
	});
	return ok;
}

export interface ValidateSpecOptions {
	/** check ops, attributes, weights and shapes against these ops (CORE_OPS or a package's list) */
	ops?: OpRegistry | OpDef[];
	/** read v1 specs through normalizeSpec first (default true) */
	normalize?: boolean;
	/** byte length of weights.bin, checked against spec.tensors */
	weightsBytes?: number;
	/** the folder name the spec lives in (slug must equal it) */
	folder?: string;
}

export function validateSpec(raw: unknown, o: ValidateSpecOptions = {}): SchemaIssue[] {
	const is = new Issues();
	if (!isObj(raw)) {
		is.err('', 'spec must be a JSON object');
		return is.list;
	}
	let s: AtlasSpec;
	if (o.normalize === false) s = raw as AtlasSpec;
	else {
		try {
			s = normalizeSpec(raw);
		} catch (e) {
			is.err('', `normalizeSpec failed: ${(e as Error).message}`);
			return is.list;
		}
	}

	// top level
	if (!isStr(s.slug) || !SLUG.test(s.slug)) is.err('slug', 'must be a lowercase slug (a-z, 0-9, -)');
	else if (o.folder !== undefined && o.folder !== s.slug) is.err('slug', `slug '${s.slug}' does not match its folder '${o.folder}'`);
	checkI18n(is, 'name', s.name);
	if (!isStr(s.version) || !s.version) is.err('version', 'must be a non-empty string');
	if (!isInt(s.params) || s.params < 0) is.err('params', 'must be a non-negative integer (number of parameters)');
	if (!PRECISION.has(s.precision as string)) is.err('precision', `must be one of ${[...PRECISION].join(', ')}`);

	// dims
	const symbols = new Set<string>();
	if (s.dims !== undefined) {
		if (!isObj(s.dims)) is.err('dims', 'must be an object { <name>: { min, max, label } }');
		else
			for (const [k, d] of Object.entries(s.dims)) {
				const p = `dims.${k}`;
				if (!isObj(d) || !isInt(d.min) || !isInt(d.max) || d.min < 0 || d.max < d.min) is.err(p, 'needs integers 0 <= min <= max');
				else symbols.add(k);
				if (isObj(d)) checkI18n(is, `${p}.label`, d.label);
			}
	}

	// tensors
	const tensors = new Map<string, number[]>();
	if (!Array.isArray(s.tensors)) is.err('tensors', 'must be an array');
	else {
		const ranges: { name: string; a: number; b: number }[] = [];
		let unique = 0;
		const seenRange = new Set<string>();
		s.tensors.forEach((t, i) => {
			const p = `tensors[${i}]`;
			if (!isObj(t) || !isStr(t.name) || !t.name) return is.err(p, 'needs { name, shape, offset, length }');
			if (tensors.has(t.name)) is.err(`${p}.name`, `duplicate tensor name '${t.name}'`);
			if (!Array.isArray(t.shape) || !t.shape.every((d) => isInt(d) && d >= 0)) return is.err(`${p}.shape`, 'must be non-negative integers');
			const n = t.shape.reduce((a, b) => a * b, 1);
			if (!isInt(t.offset) || t.offset < 0) is.err(`${p}.offset`, 'must be a non-negative integer (in floats)');
			if (t.length !== n) is.err(`${p}.length`, `is ${t.length}, shape [${t.shape}] has ${n} floats`);
			tensors.set(t.name, t.shape.slice());
			const key = `${t.offset}:${t.length}`;
			if (!seenRange.has(key)) {
				seenRange.add(key);
				unique += n;
				ranges.push({ name: t.name, a: t.offset, b: t.offset + t.length });
			}
		});
		ranges.sort((x, y) => x.a - y.a);
		for (let i = 1; i < ranges.length; i++) {
			if (ranges[i].a < ranges[i - 1].b) is.err('tensors', `'${ranges[i].name}' overlaps '${ranges[i - 1].name}' (only identical ranges may be shared, for tied weights)`);
		}
		if (isInt(s.params) && unique !== s.params) is.warn('params', `spec.params is ${s.params} but the distinct tensors hold ${unique} floats`);
		const end = ranges.reduce((m, r) => Math.max(m, r.b), 0);
		if (o.weightsBytes !== undefined && o.weightsBytes < end * 4) is.err('tensors', `weights file has ${o.weightsBytes} bytes, the tensors need ${end * 4}`);
	}

	// inputs
	const inputNames = new Map<string, Dim[]>();
	if (s.inputs !== undefined) {
		if (!Array.isArray(s.inputs)) is.err('inputs', 'must be an array of declared inputs');
		else
			s.inputs.forEach((d, i) => {
				const p = `inputs[${i}]`;
				if (!isObj(d) || !isStr(d.name) || !d.name) return is.err(p, 'needs { name, adapter, kind, shape }');
				if (inputNames.has(d.name)) is.err(`${p}.name`, `duplicate input '${d.name}'`);
				if (!isStr(d.adapter) || !d.adapter) is.err(`${p}.adapter`, 'must name an input adapter (drawing28, text-bpe, …)');
				if (!isStr(d.kind) || !d.kind) is.err(`${p}.kind`, 'must be a catalog InputKind (drawing, text, …)');
				if (d.dtype !== undefined && d.dtype !== 'f32' && d.dtype !== 'i32') is.err(`${p}.dtype`, "must be 'f32' or 'i32'");
				if (checkShape(is, `${p}.shape`, d.shape, symbols)) inputNames.set(d.name, d.shape);
				if (d.axes !== undefined && (!Array.isArray(d.axes) || d.axes.length !== (d.shape?.length ?? -1) || !d.axes.every((a) => AXES.has(a)))) is.err(`${p}.axes`, 'must list one known axis per dim');
			});
	} else if (!isObj(s.input)) is.err('inputs', 'no declared inputs (spec 1.1 `inputs` or v1 `input`)');

	// layers
	const layerIds = new Map<string, AtlasLayer>();
	if (!Array.isArray(s.layers) || s.layers.length === 0) is.err('layers', 'must be a non-empty array');
	else
		s.layers.forEach((l, i) => {
			const p = `layers[${i}]`;
			if (!isObj(l) || !isStr(l.id) || !l.id) return is.err(p, 'needs { id, op, label, in, outShape }');
			const pid = `${p} '${l.id}'`;
			if (l.id.includes('#')) is.err(`${pid}.id`, "'#' is reserved for sub-steps");
			if (layerIds.has(l.id)) is.err(`${pid}.id`, `duplicate layer id '${l.id}'`);
			if (!isStr(l.op) || !l.op) is.err(`${pid}.op`, 'must name an op');
			checkI18n(is, `${pid}.label`, l.label);
			if (!Array.isArray(l.in) || !l.in.every(isStr)) is.err(`${pid}.in`, 'must be an array of layer ids or input names');
			else
				for (const n of l.in) {
					if (!layerIds.has(n) && !inputNames.has(n)) is.err(`${pid}.in`, `'${n}' is neither an earlier layer nor a declared input`, 'list layers in execution order');
				}
			checkShape(is, `${pid}.outShape`, l.outShape, symbols);
			if (l.params !== undefined && !isObj(l.params)) is.err(`${pid}.params`, 'must be an object');
			if (l.w !== undefined) {
				if (!isObj(l.w)) is.err(`${pid}.w`, 'must map weight slots to tensor names');
				else for (const [slot, name] of Object.entries(l.w)) if (!isStr(name) || !tensors.has(name)) is.err(`${pid}.w.${slot}`, `tensor '${String(name)}' is not in spec.tensors`);
			}
			if (l.desc !== undefined) checkI18n(is, `${pid}.desc`, l.desc);
			if (l.axes !== undefined && (!Array.isArray(l.axes) || l.axes.length !== (l.outShape?.length ?? -1) || !l.axes.every((a) => AXES.has(a)))) is.err(`${pid}.axes`, 'must list one known axis per output dim');
			if (l.sub !== undefined && (!Array.isArray(l.sub) || !l.sub.every((x) => isStr(x) && x && !x.includes('#')))) is.err(`${pid}.sub`, "must be sub-step names without '#'");
			if (l.code !== undefined && (!isObj(l.code) || !isStr(l.code.file) || !isInt(l.code.line))) is.err(`${pid}.code`, 'must be { file, line, end? }');
			if (l.teach !== undefined) {
				if (!isObj(l.teach)) is.err(`${pid}.teach`, 'must be an object');
				else {
					if (l.teach.lesson !== undefined && !['step', 'fold', 'skip'].includes(l.teach.lesson)) is.err(`${pid}.teach.lesson`, "must be 'step', 'fold' or 'skip'");
					if (l.teach.title !== undefined) checkI18n(is, `${pid}.teach.title`, l.teach.title);
				}
			}
			if (l.ui !== undefined && !isObj(l.ui)) is.err(`${pid}.ui`, 'must be an object');
			if (l.rf !== undefined && (!isObj(l.rf) || !isInt(l.rf.size) || !isInt(l.rf.jump))) is.err(`${pid}.rf`, 'must be { size, jump }');
			layerIds.set(l.id, l);
		});

	if (s.outputs !== undefined && (!Array.isArray(s.outputs) || !s.outputs.every((x) => layerIds.has(x)))) is.err('outputs', 'must list layer ids');
	if (s.files !== undefined && (!isObj(s.files) || !Object.values(s.files).every(isStr))) is.err('files', 'must map roles to file names');
	if (s.atlas !== undefined) {
		if (!isObj(s.atlas) || s.atlas.spec !== '1.1') is.err('atlas.spec', "must be '1.1'");
		else {
			if (!Array.isArray(s.atlas.kinds) || !s.atlas.kinds.length || !s.atlas.kinds.every((k) => KINDS.has(k))) is.err('atlas.kinds', 'must list trace / train / process');
			if (!isStr(s.atlas.exporter)) is.err('atlas.exporter', 'must name the exporter');
		}
	}

	// parity
	if (!isObj(s.parity)) is.err('parity', 'must be an object');
	else {
		const p = s.parity;
		if (!isStr(p.against)) is.err('parity.against', 'must name the reference (pytorch-fp32-cpu, …)');
		if (p.max_abs_diff !== null && !(typeof p.max_abs_diff === 'number' && Number.isFinite(p.max_abs_diff))) is.err('parity.max_abs_diff', 'must be a number, or null until `atlas parity` runs');
		if (p.checked !== null && !isStr(p.checked)) is.err('parity.checked', 'must be a date string or null');
		if (p.bar !== undefined && !(typeof p.bar === 'number' && p.bar > 0)) is.err('parity.bar', 'must be a positive number');
		if (p.pass !== undefined && typeof p.pass !== 'boolean') is.err('parity.pass', 'must be a boolean');
		if (p.perLayer !== undefined && (!isObj(p.perLayer) || !Object.values(p.perLayer).every((v) => typeof v === 'number'))) is.err('parity.perLayer', 'must map ids to numbers');
	}

	if (o.ops && Array.isArray(s.layers)) checkOps(is, s, toRegistry(o.ops), tensors, inputNames);
	return is.list;
}

function checkOps(is: Issues, s: AtlasSpec, ops: OpRegistry, tensors: Map<string, number[]>, inputs: Map<string, Dim[]>): void {
	const ready: { layer: AtlasLayer; def: OpDef; params: Record<string, unknown>; w: Record<string, number[]> }[] = [];
	let broken = false;
	s.layers.forEach((l, i) => {
		const pid = `layers[${i}] '${l.id}'`;
		if (!ops.has(l.op)) {
			is.err(`${pid}.op`, `op '${l.op}' is not registered`, "add it to the package's ops.ts (or `atlas new op`)");
			broken = true;
			return;
		}
		const r = ops.get(l.op);
		const params = mergeParams(r.def, r.params, l.params);
		for (const [k, v] of Object.entries(l.params ?? {})) {
			const a = r.def.attrs[k];
			if (!a) is.err(`${pid}.params.${k}`, `op '${r.def.op}' has no attribute '${k}' (known: ${Object.keys(r.def.attrs).join(', ') || 'none'})`);
			else if (!checkAttrType(a.type, v)) is.err(`${pid}.params.${k}`, `must be ${a.type}, got ${JSON.stringify(v)}`);
		}
		for (const [k, a] of Object.entries(r.def.attrs)) if (a.required && params[k] === undefined) is.err(`${pid}.params.${k}`, `required attribute '${k}' of '${r.def.op}' is missing`);
		const w: Record<string, number[]> = {};
		for (const ws of r.def.weights ?? []) {
			const name = weightName(l, ws.slot);
			const shape = tensors.get(name);
			if (shape) w[ws.slot] = shape;
			else if (!ws.optional) {
				is.err(`${pid}`, `weight slot '${ws.slot}' -> tensor '${name}' is not in spec.tensors`);
				broken = true;
			}
		}
		if (l.sub) for (const sub of l.sub) if (!(r.def.subs ?? []).some((x) => x.name === sub)) is.err(`${pid}.sub`, `op '${r.def.op}' emits no sub-step '${sub}'`);
		ready.push({ layer: l, def: r.def, params, w });
	});
	if (broken) return;
	// inferShape at the extremes of every symbolic dim
	const extremes: Record<string, number>[] = [];
	const names = Object.keys(s.dims ?? {});
	if (names.length) {
		extremes.push(Object.fromEntries(names.map((k) => [k, s.dims![k].min])));
		extremes.push(Object.fromEntries(names.map((k) => [k, s.dims![k].max])));
	} else extremes.push({});
	for (const dims of extremes) {
		const label = names.length ? ` at ${names.map((k) => `${k}=${dims[k]}`).join(', ')}` : '';
		const sub = (shape: Dim[]): Dim[] => shape.map((d) => (typeof d === 'string' && dims[d] !== undefined ? dims[d] : d));
		const shapes = new Map<string, Dim[]>();
		for (const [k, v] of inputs) shapes.set(k, sub(v));
		for (const { layer, def, params, w } of ready) {
			const ins = layer.in.map((n) => shapes.get(n));
			if (ins.some((x) => !x)) break;
			let got: Dim[];
			try {
				got = def.inferShape(ins as Dim[][], params, w);
			} catch (e) {
				is.err(`layer '${layer.id}'`, `inferShape failed${label}: ${(e as Error).message}`, undefined, 'AK302');
				break;
			}
			const want = sub(layer.outShape);
			if (got.length !== want.length || got.some((d, i) => d !== want[i])) is.err(`layer '${layer.id}'.outShape`, `inferShape gives [${got}] but outShape is [${want}]${label}`, undefined, 'AK302');
			shapes.set(layer.id, want);
		}
	}
}

/** Validate and return the normalized spec; throws one error listing every error-level issue. */
export function assertSpec(raw: unknown, o: ValidateSpecOptions = {}): AtlasSpec {
	const issues = errorsOf(validateSpec(raw, o));
	if (issues.length) throw new Error(`spec.json is invalid:\n${formatIssues(issues)}`);
	return o.normalize === false ? (raw as AtlasSpec) : normalizeSpec(raw);
}
