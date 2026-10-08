// Hand-written validator for atlas-tape/1 headers (design §4.2, §6.3 "Schema" gate). The layout is documented in
// process/io.ts and pipeline/atlas_kit/tape.py.
// Returns a list of problems ([] = valid). Structural only: it never decodes the blob.

const PROVENANCE = new Set(['live', 'precomputed', 'illustrative']);
const FIELD_KINDS = new Set(['tensor', 'scalar', 'int', 'index', 'text', 'json']);
const ROLES = new Set(['activation', 'param', 'grad', 'optimizer', 'data', 'target', 'output']);
const SCALES = new Set(['diverging', 'sequential', 'categorical']);
const PHASE_KINDS = new Set(['input', 'forward', 'loss', 'backward', 'update', 'observe', 'act', 'reward', 'message', 'aggregate', 'sample', 'merge', 'output', 'other']);
const DTYPE_BYTES: Record<string, number> = { f32: 4, f64: 8, i32: 4 };
const NUMS = new Set(['NaN', 'Infinity', '-Infinity']);

const isObj = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);
const isI18n = (v: unknown) => isObj(v) && typeof v.en === 'string' && (v.zh === undefined || typeof v.zh === 'string');

function checkValue(v: unknown, where: string, blobBytes: number | undefined, out: string[]): void {
	if (out.length > 50) return;
	if (Array.isArray(v)) {
		v.forEach((x, i) => checkValue(x, `${where}[${i}]`, blobBytes, out));
		return;
	}
	if (!isObj(v)) {
		if (typeof v === 'number' && !Number.isFinite(v)) out.push(`${where}: non-finite number must be written as {"$f": ...}`);
		return;
	}
	if (typeof v.$t === 'string') {
		const bytes = DTYPE_BYTES[v.$t];
		const shape = v.shape;
		if (!bytes) out.push(`${where}: unknown dtype ${String(v.$t)} (f32 | f64 | i32)`);
		else if (!Number.isInteger(v.o) || !Number.isInteger(v.n) || (v.o as number) < 0 || (v.n as number) < 0) out.push(`${where}: o and n must be non-negative integers`);
		else if ((v.o as number) % 8 !== 0) out.push(`${where}: offset ${v.o} is not 8-byte aligned`);
		else if (blobBytes !== undefined && (v.o as number) + (v.n as number) * bytes > blobBytes) out.push(`${where}: array ends at byte ${(v.o as number) + (v.n as number) * bytes}, blob has ${blobBytes}`);
		if (!Array.isArray(shape) || shape.some((d) => !Number.isInteger(d) || d < 0)) out.push(`${where}: shape must be non-negative integers`);
		else if (shape.reduce((a: number, b: number) => a * b, 1) !== v.n) out.push(`${where}: shape [${shape.join(', ')}] does not hold n = ${v.n} values`);
		return;
	}
	if (typeof v.$f === 'string' && Object.keys(v).length === 1) {
		if (!NUMS.has(v.$f)) out.push(`${where}: unknown $f ${String(v.$f)}`);
		return;
	}
	for (const [k, x] of Object.entries(v)) checkValue(x, `${where}.${k}`, blobBytes, out);
}

export function validateTapeHeader(h: unknown, blobBytes?: number): string[] {
	const out: string[] = [];
	if (!isObj(h)) return ['header is not an object'];
	if (h.format !== 'atlas-tape/1') out.push(`format must be 'atlas-tape/1' (got ${JSON.stringify(h.format)})`);
	if (typeof h.process !== 'string' || !h.process) out.push('process must be a non-empty string');
	if (typeof h.version !== 'string') out.push('version must be a string');
	if (typeof h.seed !== 'number' || !Number.isInteger(h.seed)) out.push('seed must be an integer');
	if (!PROVENANCE.has(h.provenance as string)) out.push(`provenance must be live | precomputed | illustrative (got ${JSON.stringify(h.provenance)})`);
	const schema = h.schema;
	if (!isObj(schema)) out.push('schema must be an object');
	else
		for (const [k, d] of Object.entries(schema)) {
			if (!isObj(d)) {
				out.push(`schema.${k} must be an object`);
				continue;
			}
			if (!FIELD_KINDS.has(d.kind as string)) out.push(`schema.${k}.kind ${JSON.stringify(d.kind)} is not one of ${[...FIELD_KINDS].join(' ')}`);
			if (!isI18n(d.label)) out.push(`schema.${k}.label must be { en, zh? }`);
			if (d.role !== undefined && !ROLES.has(d.role as string)) out.push(`schema.${k}.role ${JSON.stringify(d.role)} is unknown`);
			if (d.scale !== undefined && !SCALES.has(d.scale as string)) out.push(`schema.${k}.scale ${JSON.stringify(d.scale)} is unknown`);
			if (d.track !== undefined && d.track !== 'snapshot' && d.track !== 'changes') out.push(`schema.${k}.track must be snapshot | changes`);
			if (d.tol !== undefined && d.tol !== 'exact' && typeof d.tol !== 'number') out.push(`schema.${k}.tol must be a number or 'exact'`);
			if ('derive' in d) out.push(`schema.${k}: functions cannot be stored (write derived: true)`);
		}
	const phases = h.phases;
	if (!isObj(phases)) out.push('phases must be an object');
	else
		for (const [k, p] of Object.entries(phases)) {
			if (!isObj(p) || !isI18n(p.label) || !PHASE_KINDS.has(p.kind as string)) out.push(`phases.${k} must be { label: { en, zh? }, kind: ${[...PHASE_KINDS].join('|')}, tex? }`);
			else if (p.tex !== undefined && typeof p.tex !== 'string') out.push(`phases.${k}.tex must be a string`);
		}
	if (h.keyEvery !== undefined && (!Number.isInteger(h.keyEvery) || (h.keyEvery as number) < 1)) out.push('keyEvery must be a positive integer');
	if (h.bin !== undefined) {
		const b = h.bin as Record<string, unknown>;
		if (!isObj(b) || typeof b.file !== 'string' || !Number.isInteger(b.bytes)) out.push('bin must be { file, bytes }');
		else if (blobBytes !== undefined && b.bytes !== blobBytes) out.push(`bin.bytes is ${b.bytes} but the blob has ${blobBytes} bytes`);
	}
	if (!Array.isArray(h.frames)) {
		out.push('frames must be an array');
		return out;
	}
	const derived = new Set(isObj(schema) ? Object.entries(schema).filter(([, d]) => isObj(d) && d.derived === true).map(([k]) => k) : []);
	h.frames.forEach((f, i) => {
		if (out.length > 50) return;
		const at = `frames[${i}]`;
		if (!isObj(f)) return out.push(`${at} is not an object`);
		if (typeof f.phase !== 'string') out.push(`${at}.phase must be a string`);
		else if (isObj(phases) && Object.keys(phases).length && !(f.phase in phases)) out.push(`${at}.phase '${f.phase}' is not declared in phases`);
		for (const k of ['step', 'sub'] as const) if (f[k] !== undefined && typeof f[k] !== 'number') out.push(`${at}.${k} must be a number`);
		if (f.key !== undefined && f.key !== true) out.push(`${at}.key must be true when present`);
		if (!isObj(f.state)) out.push(`${at}.state must be an object`);
		else {
			for (const k of Object.keys(f.state)) {
				if (derived.has(k)) out.push(`${at}.state.${k} is a derived field and must not be stored`);
				else if (isObj(schema) && Object.keys(schema).length && !(k in schema)) out.push(`${at}.state.${k} is not in the schema`);
			}
			checkValue(f.state, `${at}.state`, blobBytes, out);
		}
		if (f.changes !== undefined) {
			if (!Array.isArray(f.changes)) out.push(`${at}.changes must be an array`);
			else
				f.changes.forEach((c, j) => {
					if (!isObj(c) || typeof c.field !== 'string') out.push(`${at}.changes[${j}] needs a field`);
					else if (c.index !== undefined && (!Array.isArray(c.index) || c.index.some((x) => !Number.isInteger(x)))) out.push(`${at}.changes[${j}].index must be integers`);
				});
		}
		if (f.metrics !== undefined) {
			if (!isObj(f.metrics)) out.push(`${at}.metrics must be an object`);
			else checkValue(f.metrics, `${at}.metrics`, blobBytes, out);
		}
		if (f.draws !== undefined && (!Array.isArray(f.draws) || f.draws.some((x) => typeof x !== 'number' && !(isObj(x) && typeof x.$f === 'string')))) out.push(`${at}.draws must be numbers`);
		if (f.caption !== undefined && !isI18n(f.caption)) out.push(`${at}.caption must be { en, zh? }`);
	});
	return out;
}
