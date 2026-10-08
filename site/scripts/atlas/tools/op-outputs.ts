// Runs every TS op (CORE_OPS, or a package's ops.ts) on the inputs of its torch goldens and prints what the TS op
// computed, so the Python mirrors can be compared with the TS ops on exactly the same numbers (W1-E2 acceptance 2;
// pipeline/atlas_kit/tests/test_mirrors.py calls it).
//
//   node scripts/atlas/tools/op-outputs.ts [--op conv2d] [--goldens <dir>] [--ops <package ops.ts>] [--out file.json]
//
// Output: { format: 'atlas-op-outputs/1', ops: { <op>: [{ id, op, out: {shape, dtype, b64}, subs?: {...} }] } }
// dtype is 'float32' or 'float64' (whatever the op stored). Goldens: tests/atlas/ops/<op>.json (atlas-golden/1).
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import type { AnyTensor, AtlasLayer, OpCtx, OpDef, Tensor } from '../../../src/lib/atlas/types/index.ts';
import { writeJson } from '../lib/io.ts';
import { roots } from '../root.ts';

interface B64 {
	shape: number[];
	dtype: string;
	b64: string;
}
interface GoldenCase {
	id: string;
	op: string;
	params: Record<string, unknown>;
	inputs: B64[];
	weights: Record<string, B64>;
	out: B64;
	subs?: Record<string, B64>;
}

function bytes(b64: string): Uint8Array {
	return new Uint8Array(Buffer.from(b64, 'base64'));
}

export function decode(t: B64): AnyTensor {
	const b = bytes(t.b64);
	const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
	if (t.dtype === 'int32') return { shape: t.shape, data: new Int32Array(buf), dtype: 'i32' };
	if (t.dtype === 'float64') return { shape: t.shape, data: new Float64Array(buf) };
	return { shape: t.shape, data: new Float32Array(buf) };
}

export function encode(t: Tensor): B64 {
	const d = t.data;
	return { shape: t.shape.slice(), dtype: d instanceof Float64Array ? 'float64' : 'float32', b64: Buffer.from(d.buffer, d.byteOffset, d.byteLength).toString('base64') };
}

export interface OpOutputs {
	format: 'atlas-op-outputs/1';
	ops: Record<string, { id: string; op: string; out: B64; subs?: Record<string, B64> }[]>;
}

/** Run the ops (by name) over their golden files. `ops` is CORE_OPS or a package's list. */
export async function opOutputs(site: string, o: { only?: string[]; goldens?: string; ops?: OpDef[] } = {}): Promise<OpOutputs> {
	const atlas = join(site, 'src', 'lib', 'atlas');
	const reg = (await import(pathToFileURL(join(atlas, 'ops', 'registry.ts')).href)) as typeof import('../../../src/lib/atlas/ops/registry.ts');
	const common = (await import(pathToFileURL(join(atlas, 'ops', 'common.ts')).href)) as typeof import('../../../src/lib/atlas/ops/common.ts');
	const ops = o.ops ?? ((await import(pathToFileURL(join(atlas, 'ops', 'index.ts')).href)) as { CORE_OPS: OpDef[] }).CORE_OPS;
	const registry = reg.createRegistry(...ops);
	const dir = o.goldens ?? join(site, 'tests', 'atlas', 'ops');
	const names = o.only ?? ops.map((d) => d.op);
	const out: OpOutputs = { format: 'atlas-op-outputs/1', ops: {} };
	for (const name of names) {
		const f = join(dir, `${name}.json`);
		if (!existsSync(f)) continue;
		const g = JSON.parse(readFileSync(f, 'utf8')) as { cases: GoldenCase[] };
		out.ops[name] = g.cases.map((c) => {
			const r = registry.get(c.op);
			const params = reg.mergeParams(r.def, r.params, c.params);
			const ins = c.inputs.map(decode);
			const weights: Record<string, Tensor> = {};
			for (const [k, v] of Object.entries(c.weights)) weights[k] = decode(v) as Tensor;
			const layer: AtlasLayer = { id: 'g', op: c.op, label: { en: 'golden', zh: '金标' }, in: ins.map((_, i) => `x${i}`), outShape: c.out.shape, params: c.params };
			const subs: Record<string, Tensor> = {};
			const ctx: OpCtx = {
				layer,
				dims: {},
				w: (slot) => {
					const t = weights[slot];
					if (!t) throw new common.MissingWeightError(slot, `g.${slot}`, 'g');
					return t;
				},
				emit: (s, t) => (subs[s] = t)
			};
			const y = r.def.forward(ins, params, ctx);
			const row: { id: string; op: string; out: B64; subs?: Record<string, B64> } = { id: c.id, op: c.op, out: encode(y) };
			if (Object.keys(subs).length) row.subs = Object.fromEntries(Object.entries(subs).map(([k, v]) => [k, encode(v)]));
			return row;
		});
	}
	return out;
}

/** Golden files present in a folder (op names). */
export function goldenOps(dir: string): string[] {
	return existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)).sort() : [];
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
	const args = process.argv.slice(2);
	const val = (flag: string) => {
		const i = args.indexOf(flag);
		return i >= 0 ? args[i + 1] : undefined;
	};
	const only = args.flatMap((a, i) => (args[i - 1] === '--op' ? [a] : []));
	const opsFile = val('--ops');
	const ops = opsFile ? ((await import(pathToFileURL(resolve(opsFile)).href)) as { default: OpDef[] }).default : undefined;
	const res = await opOutputs(roots().site, { only: only.length ? only : undefined, goldens: val('--goldens'), ops });
	const file = val('--out');
	if (file) writeJson(file, res);
	else process.stdout.write(JSON.stringify(res) + '\n');
}
