// Generates site/src/lib/atlas/schema/ops.json from CORE_OPS (design §4.3: W1-A owns ops.json; the Python mirrors in
// pipeline/atlas_kit/ops are checked against it).
//   node src/lib/atlas/ops/manifest.ts           print the JSON
//   node src/lib/atlas/ops/manifest.ts --write   write schema/ops.json (LF, no BOM)
//   node src/lib/atlas/ops/manifest.ts --check   exit 1 when schema/ops.json is out of date
// Format: a JSON array, one entry per op in CORE_OPS order:
//   { op, version, attrs: { <name>: { type, default?, required?, doc? } }, aliases: { <alias>: { <attr>: value } },
//     weights: [{ slot, optional? }], subs: [{ name, label, axes? }], category, zoom, label, parityBar }
// Importing this module has no side effects (tests call buildManifest()).
import type { OpDef } from '../types/index.ts';
import { CORE_OPS } from './index.ts';

export interface OpManifestEntry {
	op: string;
	version: number;
	attrs: OpDef['attrs'];
	aliases: Record<string, Record<string, unknown>>;
	weights: { slot: string; optional?: boolean }[];
	subs: { name: string; label: { en: string; zh?: string }; axes?: string[] }[];
	category: string;
	zoom: string;
	label: { en: string; zh?: string };
	parityBar: number;
}

export function manifestEntry(def: OpDef): OpManifestEntry {
	const attrs: OpDef['attrs'] = {};
	for (const [k, a] of Object.entries(def.attrs)) {
		const e: OpDef['attrs'][string] = { type: a.type };
		if (a.default !== undefined) e.default = a.default;
		if (a.required) e.required = true;
		if (a.doc) e.doc = a.doc;
		attrs[k] = e;
	}
	return {
		op: def.op,
		version: def.version,
		attrs,
		aliases: Object.fromEntries(Object.entries(def.aliases ?? {}).map(([k, v]) => [k, { ...(v as Record<string, unknown>) }])),
		weights: (def.weights ?? []).map((w) => (w.optional ? { slot: w.slot, optional: true } : { slot: w.slot })),
		subs: (def.subs ?? []).map((s) => (s.axes ? { name: s.name, label: s.label, axes: s.axes.slice() } : { name: s.name, label: s.label })),
		category: def.category,
		zoom: def.zoom,
		label: def.label,
		parityBar: def.parity.bar
	};
}

/** The manifest for a list of ops (default CORE_OPS). */
export function buildManifest(ops: OpDef[] = CORE_OPS): OpManifestEntry[] {
	return ops.map(manifestEntry);
}

/** The exact text written to schema/ops.json. */
export function manifestText(ops: OpDef[] = CORE_OPS): string {
	return JSON.stringify(buildManifest(ops), null, 2) + '\n';
}

async function main(argv: string[]): Promise<void> {
	const fsName = 'node:fs';
	const urlName = 'node:url';
	const fs = (await import(/* @vite-ignore */ fsName)) as typeof import('node:fs');
	const { fileURLToPath } = (await import(/* @vite-ignore */ urlName)) as typeof import('node:url');
	const target = fileURLToPath(new URL('../schema/ops.json', import.meta.url));
	const text = manifestText();
	if (argv.includes('--write')) {
		fs.writeFileSync(target, text, { encoding: 'utf8' });
		console.log(`wrote ${target} (${CORE_OPS.length} ops)`);
	} else if (argv.includes('--check')) {
		const cur = fs.existsSync(target) ? fs.readFileSync(target, 'utf8') : '';
		if (cur !== text) {
			console.error(`${target} is out of date: run node src/lib/atlas/ops/manifest.ts --write`);
			(globalThis as { process?: { exitCode?: number } }).process!.exitCode = 1;
		} else console.log(`ops.json is up to date (${CORE_OPS.length} ops)`);
	} else process.stdout.write(text);
}

const proc = (globalThis as { process?: { argv?: string[] } }).process;
if (proc?.argv?.[1] && /manifest\.ts$/.test(proc.argv[1].replace(/\\/g, '/')) && import.meta.url.endsWith('/ops/manifest.ts')) {
	await main(proc.argv.slice(2));
}
