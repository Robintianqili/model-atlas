// Generates the reference sections of site/DEVKIT.md from the code, so the guide never drifts from the kit:
//   commands  scripts/atlas/cli.ts COMMANDS
//   codes     scripts/atlas/errors.ts CODES (each row links to its explanation below the table)
//   ports     scripts/atlas/ports.ts PORTS
//   ops       src/lib/atlas/schema/ops.json + the CORE_OPS narration (what each auto step says first)
//   views     every kit view meta (src/lib/atlas/ui/views/<key>/meta.ts); package views are not listed (K0)
//
//   node scripts/atlas/docs.ts --write     rewrite the blocks between <!-- atlas:generated NAME --> and <!-- /atlas:generated -->
//   node scripts/atlas/docs.ts --check     exit 1 when DEVKIT.md is out of date (the text of the blocks differs)
// Run it after adding an op, a view, a command or a port (the main session runs it at integration).
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { OpDef, ViewMeta } from '../../src/lib/atlas/types/index.ts';
import { CODES } from './errors.ts';
import { writeText } from './lib/io.ts';
import { PORTS } from './ports.ts';
import { cliSite } from './root.ts';

interface OpEntry {
	op: string;
	version: number;
	attrs: Record<string, { type: string; default?: unknown; required?: boolean; doc?: string }>;
	aliases: Record<string, Record<string, unknown>>;
	weights: { slot: string; optional?: boolean }[];
	subs: (string | { name: string })[];
	category: string;
	zoom: string;
	label: { en: string; zh?: string };
	parityBar: number;
}

const cell = (s: string) => s.replace(/\|/g, '\\|').replace(/\n/g, ' ');
const firstSentence = (s: string) => {
	const m = /^(.+?[.!?])(\s|$)/.exec(s.trim());
	return m ? m[1] : s.trim();
};

export function commandsBlock(cmds: Record<string, { summary: string }>): string {
	const rows = Object.entries(cmds).map(([k, c]) => {
		const [usage, ...rest] = c.summary.split(': ');
		return `| \`${k}\` | \`npm run atlas -- ${cell(usage)}\` | ${cell(rest.join(': ') || usage)} |`;
	});
	return ['| Command | Usage | What it does |', '|---|---|---|', ...rows].join('\n');
}

export function codesBlock(): string {
	const rows = Object.entries(CODES).map(([k, v]) => `| [${k}](#${k.toLowerCase()}) | ${cell(v)} |`);
	return ['| Code | Meaning |', '|---|---|', ...rows].join('\n');
}

export function portsBlock(): string {
	const rows = Object.entries(PORTS).map(([k, p]) => `| \`${k}\` | ${p} | \`npm run dev -- --port ${p}\` |`);
	return ['| Package or lane | Port | Dev server |', '|---|---|---|', ...rows, '| anything else | 5210 and up: `atlas new` picks a free one and records it in `notes/<slug>.md` | `npm run dev -- --port <that port>` |'].join('\n');
}

export async function opsBlock(site: string): Promise<string> {
	const file = join(site, 'src', 'lib', 'atlas', 'schema', 'ops.json');
	if (!existsSync(file)) return '_schema/ops.json is missing in this tree._';
	const ops = JSON.parse(readFileSync(file, 'utf8')) as OpEntry[];
	let narr: Record<string, string> = {};
	const idx = join(site, 'src', 'lib', 'atlas', 'ops', 'index.ts');
	if (existsSync(idx)) {
		const m = (await import(pathToFileURL(idx).href)) as { CORE_OPS: OpDef[] };
		narr = Object.fromEntries(m.CORE_OPS.map((d) => [d.op, firstSentence(d.narration.what.en)]));
	}
	const attr = (e: OpEntry) =>
		Object.entries(e.attrs)
			.map(([k, a]) => `\`${k}\`: ${a.type}${a.required ? ' (required)' : a.default !== undefined ? ` = ${JSON.stringify(a.default)}` : ''}`)
			.join('<br>') || '-';
	const rows = ops.map((e) => {
		const w = e.weights.map((x) => `\`${x.slot}\`${x.optional ? '?' : ''}`).join(', ') || '-';
		const subs = e.subs.map((s) => (typeof s === 'string' ? s : s.name)).join(' ') || '-';
		const al = Object.entries(e.aliases)
			.map(([k, p]) => `\`${k}\`${Object.keys(p).length ? ` (${Object.entries(p).map(([a, v]) => `${a}=${JSON.stringify(v)}`).join(', ')})` : ''}`)
			.join(', ');
		return `| \`${e.op}\`${al ? `<br>aliases ${al}` : ''} | ${cell(e.label.en)}<br>${cell(e.label.zh ?? '')} | ${e.category} | \`zoom.${e.zoom}\` | ${attr(e)} | ${w} | ${subs} | ${cell(narr[e.op] ?? '')} |`;
	});
	return [
		`${ops.length} ops. Python: \`b.<op>(id, ...)\` builder calls (\`pipeline/atlas_kit/spec.py\`); attributes are checked against this table (AK106). Parity bar for every op: ${[...new Set(ops.map((o) => o.parityBar))].join(', ')}.`,
		'',
		'| Op | Label (en / zh) | Category | Zoom view | Attributes | Weights (`?` optional) | Sub-steps | The auto step starts with |',
		'|---|---|---|---|---|---|---|---|',
		...rows
	].join('\n');
}

export async function viewsBlock(site: string): Promise<string> {
	const found: { key: string; meta: ViewMeta | null; err?: string }[] = [];
	const kit = join(site, 'src', 'lib', 'atlas', 'ui', 'views');
	if (existsSync(kit)) for (const k of readdirSync(kit).sort()) if (existsSync(join(kit, k, 'meta.ts'))) found.push({ key: k, meta: null });
	// kit views only (K0): package views ('<slug>/<key>') belong to their package, so merging a package with a local view
	// no longer makes this generated table (and the docs gate) stale; the gallery /dev/atlas/ lists them all
	const ex = join(site, 'src', 'lib', 'explainers');
	for (const f of found) {
		const [slug, local] = f.key.split('/');
		const dir = local ? join(ex, slug, 'views', local) : join(kit, f.key);
		try {
			f.meta = ((await import(pathToFileURL(join(dir, 'meta.ts')).href)) as { default: ViewMeta }).default;
		} catch (e) {
			f.err = (e as Error).message.split('\n')[0];
		}
	}
	if (!found.length)
		return [
			'_No view metas in this tree yet._ The kit views come with wave 1b (W1-D2: `zoom.weighted-sum`, `zoom.reduce`, `zoom.elementwise`,',
			'`zoom.normalize`, `zoom.softmax`, `zoom.lookup`, `zoom.index-map`, `zoom.gate`, `zoom.formula`, `layer.tensor`, `compare`,',
			'`tape.table`, `kit.*`; W1-B2: `plane2d`, `curve1d`, `loss-curve`, `train-step`, `network`, `train-controls`).',
			'Regenerate this table after integration: `node scripts/atlas/docs.ts --write`.'
		].join('\n');
	const rows = found.map((f) =>
		f.meta
			? `| \`${f.key}\` | ${cell(f.meta.title.en)}<br>${cell(f.meta.title.zh ?? '')} | ${f.meta.kinds.join(', ')} | ${f.meta.minHeight ?? '-'} |`
			: `| \`${f.key}\` | (meta.ts does not load in Node: ${cell(f.err ?? '')}) | | |`
	);
	return [
		`${found.length} kit views. Use the key as a step's \`view\`; open \`/dev/atlas/\` for the gallery with demo data. A package's own views`,
		"(`explainers/<slug>/views/<key>/`, key `'<slug>/<key>'`) are in the gallery too; they are not listed here.",
		'',
		'| Key | Title (en / zh) | Kinds | Min height (px) |',
		'|---|---|---|---|',
		...rows
	].join('\n');
}

export async function generate(site: string, text: string): Promise<{ text: string; blocks: string[] }> {
	const { COMMANDS } = (await import('./cli.ts')) as { COMMANDS: Record<string, { summary: string }> };
	const blocks: Record<string, string> = {
		commands: commandsBlock(COMMANDS),
		codes: codesBlock(),
		ports: portsBlock(),
		ops: await opsBlock(site),
		views: await viewsBlock(site)
	};
	const seen: string[] = [];
	const out = text.replace(/<!-- atlas:generated (\w+) -->\n[\s\S]*?<!-- \/atlas:generated -->/g, (m, name: string) => {
		if (!(name in blocks)) return m;
		seen.push(name);
		return `<!-- atlas:generated ${name} -->\n${blocks[name]}\n<!-- /atlas:generated -->`;
	});
	return { text: out, blocks: seen };
}

if (process.argv[1] && /docs\.ts$/.test(process.argv[1])) {
	const site = cliSite();
	const file = join(site, 'DEVKIT.md');
	const text = readFileSync(file, 'utf8');
	const r = await generate(site, text);
	if (process.argv.includes('--check')) {
		const ok = r.text === text;
		console.log(ok ? `DEVKIT.md is up to date (${r.blocks.join(', ')})` : 'DEVKIT.md is out of date: run node scripts/atlas/docs.ts --write');
		process.exitCode = ok ? 0 : 1;
	} else if (process.argv.includes('--write')) {
		writeText(file, r.text);
		console.log(`DEVKIT.md: regenerated ${r.blocks.join(', ')}`);
	} else console.log('usage: node scripts/atlas/docs.ts --write | --check');
}
