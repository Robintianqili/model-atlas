// atlas new (design §4.12):
//   new model <slug> --kind trace|train|process[,...] [--input <adapter>] [--template <name>] [--dev] [--no-worker] [--port n]
//   new model <slug> --like <existing-slug> [--dev] [--port n]   (a copy of an existing package, renamed: the closest start
//                                                  for a variant, as hello-xor-2 of hello-xor)
//   new concept <slug> [--kind trace|train|process[,...]] [--input <adapter>] [--dev] [--port n]   (K2: a concept package,
//                                                  routed at /train/<slug>/; default --kind process)
//   new concept <slug> --like <package> [--dev] [--port n]   (a renamed copy of a concept package, or of a model package
//                                                  turned into a concept: type 'concept', concept.ts and its test added)
//   new worker <slug>                              (adds worker.ts to an existing package once the engine exists)
//   new op <name> [--local <slug>] [--like <op>]
//   new view <key> [--local <slug>]
// `new model` writes src/lib/explainers/<slug>/ in ATLAS_SITE, pipeline/<slug>/ and notes/<slug>.md in ATLAS_ROOT.
// The slug must be in the catalog (AK701: models.ts for a model, concepts.ts for a concept) unless --dev, which makes a
// lab-only devOnly package. Nothing is overwritten.
// `new concept` writes the same folders: the base scaffold with the templates/concept/ overlay (explainer.ts with type
// 'concept', lesson.ts, concept.ts, concept.test.ts, notes). A concept package lives in src/lib/explainers/<slug>/ like a
// model's, so every loader, source kind, view, lint and gate works on it unchanged; catalog slugs of models and concepts
// are disjoint (scripts/atlas/tests/concept.test.ts checks it).
// The dev-server port: --port, else the package's row in the port table, else the first free port from 5210 up that no
// other package claims (ports.ts pickPort); the notes record it and the "Next:" list prints it.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { CONFIG_DIR } from '../config.ts';
import { AtlasError } from '../errors.ts';
import { posix, readJson, writeText } from '../lib/io.ts';
import { render, type Vars } from '../lib/template.ts';
import { PORTS, pickPort } from '../ports.ts';
import { roots, type Roots } from '../root.ts';

const TEMPLATES = join(CONFIG_DIR, 'templates');
/** The concept overlay (templates/concept/): used by `new concept`, never offered as a model template. */
export const CONCEPT_TEMPLATE = 'concept';
/** Concepts with a hand-built page (src/routes/train/<slug>/), which takes precedence over the concept route. */
const STATIC_CONCEPTS = ['tokenization', 'next-token-prediction', 'transfer-learning'];
const KINDS = ['trace', 'train', 'process'] as const;
type Kind = (typeof KINDS)[number];

/** adapter id -> catalog InputKind (design §4.9) */
export const ADAPTER_KIND: Record<string, string> = {
	drawing28: 'drawing',
	'text-bpe': 'text',
	'text-char': 'text',
	points2d: 'points2d',
	series: 'series',
	graph: 'graph',
	gridworld: 'game',
	'shapes-captions': 'drawing'
};

const SLUG = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const today = () => new Date().toISOString().slice(0, 10);
const camel = (s: string) => s.replace(/[-_.]+([a-z0-9])/g, (_, c: string) => c.toUpperCase()).replace(/^[^a-zA-Z_$]/, '_$&');

interface Opts {
	kinds?: Kind[];
	input?: string;
	template?: string;
	dev: boolean;
	noWorker: boolean;
	local?: string;
	like?: string;
	port?: number;
}

function parse(args: string[]): { what: string; name: string; o: Opts } {
	const pos: string[] = [];
	const o: Opts = { dev: false, noWorker: false };
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		const val = () => {
			const v = args[++i];
			if (!v || v.startsWith('--')) throw new AtlasError('AK301', `${a} needs a value`);
			return v;
		};
		if (a === '--kind' || a === '--kinds') {
			const ks = val().split(',').map((s) => s.trim());
			for (const k of ks) if (!(KINDS as readonly string[]).includes(k)) throw new AtlasError('AK301', `unknown kind '${k}'`, { fix: 'kinds: trace, train, process (comma-separated for combinations)' });
			o.kinds = [...new Set(ks)] as Kind[];
		} else if (a === '--input') o.input = val();
		else if (a === '--template') o.template = val();
		else if (a === '--dev') o.dev = true;
		else if (a === '--no-worker') o.noWorker = true;
		else if (a === '--local') o.local = val();
		else if (a === '--like') o.like = val();
		else if (a === '--port') {
			const p = Number(val());
			if (!Number.isInteger(p) || p < 1024 || p > 65535) throw new AtlasError('AK301', `--port needs a port number (1024-65535)`);
			o.port = p;
		} else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`);
		else pos.push(a);
	}
	if (pos.length !== 2) throw new AtlasError('AK301', 'usage: atlas new model <slug> --kind ... | new concept <slug> [--like <package>] | new worker <slug> | new op <name> | new view <key>');
	return { what: pos[0], name: pos[1], o };
}

interface CatalogEntry {
	slug: string;
	name: { en: string; zh?: string };
	hook: { en: string; zh?: string };
	shows: { en: string; zh?: string };
	inputs: string[];
	status: string;
}

interface ConceptCatalogEntry {
	slug: string;
	title: { en: string; zh?: string };
	interaction: { en: string; zh?: string };
	status: string;
}

/** The concept's catalog entry (src/lib/catalog/concepts.ts), or null. */
async function conceptEntry(r: Roots, slug: string): Promise<ConceptCatalogEntry | null> {
	const f = join(r.site, 'src', 'lib', 'catalog', 'concepts.ts');
	if (!existsSync(f)) return null;
	const m = (await import(pathToFileURL(f).href)) as { CONCEPTS?: ConceptCatalogEntry[] };
	return m.CONCEPTS?.find((e) => e.slug === slug) ?? null;
}

async function catalogEntry(r: Roots, slug: string): Promise<CatalogEntry | null> {
	const f = join(r.site, 'src', 'lib', 'catalog', 'models.ts');
	if (!existsSync(f)) return null;
	const m = (await import(pathToFileURL(f).href)) as { MODELS?: CatalogEntry[] };
	return m.MODELS?.find((e) => e.slug === slug) ?? null;
}

/** One line of text safe inside a '...' TS string and a "..." Python string. */
const lit = (s: string) => s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\s+/g, ' ').trim();

interface FileSpec {
	from?: string;
	to: string;
	when?: string;
	/** drop a file the base template would write (an overlay that does not need it) */
	skip?: boolean;
}
interface TemplateJson {
	name: string;
	description: string;
	/** defaults used when --kind / --input are not given */
	kinds?: Kind[];
	input?: string;
	files: FileSpec[];
}

/** Named templates (folders with a template.json, except base). */
export function templateNames(): string[] {
	return readdirSync(TEMPLATES)
		.filter((d) => d !== 'base' && d !== CONCEPT_TEMPLATE && existsSync(join(TEMPLATES, d, 'template.json')))
		.sort();
}

export function readTemplate(name: string): TemplateJson {
	const dir = join(TEMPLATES, name);
	if (!existsSync(join(dir, 'template.json'))) throw new AtlasError('AK301', `no template '${name}'`, { fix: `templates: ${templateNames().join(', ')}` });
	return readJson<TemplateJson>(join(dir, 'template.json'));
}

function templateFiles(name: string): { dir: string; files: FileSpec[] }[] {
	const chain = ['base', ...(name && name !== 'base' ? [name] : [])];
	return chain.map((t) => ({ dir: join(TEMPLATES, t), files: readTemplate(t).files }));
}

/** The dev-server port of a new package: --port, else its table row, else a free port no other package claims. */
async function portFor(slug: string, o: Opts, r: Roots): Promise<number> {
	return o.port ?? PORTS[slug] ?? (await pickPort({ slug, root: r.root }));
}

export async function newModel(slug: string, o: Opts, r: Roots = roots()): Promise<{ written: string[]; notes: string[]; kinds: Kind[]; port: number; skipped?: string[] }> {
	if (!SLUG.test(slug)) throw new AtlasError('AK301', `'${slug}' is not a slug (lowercase letters, digits and single hyphens)`);
	if (o.like) return newModelLike(slug, o, r);
	if (o.template === CONCEPT_TEMPLATE) throw new AtlasError('AK301', `'${CONCEPT_TEMPLATE}' is the concept overlay, not a model template`, { fix: `npm run atlas -- new concept ${slug}` });
	if (o.template && o.template !== 'base') {
		const tj = readTemplate(o.template);
		if (!o.kinds?.length && tj.kinds?.length) o = { ...o, kinds: tj.kinds };
		if (!o.input && tj.input) o = { ...o, input: tj.input };
	}
	if (!o.kinds?.length) throw new AtlasError('AK301', '--kind is required: trace, train, process (or a comma-separated mix)', { fix: `or pick a template that sets them: --template ${templateNames().join('|')}` });
	const notes: string[] = [];
	const entry = await catalogEntry(r, slug);
	if (!o.dev && !entry)
		throw new AtlasError('AK701', `'${slug}' is not in the catalog (src/lib/catalog/models.ts)`, {
			fix: `use --dev for a lab-only package, or propose the catalog entry first (maintainers add it; see CONTRIBUTING.md)`
		});
	if (o.dev && entry) notes.push(`'${slug}' is in the catalog, but --dev makes a lab-only package; drop --dev to build its real page`);
	const kinds = o.kinds;
	const has = (k: Kind) => kinds.includes(k);
	if (has('trace') && !o.input) throw new AtlasError('AK301', 'kind trace needs --input <adapter>', { fix: `adapters: ${Object.keys(ADAPTER_KIND).join(', ')}` });
	if (o.input) {
		const kitAdapter = join(r.site, 'src', 'lib', 'atlas', 'adapters', o.input, 'adapter.ts');
		if (!existsSync(kitAdapter)) {
			// a local adapter (K0: this said "not in this tree yet (W1-D1/W1-B2 ...)", a stale wave-1 message)
			const adaptersDir = join(r.site, 'src', 'lib', 'atlas', 'adapters');
			const kit = existsSync(adaptersDir) ? readdirSync(adaptersDir).filter((d) => existsSync(join(adaptersDir, d, 'adapter.ts'))) : [];
			const fromTemplate = templateFiles(o.template ?? 'base').some((t) => t.files.some((f) => f.to === `package/adapters/${o.input}/adapter.ts`));
			notes.push(
				fromTemplate
					? `adapters/${o.input}/adapter.ts is a local adapter from the template: add its Capture.svelte (the input widget, props CaptureProps in atlas/types/explainer.ts; hello-xor's adapters/xor-point/Capture.svelte is a small example; DEVKIT.md 5.4)`
					: `'${o.input}' is not a kit adapter (kit: ${kit.join(', ') || 'none'}): write a local one in adapters/${o.input}/: adapter.ts (pure TS: raw input -> tensors) and Capture.svelte (the input widget); hello-xor's adapters/xor-point/ is a worked example (DEVKIT.md 5.4)`
			);
		}
		if (entry && ADAPTER_KIND[o.input] && !entry.inputs.includes(ADAPTER_KIND[o.input]))
			notes.push(`the catalog lists inputs [${entry.inputs.join(', ')}] for ${slug}; adapter '${o.input}' gives '${ADAPTER_KIND[o.input]}'`);
	}
	const pkgDir = join(r.site, 'src', 'lib', 'explainers', slug);
	const pipeDir = join(r.root, 'pipeline', slug);
	const notesFile = join(r.root, 'notes', `${slug}.md`);
	const enginePresent = existsSync(join(r.site, 'src', 'lib', 'atlas', 'engine', 'host.ts'));
	const wantWorker = (has('train') || has('process')) && !o.noWorker;
	const worker = wantWorker && enginePresent;
	if (wantWorker && !enginePresent)
		notes.push(`worker.ts not generated: atlas/engine/host.ts (W1-B2, wave 1b) is not in this tree. Add it later with \`npm run atlas -- new worker ${slug}\``);
	if (has('train') && !existsSync(join(r.site, 'src', 'lib', 'atlas', 'autograd', 'nn.ts')))
		notes.push('trainer.ts imports atlas/autograd/* (W1-B1), which is not in this tree yet: it loads once W1-B1 is merged');

	const v: Vars = {
		slug,
		today: today(),
		dev: o.dev,
		catalog: !!entry,
		trace: has('trace'),
		train: has('train'),
		process: has('process'),
		worker,
		hasInput: !!o.input,
		adapter: o.input ?? 'TODO-adapter',
		inputKind: (o.input && ADAPTER_KIND[o.input]) || entry?.inputs[0] || 'TODO',
		kindsList: kinds.map((k) => `'${k}'`).join(', '),
		kindsCsv: kinds.join(', '),
		kindsPy: kinds.map((k) => `"${k}"`).join(', ') + (kinds.length === 1 ? ',' : ''),
		processId: camel(slug),
		processIdent: camel(slug),
		trainerId: camel(slug),
		trainerIdent: camel(slug), // the export name equals the id, so the loader registers it without a note
		serveArgs: [has('train') ? 'trainers' : '', has('process') ? 'processes' : ''].filter(Boolean).join(', '),
		nameEn: lit(entry?.name.en ?? slug),
		nameZh: lit(entry?.name.zh ?? entry?.name.en ?? slug),
		hookEn: lit(entry?.hook.en ?? `(dev package: no catalog entry)`),
		hookZh: lit(entry?.hook.zh ?? ''),
		showsEn: lit(entry?.shows.en ?? `(dev package: no catalog entry)`),
		showsZh: lit(entry?.shows.zh ?? ''),
		port: await portFor(slug, o, r),
		template: o.template ?? 'base'
	};

	// plan every file first; refuse before writing anything if one exists
	const plan = new Map<string, string>();
	for (const { dir, files } of templateFiles(o.template ?? 'base')) {
		for (const f of files) {
			if (f.when && !v[f.when]) continue;
			const dest = f.to === 'notes' ? notesFile : f.to.startsWith('package/') ? join(pkgDir, f.to.slice(8)) : f.to.startsWith('pipeline/') ? join(pipeDir, f.to.slice(9)) : null;
			if (!dest) throw new AtlasError('AK301', `template ${posix(dir)}: bad destination '${f.to}'`);
			if (f.skip) {
				plan.delete(dest);
				continue;
			}
			if (!f.from) throw new AtlasError('AK301', `template ${posix(dir)}: '${f.to}' has no 'from'`);
			plan.set(dest, render(readFileSync(join(dir, f.from), 'utf8'), v, posix(join(dir, f.from))));
		}
	}
	const clash = [...plan.keys()].filter((p) => existsSync(p));
	if (existsSync(pkgDir) || clash.length)
		throw new AtlasError('AK301', `'${slug}' already exists`, { where: [pkgDir, ...clash].filter(existsSync).map(posix).join(', '), fix: 'pick another slug, or remove the old files first (nothing was written)' });
	for (const [p, text] of plan) writeText(p, text);
	return { written: [...plan.keys()], notes, kinds, port: Number(v.port) };
}

// ------------------------------------------------------------------ new model <slug> --like <existing>
/** Text files copied from a package (everything under explainers/<old>/) and from its pipeline folder (sources only:
 *  checkpoints, metrics, data and caches are outputs of the old model, so they are left out). */
const PIPE_TEXT = /\.(py|md|txt|toml|cfg|ini|ya?ml)$/i;
const PKG_TEXT = /\.(ts|svelte|js|md|json|css|txt)$/i;
const SKIP_DIR = new Set(['__pycache__', 'runs', 'data', 'node_modules', '.ipynb_checkpoints']);
const MAX_COPY = 256 * 1024;

function filesUnder(dir: string, base = dir): string[] {
	if (!existsSync(dir)) return [];
	const out: string[] = [];
	for (const n of readdirSync(dir).sort()) {
		const p = join(dir, n);
		if (statSync(p).isDirectory()) {
			if (!SKIP_DIR.has(n)) out.push(...filesUnder(p, base));
		} else out.push(relative(base, p));
	}
	return out;
}

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Rename every whole-word use of the old slug (kebab, camelCase and snake_case forms) to the new one. */
export function renameSlug(text: string, from: string, to: string): { text: string; count: number } {
	const pairs: [string, string][] = [
		[from, to],
		[camel(from), camel(to)],
		[from.replace(/-/g, '_'), to.replace(/-/g, '_')]
	];
	const seen = new Set<string>();
	let count = 0;
	let out = text;
	for (const [a, b] of pairs.sort((x, y) => y[0].length - x[0].length)) {
		if (seen.has(a) || a === b) continue;
		seen.add(a);
		out = out.replace(new RegExp(`(?<![A-Za-z0-9_])${esc(a)}(?![A-Za-z0-9_])`, 'g'), () => {
			count++;
			return b;
		});
	}
	return { text: out, count };
}

export async function newModelLike(slug: string, o: Opts, r: Roots = roots(), as: 'model' | 'concept' = 'model'): Promise<{ written: string[]; notes: string[]; kinds: Kind[]; port: number; skipped: string[] }> {
	const old = o.like!;
	if (o.kinds?.length || o.input || o.template) throw new AtlasError('AK301', '--like copies a whole package: drop --kind, --input and --template', { fix: `npm run atlas -- new ${as} ${slug} --like ${old} [--dev] [--port n]` });
	const srcPkg = join(r.site, 'src', 'lib', 'explainers', old);
	if (!existsSync(join(srcPkg, 'explainer.ts'))) {
		const { listPackages } = await import('../load.ts');
		throw new AtlasError('AK301', `--like ${old}: no package '${old}'`, { fix: `packages: ${listPackages(r.site).join(', ') || 'none yet'}` });
	}
	const notes: string[] = [];
	const entry = as === 'concept' ? await conceptEntry(r, slug) : await catalogEntry(r, slug);
	if (!o.dev && !entry)
		throw new AtlasError('AK701', `'${slug}' is not in the catalog (src/lib/catalog/${as === 'concept' ? 'concepts' : 'models'}.ts)`, {
			fix: `use --dev for a lab-only package, or propose the catalog entry first (maintainers add it; see CONTRIBUTING.md)`
		});
	const { loadPackage } = await import('../load.ts');
	const kinds = (await loadPackage(old, r.site)).pkg.manifest.kinds.filter((k): k is Kind => (KINDS as readonly string[]).includes(k));
	const pkgDir = join(r.site, 'src', 'lib', 'explainers', slug);
	const pipeDir = join(r.root, 'pipeline', slug);
	const notesFile = join(r.root, 'notes', `${slug}.md`);
	const plan = new Map<string, string>();
	const skipped: string[] = [];
	let renamed = 0;
	let copied = 0;
	const copy = (fromDir: string, toDir: string, ok: RegExp) => {
		for (const f of filesUnder(fromDir)) {
			const src = join(fromDir, f);
			const rel = posix(relative(r.root, src));
			if (!ok.test(f) || statSync(src).size > MAX_COPY) {
				skipped.push(rel);
				continue;
			}
			const { text, count } = renameSlug(readFileSync(src, 'utf8'), old, slug);
			renamed += count;
			copied++;
			plan.set(join(toDir, f), text);
		}
	};
	copy(srcPkg, pkgDir, PKG_TEXT);
	copy(join(r.root, 'pipeline', old), pipeDir, PIPE_TEXT);
	// the manifest: lab-only or not, and a fresh quality level
	const ex = join(pkgDir, 'explainer.ts');
	let m = plan.get(ex)!;
	if (o.dev && !/devOnly:\s*true/.test(m)) m = m.replace(/^(\s*)slug:\s*'[^']*',\s*$/m, (line, ind: string) => `${line}\n${ind}devOnly: true,`);
	if (!o.dev && /devOnly:\s*true/.test(m)) {
		m = m.replace(/devOnly:\s*true/, 'devOnly: false');
		notes.push(`'${old}' is lab-only (devOnly); '${slug}' is a catalog page, so its manifest says devOnly: false`);
	}
	if (/quality:\s*'(reviewed|flagship)'/.test(m)) m = m.replace(/quality:\s*'(reviewed|flagship)'/, "quality: 'draft'");
	if (as === 'concept') m = asConceptManifest(m, notes, old);
	plan.set(ex, m);
	// a package made before the engine had no worker.ts: give the copy one (its trainers and processes run in a worker)
	const wk = join(pkgDir, 'worker.ts');
	const train = plan.has(join(pkgDir, 'trainer.ts'));
	const proc = plan.has(join(pkgDir, 'process.ts'));
	if (!plan.has(wk) && (train || proc) && existsSync(join(r.site, 'src', 'lib', 'atlas', 'engine', 'host.ts')))
		plan.set(wk, render(readFileSync(join(TEMPLATES, 'base', 'package', 'worker.ts.tmpl'), 'utf8'), { slug, train, process: proc, serveArgs: [train ? 'trainers' : '', proc ? 'processes' : ''].filter(Boolean).join(', ') }));
	const port = await portFor(slug, o, r);
	const v: Vars = {
		slug,
		today: today(),
		dev: o.dev,
		trace: kinds.includes('trace'),
		train: kinds.includes('train'),
		process: kinds.includes('process'),
		kindsCsv: kinds.join(', '),
		port,
		hookEn: lit((entry as CatalogEntry | null)?.hook?.en ?? `(dev package: no catalog entry; a copy of ${old})`),
		interactionEn: lit((entry as ConceptCatalogEntry | null)?.interaction?.en ?? `(dev package: no catalog entry; a copy of ${old})`)
	};
	if (as === 'concept') {
		// a model turned into a concept gets the concept page parts it lacks (the template's, to fill)
		const first = Object.keys((await loadPackage(old, r.site)).pkg.manifest.sources ?? {})[0] ?? 'fwd';
		const cv: Vars = { slug, today: today(), toySource: first, toyView: 'tape.table' };
		for (const f of ['concept.ts', 'concept.test.ts']) {
			const dest = join(pkgDir, f);
			if (!plan.has(dest)) {
				plan.set(dest, render(readFileSync(join(TEMPLATES, CONCEPT_TEMPLATE, 'package', `${f}.tmpl`), 'utf8'), cv, `${f}.tmpl`));
				notes.push(`added ${f} from the concept template: fill its hook and panels`);
			}
		}
	}
	const notesTpl = as === 'concept' ? join(TEMPLATES, CONCEPT_TEMPLATE, 'notes.md.tmpl') : join(TEMPLATES, 'base', 'notes.md.tmpl');
	plan.set(notesFile, render(readFileSync(notesTpl, 'utf8'), v, 'notes.md.tmpl').replace(`Scaffolded by\n\`atlas new ${as} ${slug}\``, `Scaffolded by\n\`atlas new ${as} ${slug} --like ${old}\``));
	const clash = [...plan.keys()].filter((p) => existsSync(p));
	if (existsSync(pkgDir) || clash.length)
		throw new AtlasError('AK301', `'${slug}' already exists`, { where: [pkgDir, ...clash].filter(existsSync).map(posix).join(', '), fix: 'pick another slug, or remove the old files first (nothing was written)' });
	for (const [p, text] of plan) writeText(p, text);
	notes.push(`copied ${copied} files of '${old}' and renamed '${old}' to '${slug}' in ${renamed} places (ids such as trainer and process ids are per package: they need no renaming)`);
	if (skipped.length) notes.push(`not copied (outputs of the old model, binary or large; train and export make your own): ${skipped.join(', ')}`);
	if (existsSync(join(pkgDir, 'claims.test.ts'))) notes.push(`claims.test.ts states facts about '${old}': rewrite each claim for '${slug}' before atlas check`);
	return { written: [...plan.keys()], notes, kinds, port, skipped };
}

// ------------------------------------------------------------------ new concept <slug> (K2)
/** A model manifest's text turned into a concept manifest: type 'concept', typed with the ConceptManifest alias (the
 *  frozen ExplainerManifest says type 'model'). An already-concept manifest is left as it is. */
export function asConceptManifest(text: string, notes: string[] = [], from = 'the source'): string {
	if (/type:\s*'concept'/.test(text)) return text;
	let m = text.replace(/type:\s*'model'/, "type: 'concept'");
	const alias =
		"/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */\n" +
		"type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };\n\n";
	if (/const manifest:\s*ExplainerManifest\s*=/.test(m)) m = m.replace(/const manifest:\s*ExplainerManifest\s*=/, `${alias}const manifest: ConceptManifest =`);
	else notes.push(`explainer.ts of ${from} is not written as \`const manifest: ExplainerManifest = {...}\`: give it the ConceptManifest type by hand`);
	notes.push(`explainer.ts now says type: 'concept' (a copy of the model package ${from}): rewrite its goals, prequestion and sources for the concept`);
	return m;
}

export async function newConcept(slug: string, o: Opts, r: Roots = roots()): Promise<{ written: string[]; notes: string[]; kinds: Kind[]; port: number; skipped?: string[] }> {
	if (!SLUG.test(slug)) throw new AtlasError('AK301', `'${slug}' is not a slug (lowercase letters, digits and single hyphens)`);
	if (o.template) throw new AtlasError('AK301', 'new concept takes no --template (its overlay is templates/concept/)', { fix: `npm run atlas -- new concept ${slug} [--kind ...] [--input <adapter>], or --like <package>` });
	if (o.like) return newModelLike(slug, o, r, 'concept');
	const notes: string[] = [];
	const entry = await conceptEntry(r, slug);
	if (!o.dev && !entry)
		throw new AtlasError('AK701', `'${slug}' is not in the catalog (src/lib/catalog/concepts.ts)`, {
			fix: `use --dev for a lab-only package, or propose the catalog entry first (maintainers add it; see CONTRIBUTING.md)`
		});
	if (!o.dev && STATIC_CONCEPTS.includes(slug)) notes.push(`'${slug}' has a hand-built page (src/routes/train/${slug}/), which takes precedence over the concept route: this package shows only in the Lesson Lab`);
	if (o.dev && entry) notes.push(`'${slug}' is in the catalog, but --dev makes a lab-only package; drop --dev to build its real page`);
	if (await catalogEntry(r, slug))
		throw new AtlasError('AK701', `'${slug}' is both a model and a concept slug in the catalog: one explainers/<slug>/ folder cannot hold both`, {
			fix: 'rename the concept (or the model) in the catalog first (maintainers; src/lib/catalog/concepts.ts)'
		});
	const kinds = o.kinds?.length ? o.kinds : (['process'] as Kind[]);
	const has = (k: Kind) => kinds.includes(k);
	if (has('trace') && !o.input) throw new AtlasError('AK301', 'kind trace needs --input <adapter>', { fix: `adapters: ${Object.keys(ADAPTER_KIND).join(', ')}` });
	if (o.input && !existsSync(join(r.site, 'src', 'lib', 'atlas', 'adapters', o.input, 'adapter.ts')))
		notes.push(`'${o.input}' is not a kit adapter: write a local one in adapters/${o.input}/ (adapter.ts + Capture.svelte; DEVKIT.md 5.4)`);
	const pkgDir = join(r.site, 'src', 'lib', 'explainers', slug);
	const pipeDir = join(r.root, 'pipeline', slug);
	const notesFile = join(r.root, 'notes', `${slug}.md`);
	const enginePresent = existsSync(join(r.site, 'src', 'lib', 'atlas', 'engine', 'host.ts'));
	const wantWorker = (has('train') || has('process')) && !o.noWorker;
	const worker = wantWorker && enginePresent;
	if (wantWorker && !enginePresent) notes.push(`worker.ts not generated: atlas/engine/host.ts is not in this tree. Add it later with \`npm run atlas -- new worker ${slug}\``);
	const title = { en: entry?.title.en ?? slug, zh: entry?.title.zh ?? entry?.title.en ?? slug };
	const interaction = { en: entry?.interaction.en ?? '(dev package: no catalog entry)', zh: entry?.interaction.zh ?? '' };
	const v: Vars = {
		slug,
		today: today(),
		dev: o.dev,
		catalog: !!entry,
		trace: has('trace'),
		train: has('train'),
		process: has('process'),
		worker,
		hasInput: !!o.input,
		adapter: o.input ?? 'TODO-adapter',
		inputKind: (o.input && ADAPTER_KIND[o.input]) || 'none',
		kindsList: kinds.map((k) => `'${k}'`).join(', '),
		kindsCsv: kinds.join(', '),
		kindsPy: kinds.map((k) => `"${k}"`).join(', ') + (kinds.length === 1 ? ',' : ''),
		processId: camel(slug),
		processIdent: camel(slug),
		trainerId: camel(slug),
		trainerIdent: camel(slug),
		serveArgs: [has('train') ? 'trainers' : '', has('process') ? 'processes' : ''].filter(Boolean).join(', '),
		// the base templates' names (pipeline README, model.py) read the concept's title and interaction
		nameEn: lit(title.en),
		nameZh: lit(title.zh),
		hookEn: lit(interaction.en),
		hookZh: lit(interaction.zh),
		showsEn: lit(interaction.en),
		showsZh: lit(interaction.zh),
		titleEn: lit(title.en),
		titleZh: lit(title.zh),
		interactionEn: lit(interaction.en),
		toySource: has('process') ? 'run' : has('train') ? 'train' : 'fwd',
		toyView: has('process') ? 'tape.table' : has('train') ? 'loss-curve' : 'layer.tensor',
		port: await portFor(slug, o, r),
		template: CONCEPT_TEMPLATE
	};
	// plan every file first (the base scaffold, then the concept overlay); refuse before writing anything if one exists
	const plan = new Map<string, string>();
	for (const { dir, files } of templateFiles(CONCEPT_TEMPLATE)) {
		for (const f of files) {
			if (f.when && !v[f.when]) continue;
			const dest = f.to === 'notes' ? notesFile : f.to.startsWith('package/') ? join(pkgDir, f.to.slice(8)) : f.to.startsWith('pipeline/') ? join(pipeDir, f.to.slice(9)) : null;
			if (!dest) throw new AtlasError('AK301', `template ${posix(dir)}: bad destination '${f.to}'`);
			if (f.skip) {
				plan.delete(dest);
				continue;
			}
			if (!f.from) throw new AtlasError('AK301', `template ${posix(dir)}: '${f.to}' has no 'from'`);
			plan.set(dest, render(readFileSync(join(dir, f.from), 'utf8'), v, posix(join(dir, f.from))));
		}
	}
	const clash = [...plan.keys()].filter((p) => existsSync(p));
	if (existsSync(pkgDir) || clash.length)
		throw new AtlasError('AK301', `'${slug}' already exists`, { where: [pkgDir, ...clash].filter(existsSync).map(posix).join(', '), fix: 'pick another slug, or remove the old files first (nothing was written)' });
	for (const [p, text] of plan) writeText(p, text);
	return { written: [...plan.keys()], notes, kinds, port: Number(v.port) };
}

export async function newWorker(slug: string, r: Roots = roots()): Promise<string> {
	const pkgDir = join(r.site, 'src', 'lib', 'explainers', slug);
	const dest = join(pkgDir, 'worker.ts');
	if (!existsSync(join(pkgDir, 'explainer.ts'))) throw new AtlasError('AK301', `no package '${slug}'`);
	if (existsSync(dest)) throw new AtlasError('AK301', `${posix(dest)} already exists`);
	const train = existsSync(join(pkgDir, 'trainer.ts'));
	const proc = existsSync(join(pkgDir, 'process.ts'));
	if (!train && !proc) throw new AtlasError('AK301', `${slug} has no trainer.ts or process.ts: it needs no worker`);
	if (!existsSync(join(r.site, 'src', 'lib', 'atlas', 'engine', 'host.ts')))
		throw new AtlasError('AK301', 'atlas/engine/host.ts (W1-B2) is not in this tree yet', { fix: 'run this again after wave 1b is merged' });
	const tpl = readFileSync(join(TEMPLATES, 'base', 'package', 'worker.ts.tmpl'), 'utf8');
	writeText(dest, render(tpl, { slug, train, process: proc, serveArgs: [train ? 'trainers' : '', proc ? 'processes' : ''].filter(Boolean).join(', ') }));
	return dest;
}

export function newOp(name: string, o: Opts, r: Roots = roots()): string[] {
	if (!/^[a-z][a-z0-9_]*$/.test(name)) throw new AtlasError('AK301', `op name '${name}' must be lowercase snake_case (conv1d, gcn_conv)`);
	const local = o.local;
	if (local && !existsSync(join(r.site, 'src', 'lib', 'explainers', local, 'explainer.ts'))) throw new AtlasError('AK301', `no package '${local}'`);
	const opsDir = local ? join(r.site, 'src', 'lib', 'explainers', local, 'ops') : join(r.site, 'src', 'lib', 'atlas', 'ops');
	const tsFile = join(opsDir, `${name}.ts`);
	const testFile = join(opsDir, `${name}.test.ts`);
	const pyFile = local ? join(r.root, 'pipeline', local, `ops_${name}.py`) : join(r.root, 'pipeline', 'atlas_kit', 'ops', `${name}.py`);
	const clash = [tsFile, testFile, pyFile].filter(existsSync);
	if (clash.length) throw new AtlasError('AK301', `op '${name}' already exists`, { where: clash.map(posix).join(', ') });
	const typesPath = posix(relative(opsDir, join(r.site, 'src', 'lib', 'atlas', 'types', 'index.ts')));
	const v: Vars = {
		op: name,
		opIdent: camel(name),
		opFile: `${name}.ts`,
		today: today(),
		local: !!local,
		slug: local ?? '',
		like: o.like ?? '',
		typesPath: typesPath.startsWith('.') ? typesPath : `./${typesPath}`,
		mirrorPath: posix(relative(r.root, pyFile)),
		// K4: repo-relative through the site (a lane outside the repo gave an absolute D:/... path here)
		tsPath: `site/${posix(relative(r.site, tsFile))}`
	};
	let ts = render(readFileSync(join(TEMPLATES, 'op', 'op.ts.tmpl'), 'utf8'), v, 'op.ts.tmpl');
	let test = render(readFileSync(join(TEMPLATES, 'op', 'op.test.ts.tmpl'), 'utf8'), v, 'op.test.ts.tmpl');
	let py = render(readFileSync(join(TEMPLATES, 'op', 'mirror.py.tmpl'), 'utf8'), v, 'mirror.py.tmpl');
	if (o.like) {
		const src = [join(r.site, 'src', 'lib', 'atlas', 'ops', `${o.like}.ts`), ...(local ? [join(opsDir, `${o.like}.ts`)] : [])].find(existsSync);
		if (!src) throw new AtlasError('AK301', `--like ${o.like}: no op file ${o.like}.ts`, { fix: 'name an existing op in src/lib/atlas/ops/ (or the package ops/ folder)' });
		ts = likeCopy(readFileSync(src, 'utf8'), { name, like: o.like, from: dirname(src), to: opsDir, note: posix(relative(r.site, src)) });
		test = render(readFileSync(join(TEMPLATES, 'op', 'op.like.test.ts.tmpl'), 'utf8'), v, 'op.like.test.ts.tmpl');
		// the Python mirror of the --like op, renamed (core mirrors live in pipeline/atlas_kit/ops/<op>.py)
		const pySrc = join(r.root, 'pipeline', 'atlas_kit', 'ops', `${o.like}.py`);
		if (existsSync(pySrc))
			py =
				`"""Python mirror of op '${name}' (${v.tsPath}), copied from pipeline/atlas_kit/ops/${o.like}.py by \`atlas new op ${name} --like ${o.like}\`.\n\n` +
				`Use it from describe(): ${local ? `from ops_${name} import reference as ${camel(name)}_ref; y = b.op("${name}", "my_layer", [x], params=..., ref=${camel(name)}_ref)` : 'register it in pipeline/atlas_kit/ops/__init__.py MIRRORS'}\n"""\n` +
				readFileSync(pySrc, 'utf8').replace(/^"""[\s\S]*?"""\n/, '').replace(new RegExp(`^OP = (['"])${o.like}\\1`, 'm'), `OP = "${name}"`);
	}
	writeText(tsFile, ts);
	writeText(testFile, test);
	writeText(pyFile, py);
	const out = [tsFile, testFile, pyFile];
	// a local op is listed in the package's ops.ts; write one when the package has none (K0: `new op --local` said "list
	// it in ops.ts" but the package had no ops.ts)
	if (local) {
		const opsTs = join(r.site, 'src', 'lib', 'explainers', local, 'ops.ts');
		if (!existsSync(opsTs)) {
			writeText(
				opsTs,
				`// ${local}: the ops its spec uses, the core ops and the package's own (DEVKIT.md 5.6). Written by \`atlas new op ${name} --local ${local}\`.\n` +
					`import { CORE_OPS } from '../../atlas/ops/index.ts';\nimport { ${camel(name)} } from './ops/${name}.ts';\n\nexport default [...CORE_OPS, ${camel(name)}];\n`
			);
			out.push(opsTs);
		}
	}
	return out;
}

/** A --like copy of an op file for another folder: its relative imports re-pointed from `from` to `to`, its OpDef export
 *  renamed to the new op's identifier, and its op name (the `op:` field only; terms and glossary ids keep theirs) changed. */
export function likeCopy(text: string, o: { name: string; like: string; from: string; to: string; note: string }): string {
	const rel = (spec: string) => {
		const p = posix(relative(o.to, resolve(o.from, spec)));
		return p.startsWith('.') ? p : `./${p}`;
	};
	let s = text.replace(/(\bfrom\s+|\bimport\s*\(\s*)(['"])(\.{1,2}\/[^'"]+)\2/g, (_m, pre: string, q: string, spec: string) => `${pre}${q}${rel(spec)}${q}`);
	const exports = [...s.matchAll(/export const (\w+)\s*:\s*OpDef\b/g)].map((m) => m[1]);
	const ident = exports.includes(camel(o.like)) ? camel(o.like) : exports[0];
	if (ident) {
		const to = camel(o.name);
		s = s.replace(new RegExp(`export const ${ident}\\b`), `export const ${to}`);
		s = s.replace(new RegExp(`(export default |export \\{[^}]*?)\\b${ident}\\b`, 'g'), `$1${to}`);
		// self-references in code (linear.explain(...)), not words in strings ('linear: weight must be ...')
		s = s.replace(new RegExp(`(?<![\\w'"\`.-])${ident}(?=\\.\\w|\\()`, 'g'), to);
	}
	s = s.replace(new RegExp(`(\\bop:\\s*)(['"])${o.like}\\2`, 'g'), `$1$2${o.name}$2`);
	return `// Scaffolded by \`atlas new op ${o.name} --like ${o.like}\`: a copy of ${o.note} (imports re-pointed, export renamed to ${camel(o.name)}). Rename what is left of '${o.like}'.\n` + s;
}

export function newView(key: string, o: Opts, r: Roots = roots()): string[] {
	if (!/^[a-z0-9]+([.-][a-z0-9]+)*$/.test(key)) throw new AtlasError('AK301', `view key '${key}' must be lowercase with dots or hyphens (plane2d, zoom.gate, grid-board)`);
	const local = o.local;
	if (local && !existsSync(join(r.site, 'src', 'lib', 'explainers', local, 'explainer.ts'))) throw new AtlasError('AK301', `no package '${local}'`);
	const dir = local ? join(r.site, 'src', 'lib', 'explainers', local, 'views', key) : join(r.site, 'src', 'lib', 'atlas', 'ui', 'views', key);
	if (existsSync(dir)) throw new AtlasError('AK301', `view folder already exists: ${posix(dir)}`);
	const typesPath = posix(relative(dir, join(r.site, 'src', 'lib', 'atlas', 'types', 'index.ts')));
	const fakePath = posix(relative(dir, join(r.site, 'src', 'lib', 'atlas', 'runtime', 'fake.ts')));
	const fake = existsSync(join(r.site, 'src', 'lib', 'atlas', 'runtime', 'fake.ts'));
	const v: Vars = { key, viewKey: local ? `${local}/${key}` : key, today: today(), typesPath, fakePath, fake };
	const files = [join(dir, 'meta.ts'), join(dir, 'View.svelte')];
	writeText(files[0], render(readFileSync(join(TEMPLATES, 'view', 'meta.ts.tmpl'), 'utf8'), v, 'meta.ts.tmpl'));
	writeText(files[1], render(readFileSync(join(TEMPLATES, 'view', 'View.svelte.tmpl'), 'utf8'), v, 'View.svelte.tmpl'));
	return files;
}

export async function main(args: string[]): Promise<number> {
	const { what, name, o } = parse(args);
	const r = roots();
	const show = (files: string[]) => files.forEach((f) => console.log(`  + ${posix(relative(r.root, f))}`));
	if (what === 'model') {
		const { written, notes, kinds, port } = await newModel(name, o, r);
		console.log(`atlas new model ${name} (${kinds.join(', ')}${o.dev ? ', dev-only' : ''}${o.template ? `, template ${o.template}` : ''}${o.like ? `, like ${o.like}` : ''})${r.lane ? ` in lane ${r.lane}` : ''}:`);
		show(written);
		for (const n of notes) console.log(`  note: ${n}`);
		const has = (k: Kind) => kinds.includes(k);
		let sources: [string, string][] = [];
		try {
			const { loadPackage } = await import('../load.ts');
			const l = await loadPackage(name, r.site);
			sources = Object.entries(l.pkg.manifest.sources ?? {}).map(([k, d]) => [k, d.kind]);
		} catch {
			// the kit is not complete in this tree: the default source names
			if (has('train')) sources.push(['train', 'train']);
			if (has('process')) sources.push(['run', 'process']);
		}
		const next: string[] = [];
		if (has('trace'))
			next.push(
				o.like
					? `adapt pipeline/${name}/model.py, train it (npm run atlas -- py pipeline/${name}/train.py), then npm run atlas -- export ${name}`
					: `write pipeline/${name}/model.py and train it (npm run atlas -- py <your script>); then npm run atlas -- export ${name}`
			);
		for (const [src, kind] of sources) if (kind === 'train' || kind === 'process') next.push(`npm run atlas -- run ${name} ${src} --frames 0..20`);
		next.push(`npm run dev -- --port ${port}  ->  /dev/atlas/lab/${name}/${o.dev ? '' : ` and /m/${name}/`}   (port recorded in notes/${name}.md)`);
		next.push(o.like ? `rewrite explainer.ts, lesson.ts and claims.test.ts for ${name}; npm run atlas -- lesson ${name} lint` : `fill the TODOs in explainer.ts and lesson.ts; npm run atlas -- lesson ${name} lint`);
		next.push(`npm run atlas -- check ${name} --fast while you work, then npm run atlas -- check ${name}; write notes/${name}.md`);
		console.log('\nNext:');
		next.forEach((line, i) => console.log(`  ${i + 1}. ${line}`));
		return 0;
	}
	if (what === 'concept') {
		const { written, notes, kinds, port } = await newConcept(name, o, r);
		console.log(`atlas new concept ${name} (${kinds.join(', ')}${o.dev ? ', dev-only' : ''}${o.like ? `, like ${o.like}` : ''})${r.lane ? ` in lane ${r.lane}` : ''}:`);
		show(written);
		for (const n of notes) console.log(`  note: ${n}`);
		const next: string[] = [];
		if (kinds.includes('trace') && !o.like) next.push(`export a model (npm run atlas -- export ${name}), or point the trace source at a live model's folder: { kind: 'trace', assets: 'models/tinycnn' }`);
		next.push(`npm run atlas -- run ${name} <source> --frames 0..20   (the sources in explainer.ts)`);
		next.push(`npm run dev -- --port ${port}  ->  ${o.dev ? `/dev/atlas/lab/${name}/` : `/train/${name}/ (the concept page; drafts are included in dev)`}   (port recorded in notes/${name}.md)`);
		next.push(`fill the TODOs in explainer.ts, lesson.ts and concept.ts; npm run atlas -- lesson ${name} lint; node --test src/lib/explainers/${name}/concept.test.ts`);
		next.push(`write claims.test.ts for every sentence that is not computed live; npm run atlas -- check ${name} --fast; npm run atlas -- e2e ${name} --url /train/${name}/`);
		console.log('\nNext:');
		next.forEach((line, i) => console.log(`  ${i + 1}. ${line}`));
		return 0;
	}
	if (what === 'worker') {
		const f = await newWorker(name, r);
		show([f]);
		return 0;
	}
	if (what === 'op') {
		console.log(`atlas new op ${name}${o.local ? ` (local to ${o.local})` : ' (core)'}:`);
		show(newOp(name, o, r));
		console.log(o.local ? `\nList it in src/lib/explainers/${o.local}/ops.ts: export default [...CORE_OPS, ${camel(name)}];` : `\nAdd it to CORE_OPS (src/lib/atlas/ops/index.ts), regenerate schema/ops.json, and register the mirror in pipeline/atlas_kit/ops/__init__.py.`);
		return 0;
	}
	if (what === 'view') {
		console.log(`atlas new view ${name}${o.local ? ` (local to ${o.local}, key '${o.local}/${name}')` : ' (kit)'}:`);
		show(newView(name, o, r));
		return 0;
	}
	throw new AtlasError('AK301', `atlas new ${what}: expected model, concept, worker, op or view`);
}
