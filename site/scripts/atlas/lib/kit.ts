// The kit modules of the site a command works on (ATLAS_SITE), imported by path.
// Commands import the kit dynamically from roots().site instead of statically from this CLI's own site, so a package and
// the kit it runs on always share one copy of every module (the executor's WeakMap of trace inputs, MissingWeightError,
// the recorder). The types come from this CLI's site (they are frozen).
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { AtlasError } from '../errors.ts';
import { roots } from '../root.ts';

type Compile = typeof import('../../../src/lib/atlas/lesson/compile.ts');
type Lint = typeof import('../../../src/lib/atlas/lesson/lint.ts');
type Plan = typeof import('../../../src/lib/atlas/lesson/plan.ts');
type Narrate = typeof import('../../../src/lib/atlas/lesson/narrate.ts');
type Runtime = typeof import('../../../src/lib/atlas/runtime/runtime.ts');
type Sources = typeof import('../../../src/lib/atlas/runtime/sources.ts');
type Local = typeof import('../../../src/lib/atlas/runtime/local.ts');
type Executor = typeof import('../../../src/lib/atlas/graph/executor.ts');
type Explain = typeof import('../../../src/lib/atlas/graph/explain.ts');
type Legacy = typeof import('../../../src/lib/atlas/graph/legacy.ts');
type Refs = typeof import('../../../src/lib/atlas/graph/refs.ts');
type CoreIo = typeof import('../../../src/lib/atlas/core/io.ts');
type Compare = typeof import('../../../src/lib/atlas/core/compare.ts');
type SchemaSpec = typeof import('../../../src/lib/atlas/schema/spec.ts');
type SchemaRef = typeof import('../../../src/lib/atlas/schema/ref.ts');
type SchemaFacts = typeof import('../../../src/lib/atlas/schema/facts.ts');
type SchemaTape = typeof import('../../../src/lib/atlas/schema/tape.ts');
type Record_ = typeof import('../../../src/lib/atlas/process/record.ts');
type ProcIo = typeof import('../../../src/lib/atlas/process/io.ts');
type Determinism = typeof import('../../../src/lib/atlas/process/determinism.ts');
type Ops = typeof import('../../../src/lib/atlas/ops/index.ts');

export interface Kit {
	site: string;
	compile: Compile;
	lint: Lint;
	plan: Plan;
	narrate: Narrate;
	runtime: Runtime;
	sources: Sources;
	/** K1: localRunner, sharedRunner (runs shared between the runtimes of lint fixtures) */
	local: Local;
	executor: Executor;
	explain: Explain;
	legacy: Legacy;
	refs: Refs;
	io: CoreIo;
	compare: Compare;
	schemaSpec: SchemaSpec;
	schemaRef: SchemaRef;
	schemaFacts: SchemaFacts;
	schemaTape: SchemaTape;
	record: Record_;
	procIo: ProcIo;
	determinism: Determinism;
	ops: Ops;
}

const cache = new Map<string, Promise<Kit>>();

async function mod<T>(site: string, rel: string): Promise<T> {
	const f = join(site, 'src', 'lib', 'atlas', ...rel.split('/'));
	if (!existsSync(f))
		throw new AtlasError('AK301', `the kit module atlas/${rel} is missing in ${site}`, {
			fix: 'run the CLI from a site with the merged kit (wave 1a or later), or sync the lane (npm run atlas -- worktree <name> --sync)'
		});
	return (await import(pathToFileURL(f).href)) as T;
}

/** Every kit module the commands use, from `site` (default ATLAS_SITE). Loaded once per site. */
export function kit(site: string = roots().site): Promise<Kit> {
	let p = cache.get(site);
	if (!p) {
		p = (async () => ({
			site,
			compile: await mod<Compile>(site, 'lesson/compile.ts'),
			lint: await mod<Lint>(site, 'lesson/lint.ts'),
			plan: await mod<Plan>(site, 'lesson/plan.ts'),
			narrate: await mod<Narrate>(site, 'lesson/narrate.ts'),
			runtime: await mod<Runtime>(site, 'runtime/runtime.ts'),
			sources: await mod<Sources>(site, 'runtime/sources.ts'),
			local: await mod<Local>(site, 'runtime/local.ts'),
			executor: await mod<Executor>(site, 'graph/executor.ts'),
			explain: await mod<Explain>(site, 'graph/explain.ts'),
			legacy: await mod<Legacy>(site, 'graph/legacy.ts'),
			refs: await mod<Refs>(site, 'graph/refs.ts'),
			io: await mod<CoreIo>(site, 'core/io.ts'),
			compare: await mod<Compare>(site, 'core/compare.ts'),
			schemaSpec: await mod<SchemaSpec>(site, 'schema/spec.ts'),
			schemaRef: await mod<SchemaRef>(site, 'schema/ref.ts'),
			schemaFacts: await mod<SchemaFacts>(site, 'schema/facts.ts'),
			schemaTape: await mod<SchemaTape>(site, 'schema/tape.ts'),
			record: await mod<Record_>(site, 'process/record.ts'),
			procIo: await mod<ProcIo>(site, 'process/io.ts'),
			determinism: await mod<Determinism>(site, 'process/determinism.ts'),
			ops: await mod<Ops>(site, 'ops/index.ts')
		}))();
		cache.set(site, p);
	}
	return p;
}

/** The folder under static/ that holds a package's assets. */
export function assetsDir(site: string, slug: string, dir?: string): string {
	return join(site, 'static', ...(dir ?? `models/${slug}`).split('/'));
}
