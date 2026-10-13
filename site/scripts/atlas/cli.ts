// The DevKit CLI (design §4.12):  npm run atlas -- <command> [args]   (from site/ or a lane's site/)
// Every failure prints an AKxxx block (code, where, likely, fix, docs) and exits 1.
import { AtlasError } from './errors.ts';
import { roots } from './root.ts';

type Cmd = { summary: string; run: (args: string[]) => Promise<number> };

export const COMMANDS: Record<string, Cmd> = {
	doctor: {
		summary: 'doctor [--node-only] [--json]: check Node, the site packages, where lanes live and their shared packages, Python (+ torch, numpy, networkx, atlas_kit), playwright msedge, ports and git; print fixes',
		run: async (a) => (await import('./commands/doctor.ts')).main(a)
	},
	new: {
		summary: 'new model <slug> --kind trace|train|process[,...] [--input <adapter>] [--template <name>] [--dev] [--port n] | new model <slug> --like <package> [--dev] [--port n] | new concept <slug> [--kind trace|train|process[,...]] [--input <adapter>] [--dev] [--port n] | new concept <slug> --like <package> | new worker <slug> | new op <name> [--local <slug>] [--like <op>] | new view <key> [--local <slug>]: scaffold a model package (manifest, lesson, trainer/process, Python side, notes with its dev-server port), a concept package (type concept with concept.ts, routed at /train/<slug>/; --like turns a model package into a concept), a renamed copy of an existing package, a worker entry, an op (TS + test + Python mirror) or a view (meta + component); never overwrites',
		run: async (a) => (await import('./commands/new.ts')).main(a)
	},
	py: {
		summary: 'py <script.py | -m module | -c code> [args...]: run a pipeline script with the configured Python (the one export and parity use), PYTHONPATH=pipeline, ATLAS_SITE and ATLAS_ROOT set',
		run: async (a) => (await import('./commands/py.ts')).main(a)
	},
	export: {
		summary: 'export <slug> [--no-parity] [--no-draft] [-- <export.py args>]: run pipeline/<slug>/export.py with the configured Python, then parity --write and lesson draft',
		run: async (a) => (await import('./commands/export.ts')).main(a)
	},
	parity: {
		summary: 'parity <slug> [--write] [--json] [--dir <folder under static/>] [--steps n] [--fixtures]: kind A against reference.json, kind B against the twin (one run per trainer and options), kind C against the Python reference (--fixtures: on every fixture input; files in .atlas/<slug>/parity/); --write records spec.parity and explainers/<slug>/parity.json (the badge); legacy scripts with --no-write',
		run: async (a) => (await import('./commands/parity.ts')).main(a)
	},
	run: {
		summary: "run <slug> <source> [--fixture id] [--frames a..b] [--seed n] [--opt '<json>'] [--detail coarse|fine] [--save] [--precompute]: print a process or trainer tape in Node, its input from a real runtime on the fixture (--opt overrides source options, e.g. '{\"hidden\":3}')",
		run: async (a) => (await import('./commands/run.ts')).main(a)
	},
	lesson: {
		summary: 'lesson <slug> draft|lint|print [--lang en|zh] [--fixture id] [--json] [--cap n]: the auto plan, the lint over every fixture (identical runs shared between fixtures; --cap n caps training while you iterate, never a gate), or the narration tables',
		run: async (a) => (await import('./commands/lesson.ts')).main(a)
	},
	check: {
		summary: 'check <slug> [<slug>...] [--quick|--fast] [--passes <list>] [--json] [--only gate,gate] [--port n] [--keep-server|--no-keep-server] [--stop-server] | check --all [<slug>...] [--e2e <slugs|all>] [--passes <list>] [--fast] [--json]: every gate (schema, parity, op tests, lesson, determinism, honesty, budgets, types, site, e2e, paperwork, ownership) with a package-only build and the default e2e passes; --quick while iterating (no build or browser; parity and op tests from a cache while their inputs are unchanged); --fast without types, site and e2e; several slugs share one build and one e2e sequence; --all runs the fast gates of every package, the kit tests, docs --check, legacy parity, one full build and one e2e sequence over the packages given (the default passes; the integrator gives none)',
		run: async (a) => (await import('./commands/check.ts')).main(a)
	},
	build: {
		summary: 'build <slug>[,<slug>...] [--no-types] | build --all [--no-types]: a package-only build (ATLAS_ONLY: the home page and the pages of the packages) into .atlas/build-<slug>/ after svelte-check of the packages and check:catalog, then check:build; --all: npm run check and the full build',
		run: async (a) => (await import('./commands/build.ts')).main(a)
	},
	e2e: {
		summary: 'e2e <slug>[,<slug>...] [--passes default|all|full|layout|<width>[x<height>]-<lang>-<scheme>[-reduced],...] [--full] [--phone px] [--url <path>] [--port n] [--base <url> | --build <dir>] [--keep-server|--no-keep-server] [--stop-server] [--axe fail|warn|off] [--no-contact] [--wait ms] [--shots <dir>] [--cap n|off] [--as-live]: Playwright (Edge, headless) through the lesson; default passes 1280 px en light and 390 px zh dark (with a 360 px scroll probe), all or full for more, layout for the lesson app\'s windows (the app/flow mode check, AK605); an axe scan of every step (AK604); contact sheets, lesson-print.txt and axe.txt in notes/shots/<slug>/contact/ (a lane outside the repo: <lane>/shots/<slug>/contact/); reuses a dev server of this site or starts one and keeps it (it stops itself after 20 idle minutes); --build <dir> serves a built site instead and measures the JS the lesson loads (AK606); training capped at 200 steps (--cap); --as-live (a lane) shows the package as its promoted catalog entry for the run',
		run: async (a) => (await import('./commands/e2e.ts')).main(a)
	},
	worktree: {
		summary: 'worktree <name> [--owns <glob,...>] [--from-git <rev>] | worktree <name> --sync [--from-git <rev>]: make a lane copy <lanes>/<name>/site (the lanes folder: ATLAS_LANES_DIR or "lanesDir" in atlas.config(.local).json, default site/.work) from the main tree or from a committed revision (hash snapshot; lane.json records the repo root and the main site; the local CLI config is copied, never merged), or refresh its non-owned files (AK703)',
		run: async (a) => (await import('./commands/worktree.ts')).main(a)
	},
	lanes: {
		summary: 'lanes [--json] | lanes --deps | lanes --rm <name> [--force] | lanes --clean <name|main>: where lanes live and the lanes there (size, source, merged or not); --deps installs the node_modules that lanes outside the main site share (npm ci from site/package.json and package-lock.json into <lanes>/node_modules); --rm deletes a lane with its e2e shots, refused while it holds owned changes main does not have (--force deletes anyway); --clean deletes the build output and caches of a lane or of main (build, .svelte-kit, .atlas/build*, node_modules/.vite) after stopping its kept dev server',
		run: async (a) => (await import('./commands/lanes.ts')).main(a)
	},
	merge: {
		summary: "merge <name> [--dry-run]: copy a lane's owned files back into the main tree; refuses (AK702) if the lane changed anything else",
		run: async (a) => (await import('./commands/worktree.ts')).mergeMain(a)
	},
	review: {
		summary: 'review <slug>: write notes/review-<slug>.md (checklist, the last check with its mode, the catalog patch, contact sheets and screenshots, every narration on 3 inputs in en and zh)',
		run: async (a) => (await import('./commands/review.ts')).main(a)
	},
	promote: {
		summary: "promote <slug> [--dry-run | --verify | --apply] [--all-terms]: the catalog patch from the manifest (models.ts or concepts.ts entry: status, lesson and manifest.catalog's fields; tech-zh.ts rows to add and the rows it leaves unused; glossary.ts terms another page also uses) with checks by hand (measured weights size in 1,024-byte KB, params, the page's input kinds, mode, similar tech-zh rows); appends credits to static/licenses.json (not with --dry-run or --verify); --verify applies the patch to a copy of the catalog and runs check:catalog on it (a lane too); --apply writes it into the catalog files in the main tree, keeping each file's line endings",
		run: async (a) => (await import('./commands/promote.ts')).main(a)
	},
	thumbs: {
		summary: 'thumbs [<slug>...] [--build <dir> | --base <url>] [--port n] [--out <dir>] [--sheets <dir>] [--cols n]: the catalog thumbnails: each live lesson\'s poster stage (manifest.page.poster, else the first step with a figure) in headless Edge at 960 x 600, light and dark, cropped to its largest figure, composed into 4 webp sprites (rows 224 x 140 for every lesson, cards 480 x 300 for the home set) and index.json in static/thumbs/, with labelled contact sheets for the review in .atlas/thumbs/',
		run: async (a) => (await import('./commands/thumbs.ts')).main(a)
	},
	glossary: {
		summary: 'glossary [--json]: the site glossary\'s size (gzip, in every page that uses Term, the flagships too) and the terms only one package uses and also defines in its manifest (safe to take out of glossary.ts); read-only',
		run: async (a) => (await import('./commands/glossary.ts')).main(a)
	}
};

function help(): string {
	const lines = ['Model Atlas DevKit CLI', '', 'usage: npm run atlas -- <command> [args]', ''];
	for (const [name, c] of Object.entries(COMMANDS)) lines.push(`  ${name.padEnd(9)} ${c.summary}`);
	lines.push('', 'Guide: site/DEVKIT.md (error codes: site/DEVKIT.md#ak<code>). Design: research/devkit/DEVKIT_DESIGN.md §4.12.');
	return lines.join('\n');
}

export async function cli(argv: string[]): Promise<number> {
	const [name, ...args] = argv;
	if (!name || name === 'help' || name === '--help' || name === '-h') {
		console.log(help());
		return name ? 0 : 1;
	}
	const cmd = COMMANDS[name];
	if (!cmd) {
		console.error(`unknown command '${name}'\n\n${help()}`);
		return 1;
	}
	try {
		const r = roots(); // resolve ATLAS_SITE / ATLAS_ROOT once, early (AK002 if the layout is wrong)
		// a lane outside the repo: no child (node --test, vite, svelte-check, Python) can find the repo by walking up, so
		// every child gets the root lane.json records
		if (r.rootFrom === 'lane') process.env.ATLAS_ROOT = r.root;
		return await cmd.run(args);
	} catch (e) {
		if (e instanceof AtlasError) {
			console.error(e.format());
			return 1;
		}
		throw e;
	}
}

if (process.argv[1] && /cli\.ts$/.test(process.argv[1])) {
	const code = await cli(process.argv.slice(2));
	process.exitCode = code;
}
