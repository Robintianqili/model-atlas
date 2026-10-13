// CLI configuration. atlas.config.json is committed and machine-neutral ({"python": "python"}: Python from PATH);
// atlas.config.local.json is git-ignored and overrides it on one machine (for example a virtual environment's Python).
// Keys:
//   python    the Python the CLI runs. The ATLAS_PYTHON environment variable overrides both files.
//   lanesDir  the folder that holds the lanes (<lanesDir>/<name>/site; default: site/.work). The ATLAS_LANES_DIR
//             environment variable overrides both files (set but empty: the default). A relative path is relative to
//             the main site folder. It is read from the MAIN site's scripts/atlas/ (lanesChoice), so the main tree and
//             its lanes agree on it.
import { existsSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AtlasError } from './errors.ts';
import { readJson } from './lib/io.ts';

export interface AtlasConfig {
	python: string;
	lanesDir: string;
}
export interface PythonChoice {
	python: string;
	from: 'ATLAS_PYTHON' | 'atlas.config.local.json' | 'atlas.config.json' | 'default';
}

export const CONFIG_DIR = dirname(fileURLToPath(import.meta.url));

function readConfig(file: string): Partial<AtlasConfig> | null {
	if (!existsSync(file)) return null;
	try {
		return readJson<Partial<AtlasConfig>>(file);
	} catch (e) {
		throw new AtlasError('AK002', `cannot parse ${file}: ${(e as Error).message}`, { fix: 'fix the JSON (its keys: "python" and "lanesDir", both optional)' });
	}
}

/** The Python the CLI runs, and where that choice came from. `dir` is for tests. */
export function pythonChoice(env: NodeJS.ProcessEnv = process.env, dir: string = CONFIG_DIR): PythonChoice {
	if (env.ATLAS_PYTHON !== undefined) return { python: env.ATLAS_PYTHON.trim(), from: 'ATLAS_PYTHON' };
	const local = readConfig(join(dir, 'atlas.config.local.json'));
	if (local && typeof local.python === 'string') return { python: local.python, from: 'atlas.config.local.json' };
	const shared = readConfig(join(dir, 'atlas.config.json'));
	if (shared && typeof shared.python === 'string') return { python: shared.python, from: 'atlas.config.json' };
	return { python: 'python', from: 'default' };
}

/** A configured path relative to the config folder is resolved; a bare command ('python', 'python3') is kept. */
export function pythonCommand(c: PythonChoice, dir: string = CONFIG_DIR): string {
	const p = c.python;
	if (!p) return p;
	if (isAbsolute(p) || !/[\\/]/.test(p)) return p;
	return resolve(dir, p);
}

// ------------------------------------------------------------------ where the lanes live
export interface LanesChoice {
	/** the absolute folder that holds the lanes: <dir>/<name>/site and <dir>/<name>/lane.json */
	dir: string;
	/** 'lane': the folder a lane is in (doctor, inside a lane whose folder is not the configured one) */
	from: 'ATLAS_LANES_DIR' | 'atlas.config.local.json' | 'atlas.config.json' | 'default' | 'lane';
	/** the default folder, <main>/.work. Lanes made there before a lanesDir was set are still found. */
	fallback: string;
	/** true when `dir` is not inside the main site: a lane cannot find the main site's node_modules by walking up, so the
	 *  lanes share `<dir>/node_modules` (npm run atlas -- lanes --deps) */
	outside: boolean;
	/** the node_modules folder the lanes resolve: <main>/node_modules, or <dir>/node_modules for lanes outside */
	deps: string;
}

/** Is `p` the folder `dir` or inside it? (false on another drive) */
export function isInside(p: string, dir: string): boolean {
	const r = relative(resolve(dir), resolve(p));
	return r === '' || (!r.startsWith('..') && !isAbsolute(r));
}

/** Where the lanes of the main site `main` live: ATLAS_LANES_DIR, else "lanesDir" in the main site's
 *  atlas.config.local.json, else in its atlas.config.json, else <main>/.work. */
export function lanesChoice(main: string, env: NodeJS.ProcessEnv = process.env): LanesChoice {
	const fallback = join(main, '.work');
	let raw: string | null = null;
	let from: LanesChoice['from'] = 'default';
	const set = env.ATLAS_LANES_DIR;
	if (set !== undefined) {
		if (set.trim()) {
			raw = set.trim();
			from = 'ATLAS_LANES_DIR';
		}
	} else {
		const cfgDir = join(main, 'scripts', 'atlas');
		for (const f of ['atlas.config.local.json', 'atlas.config.json'] as const) {
			const c = readConfig(join(cfgDir, f));
			if (c && typeof c.lanesDir === 'string' && c.lanesDir.trim()) {
				raw = c.lanesDir.trim();
				from = f;
				break;
			}
		}
	}
	const dir = raw ? resolve(main, raw) : fallback;
	const outside = !isInside(dir, main);
	return { dir, from, fallback, outside, deps: outside ? join(dir, 'node_modules') : join(main, 'node_modules') };
}
