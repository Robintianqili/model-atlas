// CLI configuration. atlas.config.json is committed and machine-neutral ({"python": "python"}: Python from PATH);
// atlas.config.local.json is git-ignored and overrides it on one machine (for example a virtual environment's Python).
// The ATLAS_PYTHON environment variable overrides both.
import { existsSync } from 'node:fs';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AtlasError } from './errors.ts';
import { readJson } from './lib/io.ts';

export interface AtlasConfig {
	python: string;
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
		throw new AtlasError('AK002', `cannot parse ${file}: ${(e as Error).message}`, { fix: 'fix the JSON (it holds one key: "python")' });
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
