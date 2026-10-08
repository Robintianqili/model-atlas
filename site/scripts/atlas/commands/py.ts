// atlas py <script.py | -m module | -c code> [args...]   (afternoon test: "python" on PATH is often not the Python the kit
// needs, and nothing ran an arbitrary pipeline script with the configured one)
// Runs the configured Python (ATLAS_PYTHON, else atlas.config.local.json, else atlas.config.json; the same choice as
// `export` and `parity`) with PYTHONPATH=<root>/pipeline, ATLAS_SITE and ATLAS_ROOT, so `import atlas_kit as ak` works and
// files land in this site (a lane's copy inside a lane). The working folder is the one you ran npm from (INIT_CWD), as for a
// plain `python` call; a script path is found relative to it, else relative to the repo root, so from site/ both
//   npm run atlas -- py ../pipeline/hello-xor/train.py --seed 0      and      npm run atlas -- py pipeline/hello-xor/train.py
// work. Output streams through; the exit code is the script's.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { isAbsolute, resolve } from 'node:path';
import { AtlasError } from '../errors.ts';
import { python } from '../lib/child.ts';
import { posix } from '../lib/io.ts';
import { pythonEnv, roots } from '../root.ts';

export const USAGE = 'atlas py <script.py | -m module | -c code> [args...]';

/** The script a `py` call runs: as given relative to `cwd`, else relative to the repo root. */
export function resolveScript(arg: string, cwd: string, root: string): string | null {
	if (isAbsolute(arg)) return existsSync(arg) ? arg : null;
	for (const base of [cwd, root]) {
		const p = resolve(base, arg);
		if (existsSync(p)) return p;
	}
	return null;
}

export async function main(args: string[]): Promise<number> {
	if (!args.length || args[0] === '--help' || args[0] === '-h') throw new AtlasError('AK301', `usage: ${USAGE}`, { fix: 'npm run atlas -- py pipeline/<slug>/train.py --seed 0' });
	const r = roots();
	const cwd = process.env.INIT_CWD && existsSync(process.env.INIT_CWD) ? process.env.INIT_CWD : process.cwd();
	const argv = args.slice();
	if (argv[0] !== '-m' && argv[0] !== '-c') {
		const script = resolveScript(argv[0], cwd, r.root);
		if (!script) throw new AtlasError('AK301', `no script '${argv[0]}'`, { where: `looked in ${posix(cwd)} and ${posix(r.root)}`, fix: `${USAGE}; e.g. npm run atlas -- py pipeline/<slug>/train.py` });
		argv[0] = script;
	}
	const cmd = python();
	console.error(`atlas py: ${posix(cmd)} ${argv.map((a) => (/\s/.test(a) ? `"${a}"` : a)).join(' ')}${r.lane ? ` (lane ${r.lane})` : ''}`);
	return new Promise<number>((res, rej) => {
		const p = spawn(cmd, argv, { cwd, env: pythonEnv(r), stdio: 'inherit', windowsHide: true });
		p.on('error', rej);
		p.on('close', (code) => res(code ?? 1));
	});
}
