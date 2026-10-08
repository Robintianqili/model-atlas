// atlas doctor: checks the environment and prints the exact fix for each failure (design §4.12).
//   npm run atlas -- doctor [--node-only] [--json]
// --node-only: Python problems are warnings (kinds B and C need no Python).
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { pythonChoice, pythonCommand, type PythonChoice } from '../config.ts';
import { formatIssue, type AkCode, type AtlasErrorInfo } from '../errors.ts';
import { posix } from '../lib/io.ts';
import { DEFAULT_DEV_PORT, PORTS, portFree } from '../ports.ts';
import { pythonEnv, roots, type Roots } from '../root.ts';

export type Status = 'ok' | 'warn' | 'fail';
export interface Check {
	name: string;
	status: Status;
	detail: string;
	code?: AkCode;
	issue?: AtlasErrorInfo & { message: string };
}

const PY_SCRIPT = `
import json, sys
out = {"version": sys.version.split()[0], "executable": sys.executable, "ok310": sys.version_info >= (3, 10), "pkgs": {}, "missing": []}
for name in ("torch", "numpy", "networkx"):
    try:
        m = __import__(name)
        out["pkgs"][name] = getattr(m, "__version__", "?")
    except Exception as e:
        out["missing"].append(name + ": " + type(e).__name__ + ": " + str(e)[:200])
try:
    import atlas_kit
    out["atlas_kit"] = atlas_kit.__version__
    from atlas_kit import numerics
    out["tf32"] = numerics.tf32_state()
    out["site"] = str(atlas_kit.atlas_site())
    out["root"] = str(atlas_kit.atlas_root())
except Exception as e:
    out["atlas_kit_error"] = type(e).__name__ + ": " + str(e)[:400]
try:
    import torch
    out["cuda"] = bool(torch.cuda.is_available())
except Exception:
    pass
print("ATLAS_DOCTOR " + json.dumps(out))
`;

function fixPython(c: PythonChoice): string {
	return (
		`install Python >= 3.10 and run \`pip install -r pipeline/requirements.txt\`, then point the CLI at it: ` +
		`create site/scripts/atlas/atlas.config.local.json (git-ignored) with {"python": "C:/path/to/python.exe"}, ` +
		`or set ATLAS_PYTHON. Now: '${c.python}' (from ${c.from}). Kinds B and C need no Python: \`doctor --node-only\``
	);
}

export function checkPython(r: Roots, env: NodeJS.ProcessEnv = process.env, choice: PythonChoice = pythonChoice(env)): Check[] {
	const cmd = pythonCommand(choice);
	if (!cmd) {
		return [
			{
				name: 'python',
				status: 'fail',
				detail: `no Python configured (${choice.from} is empty)`,
				code: 'AK002',
				issue: { message: `no Python configured (${choice.from} is empty)`, fix: fixPython(choice) }
			}
		];
	}
	const res = spawnSync(cmd, ['-c', PY_SCRIPT], { env: pythonEnv(r, env), encoding: 'utf8', timeout: 180_000, windowsHide: true });
	const out = `${res.stdout ?? ''}`;
	const err = `${res.stderr ?? ''}`;
	const line = out.split(/\r?\n/).find((l) => l.startsWith('ATLAS_DOCTOR '));
	if (res.error || !line) {
		const enoent = (res.error as NodeJS.ErrnoException | undefined)?.code === 'ENOENT';
		const store = /Microsoft Store|was not found; run without arguments/i.test(out + err) || res.status === 9009;
		const why = enoent
			? `Python not found: '${cmd}' (from ${choice.from})`
			: store
				? `'${cmd}' is the Windows Store alias, not an installed Python (from ${choice.from})`
				: `'${cmd}' did not run (from ${choice.from}): ${(res.error?.message ?? err.trim().split(/\r?\n/).slice(-1)[0] ?? `exit ${res.status}`).slice(0, 300)}`;
		return [{ name: 'python', status: 'fail', detail: why, code: 'AK002', issue: { message: why, fix: fixPython(choice) } }];
	}
	const info = JSON.parse(line.slice('ATLAS_DOCTOR '.length)) as {
		version: string;
		executable: string;
		ok310: boolean;
		pkgs: Record<string, string>;
		missing: string[];
		atlas_kit?: string;
		atlas_kit_error?: string;
		tf32?: Record<string, boolean>;
		site?: string;
		root?: string;
		cuda?: boolean;
	};
	const checks: Check[] = [];
	const where = `${posix(info.executable)} (from ${choice.from})`;
	if (!info.ok310) {
		const m = `Python ${info.version} is too old (need >= 3.10)`;
		checks.push({ name: 'python', status: 'fail', detail: `${m}: ${where}`, code: 'AK002', issue: { message: m, where, fix: fixPython(choice) } });
		return checks;
	}
	checks.push({ name: 'python', status: 'ok', detail: `Python ${info.version}: ${where}` });
	// minimum versions (pipeline/requirements.txt)
	const MIN: Record<string, [number, number]> = { torch: [2, 1], numpy: [1, 24], networkx: [3, 0] };
	for (const [name, [a, b]] of Object.entries(MIN)) {
		const v = info.pkgs[name];
		if (!v) continue;
		const [x, y] = v.split(/[.+]/).map(Number);
		if (x < a || (x === a && y < b)) {
			info.missing.push(`${name}: version ${v} is older than ${a}.${b} (pipeline/requirements.txt)`);
			delete info.pkgs[name];
		}
	}
	if (info.missing.length) {
		const names = info.missing.map((s) => s.split(':')[0]);
		const m = `Python packages missing or too old: ${names.join(', ')}`;
		checks.push({
			name: 'py-packages',
			status: 'fail',
			detail: m,
			code: 'AK002',
			issue: {
				message: m,
				where,
				likely: info.missing.join('; ').slice(0, 300),
				fix: `"${posix(info.executable)}" -m pip install -r "${posix(join(r.root, 'pipeline', 'requirements.txt'))}" (or point the CLI at a Python that has them: atlas.config.local.json)`
			}
		});
	} else {
		checks.push({
			name: 'py-packages',
			status: 'ok',
			detail: Object.entries(info.pkgs)
				.map(([k, v]) => `${k} ${v}`)
				.join(', ') + (info.cuda ? ' (CUDA available; exports still run on the CPU)' : '')
		});
	}
	if (info.atlas_kit_error) {
		const m = `atlas_kit does not import: ${info.atlas_kit_error}`;
		checks.push({ name: 'atlas_kit', status: 'fail', detail: m, code: 'AK002', issue: { message: m, where: posix(join(r.root, 'pipeline', 'atlas_kit')) } });
	} else if (info.tf32 && Object.values(info.tf32).some(Boolean)) {
		const m = `TF32 is still on after importing atlas_kit: ${JSON.stringify(info.tf32)}`;
		checks.push({ name: 'atlas_kit', status: 'fail', detail: m, code: 'AK102', issue: { message: m } });
	} else {
		const same = info.site && info.root && posix(info.site) === posix(r.site) && posix(info.root) === posix(r.root);
		checks.push({
			name: 'atlas_kit',
			status: same ? 'ok' : 'fail',
			detail: same
				? `atlas_kit ${info.atlas_kit} imports; TF32 off; Python sees the same ATLAS_SITE and ATLAS_ROOT`
				: `Python resolves ATLAS_SITE=${info.site}, ATLAS_ROOT=${info.root}; the CLI uses ${r.site}, ${r.root}`,
			...(same ? {} : { code: 'AK002' as AkCode, issue: { message: 'Python and the CLI disagree on ATLAS_SITE/ATLAS_ROOT' } })
		});
	}
	return checks;
}

/** node_modules/<pkg>/package.json, searched upward from the site (lanes resolve the main site's node_modules). */
function findPackage(site: string, pkg: string): { version: string; dir: string } | null {
	let d = site;
	for (;;) {
		const p = join(d, 'node_modules', ...pkg.split('/'), 'package.json');
		if (existsSync(p)) return { version: (JSON.parse(readFileSync(p, 'utf8')) as { version: string }).version, dir: join(d, 'node_modules') };
		const up = dirname(d);
		if (up === d) return null;
		d = up;
	}
}

export function checkNode(r: Roots): Check[] {
	const checks: Check[] = [];
	const major = Number(process.versions.node.split('.')[0]);
	if (major < 24) {
		const m = `Node ${process.versions.node} (need >= 24: the kit runs .ts files directly through type stripping)`;
		checks.push({ name: 'node', status: 'fail', detail: m, code: 'AK001', issue: { message: m, fix: 'install Node 24 LTS (https://nodejs.org) and run `npm ci` in site/' } });
	} else {
		// npm sets npm_config_user_agent ("npm/11.8.0 node/v24.13.1 win32 x64"); otherwise ask npm itself
		let npmVersion = /npm\/(\S+)/.exec(process.env.npm_config_user_agent ?? '')?.[1];
		if (!npmVersion) {
			const npm = spawnSync('npm --version', { encoding: 'utf8', shell: true, timeout: 60_000, windowsHide: true });
			npmVersion = npm.status === 0 ? npm.stdout.trim() : undefined;
		}
		checks.push({ name: 'node', status: 'ok', detail: `Node ${process.versions.node}${npmVersion ? `, npm ${npmVersion}` : ' (npm not found)'}` });
	}
	const want = ['svelte', '@sveltejs/kit', 'vite', 'typescript', 'katex', 'playwright-core'];
	const found = want.map((p) => [p, findPackage(r.site, p)] as const);
	const missing = found.filter(([, f]) => !f).map(([p]) => p);
	if (missing.length) {
		const m = `site packages not installed: ${missing.join(', ')}`;
		checks.push({
			name: 'packages',
			status: 'fail',
			detail: m,
			code: 'AK001',
			issue: { message: m, fix: `cd "${posix(r.lane ? join(r.root, 'site') : r.site)}" && npm ci   (lanes resolve the main site's node_modules)` }
		});
	} else {
		const dirs = [...new Set(found.map(([, f]) => posix(f!.dir)))];
		checks.push({ name: 'packages', status: 'ok', detail: `${found.map(([p, f]) => `${p} ${f!.version}`).join(', ')} (${dirs.join(', ')})` });
	}
	const types = join(r.site, 'src', 'lib', 'atlas', 'types', 'index.ts');
	checks.push(
		existsSync(types)
			? { name: 'atlas types', status: 'ok', detail: posix(types) }
			: { name: 'atlas types', status: 'fail', detail: `missing ${posix(types)}`, code: 'AK001', issue: { message: 'the frozen atlas types are missing', fix: 'sync the lane from main' } }
	);
	return checks;
}

export async function checkPorts(r: Roots): Promise<Check[]> {
	const table = Object.entries(PORTS);
	const all: [string, number][] = [['default dev', DEFAULT_DEV_PORT], ...table];
	const busy: string[] = [];
	for (const [who, port] of all) if (!(await portFree(port))) busy.push(`${port} (${who})`);
	const mine = r.lane ? PORTS[r.lane] : undefined;
	const mineBusy = mine !== undefined && busy.some((b) => b.startsWith(`${mine} `));
	if (!busy.length) return [{ name: 'ports', status: 'ok', detail: `all ${all.length} table ports free (${DEFAULT_DEV_PORT}, ${table[0][1]}..${table[table.length - 1][1]})` }];
	const m = `busy: ${busy.join(', ')}`;
	return [
		{
			name: 'ports',
			status: 'warn',
			detail: `${m}${mineBusy ? ` (including this lane's ${mine})` : ''}`,
			code: 'AK003',
			issue: { message: m, likely: 'dev servers of other lanes, or another program', fix: 'stop the server (and stop dev servers before any vite build: exFAT locks), or run on the free port the table gives your package' }
		}
	];
}

export function checkGit(r: Roots): Check[] {
	const run = (args: string[]) => spawnSync('git', args, { encoding: 'utf8', timeout: 60_000, windowsHide: true });
	const plain = run(['-C', r.root, 'rev-parse', '--short', 'HEAD']);
	if ((plain.error as NodeJS.ErrnoException | undefined)?.code === 'ENOENT')
		return [{ name: 'git', status: 'warn', detail: 'git not found on PATH', code: 'AK004', issue: { message: 'git not found on PATH', fix: 'install git (needed for merges and for contributing)' } }];
	if (plain.status === 0) return [{ name: 'git', status: 'ok', detail: `HEAD ${plain.stdout.trim()}` }];
	const safe = `safe.directory=${posix(r.root)}`;
	const dubious = /dubious ownership|safe\.directory/i.test(plain.stderr);
	const withSafe = run(['-c', safe, '-C', r.root, 'rev-parse', '--short', 'HEAD']);
	if (dubious && withSafe.status === 0)
		return [
			{
				name: 'git',
				status: 'ok',
				detail: `HEAD ${withSafe.stdout.trim()}; this drive (exFAT) needs: git -c "${safe}" ... on every command (never change the global config)`
			}
		];
	const m = `git cannot read the repo at ${posix(r.root)}: ${(withSafe.stderr || plain.stderr).trim().split(/\r?\n/)[0]}`;
	return [{ name: 'git', status: 'warn', detail: m, code: 'AK004', issue: { message: m, fix: `use git -c "${safe}" ...; if the repo is not initialized yet, ignore this` } }];
}

export async function checkPlaywright(r: Roots): Promise<Check[]> {
	const pkg = findPackage(r.site, 'playwright-core');
	if (!pkg) return [{ name: 'playwright', status: 'warn', detail: 'playwright-core not installed (e2e only)', code: 'AK001', issue: { message: 'playwright-core missing', fix: 'npm ci in site/' } }];
	try {
		const entry = join(pkg.dir, 'playwright-core', 'index.mjs');
		const pw = (await import(pathToFileURL(existsSync(entry) ? entry : join(pkg.dir, 'playwright-core', 'index.js')).href)) as {
			chromium: { launch(o: object): Promise<{ version(): string; close(): Promise<void> }> };
		};
		const b = await pw.chromium.launch({ channel: 'msedge', headless: true, timeout: 30_000 });
		const v = b.version();
		await b.close();
		return [{ name: 'playwright', status: 'ok', detail: `msedge ${v} launches headless (atlas e2e)` }];
	} catch (e) {
		const m = `playwright cannot launch msedge: ${(e as Error).message.split('\n')[0].slice(0, 200)}`;
		return [
			{
				name: 'playwright',
				status: 'warn',
				detail: m,
				code: 'AK001',
				issue: { message: m, fix: 'install Microsoft Edge (e2e uses channel msedge); on Linux CI use Playwright Chromium' }
			}
		];
	}
}

export async function doctor(args: string[], env: NodeJS.ProcessEnv = process.env): Promise<{ checks: Check[]; ok: boolean }> {
	const nodeOnly = args.includes('--node-only');
	const r = roots();
	const checks: Check[] = [];
	checks.push({
		name: 'paths',
		status: 'ok',
		detail: `ATLAS_SITE ${posix(r.site)} (${r.siteFrom}${r.lane ? `, lane ${r.lane}` : ', main tree'}); ATLAS_ROOT ${posix(r.root)} (${r.rootFrom})`
	});
	checks.push(...checkNode(r));
	const py = checkPython(r, env);
	checks.push(...(nodeOnly ? py.map((c) => (c.status === 'fail' ? { ...c, status: 'warn' as Status } : c)) : py));
	checks.push(...(await checkPlaywright(r)));
	checks.push(...(await checkPorts(r)));
	checks.push(...checkGit(r));
	return { checks, ok: !checks.some((c) => c.status === 'fail') };
}

export async function main(args: string[]): Promise<number> {
	const { checks, ok } = await doctor(args);
	if (args.includes('--json')) {
		console.log(JSON.stringify({ ok, checks }, null, 1));
		return ok ? 0 : 1;
	}
	console.log('atlas doctor');
	for (const c of checks) console.log(`  ${c.status.padEnd(4)}  ${c.name.padEnd(11)} ${c.detail}`);
	const issues = checks.filter((c) => c.status !== 'ok' && c.issue);
	if (issues.length) console.log('');
	for (const c of issues) console.log(formatIssue(c.code ?? 'AK002', c.issue!.message, c.issue!) + '\n');
	const fails = checks.filter((c) => c.status === 'fail').length;
	const warns = checks.filter((c) => c.status === 'warn').length;
	console.log(ok ? `doctor: OK${warns ? ` (${warns} warning${warns > 1 ? 's' : ''})` : ''}` : `doctor: ${fails} problem${fails > 1 ? 's' : ''} to fix (see the fix lines above)`);
	return ok ? 0 : 1;
}
