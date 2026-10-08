// Child processes for the CLI: the configured Python (with ATLAS_SITE, ATLAS_ROOT, PYTHONPATH), node scripts, npm.
import { spawn, spawnSync } from 'node:child_process';
import { pythonChoice, pythonCommand } from '../config.ts';
import { AtlasError } from '../errors.ts';
import { pythonEnv, roots } from '../root.ts';

export interface RunResult {
	code: number;
	stdout: string;
	stderr: string;
	ms: number;
}

/** Run a command; with `echo`, its output is also streamed to this process (line by line, prefixed). */
export function runCmd(cmd: string, args: string[], o: { cwd?: string; env?: NodeJS.ProcessEnv; echo?: boolean | string; timeoutMs?: number; shell?: boolean } = {}): Promise<RunResult> {
	const t0 = performance.now();
	return new Promise((resolve, reject) => {
		// with a shell (npm.cmd on Windows), pass one command line: Node deprecates separate args with shell: true
		const quote = (x: string) => (/[\s"]/.test(x) ? `"${x.replace(/"/g, '\\"')}"` : x);
		const p = o.shell ? spawn([cmd, ...args].map(quote).join(' '), { cwd: o.cwd, env: o.env ?? process.env, shell: true, windowsHide: true }) : spawn(cmd, args, { cwd: o.cwd, env: o.env ?? process.env, windowsHide: true });
		let out = '';
		let err = '';
		const prefix = typeof o.echo === 'string' ? o.echo : '';
		const relay = (chunk: string, stream: NodeJS.WriteStream) => {
			if (!o.echo) return;
			stream.write(prefix ? chunk.replace(/^(?=.)/gm, prefix) : chunk);
		};
		p.stdout.setEncoding('utf8');
		p.stderr.setEncoding('utf8');
		p.stdout.on('data', (c: string) => {
			out += c;
			relay(c, process.stdout);
		});
		p.stderr.on('data', (c: string) => {
			err += c;
			relay(c, process.stderr);
		});
		let timer: NodeJS.Timeout | undefined;
		if (o.timeoutMs) timer = setTimeout(() => p.kill(), o.timeoutMs);
		p.on('error', (e) => {
			if (timer) clearTimeout(timer);
			reject(e);
		});
		p.on('close', (code) => {
			if (timer) clearTimeout(timer);
			resolve({ code: code ?? 1, stdout: out, stderr: err, ms: performance.now() - t0 });
		});
	});
}

/** The configured Python command, after a quick check that it starts. AK002 with the fix otherwise. */
export function python(): string {
	const choice = pythonChoice();
	const cmd = pythonCommand(choice);
	if (!cmd)
		throw new AtlasError('AK002', 'no Python configured', {
			fix: 'set "python" in site/scripts/atlas/atlas.config.local.json (git-ignored) or ATLAS_PYTHON; see npm run atlas -- doctor'
		});
	const r = spawnSync(cmd, ['-c', 'import sys; print(sys.version_info[:2] >= (3, 10))'], { encoding: 'utf8', timeout: 60_000, windowsHide: true });
	if (r.error || r.status !== 0)
		throw new AtlasError('AK002', `Python not usable: '${cmd}' (from ${choice.from})`, {
			likely: r.error?.message ?? r.stderr.trim().split('\n').pop(),
			fix: 'npm run atlas -- doctor prints the exact fix'
		});
	return cmd;
}

/** Run the configured Python with the atlas environment (ATLAS_SITE, ATLAS_ROOT, PYTHONPATH=<root>/pipeline, UTF-8). */
export function runPython(args: string[], o: { cwd?: string; echo?: boolean | string; env?: NodeJS.ProcessEnv; timeoutMs?: number } = {}): Promise<RunResult> {
	return runCmd(python(), args, { cwd: o.cwd ?? roots().root, env: { ...pythonEnv(), ...(o.env ?? {}) }, echo: o.echo, timeoutMs: o.timeoutMs });
}

/** Run a .ts file with this Node (type stripping). */
export function runNode(args: string[], o: { cwd?: string; echo?: boolean | string; env?: NodeJS.ProcessEnv; timeoutMs?: number } = {}): Promise<RunResult> {
	return runCmd(process.execPath, args, { cwd: o.cwd ?? roots().site, env: { ...process.env, ...(o.env ?? {}) }, echo: o.echo, timeoutMs: o.timeoutMs });
}

/** npm run <script> in the site (npm is a .cmd on Windows, so it goes through the shell). */
export function runNpm(script: string, args: string[] = [], o: { cwd?: string; echo?: boolean | string; env?: NodeJS.ProcessEnv; timeoutMs?: number } = {}): Promise<RunResult> {
	const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
	const all = ['run', script, ...(args.length ? ['--', ...args] : [])];
	return runCmd(npm, all, { cwd: o.cwd ?? roots().site, env: { ...process.env, ...(o.env ?? {}) }, echo: o.echo, timeoutMs: o.timeoutMs, shell: process.platform === 'win32' });
}
