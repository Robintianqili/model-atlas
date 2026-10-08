// One dev server for `atlas e2e` and `atlas check` (K2 speed pack): reuse a running one, else start one, never orphan it.
//
// Which server a run uses, in this order:
//   1. --base <url>: that server, as given (never stopped).
//   2. the server this site's record names (.atlas/dev-server.json, written when atlas keeps a server) if its process
//      is alive and it answers the identity probe;
//   3. a server on the wanted port (--port, else the package's port) that answers the identity probe: a dev server of
//      THIS site folder that someone started (npm run dev -- --port N); it is reused and never stopped;
//   4. otherwise atlas starts one: node --import lib/devwatch.ts <vite>/bin/vite.js dev --port N --strictPort in this
//      site (one process, no npm or shell in between, so stopping it stops vite; its esbuild child exits with it).
//      The port: --port (AK003 when something else listens there), else the package's port when free, else the first
//      free port from 5210 up.
// The identity probe: a random token written to .svelte-kit/atlas-dev-probe.txt must come back from GET
// /.svelte-kit/atlas-dev-probe.txt, which a vite dev server serves from its own root only, so a server of another lane or of
// the main tree never matches. (.svelte-kit, not .atlas: the main tree's vite.config.ts does not allow .atlas, so the probe
// failed there and a kept server was never reused; SvelteKit allows its own .svelte-kit in every site.)
// A server atlas starts is stopped when the command ends, unless it is kept (--keep-server): then it runs detached, is
// recorded in .atlas/dev-server.json and stops itself after `idleMin` minutes without a request (default 20); stop it
// with --stop-server. A server that is not kept watches the atlas process (ATLAS_DEV_PARENT) and exits when it dies, so
// a killed command leaves nothing listening.
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { closeSync, existsSync, openSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { AtlasError } from '../errors.ts';
import { packagePort, pickPort, portFree } from '../ports.ts';
import { roots } from '../root.ts';
import { posix, readJson, writeJson, writeText } from './io.ts';

export const RECORD_FORMAT = 'atlas-dev/1';
export const DEFAULT_IDLE_MIN = 20;

export interface DevRecord {
	format: typeof RECORD_FORMAT;
	pid: number;
	port: number;
	site: string;
	started: string;
	idleMin: number;
}

export interface DevServer {
	base: string;
	port: number;
	/** given (--base), reused (a running server of this site), started (by this command) */
	how: 'given' | 'reused' | 'started';
	/** why this server: printed in the e2e summary */
	why: string;
	kept: boolean;
	/** stops a server this command started and did not keep; a no-op otherwise */
	stop(): void;
}

const recordFile = (site: string) => join(site, '.atlas', 'dev-server.json');
/** one log per port: a kept server holds its log open, so a server started next to it (another port) needs its own */
const logFile = (site: string, port: number) => join(site, '.atlas', `dev-server-${port}.log`);
const now = () => new Date().toLocaleString('sv-SE');

export function pidAlive(pid: number): boolean {
	if (!pid) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (e) {
		return (e as NodeJS.ErrnoException).code === 'EPERM';
	}
}

/** Kill a process and its children (Windows: taskkill /T /F; elsewhere SIGTERM). */
export function killTree(pid: number): void {
	if (!pid) return;
	if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true });
	else
		try {
			process.kill(pid, 'SIGTERM');
		} catch {
			/* gone */
		}
}

export function readRecord(site = roots().site): DevRecord | null {
	const f = recordFile(site);
	if (!existsSync(f)) return null;
	try {
		const r = readJson<DevRecord>(f);
		return r.format === RECORD_FORMAT ? r : null;
	} catch {
		return null;
	}
}

/** Is the server on `port` a vite dev server of `site`? (the token round trip; false on any error) */
export async function probeSite(port: number, site = roots().site, timeoutMs = 4000): Promise<boolean> {
	const token = randomBytes(12).toString('hex');
	const f = join(site, '.svelte-kit', 'atlas-dev-probe.txt');
	try {
		writeText(f, token);
		const ctl = new AbortController();
		const t = setTimeout(() => ctl.abort(), timeoutMs);
		try {
			for (const host of ['localhost', '127.0.0.1']) {
				try {
					const r = await fetch(`http://${host}:${port}/.svelte-kit/atlas-dev-probe.txt`, { signal: ctl.signal });
					if (r.ok && (await r.text()).trim() === token) return true;
				} catch {
					/* next host */
				}
			}
			return false;
		} finally {
			clearTimeout(t);
		}
	} finally {
		rmSync(f, { force: true });
	}
}

export async function waitHttp(url: string, ms: number, alive?: () => boolean): Promise<boolean> {
	const t0 = Date.now();
	while (Date.now() - t0 < ms) {
		if (alive && !alive()) return false;
		try {
			const r = await fetch(url);
			if (r.status < 500) return true;
		} catch {
			/* not up yet */
		}
		await new Promise((res) => setTimeout(res, 400));
	}
	return false;
}

/** <vite>/bin/vite.js as the site resolves it (a lane finds main's node_modules upward). */
export function viteBin(site = roots().site): string {
	const req = createRequire(join(site, 'package.json'));
	let pkg: string;
	try {
		pkg = req.resolve('vite/package.json');
	} catch {
		throw new AtlasError('AK001', 'vite is not installed in the site', { fix: 'npm install in site/' });
	}
	return join(dirname(pkg), 'bin', 'vite.js');
}

const WATCH = join(dirname(fileURLToPath(import.meta.url)), 'devwatch.ts');

/** Start `vite dev` on `port` in `site`. keep: detached, recorded, idle timeout; else it watches this process. */
export async function startServer(site: string, port: number, o: { keep?: boolean; idleMin?: number } = {}): Promise<{ pid: number; child: ChildProcess }> {
	const idleMin = o.idleMin ?? DEFAULT_IDLE_MIN;
	const env = { ...process.env, ATLAS_INCLUDE_DRAFTS: '1', ATLAS_DEV_PARENT: o.keep ? '' : String(process.pid), ATLAS_DEV_IDLE_MIN: o.keep ? String(idleMin) : '' };
	const args = ['--import', pathToFileURL(WATCH).href, viteBin(site), 'dev', '--port', String(port), '--strictPort'];
	let child: ChildProcess;
	if (o.keep) {
		writeText(logFile(site, port), `atlas dev server, port ${port}, started ${now()}\n`);
		const fd = openSync(logFile(site, port), 'a');
		child = spawn(process.execPath, args, { cwd: site, env, detached: true, stdio: ['ignore', fd, fd], windowsHide: true });
		closeSync(fd);
		child.unref();
	} else {
		child = spawn(process.execPath, args, { cwd: site, env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
		child.stdout?.resume();
		child.stderr?.resume();
	}
	const pid = child.pid ?? 0;
	let exited = false;
	child.once('exit', () => (exited = true));
	const up = await waitHttp(`http://localhost:${port}/`, 120_000, () => !exited);
	if (!up) {
		killTree(pid);
		throw new AtlasError('AK003', `the dev server did not answer on port ${port}${exited ? ' (it exited)' : ''}`, {
			where: o.keep ? posix(logFile(site, port)) : undefined,
			fix: `free the port, or start it yourself (npm run dev -- --port ${port}) and pass --base http://localhost:${port}`
		});
	}
	if (o.keep) writeJson(recordFile(site), { format: RECORD_FORMAT, pid, port, site: posix(site), started: now(), idleMin } satisfies DevRecord, 1);
	return { pid, child };
}

/** Wait until nothing listens on `port` (at most `ms`). */
async function portFreed(port: number, ms = 10_000): Promise<boolean> {
	const t0 = Date.now();
	while (Date.now() - t0 < ms) {
		if (await portFree(port)) return true;
		await new Promise((r) => setTimeout(r, 300));
	}
	return portFree(port);
}

/** Stop the kept server of this site (the record's process tree), then remove the record. */
export async function stopKept(site = roots().site): Promise<{ stopped: boolean; port?: number; pid?: number; note: string }> {
	const rec = readRecord(site);
	if (!rec) return { stopped: false, note: 'no kept dev server recorded for this site' };
	const alive = pidAlive(rec.pid);
	if (alive) killTree(rec.pid);
	rmSync(recordFile(site), { force: true });
	const freed = await portFreed(rec.port);
	return { stopped: alive, port: rec.port, pid: rec.pid, note: alive ? `stopped the kept dev server (pid ${rec.pid}, port ${rec.port})${freed ? '' : `; port ${rec.port} is still busy`}` : `the recorded server (pid ${rec.pid}) had already stopped` };
}

export interface ServerOpts {
	slug?: string;
	port?: number;
	base?: string;
	keep?: boolean;
	idleMin?: number;
	site?: string;
}

/** The dev server a run uses (see the header). Always call stop() when done (in a finally). */
export async function devServer(o: ServerOpts = {}): Promise<DevServer> {
	const r = roots();
	const site = o.site ?? r.site;
	const noop = () => {};
	if (o.base) return { base: o.base.replace(/\/+$/, ''), port: Number(/:(\d+)/.exec(o.base)?.[1] ?? 0), how: 'given', why: '--base', kept: false, stop: noop };
	// 2. the recorded (kept) server
	const rec = readRecord(site);
	if (rec) {
		if (pidAlive(rec.pid) && (o.port === undefined || o.port === rec.port) && (await probeSite(rec.port, site))) return { base: `http://localhost:${rec.port}`, port: rec.port, how: 'reused', why: `the kept dev server (pid ${rec.pid})`, kept: true, stop: noop };
		if (!pidAlive(rec.pid)) rmSync(recordFile(site), { force: true });
	}
	// 3. a running server of this site on the wanted port
	const own = o.slug ? packagePort(o.slug, r.root) : null;
	const wanted = o.port ?? own ?? null;
	if (wanted && !(await portFree(wanted))) {
		if (await probeSite(wanted, site)) return { base: `http://localhost:${wanted}`, port: wanted, how: 'reused', why: `a dev server of this site already on port ${wanted}`, kept: false, stop: noop };
		if (o.port !== undefined) throw new AtlasError('AK003', `port ${o.port} is busy (and not a dev server of this site)`, { fix: `stop what runs there, pick another --port, or test that server with --base http://localhost:${o.port}` });
	}
	// 4. start one
	const port = o.port ?? (await pickPort({ preferred: own, slug: o.slug, root: r.root }));
	const why = o.port !== undefined ? '--port' : own === port ? "the package's port" : own ? `the package's port ${own} is busy` : 'the first free port from 5210';
	const { pid } = await startServer(site, port, { keep: o.keep, idleMin: o.idleMin });
	let stopped = false;
	const stop = () => {
		if (stopped || o.keep) return;
		stopped = true;
		killTree(pid);
	};
	if (!o.keep) {
		// a Ctrl-C or an uncaught error still stops it (the server also watches this process: ATLAS_DEV_PARENT)
		process.once('exit', stop);
		for (const sig of ['SIGINT', 'SIGTERM'] as const)
			process.once(sig, () => {
				stop();
				process.exit(130);
			});
	}
	return { base: `http://localhost:${port}`, port, how: 'started', why: `${why}${o.keep ? `; kept (stops after ${o.idleMin ?? DEFAULT_IDLE_MIN} idle minutes, or --stop-server)` : ''}`, kept: !!o.keep, stop };
}
