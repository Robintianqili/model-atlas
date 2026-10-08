// Preloaded into a dev server that atlas starts (node --import <this file> vite.js dev ...; K2 speed pack, lib/devserver.ts).
//   ATLAS_DEV_PARENT=<pid>   exit when that process (the atlas command that started the server) is gone, so a killed or
//                            crashed command never leaves its vite server behind
//   ATLAS_DEV_IDLE_MIN=<n>   exit after n minutes without an HTTP request (a kept server stops itself when nobody uses it)
// Exiting closes vite's stdin pipe to esbuild, so its esbuild child exits too.
import http from 'node:http';

const parent = Number(process.env.ATLAS_DEV_PARENT || 0);
const idleMin = Number(process.env.ATLAS_DEV_IDLE_MIN || 0);
let last = Date.now();

const emit = http.Server.prototype.emit;
http.Server.prototype.emit = function (this: http.Server, event: string | symbol, ...args: unknown[]): boolean {
	if (event === 'request' || event === 'upgrade') last = Date.now();
	return (emit as (this: http.Server, ...a: unknown[]) => boolean).apply(this, [event, ...args]);
} as typeof emit;

const alive = (pid: number) => {
	try {
		process.kill(pid, 0);
		return true;
	} catch (e) {
		return (e as NodeJS.ErrnoException).code === 'EPERM';
	}
};

if (parent || idleMin)
	setInterval(() => {
		if (parent && !alive(parent)) process.exit(0);
		if (idleMin && Date.now() - last > idleMin * 60_000) process.exit(0);
	}, 2000).unref();
