// A static file server for a build folder (redesign wave F): `atlas e2e <slug> --build <dir>` runs the browser passes on
// the built site instead of a dev server, so the JS a lesson loads is the real, bundled one (AK606 measures it) and page
// loads are fast. It serves <dir>/<path>, <dir>/<path>/index.html for folders, and <dir>/404.html (the SPA fallback)
// for anything else, on 127.0.0.1 only.
import { createServer, type Server } from 'node:http';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { AtlasError } from '../errors.ts';
import type { DevServer } from './devserver.ts';

const TYPES: Record<string, string> = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.mjs': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.json': 'application/json; charset=utf-8',
	'.svg': 'image/svg+xml',
	'.png': 'image/png',
	'.webp': 'image/webp',
	'.jpg': 'image/jpeg',
	'.woff2': 'font/woff2',
	'.woff': 'font/woff',
	'.ttf': 'font/ttf',
	'.txt': 'text/plain; charset=utf-8',
	'.wasm': 'application/wasm',
	'.bin': 'application/octet-stream',
	'.tokb': 'application/octet-stream'
};

/** The file a request path maps to in `root`, or null. */
export function fileFor(root: string, urlPath: string): string | null {
	let p: string;
	try {
		p = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
	} catch {
		return null;
	}
	const abs = resolve(root, '.' + normalize('/' + p));
	if (!abs.startsWith(resolve(root))) return null;
	if (existsSync(abs) && statSync(abs).isFile()) return abs;
	const idx = join(abs, 'index.html');
	if (existsSync(idx)) return idx;
	return null;
}

/** Serve `root` on a free port (from `port`); stop() closes it. */
export async function staticServer(root: string, port = 5290): Promise<DevServer> {
	if (!existsSync(join(root, 'index.html'))) throw new AtlasError('AK301', `--build ${root}: no index.html there (build the site first: ATLAS_BUILD_DIR=<dir> npm run build)`);
	const server: Server = createServer((req, res) => {
		const f = fileFor(root, req.url ?? '/');
		const file = f ?? join(root, '404.html');
		if (!existsSync(file)) {
			res.writeHead(404).end('not found');
			return;
		}
		res.writeHead(f ? 200 : 404, { 'content-type': TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream', 'cache-control': 'no-store' });
		res.end(readFileSync(file));
	});
	for (let p = port; p < port + 40; p++) {
		const ok = await new Promise<boolean>((done) => {
			server.once('error', () => done(false));
			server.listen(p, '127.0.0.1', () => done(true));
		});
		if (ok)
			return {
				base: `http://127.0.0.1:${p}`,
				port: p,
				how: 'started',
				why: `the build in ${root} (--build)`,
				kept: false,
				stop: () => void server.close()
			};
	}
	throw new AtlasError('AK003', `no free port for the build server from ${port}`);
}
