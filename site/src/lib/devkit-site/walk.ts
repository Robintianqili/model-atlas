// Browser check of the built DevKit pages (after `npm run build`; ATLAS_BUILD_DIR as scripts/check-build.ts): serves the
// build on a local port and opens every /devkit/ page in headless Edge (playwright-core, as atlas e2e), at 1280 px light in
// English and 390 px dark in Chinese, scrolled to the bottom so every lazy demo loads. Fails on a console error or
// warning, a page error, a failed request, a request to another host, horizontal page scroll, or a local-machine string
// (local.ts) in the visible text. Prints whether the site header links to /devkit/.
//   node src/lib/devkit-site/walk.ts [--port 5249] [--shots <dir>]
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { devkitPagePaths, parseGuide } from './guide.ts';
import { findLocalStrings } from './local.ts';

const SITE = fileURLToPath(new URL('../../../', import.meta.url));
const BUILD = join(SITE, process.env.ATLAS_BUILD_DIR || 'build');
const arg = (name: string) => {
	const i = process.argv.indexOf(name);
	return i > 0 ? process.argv[i + 1] : undefined;
};
const PORT = Number(arg('--port') ?? 5249);
const SHOTS = arg('--shots');

const TYPES: Record<string, string> = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript',
	'.css': 'text/css',
	'.json': 'application/json',
	'.svg': 'image/svg+xml',
	'.png': 'image/png',
	'.woff2': 'font/woff2',
	'.txt': 'text/plain'
};

if (!existsSync(BUILD)) {
	console.error(`${BUILD} not found: run npm run build first`);
	process.exit(1);
}
const req = createRequire(join(SITE, 'package.json'));
const pw = req('playwright-core') as typeof import('playwright-core');

const server = createServer((rq, rs) => {
	let f = join(BUILD, decodeURIComponent(new URL(rq.url ?? '/', 'http://x').pathname));
	if (existsSync(f) && statSync(f).isDirectory()) f = join(f, 'index.html');
	if (!existsSync(f)) {
		rs.writeHead(404);
		rs.end('not found');
		return;
	}
	rs.writeHead(200, { 'content-type': TYPES[extname(f)] ?? 'application/octet-stream' });
	createReadStream(f).pipe(rs);
});
await new Promise<void>((ok, bad) => server.once('error', bad).listen(PORT, ok));

const paths = devkitPagePaths(parseGuide(readFileSync(join(SITE, 'DEVKIT.md'), 'utf8')));
const passes = [
	{ name: '1280-en-light', width: 1280, height: 900, lang: 'en', scheme: 'light' as const },
	{ name: '390-zh-dark', width: 390, height: 844, lang: 'zh', scheme: 'dark' as const }
];
const problems: string[] = [];
let loads = 0;
let navLink = false;
const origin = `http://localhost:${PORT}`;
const browser = await pw.chromium.launch({ channel: process.env.ATLAS_BROWSER_CHANNEL || 'msedge', headless: true });
try {
	for (const ps of passes) {
		const ctx = await browser.newContext({ viewport: { width: ps.width, height: ps.height }, colorScheme: ps.scheme, reducedMotion: 'reduce' });
		const page = await ctx.newPage();
		const errs: string[] = [];
		page.on('console', (m) => {
			if (m.type() === 'error' || m.type() === 'warning') errs.push(`${m.type()}: ${m.text()}`);
		});
		page.on('pageerror', (e) => errs.push(`page error: ${e.message}`));
		page.on('requestfailed', (r) => errs.push(`request failed: ${r.url()}`));
		page.on('request', (r) => {
			if (!r.url().startsWith(origin) && !r.url().startsWith('data:')) errs.push(`request to another host: ${r.url()}`);
		});
		for (const p of paths) {
			errs.length = 0;
			const where = `${ps.name} /devkit/${p}${p ? '/' : ''}`;
			await page.goto(`${origin}/devkit/${p}${p ? '/' : ''}?lang=${ps.lang}`, { waitUntil: 'networkidle' });
			await page.evaluate(async () => {
				for (let y = 0; y < document.body.scrollHeight; y += 700) {
					window.scrollTo(0, y);
					await new Promise((r) => setTimeout(r, 60));
				}
				window.scrollTo(0, 0);
			});
			await page.waitForTimeout(300);
			loads++;
			const info = await page.evaluate(() => ({
				sw: document.documentElement.scrollWidth,
				cw: document.documentElement.clientWidth,
				text: document.body.innerText,
				nav: [...document.querySelectorAll('header a[href]')].some((a) => /\/devkit\/?$/.test(new URL((a as HTMLAnchorElement).href).pathname))
			}));
			navLink ||= info.nav;
			if (info.sw > info.cw + 1) problems.push(`${where}: horizontal scroll (${info.sw} > ${info.cw} px)`);
			for (const h of findLocalStrings(info.text)) problems.push(`${where}: ${h}`);
			for (const e of errs) problems.push(`${where}: ${e}`);
			if (SHOTS) {
				mkdirSync(SHOTS, { recursive: true });
				await page.screenshot({ path: join(SHOTS, `${ps.name}-${(p || 'home').replace(/\//g, '_')}.png`) });
			}
		}
		await ctx.close();
	}
} finally {
	await browser.close();
	server.close();
}
console.log(`devkit-site walk: ${loads} page loads (${paths.length} pages x ${passes.length} passes) on ${origin}`);
console.log(`site header links to /devkit/: ${navLink ? 'yes' : 'NO (add the "Build your own / 自己动手" link)'}`);
for (const p of problems) console.log(`FAIL ${p}`);
console.log(problems.length ? `devkit-site walk: ${problems.length} problem(s)` : 'devkit-site walk: no console error, failed or third-party request, horizontal scroll or local-machine string');
process.exitCode = problems.length ? 1 : 0;
