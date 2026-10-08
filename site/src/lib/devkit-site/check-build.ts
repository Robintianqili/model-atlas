// Post-build check of the DevKit pages (after `npm run build`; ATLAS_BUILD_DIR as scripts/check-build.ts):
//   - every /devkit/ page (one per guide section included, the list follows DEVKIT.md) was prerendered;
//   - no generated link kept the site-path token (every href/src starts with the page's base);
//   - no page holds a note about the maintainers' own computer: its HTML, its __data.json and the JS it preloads are
//     searched for local-machine strings (local.ts: local paths, "this machine", ..., and this computer's own strings from
//     the git-ignored machine.local.json);
//   - the shell JS each DevKit page preloads (gzip), for the record.
//   node src/lib/devkit-site/check-build.ts
// scripts/check-build.ts covers the rest (third-party URLs, the flagship and planned-page budgets).
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { devkitPagePaths, parseGuide } from './guide.ts';
import { BASE } from './base.ts';
import { findLocalStrings } from './local.ts';

const SITE = fileURLToPath(new URL('../../../', import.meta.url));
const BUILD = join(SITE, process.env.ATLAS_BUILD_DIR || 'build');
let failed = 0;
const fail = (m: string) => {
	failed++;
	console.error(`FAIL ${m}`);
};

if (!existsSync(BUILD)) {
	console.error(`${BUILD} not found: run npm run build first`);
	process.exit(1);
}
const g = parseGuide(readFileSync(join(SITE, 'DEVKIT.md'), 'utf8'));
const paths = devkitPagePaths(g);
const walk = (d: string, out: string[] = []): string[] => {
	for (const n of readdirSync(d)) {
		const p = join(d, n);
		if (statSync(p).isDirectory()) walk(p, out);
		else out.push(p);
	}
	return out;
};
const gz = new Map(
	walk(join(BUILD, '_app'))
		.filter((f) => f.endsWith('.js'))
		.map((f) => [relative(BUILD, f).split('\\').join('/'), gzipSync(readFileSync(f), { level: 9 }).length])
);
let localScanned = 0;
const shell = (html: string) => {
	let total = 0;
	for (const m of new Set([...html.matchAll(/(?:href|import\()\s*=?\s*["']([^"']+\.js)["']/g)].map((x) => x[1]))) {
		const key = [...gz.keys()].find((k) => k.endsWith(m.replace(/^.*?_app\//, '_app/')));
		if (key) total += gz.get(key)!;
	}
	return total;
};
for (const p of paths) {
	const file = join(BUILD, 'devkit', ...p.split('/').filter(Boolean), 'index.html');
	if (!existsSync(file)) {
		fail(`/devkit/${p}${p ? '/' : ''} was not prerendered (${relative(SITE, file)})`);
		continue;
	}
	const html = readFileSync(file, 'utf8');
	if (!/<h1[\s>]/.test(html)) fail(`/devkit/${p}/ has no <h1>`);
	const token = [...html.matchAll(new RegExp(`(?:href|src)="${BASE}`, 'g'))].length;
	if (token) fail(`/devkit/${p}/ has ${token} link(s) that still start with ${BASE}`);
	// notes about the maintainers' own computer: the page, its data and the JS it loads
	const data = join(file, '..', '__data.json');
	const scanned: [string, string][] = [[relative(BUILD, file), html]];
	if (existsSync(data)) scanned.push([relative(BUILD, data), readFileSync(data, 'utf8')]);
	for (const m of new Set([...html.matchAll(/(?:href|import\()\s*=?\s*["']([^"']+\.js)["']/g)].map((x) => x[1]))) {
		const key = [...gz.keys()].find((k) => k.endsWith(m.replace(/^.*?_app\//, '_app/')));
		if (key) scanned.push([key, readFileSync(join(BUILD, key), 'utf8')]);
	}
	for (const [name, text] of scanned) {
		const hits = findLocalStrings(text);
		if (hits.length) fail(`/devkit/${p}${p ? '/' : ''}: ${name.replace(/\\/g, '/')} has local-machine strings: ${hits.slice(0, 3).join('; ')}`);
	}
	localScanned += scanned.length;
	console.log(`ok  /devkit/${p}${p ? '/' : ''}  ${(statSync(file).size / 1024).toFixed(1)} KB html, shell ${(shell(html) / 1024).toFixed(1)} KB gzip`);
}
console.log(`local-machine strings: ${localScanned} files searched (HTML, __data.json, preloaded JS)`);
console.log(failed ? `devkit-site: ${failed} problem(s)` : `devkit-site: all ${paths.length} DevKit pages prerendered, none with a local-machine string`);
process.exitCode = failed ? 1 : 0;
