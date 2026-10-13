// atlas e2e <slug>[,<slug>...] [--passes <list>] [--full] [--phone px] [--url <path>] [--port n] [--base <url>]
//   [--keep-server | --no-keep-server] [--stop-server] [--axe fail|warn|off] [--no-contact] [--wait ms] [--shots <dir>]
//   [--cap n|off] [--as-live] [--build <dir>]                         (design §6.3 "E2E"; K2 speed pack; K4; redesign F)
//
// Redesign wave F: --build <dir> runs the passes on a built site (a static server on 127.0.0.1) instead of a dev server;
// then AK606 measures the JS the lesson has loaded (gzip, as check:build counts it) once it settles and after its last
// step (over 400 KB warns now; it becomes a gate after the hand-built pages move to the lesson app). Passes can name an
// exact window, <w>x<h>-<lang>-<scheme> (1280x680-en-light), and the 'layout' preset runs the lesson-app windows. A page
// whose lesson root carries data-lesson-mode (the lesson app) is checked for its layout mode (AK605): 'app' at 960 x 600
// and up with the document exactly one window tall (the panes scroll, the page never does), 'flow' below.
//
// K4 --as-live (a lane only): the run sees the package as the catalog will once it is promoted: promote's patch (status
// 'live', lesson, name, hook, inputs, mode, ...) is written into the lane's catalog files for the run and the files are
// put back afterwards (byte for byte; a run that was killed puts them back at the next e2e). Without it an unmerged
// package renders as a planned entry ("(Planned)" in the title, provenance 'precomputed' by default).
//
// Playwright (playwright-core, Edge or ATLAS_BROWSER_CHANNEL, headless) on the page's guided lesson (kit Walkthrough,
// `section.wt`), stepping through it with the step dots. The page: --url, else /dev/atlas/lab/<slug>/ for devOnly
// packages, else /train/<slug>/ for a concept package, else /m/<slug>/; `?atlas-e2e=1` is added (the runtime caps live training at a fixed step count with it).
// Several packages (a,b,c) run as one sequence: one dev server, one browser.
//
// Passes (lib/passes.ts): --passes takes pass names <desktop|phone|narrow|px>-<en|zh>-<light|dark>[-reduced] and presets.
//   default  desktop-en-light (1280 px) and phone-zh-dark (390 px; also probes horizontal scroll at 360 px)
//   all      + narrow-en-dark (360 px), desktop-zh-light, desktop-en-light-reduced (on request; no gate needs them)
//   full     + desktop-en-dark, narrow-zh-light, phone-en-light (--full)
//   --phone <px> adds <px>-en-light.
// Every pass, every step: the narration is not empty and shows no "…" (AK401); a zh pass shows Chinese (AK406); the first
// English desktop pass needs a provenance badge on the stage (AK502); a pass at <= 430 px has no horizontal page scroll
// (AK602); console errors and page errors fail (AK601); long tasks over 50 ms only warn (AK603, lean gates), counted
// once the web fonts have loaded and never during atlas's own work (the axe scans and the 360 px probe); the reduced-motion
// pass checks that nothing autoplays. Accessibility: an axe scan of every step (lib/axe.ts; WCAG 2.1 A + AA; serious and
// critical fail with AK604, moderate and minor warn; --axe warn|off while iterating).
// Output in notes/shots/<slug>/ (git-ignored; K4: <lane folder>/shots/<slug>/ for a lane outside the repo, so a lane on
// another drive writes nothing to the repo drive; or --shots <dir>): e2e-<pass>-<nn>.png per step, and contact/ with one
// contact sheet per pass (<pass>.png: every step tiled and labelled), lesson-print.txt (each step's title, narration,
// look-at hint, badge and numbers per language), axe.txt and axe-<pass>.json (lib/contact.ts, lib/axe.ts).
// Dev server (lib/devserver.ts): --base, else a running dev server of this site (the kept one, or one on the package's
// port), else one it starts: --port, else the package's port when free, else the first free port from 5210. A server it
// starts is kept for the next run (it stops itself after 20 idle minutes) unless --no-keep-server; --stop-server stops
// the kept one (with no slug: only that).
import type { Browser, Page } from 'playwright-core';
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, relative, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { AtlasError } from '../errors.ts';
import { AK_A11Y, axeIssues, axeText, loadAxe, scan, type StepScan } from '../lib/axe.ts';
import { lessonPrint, renderSheet, type StepText, type Tile } from '../lib/contact.ts';
import { devServer, readRecord, stopKept, type DevServer } from '../lib/devserver.ts';
import { posix, writeBytes, writeJson, writeText } from '../lib/io.ts';
import { legacyPasses, resolvePasses, type Pass } from '../lib/passes.ts';
import { Issues, type Issue } from '../lib/pkg.ts';
import { roots, shotsDir as defaultShotsDir } from '../root.ts';
import { staticServer } from '../lib/staticserver.ts';
import { gapMarks } from '../../../src/lib/atlas/lesson/narrate.ts';

export interface E2eOpts {
	/** the full preset (old flag) */
	full?: boolean;
	url?: string;
	port?: number;
	base?: string;
	waitMs?: number;
	echo?: boolean;
	/** screenshot folder (default shotsDir(slug): notes/shots/<slug>/, or <lane folder>/shots/<slug>/ for a lane outside the repo) */
	shots?: string;
	/** an extra pass at this phone width (light, en) */
	phone?: number;
	/** the training cap of ?atlas-e2e=1 (default 200); 0 = no cap */
	cap?: number;
	/** K4: a lane only: the run sees the packages as live catalog entries (promote's patch, put back afterwards) */
	asLive?: boolean;
	/** pass names and presets (lib/passes.ts); default 'default' */
	passes?: string;
	/** axe scans: fail (default: serious/critical fail the gate), warn (report only), off */
	axe?: 'fail' | 'warn' | 'off';
	/** contact sheets and the lesson print (default on) */
	contact?: boolean;
	/** keep a dev server this run starts for the next run (default true) */
	keepServer?: boolean;
	/** redesign F: run on this build folder (a static server) instead of a dev server; AK606 measures loaded JS there */
	build?: string;
	/** a server and a browser shared by several packages (e2eMany, check) */
	server?: DevServer;
	browser?: Browser;
}

export interface E2eResult {
	slug: string;
	issues: Issue[];
	summary: string;
	shots: string[];
	steps: number;
	passes: string[];
	ms: number;
	contact: string | null;
	/** redesign F (AK606): gzip bytes of the JS the page loaded once the lesson settled, and after its last step; null on a
	 *  dev server (unbundled modules: not comparable) */
	loadedJs?: { settled: number; end: number; files: number } | null;
}

/** AK606: the loaded-JS budget of a lesson page once it settles (gzip bytes). A warning in wave F; a gate after M3. */
export const LOADED_JS_BUDGET = 400 * 1024;

/** AK606: the gzip size of the JS files at `urls` (a built site's /_app/ chunks only; null on a dev server). */
export async function jsWeight(urls: string[], cache: Map<string, number>): Promise<{ gz: number; files: number } | null> {
	const js = [...new Set(urls)].filter((u) => /\.m?js(\?|#|$)/.test(u));
	if (!js.some((u) => u.includes('/_app/immutable/'))) return null;
	let gz = 0;
	for (const u of js) {
		let n = cache.get(u);
		if (n === undefined) {
			try {
				const r = await fetch(u);
				n = r.ok ? gzipSync(Buffer.from(await r.arrayBuffer()), { level: 9 }).length : 0;
			} catch {
				n = 0;
			}
			cache.set(u, n);
		}
		gz += n;
	}
	return { gz, files: js.length };
}

/** AK605: the lesson app's layout mode for a window (redesign spec §5.1): app at 960 x 600 and up, else flow. */
export function expectedMode(width: number, height: number): 'app' | 'flow' {
	return width >= 960 && height >= 600 ? 'app' : 'flow';
}

/** Long tasks with their start, and windows of atlas's own work (axe, the 360 px probe) that do not count. */
const LONG_TASK_INIT = `(() => { window.__atlasLong = []; window.__atlasSkip = []; try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__atlasLong.push([Math.round(e.duration), e.startTime]); }).observe({ type: 'longtask', buffered: true }); } catch {} })();`;

type W = { __atlasLong: [number, number][]; __atlasSkip: [number, number][] };

async function pageUrl(slug: string, o: E2eOpts): Promise<string> {
	if (o.url) return o.url;
	const ex = join(roots().site, 'src', 'lib', 'explainers', slug, 'explainer.ts');
	if (!existsSync(ex)) return `/m/${slug}/`;
	const { loadPackage } = await import('../load.ts');
	const l = await loadPackage(slug);
	// K2: a concept package (type 'concept') is routed at /train/<slug>/
	return l.pkg.manifest.devOnly ? `/dev/atlas/lab/${slug}/` : (l.pkg.manifest.type as string) === 'concept' ? `/train/${slug}/` : `/m/${slug}/`;
}

function playwright(): typeof import('playwright-core') {
	const req = createRequire(join(roots().site, 'package.json'));
	try {
		return req('playwright-core') as typeof import('playwright-core');
	} catch {
		throw new AtlasError('AK001', 'playwright-core is not installed in the site', { fix: 'npm install (it is a devDependency); then npm run atlas -- doctor' });
	}
}

export async function launchBrowser(): Promise<Browser> {
	const pw = playwright();
	// Edge on this machine; ATLAS_BROWSER_CHANNEL=chromium uses Playwright's own Chromium (npx playwright-core install chromium)
	const channel = process.env.ATLAS_BROWSER_CHANNEL ?? 'msedge';
	// --expose-gc: after an axe scan e2e collects axe's garbage inside the scan's own window (K2 integration: without it
	// the next step's render paid for a major GC, e.g. vae-2d step 3 at 125-231 ms instead of 50 ms, a false AK603)
	return pw.chromium.launch({ ...(channel === 'chromium' ? {} : { channel }), headless: true, args: ['--js-flags=--expose-gc'] });
}

/** Run `fn` as atlas's own work: long tasks that start meanwhile (and `settle` ms after) are not the page's. */
async function own<T>(page: Page, fn: () => Promise<T>, settle = 0): Promise<T> {
	const a = await page.evaluate(() => performance.now());
	try {
		return await fn();
	} finally {
		if (settle) await page.waitForTimeout(settle);
		await page.evaluate((a0) => (window as unknown as W).__atlasSkip.push([a0 - 1, performance.now() + 20]), a).catch(() => {});
	}
}

/** K4: wait until the page stops scrolling (a deep link or the lesson start scrolls smoothly, and once more when the model
 *  is ready), at most 2 s, so a screenshot never catches it halfway. */
async function settleScroll(page: Page): Promise<void> {
	let last = -1;
	for (let k = 0; k < 20; k++) {
		const y = (await page.evaluate(() => window.scrollY).catch(() => 0)) as number;
		if (y === last) return;
		last = y;
		await page.waitForTimeout(100);
	}
}

/** Remove what an earlier run wrote (e2e-*.png and contact/), so the folder shows this run only. */
function clearOld(shotsDir: string, contactDir: string): void {
	if (existsSync(shotsDir)) for (const f of readdirSync(shotsDir)) if (/^e2e-.*\.png$/.test(f)) rmSync(join(shotsDir, f), { force: true });
	rmSync(contactDir, { recursive: true, force: true });
}

const stamp = () => new Date().toLocaleString('sv-SE').slice(0, 16);

/** One package through the given passes on a running server and browser. */
async function runPackage(slug: string, o: E2eOpts, server: DevServer, browser: Browser): Promise<E2eResult> {
	const r = roots();
	const t0 = performance.now();
	const is = new Issues();
	const path = await pageUrl(slug, o);
	const passes = resolvePasses(legacyPasses(o));
	const shotsDir = o.shots ?? defaultShotsDir(slug, r);
	const contactDir = join(shotsDir, 'contact');
	const contact = o.contact !== false;
	mkdirSync(shotsDir, { recursive: true });
	clearOld(shotsDir, contactDir);
	if (contact) mkdirSync(contactDir, { recursive: true });
	const axeMode = o.axe ?? 'fail';
	const Axe = axeMode === 'off' ? null : loadAxe(r.site);
	const shots: string[] = [];
	const scans: StepScan[] = [];
	const printed: { pass: Pass; steps: StepText[] }[] = [];
	let steps = 0;
	const wait = o.waitMs ?? 900;
	const jsCache = new Map<string, number>();
	let loadedJs: E2eResult['loadedJs'] = undefined;
	const jsUrls = (page: Page) => page.evaluate(() => performance.getEntriesByType('resource').map((e) => e.name)).catch(() => [] as string[]);
	for (const p of passes) {
		const tp = performance.now();
		const ctx = await browser.newContext({ viewport: { width: p.width, height: p.height }, colorScheme: p.scheme, reducedMotion: p.reduced ? 'reduce' : 'no-preference', deviceScaleFactor: 1, hasTouch: p.width < 700 });
		await ctx.addInitScript(LONG_TASK_INIT);
		const page = await ctx.newPage();
		const errors: string[] = [];
		page.on('console', (m) => {
			if (m.type() === 'error') errors.push(m.text());
		});
		page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
		const q = `${path.includes('?') ? '&' : '?'}atlas-e2e=1${o.cap !== undefined ? `&atlas-e2e-cap=${o.cap}` : ''}${p.lang === 'zh' ? '&lang=zh' : ''}`;
		try {
			const resp = await page.goto(`${server.base}${path}${q}`, { waitUntil: 'networkidle', timeout: 120_000 });
			if (!resp || resp.status() >= 400) {
				is.error('AK601', `${p.name}: ${path} answered ${resp?.status() ?? 'nothing'}`, { fix: 'is the page routed (devOnly packages live under /dev/atlas/lab/<slug>/)? pass --url' });
				continue;
			}
			const wt = page.locator('section.wt').first();
			try {
				await wt.waitFor({ timeout: 60_000 });
			} catch {
				is.error('AK601', `${p.name}: no guided lesson (section.wt) on ${path}`);
				continue;
			}
			// long tasks are measured while the learner steps through the lesson, not while the page loads (a dev
			// server serves hundreds of unbundled modules; page-load cost is the build's check:build shell budget)
			await page.waitForTimeout(1000);
			// K1: the page fetches the KaTeX fonts while idle; wait for them (at most 10 s), so their font swap is not counted
			await page.evaluate(() => Promise.race([(document as Document & { fonts?: { ready: Promise<unknown> } }).fonts?.ready, new Promise((res) => setTimeout(res, 10_000))]));
			// K4: in the shots the lesson's control bar stays at the top of the lesson (not sticky): an element screenshot of a
			// lesson taller than the window drew the sticky bar over the step-1 title or narration
			if (p.shots) await page.addStyleTag({ content: 'header.hdr{position:static!important} section.wt .bar{position:relative!important;top:auto!important}' });
			// K4: a Vite error overlay (a compile error, possibly stale on a kept server) covers every step
			if (await page.locator('vite-error-overlay').count())
				is.error('AK601', `${p.name}: the dev server shows a Vite error overlay`, { fix: 'fix the compile error it names; if the code is fine now, the kept server is stale: npm run atlas -- e2e --stop-server, then run e2e again' });
			await page.evaluate(() => {
				(window as unknown as W).__atlasLong = [];
				(window as unknown as W).__atlasSkip = [];
			});
			const dots = wt.locator('.dots .dot');
			const n = await dots.count();
			steps = Math.max(steps, n);
			// AK606: the JS the page has loaded once the lesson settled (the first pass that is not reduced)
			const settledJs = loadedJs === undefined && !p.reduced ? await own(page, async () => jsWeight(await jsUrls(page), jsCache)) : undefined;
			if (p.reduced) {
				const before = await wt.locator('.pos').first().textContent();
				await page.waitForTimeout(1500);
				const after = await wt.locator('.pos').first().textContent();
				const pressed = await wt.locator('.btn.play[aria-pressed="true"]').count();
				if (before !== after || pressed) is.error('AK601', `reduced motion: the lesson ${pressed ? 'is playing' : 'moved'} on its own (${before} -> ${after})`, { fix: 'never autoplay when motion.reduced is true' });
			}
			const texts: StepText[] = [];
			const tiles: Tile[] = [];
			for (let i = 0; i < n; i++) {
				// the step dots are hidden on narrow layouts: click them through the DOM (the same handler a tap runs)
				try {
					await dots.nth(i).evaluate((b) => (b as HTMLButtonElement).click());
				} catch (e) {
					is.error('AK601', `${p.name}: step ${i + 1} cannot be opened: ${(e as Error).message.split('\n')[0]}`);
					continue;
				}
				await page.waitForTimeout(p.reduced ? 250 : wait);
				// wait while the stage says it is computing (StageHost's .busy status: "Computing…", "Training the network first…")
				for (let k = 0; k < 120 && (await wt.locator('.stage .busy').count()) > 0; k++) await page.waitForTimeout(500);
				const t = (await wt.evaluate((el) => {
					const txt = (s: string) => (el.querySelector(s)?.textContent ?? '').replace(/\s+/g, ' ').trim();
					const stage = el.querySelector('.stage') as HTMLElement | null;
					return { title: txt('.narr :is(h1,h2,h3)'), say: txt('.say'), look: txt('.look'), badge: txt('.stage .badge'), stage: (stage?.innerText ?? '').replace(/\s+/g, ' ').trim() };
				})) as Omit<StepText, 'step' | 'steps'>;
				texts.push({ step: i + 1, steps: n, ...t });
				const where = `${p.name}, step ${i + 1}/${n}`;
				const say = t.say;
				if (!say) is.error('AK401', `${where}: the narration is empty`);
				// K4: only a gap mark counts (a '…' next to no letter or digit); a literal range or word ('−1…1', '是…的首都') is text
				if (gapMarks(say)) is.error('AK401', `${where}: the narration shows "…" (an unresolved placeholder): ${say.slice(0, 80)}`);
				if (p.lang === 'zh' && !/[一-鿿]/.test(say)) is.error('AK406', `${where}: no Chinese in the narration (English fallback): ${say.slice(0, 60)}`);
				if (p.badge && !(await wt.locator('.stage .badge').count())) is.error('AK502', `${where}: the stage shows no provenance badge`);
				const scroll = async (label: string) => {
					const sw = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth }));
					if (sw.sw > sw.cw + 1) is.error('AK602', `${where}${label}: horizontal scroll (${sw.sw} px of content in ${sw.cw} px)`, { fix: 'grid-template-columns: minmax(0, 1fr); let wide views scroll inside their own box' });
				};
				if (p.scroll) await scroll('');
				// AK605: the lesson app's layout mode (a page without data-lesson-mode is not checked)
				const lm = (await page.evaluate(() => {
					const el = document.querySelector('[data-lesson-mode]');
					return el ? { mode: el.getAttribute('data-lesson-mode'), sh: document.documentElement.scrollHeight, w: document.documentElement.clientWidth, h: innerHeight } : null;
				})) as { mode: string | null; sh: number; w: number; h: number } | null;
				if (lm) {
					const want = expectedMode(lm.w, lm.h);
					if (lm.mode !== want) is.error('AK605', `${where}: the lesson is in ${lm.mode} mode in a ${lm.w} x ${lm.h} window (want ${want}: app at 960 x 600 and up, flow below)`, { fix: 'the lesson app picks its mode from the window size (redesign spec §5.1)' });
					else if (want === 'app' && lm.sh > lm.h + 1) is.error('AK605', `${where}: app mode, but the page is ${lm.sh} px tall in a ${lm.h} px window`, { fix: 'in app mode the stage and narration panes scroll on their own (overflow: auto); the page never scrolls' });
				}
				if (p.probe360)
					await own(
						page,
						async () => {
							await page.setViewportSize({ width: 360, height: p.height });
							await page.waitForTimeout(250);
							await scroll(' at 360 px');
							await page.setViewportSize({ width: p.width, height: p.height });
						},
						250
					);
				if (p.shots) {
					await settleScroll(page);
					const f = join(shotsDir, `e2e-${p.name}-${String(i + 1).padStart(2, '0')}.png`);
					await wt.screenshot({ path: f });
					shots.push(posix(f));
					tiles.push({ label: `${i + 1}/${n}  ${t.title}`, file: f });
				}
				if (Axe) {
					const scope = i === 0 ? 'page' : 'lesson';
					try {
						const res = await own(page, async () => {
							const r = await scan(Axe, page, scope === 'page' ? undefined : 'section.wt');
							// collect axe's garbage now, inside the scan's window, not during the next step (AK603)
							await page.evaluate(() => (window as unknown as { gc?: () => void }).gc?.()).catch(() => {});
							return r;
						});
						scans.push({ pass: p.name, step: i + 1, steps: n, title: t.title, scope, ms: res.ms, violations: res.violations });
					} catch (e) {
						scans.push({ pass: p.name, step: i + 1, steps: n, title: t.title, scope, ms: 0, violations: [], error: (e as Error).message.split('\n')[0].slice(0, 200) });
					}
				}
			}
			const long = (await page.evaluate(() => {
				const w = window as unknown as W;
				return (w.__atlasLong ?? []).filter(([, t]) => !(w.__atlasSkip ?? []).some(([a, b]) => t >= a && t <= b)).map(([d]) => d);
			})) as number[];
			if (settledJs !== undefined) {
				const endJs = settledJs ? await own(page, async () => jsWeight(await jsUrls(page), jsCache)) : null;
				loadedJs = settledJs && endJs ? { settled: settledJs.gz, end: endJs.gz, files: endJs.files } : null;
				if (loadedJs && loadedJs.settled > LOADED_JS_BUDGET)
					is.warn('AK606', `${p.name}: the lesson page loaded ${(loadedJs.settled / 1024).toFixed(1)} KB of JS (gzip) once it settled, over the ${LOADED_JS_BUDGET / 1024} KB budget`, { fix: 'load views, panels and model code when they are shown (dynamic import), not with the lesson' });
			}
			const worst = long.length ? Math.max(...long) : 0;
			// lean gates (2026-10-13): AK603 never fails a check and is never re-run for (the bar is load-sensitive)
			if (worst > 100) is.warn('AK603', `${p.name}: a main-thread long task of ${worst} ms (> 100 ms; a warning, never a gate)`, { fix: 'move work to the package worker, or yield every <= 8 ms' });
			else if (worst > 50) is.warn('AK603', `${p.name}: a main-thread long task of ${worst} ms (> 50 ms; the engine's bar)`);
			for (const e of [...new Set(errors)].slice(0, 6)) is.error('AK601', `${p.name}: console error: ${e.slice(0, 200)}`);
			printed.push({ pass: p, steps: texts });
			if (contact && tiles.length) await renderSheet(browser, `${slug} · ${p.name} (${p.width} px, ${p.lang}, ${p.scheme}) · ${n} steps · ${stamp()}`, p, tiles, join(contactDir, `${p.name}.png`));
			if (o.echo) console.log(`  ${slug} ${p.name}: ${n} steps in ${((performance.now() - tp) / 1000).toFixed(0)} s`);
		} finally {
			await ctx.close();
		}
	}
	let axeLine = 'axe off';
	if (Axe) {
		const before = is.list.length;
		const a = axeIssues(scans, is);
		if (axeMode === 'warn') for (const i of is.list.slice(before)) if (i.code === AK_A11Y) i.level = 'warn';
		axeLine = `axe ${scans.length} scans: ${a.blocking} blocking rule(s)${axeMode === 'warn' ? ' (reported as warnings: --axe warn)' : ''}, ${a.warnings} warning rule(s)`;
	}
	if (contact) {
		writeText(join(contactDir, 'lesson-print.txt'), lessonPrint({ slug, path, date: stamp(), passes: printed }));
		if (scans.length) {
			writeText(join(contactDir, 'axe.txt'), axeText(slug, scans));
			for (const p of passes) {
				const ps = scans.filter((s) => s.pass === p.name);
				if (ps.length) writeJson(join(contactDir, `axe-${p.name}.json`), { slug, pass: p.name, tags: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'], steps: ps }, 1);
			}
		}
	}
	const errs = is.errors.length;
	const ms = performance.now() - t0;
	const on = server.how === 'given' ? `server ${server.base}` : `${o.build ? 'build' : 'dev'} server on port ${server.port} (${server.how}: ${server.why})`;
	const where = posix(relative(r.root, shotsDir));
	const jsLine = loadedJs ? `loaded JS ${(loadedJs.settled / 1024).toFixed(1)} KB gzip once settled, ${(loadedJs.end / 1024).toFixed(1)} KB after the last step (${loadedJs.files} files; AK606 budget ${LOADED_JS_BUDGET / 1024} KB)` : 'loaded JS not measured (a dev server; --build <dir> measures it)';
	return {
		slug,
		issues: is.list,
		summary: `${passes.length} passes (${passes.map((p) => p.name).join(', ')}) x ${steps} steps on ${path}, ${on}: ${errs} error(s); ${axeLine}; ${jsLine}; ${shots.length} screenshots in ${where}/${contact ? `, contact sheets in ${where}/contact/` : ''}; ${(ms / 1000).toFixed(0)} s`,
		shots,
		steps,
		passes: passes.map((p) => p.name),
		ms,
		contact: contact ? posix(contactDir) : null,
		loadedJs: loadedJs ?? null
	};
}

/** K4: where --as-live keeps the lane's own catalog files during a run. */
const asLiveDir = () => join(roots().site, '.atlas', 'as-live');

/** K4: put back the catalog files an --as-live run replaced (also after a run that was killed). */
export function restoreCatalog(): string[] {
	const dir = asLiveDir();
	if (!existsSync(dir)) return [];
	const back: string[] = [];
	for (const rel of walkFiles(dir)) {
		copyFileSync(join(dir, rel), join(roots().site, rel));
		back.push(rel);
	}
	rmSync(dir, { recursive: true, force: true });
	return back;
}

function walkFiles(dir: string, rel = ''): string[] {
	const out: string[] = [];
	for (const d of readdirSync(join(dir, rel), { withFileTypes: true })) {
		const r2 = rel ? `${rel}/${d.name}` : d.name;
		if (d.isDirectory()) out.push(...walkFiles(dir, r2));
		else out.push(r2);
	}
	return out;
}

/** K4: write promote's patch for each package into the lane's catalog files, keeping the originals in .atlas/as-live/. */
export async function liveCatalog(slugs: string[]): Promise<string[]> {
	const r = roots();
	if (!r.lane) throw new AtlasError('AK301', 'e2e --as-live runs in a lane (in the main tree, promote --apply makes the entry live)');
	restoreCatalog();
	const { planPromote, patchedFiles } = await import('./promote.ts');
	const dir = asLiveDir();
	const changed = new Set<string>();
	for (const slug of slugs) {
		const files = patchedFiles(await planPromote(slug), r.site);
		for (const [rel, text] of Object.entries(files)) {
			const keep = join(dir, rel);
			if (!existsSync(keep)) {
				mkdirSync(join(keep, '..'), { recursive: true });
				copyFileSync(join(r.site, rel), keep);
			}
			writeBytes(join(r.site, rel), Buffer.from(text, 'utf8'));
			changed.add(rel);
		}
	}
	return [...changed];
}

/** Several packages as one sequence: one dev server and one browser (`o.server` / `o.browser` when given). */
export async function e2eMany(slugs: string[], o: E2eOpts = {}): Promise<E2eResult[]> {
	if (!slugs.length) return [];
	resolvePasses(legacyPasses(o)); // a bad --passes fails before a server starts
	if (o.axe !== 'off') loadAxe(roots().site);
	// K4: catalog files left patched by a killed --as-live run go back first
	const restored = restoreCatalog();
	if (restored.length && o.echo !== false) console.log(`  e2e: put back ${restored.join(', ')} (left patched by an --as-live run that did not finish)`);
	if (o.asLive) {
		const live = await liveCatalog(slugs);
		console.log(`  e2e --as-live: ${slugs.join(', ')} live in ${live.join(', ')} for this run`);
	}
	try {
		return await e2eRun(slugs, o);
	} finally {
		if (o.asLive) restoreCatalog();
	}
}

async function e2eRun(slugs: string[], o: E2eOpts): Promise<E2eResult[]> {
	const server = o.server ?? (o.build ? await staticServer(resolve(o.build), o.port) : await devServer({ slug: slugs[0], port: o.port, base: o.base, keep: o.keepServer ?? true }));
	let browser: Browser | null = null;
	try {
		browser = o.browser ?? (await launchBrowser());
		const out: E2eResult[] = [];
		for (const slug of slugs) out.push(await runPackage(slug, o, server, browser));
		return out;
	} finally {
		if (browser && !o.browser) await browser.close();
		if (!o.server) server.stop();
	}
}

export async function e2e(slug: string, o: E2eOpts = {}): Promise<E2eResult> {
	return (await e2eMany([slug], o))[0];
}

export function parseE2eArgs(args: string[]): { o: E2eOpts; slugs: string[]; stop: boolean } {
	const o: E2eOpts = {};
	const pos: string[] = [];
	let stop = false;
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--full') o.full = true;
		else if (a === '--url') o.url = args[++i];
		else if (a === '--port') {
			o.port = Number(args[++i]);
			if (!Number.isInteger(o.port) || o.port < 1024 || o.port > 65535) throw new AtlasError('AK301', '--port needs a port number (1024-65535)');
		} else if (a === '--base') o.base = args[++i];
		else if (a === '--wait') o.waitMs = Number(args[++i]);
		else if (a === '--shots') o.shots = args[++i];
		else if (a === '--passes') o.passes = args[++i] ?? '';
		else if (a === '--keep-server') o.keepServer = true;
		else if (a === '--no-keep-server') o.keepServer = false;
		else if (a === '--stop-server') stop = true;
		else if (a === '--no-contact') o.contact = false;
		else if (a === '--as-live') o.asLive = true;
		else if (a === '--build') {
			o.build = args[++i];
			if (!o.build || o.build.startsWith('--')) throw new AtlasError('AK301', '--build needs a build folder (ATLAS_BUILD_DIR of a build)');
		}
		else if (a === '--verbose') o.echo = true;
		else if (a === '--axe') {
			const v = args[++i];
			if (v !== 'fail' && v !== 'warn' && v !== 'off') throw new AtlasError('AK301', '--axe is fail, warn or off');
			o.axe = v;
		} else if (a === '--phone') {
			o.phone = Number(args[++i]);
			if (!Number.isInteger(o.phone) || o.phone < 280 || o.phone > 700) throw new AtlasError('AK301', '--phone needs a phone width in px (280-700), e.g. --phone 390');
		} else if (a === '--cap') {
			const v = args[++i];
			o.cap = v === 'off' ? 0 : Number(v);
			if (!Number.isInteger(o.cap) || o.cap < 0) throw new AtlasError('AK301', '--cap needs a step count or off');
		} else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`, { fix: USAGE });
		else pos.push(...a.split(',').map((s) => s.trim()).filter(Boolean));
	}
	if (o.passes !== undefined) resolvePasses(legacyPasses(o));
	if (o.build && o.base) throw new AtlasError('AK301', '--build and --base are two different servers: pick one');
	return { o, slugs: pos, stop };
}

export const USAGE = 'atlas e2e <slug>[,<slug>...] [--passes default|all|full|layout|<width>[x<height>]-<lang>-<scheme>[-reduced],...] [--full] [--phone px] [--url <path>] [--port n] [--base <url> | --build <dir>] [--keep-server|--no-keep-server] [--stop-server] [--axe fail|warn|off] [--no-contact] [--wait ms] [--shots <dir>] [--cap n|off] [--as-live]';

export async function main(args: string[]): Promise<number> {
	const { o, slugs, stop } = parseE2eArgs(args);
	if (stop && !slugs.length) {
		const s = await stopKept();
		console.log(`atlas e2e --stop-server: ${s.note}`);
		return 0;
	}
	if (!slugs.length) throw new AtlasError('AK301', `usage: ${USAGE}`);
	if (o.url && slugs.length > 1) throw new AtlasError('AK301', '--url takes one package');
	const res = await e2eMany(slugs, o);
	let errors = 0;
	for (const x of res) {
		const bad = x.issues.some((i) => i.level === 'error');
		console.log(`atlas e2e ${x.slug}: ${bad ? 'FAIL' : 'PASS'}; ${x.summary}`);
		const is = new Issues();
		is.list = x.issues;
		if (is.list.length) console.error(is.format());
		errors += is.errors.length;
	}
	if (stop) console.log(`  ${(await stopKept()).note}`);
	else {
		const rec = readRecord();
		if (rec && !o.base) console.log(`  dev server kept on http://localhost:${rec.port}/ (pid ${rec.pid}; it stops itself after ${rec.idleMin} idle minutes; stop it now: npm run atlas -- e2e --stop-server)`);
	}
	return errors ? 1 : 0;
}
