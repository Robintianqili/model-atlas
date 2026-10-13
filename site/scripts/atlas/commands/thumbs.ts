// atlas thumbs [<slug>...] [--build <dir> | --base <url>] [--port n] [--out <dir>] [--sheets <dir>] [--cols n]
//   (redesign wave F, spec §10.5): the catalog's lesson thumbnails, from each live lesson's poster stage.
//
// Each live lesson (catalog models and concepts with status 'live') opens in headless Edge at 960 x 600 (light, then dark;
// `?atlas-e2e=1` caps live training). The poster step is manifest.page.poster.step (opened by its deep link
// #lesson-<id>), else the first step whose stage shows a figure (a canvas or an svg of at least 160 x 100 px); the tile
// is that stage's largest figure element, so no UI chrome is in it. A lesson whose views are all HTML (text, tables,
// token rows) gets the top of its first view instead (the view's own box, no controls), until it names a poster step. The tiles are composed in the browser (canvas
// toBlob('image/webp'), no new dependency) into 4 sprites and an index:
//   <out>/rows.light.webp, rows.dark.webp     224 x 140 tiles, every live lesson (budget 250 KB per theme)
//   <out>/cards.light.webp, cards.dark.webp   480 x 300 tiles, the home page's curated set (budget 120 KB per theme)
//   <out>/index.json                          { format, made, rows: {...}, cards: {...}, tiles: { slug: { row, card?, step } } }
// 4 files instead of 134 (each file takes at least 1 MiB on the exFAT drive). <out> defaults to static/thumbs/. A labelled
// contact sheet per theme (sheet.light.png, sheet.dark.png) goes to --sheets <dir> (default .atlas/thumbs/) for the
// review: no UI chrome in a tile, every tile a real stage. Only these files are written; the tiles stay in memory.
// The site: --build <dir> (a static server on a built site; faster), --base <url>, else a dev server like e2e's.
import type { Browser, Page } from 'playwright-core';
import { existsSync, mkdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { AtlasError } from '../errors.ts';
import { devServer, type DevServer } from '../lib/devserver.ts';
import { posix, writeBytes, writeJson } from '../lib/io.ts';
import { staticServer } from '../lib/staticserver.ts';
import { roots } from '../root.ts';
import { launchBrowser } from './e2e.ts';

export const ROW_TILE = { w: 224, h: 140 } as const;
export const CARD_TILE = { w: 480, h: 300 } as const;
/** the home page's Start here set (spec §6.3: four lessons, four kinds of input) */
export const CARD_SLUGS = ['tinycnn', 'tinygpt', 'loss-gradient-descent', 'mlp-playground'];
export const BUDGET = { rows: 250 * 1024, cards: 120 * 1024 };
/** the smallest figure that makes a poster: a canvas or svg at least this big (CSS px) */
export const MIN_FIGURE = { w: 160, h: 100 };
/** the stage backgrounds of the two themes (tokens.css --stage), behind each tile */
export const STAGE_BG = { light: '#eef0ec', dark: '#0b0d0f' } as const;
export const VIEWPORT = { width: 960, height: 600 };

export interface Lesson {
	slug: string;
	kind: 'model' | 'concept';
	path: string;
}

/** The live lessons of the catalog, models first (catalog order). */
export function liveLessons(models: { slug: string; status: string }[], concepts: { slug: string; status: string }[]): Lesson[] {
	return [
		...models.filter((m) => m.status === 'live').map((m) => ({ slug: m.slug, kind: 'model' as const, path: `/m/${m.slug}/` })),
		...concepts.filter((c) => c.status === 'live').map((c) => ({ slug: c.slug, kind: 'concept' as const, path: `/train/${c.slug}/` }))
	];
}

/** Where tile i sits in a sprite of `cols` columns. */
export function tileAt(i: number, cols: number, tile: { w: number; h: number }): { x: number; y: number } {
	return { x: (i % cols) * tile.w, y: Math.floor(i / cols) * tile.h };
}

/** The sprite size for n tiles in `cols` columns. */
export function spriteSize(n: number, cols: number, tile: { w: number; h: number }): { w: number; h: number; rows: number } {
	const c = Math.max(1, Math.min(cols, n));
	const rows = Math.max(1, Math.ceil(n / c));
	return { w: c * tile.w, h: rows * tile.h, rows };
}

/** The poster among the steps tried: the first with a figure of MIN_FIGURE or more, else the one with the largest. */
export function pickPoster(tried: { step: number; w: number; h: number }[]): number | null {
	const ok = tried.find((t) => t.w >= MIN_FIGURE.w && t.h >= MIN_FIGURE.h);
	if (ok) return ok.step;
	const best = [...tried].filter((t) => t.w * t.h > 0).sort((a, b) => b.w * b.h - a.w * a.h)[0];
	return best ? best.step : null;
}

export interface ThumbIndex {
	format: 'atlas-thumbs/1';
	made: string;
	rows: { tile: { w: number; h: number }; cols: number; light: string; dark: string; bytes: { light: number; dark: number } };
	cards: { tile: { w: number; h: number }; cols: number; light: string; dark: string; bytes: { light: number; dark: number } };
	/** slug -> its row tile index, its card tile index (curated set only) and the step it shows (1-based, or the step id) */
	tiles: Record<string, { row: number; card?: number; step: number | string }>;
	missing: string[];
}

/** The index file (pure: what the sprites hold). */
export function thumbIndex(rows: { slug: string; step: number | string }[], cards: string[], cols: { rows: number; cards: number }, bytes: { rows: { light: number; dark: number }; cards: { light: number; dark: number } }, missing: string[], made: string): ThumbIndex {
	const tiles: ThumbIndex['tiles'] = {};
	rows.forEach((r, i) => (tiles[r.slug] = { row: i, step: r.step }));
	cards.forEach((s, j) => {
		if (tiles[s]) tiles[s].card = j;
	});
	return {
		format: 'atlas-thumbs/1',
		made,
		rows: { tile: { ...ROW_TILE }, cols: cols.rows, light: 'rows.light.webp', dark: 'rows.dark.webp', bytes: bytes.rows },
		cards: { tile: { ...CARD_TILE }, cols: cols.cards, light: 'cards.light.webp', dark: 'cards.dark.webp', bytes: bytes.cards },
		tiles,
		missing
	};
}

// ------------------------------------------------------------------ the browser side

type Shot = { light: Buffer; dark: Buffer; step: number | string };

async function waitStage(page: Page): Promise<void> {
	const wt = page.locator('section.wt').first();
	for (let k = 0; k < 120 && (await wt.locator('.stage .busy').count()) > 0; k++) await page.waitForTimeout(500);
	await page.waitForTimeout(500);
}

/** The largest visible canvas or svg in the lesson stage: its box (page coordinates), or null. */
async function largestFigure(page: Page): Promise<{ x: number; y: number; w: number; h: number } | null> {
	return page.evaluate(() => {
		const stage = document.querySelector('section.wt .stage');
		if (!stage) return null;
		let best: { x: number; y: number; w: number; h: number } | null = null;
		for (const el of stage.querySelectorAll('canvas, svg')) {
			if (el.closest('button, .badge, [role="tooltip"]')) continue;
			if (el.parentElement?.closest('svg')) continue; // an svg inside an svg
			const r = el.getBoundingClientRect();
			const cs = getComputedStyle(el);
			if (cs.visibility === 'hidden' || cs.display === 'none' || r.width < 40 || r.height < 40) continue;
			if (!best || r.width * r.height > best.w * best.h) best = { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height };
		}
		return best;
	});
}

/** The box of the step's view (the root element of the view in the stage's slot), its height capped at 0.7 x its width
 *  (the top of a tall view), or null. For lessons whose views are HTML, not canvas or svg. */
async function viewBox(page: Page): Promise<{ x: number; y: number; w: number; h: number } | null> {
	return page.evaluate(() => {
		const el = [...document.querySelectorAll('section.wt .stage .slot > *')].find((e) => !e.matches('.miss, details.as-text'));
		if (!el) return null;
		const r = el.getBoundingClientRect();
		if (r.width < 40 || r.height < 40) return null;
		return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: Math.min(r.height, r.width * 0.7) };
	});
}

async function figureShot(page: Page, view = false): Promise<Buffer | null> {
	const find = view ? viewBox : largestFigure;
	const f = await find(page);
	if (!f) return null;
	// bring it into view, then shoot its box with a little margin (still inside the stage: no chrome)
	await page.evaluate((y) => scrollTo(0, Math.max(0, y - 40)), f.y);
	await page.waitForTimeout(150);
	const g = await find(page);
	if (!g) return null;
	const sy = await page.evaluate(() => scrollY);
	const pad = 4;
	return page.screenshot({ clip: { x: Math.max(0, g.x - pad), y: Math.max(0, g.y - sy - pad), width: g.w + 2 * pad, height: g.h + 2 * pad } });
}

async function shootLesson(browser: Browser, base: string, l: Lesson, poster: string | null): Promise<Shot | null> {
	const ctx = await browser.newContext({ viewport: VIEWPORT, colorScheme: 'light', deviceScaleFactor: 2, reducedMotion: 'reduce' });
	const page = await ctx.newPage();
	try {
		const url = `${base}${l.path}?atlas-e2e=1${poster ? `#lesson-${poster}` : ''}`;
		await page.goto(url, { waitUntil: 'networkidle', timeout: 120_000 });
		const wt = page.locator('section.wt').first();
		await wt.waitFor({ timeout: 60_000 });
		await page.addStyleTag({ content: 'header.hdr{position:static!important} section.wt .bar{position:relative!important;top:auto!important}' });
		let step: number | string;
		let view = false;
		if (poster) {
			await waitStage(page);
			step = poster;
			view = !(await largestFigure(page));
		} else {
			const dots = wt.locator('.dots .dot');
			const n = await dots.count();
			const tried: { step: number; w: number; h: number }[] = [];
			for (let i = 0; i < Math.min(n, 8); i++) {
				await dots.nth(i).evaluate((b) => (b as HTMLButtonElement).click());
				await waitStage(page);
				const f = await largestFigure(page);
				tried.push({ step: i, w: f?.w ?? 0, h: f?.h ?? 0 });
				if (f && f.w >= MIN_FIGURE.w && f.h >= MIN_FIGURE.h) break;
			}
			let pick = pickPoster(tried);
			if (pick === null) {
				// no canvas or svg in the first steps: the first view big enough to show
				view = true;
				for (const t of tried) {
					await dots.nth(t.step).evaluate((b) => (b as HTMLButtonElement).click());
					await waitStage(page);
					const v = await viewBox(page);
					if (v && v.w >= MIN_FIGURE.w && v.h >= MIN_FIGURE.h) {
						pick = t.step;
						break;
					}
				}
				if (pick === null) return null;
			} else if (pick !== tried[tried.length - 1].step) {
				await dots.nth(pick).evaluate((b) => (b as HTMLButtonElement).click());
				await waitStage(page);
			}
			step = pick + 1;
		}
		const light = await figureShot(page, view);
		await page.emulateMedia({ colorScheme: 'dark' });
		await page.waitForTimeout(700);
		const dark = await figureShot(page, view);
		return light && dark ? { light, dark, step } : null;
	} finally {
		await ctx.close();
	}
}

/** Compose tiles into a sprite in the browser: each image contain-fitted on the stage colour; webp at `quality`. */
async function compose(browser: Browser, imgs: Buffer[], cols: number, tile: { w: number; h: number }, bg: string, quality: number): Promise<Buffer> {
	const page = await browser.newPage();
	try {
		const { w, h } = spriteSize(imgs.length, cols, tile);
		const urls = imgs.map((b) => `data:image/png;base64,${b.toString('base64')}`);
		const b64 = await page.evaluate(
			async ({ urls, cols, tile, w, h, bg, quality }) => {
				const c = document.createElement('canvas');
				c.width = w;
				c.height = h;
				const g = c.getContext('2d')!;
				g.fillStyle = bg;
				g.fillRect(0, 0, w, h);
				g.imageSmoothingQuality = 'high';
				const pad = Math.round(tile.w * 0.04);
				for (let i = 0; i < urls.length; i++) {
					const im = new Image();
					im.src = urls[i];
					await im.decode();
					const x0 = (i % cols) * tile.w;
					const y0 = Math.floor(i / cols) * tile.h;
					const s = Math.min((tile.w - 2 * pad) / im.width, (tile.h - 2 * pad) / im.height);
					const dw = im.width * s;
					const dh = im.height * s;
					g.drawImage(im, x0 + (tile.w - dw) / 2, y0 + (tile.h - dh) / 2, dw, dh);
				}
				const blob: Blob = await new Promise((res) => c.toBlob((b) => res(b!), 'image/webp', quality));
				const buf = new Uint8Array(await blob.arrayBuffer());
				let s = '';
				for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
				return btoa(s);
			},
			{ urls, cols, tile, w, h, bg, quality }
		);
		return Buffer.from(b64, 'base64');
	} finally {
		await page.close();
	}
}

/** A sprite under its budget: webp quality 0.82, lowered in steps until it fits (or 0.5). */
async function composeWithin(browser: Browser, imgs: Buffer[], cols: number, tile: { w: number; h: number }, bg: string, budget: number): Promise<{ buf: Buffer; quality: number }> {
	let q = 0.82;
	for (;;) {
		const buf = await compose(browser, imgs, cols, tile, bg, q);
		if (buf.length <= budget || q <= 0.5) return { buf, quality: q };
		q = Math.round((q - 0.08) * 100) / 100;
	}
}

/** A labelled contact sheet of the tiles (for the review), as a PNG. */
async function sheet(browser: Browser, title: string, tiles: { label: string; img: Buffer }[], dark: boolean): Promise<Buffer> {
	const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, colorScheme: dark ? 'dark' : 'light' });
	try {
		const cells = tiles.map((t) => `<figure><img src="data:image/png;base64,${t.img.toString('base64')}"><figcaption>${t.label.replace(/[<&]/g, ' ')}</figcaption></figure>`).join('');
		await page.setContent(`<!doctype html><style>body{margin:16px;font:13px system-ui;background:${dark ? '#0f1113' : '#f4f5f2'};color:${dark ? '#eceee9' : '#16181b'}}h1{font-size:16px}main{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}figure{margin:0}img{width:100%;aspect-ratio:224/140;object-fit:contain;background:${dark ? STAGE_BG.dark : STAGE_BG.light};border:1px solid #8888}figcaption{margin-top:2px}</style><h1>${title}</h1><main>${cells}</main>`);
		await page.waitForTimeout(300);
		return await page.screenshot({ fullPage: true });
	} finally {
		await page.close();
	}
}

async function posterOf(site: string, l: Lesson): Promise<string | null> {
	const f = join(site, 'src', 'lib', 'explainers', l.slug, 'explainer.ts');
	if (!existsSync(f)) return null;
	try {
		const m = (await import(pathToFileURL(f).href)) as { default?: { page?: { poster?: { step?: string } } } };
		return m.default?.page?.poster?.step ?? null;
	} catch {
		return null;
	}
}

export interface ThumbsOpts {
	slugs?: string[];
	build?: string;
	base?: string;
	port?: number;
	out?: string;
	sheets?: string;
	cols?: number;
	echo?: boolean;
}

export async function thumbs(o: ThumbsOpts = {}): Promise<{ index: ThumbIndex; files: string[]; over: string[] }> {
	const r = roots();
	const { MODELS } = (await import(pathToFileURL(join(r.site, 'src', 'lib', 'catalog', 'models.ts')).href)) as { MODELS: { slug: string; status: string }[] };
	const { CONCEPTS } = (await import(pathToFileURL(join(r.site, 'src', 'lib', 'catalog', 'concepts.ts')).href)) as { CONCEPTS: { slug: string; status: string }[] };
	let lessons = liveLessons(MODELS, CONCEPTS);
	if (o.slugs?.length) {
		const unknown = o.slugs.filter((s) => !lessons.some((l) => l.slug === s));
		if (unknown.length) throw new AtlasError('AK701', `not a live lesson: ${unknown.join(', ')}`);
		lessons = lessons.filter((l) => o.slugs!.includes(l.slug));
	}
	const out = resolve(o.out ?? join(r.site, 'static', 'thumbs'));
	const sheets = resolve(o.sheets ?? join(r.site, '.atlas', 'thumbs'));
	const server: DevServer = o.build ? await staticServer(resolve(o.build), o.port) : await devServer({ slug: lessons[0]?.slug, port: o.port, base: o.base, keep: true });
	const browser = await launchBrowser();
	const shots: { lesson: Lesson; shot: Shot }[] = [];
	const missing: string[] = [];
	try {
		for (const l of lessons) {
			const t0 = performance.now();
			let shot: Shot | null = null;
			try {
				shot = await shootLesson(browser, server.base, l, await posterOf(r.site, l));
			} catch (e) {
				if (o.echo) console.log(`  ${l.slug}: ${(e as Error).message.split('\n')[0]}`);
			}
			if (shot) shots.push({ lesson: l, shot });
			else missing.push(l.slug);
			if (o.echo) console.log(`  ${l.slug.padEnd(28)} ${shot ? `step ${shot.step}` : 'NO FIGURE'} (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
		}
		const cols = { rows: o.cols ?? 8, cards: 2 };
		const cardShots = CARD_SLUGS.map((s) => shots.find((x) => x.lesson.slug === s)).filter((x): x is { lesson: Lesson; shot: Shot } => !!x);
		const made: Record<string, { buf: Buffer; quality: number }> = {};
		for (const theme of ['light', 'dark'] as const) {
			made[`rows.${theme}`] = await composeWithin(browser, shots.map((x) => x.shot[theme]), cols.rows, ROW_TILE, STAGE_BG[theme], BUDGET.rows);
			made[`cards.${theme}`] = await composeWithin(browser, cardShots.map((x) => x.shot[theme]), cols.cards, CARD_TILE, STAGE_BG[theme], BUDGET.cards);
		}
		mkdirSync(out, { recursive: true });
		const files: string[] = [];
		for (const [name, m] of Object.entries(made)) {
			const f = join(out, `${name}.webp`);
			writeBytes(f, m.buf);
			files.push(posix(f));
		}
		const bytes = (k: 'rows' | 'cards') => ({ light: made[`${k}.light`].buf.length, dark: made[`${k}.dark`].buf.length });
		const index = thumbIndex(shots.map((x) => ({ slug: x.lesson.slug, step: x.shot.step })), cardShots.map((x) => x.lesson.slug), cols, { rows: bytes('rows'), cards: bytes('cards') }, missing, new Date().toISOString().slice(0, 10));
		writeJson(join(out, 'index.json'), index, 1);
		files.push(posix(join(out, 'index.json')));
		mkdirSync(sheets, { recursive: true });
		for (const theme of ['light', 'dark'] as const) {
			const png = await sheet(browser, `atlas thumbs · ${theme} · ${shots.length} lessons · ${index.made}`, shots.map((x) => ({ label: `${x.lesson.slug} · step ${x.shot.step}`, img: x.shot[theme] })), theme === 'dark');
			const f = join(sheets, `sheet.${theme}.png`);
			writeBytes(f, png);
			files.push(posix(f));
		}
		const over = Object.entries(made)
			.filter(([k, m]) => m.buf.length > (k.startsWith('rows') ? BUDGET.rows : BUDGET.cards))
			.map(([k, m]) => `${k}.webp ${(m.buf.length / 1024).toFixed(1)} KB`);
		if (o.echo) for (const [k, m] of Object.entries(made)) console.log(`  ${k}.webp ${(m.buf.length / 1024).toFixed(1)} KB (webp quality ${m.quality})`);
		return { index, files, over };
	} finally {
		await browser.close();
		server.stop();
	}
}

export const USAGE = 'atlas thumbs [<slug>...] [--build <dir> | --base <url>] [--port n] [--out <dir>] [--sheets <dir>] [--cols n]';

export async function main(args: string[]): Promise<number> {
	const o: ThumbsOpts = { echo: true, slugs: [] };
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		const val = () => {
			const v = args[++i];
			if (!v || v.startsWith('--')) throw new AtlasError('AK301', `${a} needs a value`, { fix: USAGE });
			return v;
		};
		if (a === '--build') o.build = val();
		else if (a === '--base') o.base = val();
		else if (a === '--port') o.port = Number(val());
		else if (a === '--out') o.out = val();
		else if (a === '--sheets') o.sheets = val();
		else if (a === '--cols') o.cols = Number(val());
		else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`, { fix: USAGE });
		else o.slugs!.push(...a.split(',').filter(Boolean));
	}
	if (o.build && o.base) throw new AtlasError('AK301', '--build and --base are two different servers: pick one');
	if (o.build && !existsSync(join(o.build, 'index.html'))) throw new AtlasError('AK301', `--build ${o.build}: no built site there`);
	const res = await thumbs(o);
	console.log(`atlas thumbs: ${Object.keys(res.index.tiles).length} lessons in the row sprites, ${res.index.cards.cols ? Object.values(res.index.tiles).filter((t) => t.card !== undefined).length : 0} in the card sprites${res.index.missing.length ? `; no figure for ${res.index.missing.join(', ')}` : ''}`);
	for (const f of res.files) console.log(`  ${f}${existsSync(f) ? ` (${(statSync(f).size / 1024).toFixed(1)} KB)` : ''}`);
	if (res.over.length) console.error(`  over budget (rows 250 KB, cards 120 KB per theme): ${res.over.join(', ')}`);
	return res.over.length ? 1 : 0;
}
