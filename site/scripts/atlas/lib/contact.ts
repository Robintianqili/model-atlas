// Contact sheets and the lesson print (K2 speed pack): what a reviewer reads instead of walking the page by hand.
//
// `atlas e2e` writes into notes/shots/<slug>/contact/ (git-ignored):
//   <pass>.png        one image per pass: every lesson step's screenshot, tiled and labelled "n/N  title"
//   lesson-print.txt  every step as the page showed it, per language: title, narration, the look-at hint, the badge,
//                     the numbers in the narration and the numbers on the stage
//   axe.txt, axe-<pass>.json   the accessibility scans (lib/axe.ts)
// The sheet is drawn by the same headless browser (an HTML page of the step screenshots), so no image library is needed.
import type { Browser } from 'playwright-core';
import { readFileSync } from 'node:fs';
import type { Pass } from './passes.ts';

/** What one step showed in one pass (read from the lesson's DOM after the step settled). */
export interface StepText {
	step: number;
	steps: number;
	title: string;
	say: string;
	look: string;
	badge: string;
	/** the stage's visible text (whitespace collapsed) */
	stage: string;
}

/** Numbers as printed (0.731, -1.2e-3, 83.1%, 29,626, −0.5). */
export function numbersIn(text: string): string[] {
	// not the digits of a name (x1, h2, W_k1, σ2) or the tail of a longer token; Chinese may touch a number (第1步)
	return text.match(/(?<![A-Za-zͰ-Ͽ\d_.])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?/gi) ?? [];
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export interface Tile {
	label: string;
	file: string;
}

/** Layout of a sheet: columns, tile width and the tallest a tile may be (the sheet stays under ~15,000 px). */
export function sheetLayout(pass: Pick<Pass, 'width'>, n: number): { cols: number; tileW: number; maxTileH: number; width: number } {
	const phone = pass.width < 700;
	const cols = Math.max(1, Math.min(n, phone ? 6 : 3));
	const tileW = phone ? pass.width : 600;
	const rows = Math.ceil(n / cols);
	const cap = phone ? 1300 : 900;
	const maxTileH = Math.max(240, Math.min(cap, Math.floor((15_000 - 80) / Math.max(1, rows)) - 40));
	return { cols, tileW, maxTileH, width: cols * (tileW + 16) + 16 };
}

/** The sheet as an HTML page (images inlined as data URLs). */
export function sheetHtml(title: string, tiles: { label: string; src: string }[], l: ReturnType<typeof sheetLayout>): string {
	const cells = tiles
		.map((t) => `<figure><figcaption>${esc(t.label)}</figcaption><div class="img"><img src="${t.src}" alt=""></div></figure>`)
		.join('');
	return `<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;padding:16px;background:#e9e9ec;font:600 15px/1.3 system-ui,sans-serif;color:#1b1b1f;width:${l.width}px}
h1{font-size:18px;margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(${l.cols},${l.tileW}px);gap:16px;align-items:start}
figure{margin:0;background:#fff;border:1px solid #b9b9c2;border-radius:6px;overflow:hidden}
figcaption{padding:6px 8px;border-bottom:1px solid #d5d5dc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.img{max-height:${l.maxTileH}px;overflow:hidden}img{display:block;width:100%;height:auto}
</style></head><body><h1>${esc(title)}</h1><div class="grid">${cells}</div></body></html>`;
}

/** Draw one pass's sheet from its step screenshots into `out` (PNG). */
export async function renderSheet(browser: Browser, title: string, pass: Pick<Pass, 'width'>, tiles: Tile[], out: string): Promise<void> {
	if (!tiles.length) return;
	const l = sheetLayout(pass, tiles.length);
	const html = sheetHtml(
		title,
		tiles.map((t) => ({ label: t.label, src: `data:image/png;base64,${readFileSync(t.file).toString('base64')}` })),
		l
	);
	const ctx = await browser.newContext({ viewport: { width: l.width, height: 900 }, deviceScaleFactor: 1, colorScheme: 'light' });
	try {
		const page = await ctx.newPage();
		await page.setContent(html, { waitUntil: 'load' });
		await page.screenshot({ path: out, fullPage: true, type: 'png' });
	} finally {
		await ctx.close();
	}
}

const MAX_STAGE_NUMBERS = 40;

function numbersLine(t: StepText): string {
	const said = numbersIn(`${t.say} ${t.look}`);
	const shown = numbersIn(t.stage);
	const stage = shown.length > MAX_STAGE_NUMBERS ? `${shown.slice(0, MAX_STAGE_NUMBERS).join(' ')} … (${shown.length} in all)` : shown.join(' ');
	return `narration [${said.join(', ') || 'none'}]; stage [${stage || 'none'}]`;
}

/** lesson-print.txt: per step, each language as its first pass showed it; other passes of a language are compared. */
export function lessonPrint(o: { slug: string; path: string; date: string; passes: { pass: Pass; steps: StepText[] }[] }): string {
	const out: string[] = [
		`Lesson print: ${o.slug} (${o.path}), atlas e2e ${o.date}`,
		'What the page showed on every step, per language: title, narration, look-at hint, badge, the numbers in the narration and on the stage.',
		`Passes: ${o.passes.map((p) => `${p.pass.name} (${p.pass.lang})`).join(', ')}. Sheets: contact/<pass>.png; accessibility: contact/axe.txt.`,
		''
	];
	const langs = [...new Set(o.passes.map((p) => p.pass.lang))];
	const n = Math.max(0, ...o.passes.map((p) => p.steps.length));
	for (let i = 0; i < n; i++) {
		const any = o.passes.map((p) => p.steps[i]).find(Boolean);
		out.push(`Step ${i + 1}/${any?.steps ?? n}`);
		for (const lang of langs) {
			const ps = o.passes.filter((p) => p.pass.lang === lang && p.steps[i]);
			if (!ps.length) continue;
			const t = ps[0].steps[i];
			out.push(`  ${lang}  title      ${t.title}`);
			out.push(`      narration  ${t.say}`);
			if (t.look) out.push(`      look       ${t.look}`);
			if (t.badge) out.push(`      badge      ${t.badge}`);
			out.push(`      numbers    ${numbersLine(t)}`);
			for (const other of ps.slice(1)) {
				const u = other.steps[i];
				if (u.say !== t.say) out.push(`      ${other.pass.name} narration differs: ${u.say}`);
				const a = numbersIn(t.stage).join(' ');
				const b = numbersIn(u.stage).join(' ');
				if (a !== b) out.push(`      ${other.pass.name} stage numbers differ: [${numbersIn(u.stage).slice(0, MAX_STAGE_NUMBERS).join(' ')}]`);
			}
		}
		out.push('');
	}
	return out.join('\n');
}
