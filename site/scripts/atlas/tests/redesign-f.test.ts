// Redesign wave F: the kit-level additions of the CLI (LOG.md "Redesign F"): atlas thumbs' layout and poster choice, the
// e2e windows (<w>x<h> passes, the layout preset, AK605's mode rule, AK606's weight of a built site's JS), the static
// build server, the generated glossary names (lazy glossary), and the design tokens (both themes, contrast, no legacy
// aliases in new kit code).
// Run from site/:  node --test scripts/atlas/tests/redesign-f.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { BUDGET, CARD_SLUGS, CARD_TILE, liveLessons, pickPoster, ROW_TILE, spriteSize, thumbIndex, tileAt } from '../commands/thumbs.ts';
import { expectedMode, jsWeight, LOADED_JS_BUDGET } from '../commands/e2e.ts';
import { parsePass, resolvePasses } from '../lib/passes.ts';
import { fileFor, staticServer } from '../lib/staticserver.ts';
import { namesSource, namesStale } from '../../gen-glossary-names.ts';
import { CODES } from '../errors.ts';
import { SITE } from './helpers.ts';

test('thumbs: live lessons, models first, with their routes', () => {
	const l = liveLessons([{ slug: 'a', status: 'live' }, { slug: 'b', status: 'planned' }], [{ slug: 'c', status: 'live' }]);
	assert.deepEqual(l, [
		{ slug: 'a', kind: 'model', path: '/m/a/' },
		{ slug: 'c', kind: 'concept', path: '/train/c/' }
	]);
});

test('thumbs: sprite layout', () => {
	assert.deepEqual(spriteSize(67, 8, ROW_TILE), { w: 8 * 224, h: 9 * 140, rows: 9 });
	assert.deepEqual(spriteSize(4, 2, CARD_TILE), { w: 960, h: 600, rows: 2 });
	assert.deepEqual(spriteSize(3, 8, ROW_TILE), { w: 3 * 224, h: 140, rows: 1 });
	assert.deepEqual(tileAt(9, 8, ROW_TILE), { x: 224, y: 140 });
	assert.equal(CARD_SLUGS.length, 4);
	assert.ok(BUDGET.rows === 250 * 1024 && BUDGET.cards === 120 * 1024);
});

test('thumbs: the poster is the first step with a real figure, else the largest', () => {
	assert.equal(pickPoster([{ step: 0, w: 120, h: 40 }, { step: 1, w: 400, h: 260 }, { step: 2, w: 600, h: 400 }]), 1);
	assert.equal(pickPoster([{ step: 0, w: 120, h: 60 }, { step: 1, w: 150, h: 90 }]), 1);
	assert.equal(pickPoster([{ step: 0, w: 0, h: 0 }]), null);
	assert.equal(pickPoster([]), null);
});

test('thumbs: the index says where every lesson is', () => {
	const ix = thumbIndex([{ slug: 'a', step: 2 }, { slug: 'tinycnn', step: 'conv' }], ['tinycnn'], { rows: 8, cards: 2 }, { rows: { light: 10, dark: 11 }, cards: { light: 5, dark: 6 } }, ['z'], '2026-10-13');
	assert.equal(ix.format, 'atlas-thumbs/1');
	assert.deepEqual(ix.tiles, { a: { row: 0, step: 2 }, tinycnn: { row: 1, step: 'conv', card: 0 } });
	assert.equal(ix.rows.light, 'rows.light.webp');
	assert.deepEqual(ix.missing, ['z']);
});

test('e2e: exact windows, the layout preset and the app/flow rule', () => {
	const p = parsePass('1280x680-en-light');
	assert.deepEqual([p.width, p.height, p.scroll], [1280, 680, false]);
	const n = parsePass('360x640-zh-dark');
	assert.deepEqual([n.width, n.height, n.scroll, n.probe360], [360, 640, true, false]);
	assert.throws(() => parsePass('1280x200-en-light'), /outside/);
	assert.deepEqual(resolvePasses('layout').map((x) => x.name), ['1280x800-en-light', '1280x680-en-light', '1024x768-en-dark', 'phone-zh-dark', '360x640-en-light']);
	assert.deepEqual([expectedMode(1280, 800), expectedMode(1024, 768), expectedMode(1366, 657), expectedMode(959, 800), expectedMode(1280, 599), expectedMode(768, 1024)], ['app', 'app', 'app', 'flow', 'flow', 'flow']);
	assert.ok(CODES.AK605 && CODES.AK606 && CODES.AK507);
	assert.equal(LOADED_JS_BUDGET, 400 * 1024);
});

test('static build server and AK606 on a built site (and not on a dev server)', async () => {
	const dir = mkdtempSync(join(tmpdir(), 'atlas-build-'));
	try {
		mkdirSync(join(dir, '_app', 'immutable'), { recursive: true });
		mkdirSync(join(dir, 'm', 'x'), { recursive: true });
		writeFileSync(join(dir, 'index.html'), '<!doctype html><title>home</title>');
		writeFileSync(join(dir, 'm', 'x', 'index.html'), '<!doctype html><title>x</title>');
		writeFileSync(join(dir, '404.html'), 'nope');
		writeFileSync(join(dir, '_app', 'immutable', 'a.js'), 'export const a = "' + 'x'.repeat(5000) + '";');
		assert.ok(fileFor(dir, '/m/x/')?.endsWith('index.html'));
		assert.equal(fileFor(dir, '/../secret'), null);
		const s = await staticServer(dir, 5390);
		try {
			const home = await fetch(`${s.base}/`);
			assert.equal(home.status, 200);
			assert.match(home.headers.get('content-type') ?? '', /text\/html/);
			assert.equal((await fetch(`${s.base}/nothing/`)).status, 404);
			const js = await jsWeight([`${s.base}/_app/immutable/a.js`, `${s.base}/_app/immutable/a.js`, `${s.base}/page.css`], new Map());
			assert.ok(js && js.files === 1 && js.gz > 0 && js.gz < 200, `gzip of a repetitive file is small: ${JSON.stringify(js)}`);
			assert.equal(await jsWeight([`${s.base}/src/lib/x.ts?t=1`, `${s.base}/node_modules/.vite/deps/svelte.js`], new Map()), null);
		} finally {
			s.stop();
		}
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('glossary names: generated from glossary.ts, up to date, quotes escaped', async () => {
	const src = namesSource([
		{ id: 'tensor', term: { en: 'tensor', zh: '张量' }, def: { en: 'd', zh: 'd' } },
		{ id: 'feature-map', term: { en: "it's", zh: '特征图' }, def: { en: 'd', zh: 'd' } }
	]);
	assert.match(src, /\ttensor: \['tensor', '张量'\]/);
	assert.match(src, /\t'feature-map': \['it\\'s', '特征图'\]/);
	assert.equal(await namesStale(SITE), false, 'src/lib/catalog/glossary-names.ts is stale: node scripts/gen-glossary-names.ts');
});

// ------------------------------------------------------------------ tokens (redesign spec §2)

/** The custom properties of one CSS block (the first block whose selector line matches `sel`). */
function block(css: string, sel: RegExp): Record<string, string> {
	const lines = css.split('\n');
	const start = lines.findIndex((l) => sel.test(l));
	assert.ok(start >= 0, `no block ${sel}`);
	const out: Record<string, string> = {};
	let depth = 0;
	for (let i = start; i < lines.length; i++) {
		const l = lines[i];
		for (const m of l.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) out[m[1]] = m[2].trim();
		depth += (l.match(/\{/g) ?? []).length - (l.match(/\}/g) ?? []).length;
		if (i > start && depth <= 0) break;
	}
	return out;
}

const TOKENS = readFileSync(join(SITE, 'src', 'lib', 'styles', 'tokens.css'), 'utf8').replace(/\r\n/g, '\n');

test('tokens: both dark blocks set the same values, and every themed token of the light block', () => {
	const light = block(TOKENS, /^:root \{/);
	const media = block(TOKENS, /:root:not\(\[data-theme='light'\]\) \{/);
	const forced = block(TOKENS, /^:root\[data-theme='dark'\] \{/);
	assert.deepEqual(media, forced, 'the two dark blocks differ');
	const themed = Object.keys(light).filter((k) => /^#|^rgb/.test(light[k]) && !/^--(cat|q|k|v|real|generated|div|seq|param|grad|mask)/.test(k) && !/^--(mark|mark-band)$/.test(k));
	const missing = themed.filter((k) => !(k in forced));
	assert.deepEqual(missing, [], 'light tokens with no dark value');
	for (const a of ['--bg', '--surface', '--surface-2', '--surface-3', '--accent', '--accent-soft', '--accent-ink', '--font-body', '--font-display', '--fs-xs', '--fs-3xl', '--lh-body', '--highlight', '--planned', '--planned-soft'])
		assert.ok(a in light, `legacy alias ${a} is gone`);
	assert.ok(!/'Fraunces/.test(TOKENS), 'Fraunces is removed');
});

test('tokens: the contrast of the spec (§2.2) holds in both themes', async () => {
	const { contrast, hexToRgb } = await import('../../../src/lib/kit/color.ts');
	const light = block(TOKENS, /^:root \{/);
	const dark = { ...light, ...block(TOKENS, /^:root\[data-theme='dark'\] \{/) };
	const c = (t: Record<string, string>, fg: string, bg: string) => contrast(hexToRgb(t[fg]), hexToRgb(t[bg]));
	for (const [name, t] of [['light', light], ['dark', dark]] as const) {
		for (const bg of ['--panel', '--paper', '--stage', '--well', '--here-bg']) assert.ok(c(t, '--muted', bg) >= 5.1, `${name}: muted on ${bg} is ${c(t, '--muted', bg).toFixed(2)}`);
		for (const bg of ['--panel', '--paper']) assert.ok(c(t, '--line-control', bg) >= 3, `${name}: control outline on ${bg} is ${c(t, '--line-control', bg).toFixed(2)}`);
		assert.ok(c(t, '--act-ink', '--act') >= 15, `${name}: the Next button`);
		assert.ok(c(t, '--mark-band-ink', '--mark-band') >= 12, `${name}: the headline band`);
		assert.ok(c(t, '--ok', '--ok-soft') >= 4.5 && c(t, '--no', '--no-soft') >= 4.5, `${name}: feedback`);
		for (const k of ['--live', '--pre', '--illus']) assert.ok(c(t, k, '--panel') >= 4.5, `${name}: ${k} on panel`);
		for (const k of ['--div-pos-text', '--div-neg-text']) for (const bg of ['--panel', '--paper', '--stage', '--well']) assert.ok(c(t, k, bg) >= 4.5, `${name}: ${k} on ${bg} is ${c(t, k, bg).toFixed(2)}`);
	}
});

test('new kit code uses the canonical token names (no legacy aliases)', () => {
	const kit = join(SITE, 'src', 'lib', 'kit');
	const files = ['SiteHeader.svelte', 'SiteFooter.svelte', 'Sheet.svelte', 'Popover.svelte', 'Icon.svelte', 'Mark.svelte', 'LangToggle.svelte', 'ThemeToggle.svelte', 'ProvenanceBadge.svelte', 'Term.svelte'].map((f) => join(kit, f));
	for (const d of ['lesson', 'catalog']) {
		const dir = join(kit, d);
		if (existsSync(dir)) for (const f of readdirSync(dir, { recursive: true }) as string[]) if (/\.(svelte|ts|css)$/.test(f)) files.push(join(dir, f));
	}
	const LEGACY = /var\(--(bg|surface(-[23])?|accent(-strong|-soft|-ink)?|font-body|font-display|fs-[a-z0-9]+|lh-body|highlight(-halo)?|planned(-soft)?)\)/g;
	const bad: string[] = [];
	for (const f of files) {
		const src = readFileSync(f, 'utf8');
		for (const m of src.matchAll(LEGACY)) bad.push(`${f.slice(kit.length + 1)}: ${m[0]}`);
	}
	assert.deepEqual(bad, [], 'use --paper/--panel/--well/--ink/--here-bg/--t-* (KIT.md section 4)');
});
