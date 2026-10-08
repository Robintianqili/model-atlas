// The DevKit guide on the website (/devkit/guide/): site/DEVKIT.md split into its top-level sections at build time.
// Pure functions on the file's text (Node-safe; used by the route's server load and by devkit-site.test.ts).
//
//   parseGuide(md)          title, intro, sections (number, title, slug, markdown, hash), every heading id
//   renderSection(g, s)     the section as HTML; DEVKIT.md's own #anchors become links to the right section page
//   sectionHash(markdown)   what a Chinese summary was written for (generated tables and local-only blocks left out;
//						   see summaries.ts)
// Blocks between <!-- local-only:start --> and <!-- local-only:end --> (notes about the maintainers' own computer) are
// cut out before the file is split (local.ts): no page renders them, no hash counts them.
import { createHash } from 'node:crypto';
import { BASE, Slugger, plain, render, type Heading, type RenderOpts } from './md.ts';
import { HIDDEN_HEADING, publicText, stripLocalOnly } from './local.ts';

export interface GuideSection {
	n: number;
	/** the heading without its number: "Set up" */
	title: string;
	/** URL slug: the title's slug ("set-up"), stable when sections are renumbered */
	slug: string;
	/** GitHub id of the heading in DEVKIT.md ("1-set-up") */
	id: string;
	/** the section's Markdown, without its heading and the closing --- */
	md: string;
	hash: string;
	/** its ### headings */
	subs: { id: string; text: string }[];
	/** every heading text of the file up to this section's own heading: replayed so ids match the file's (GitHub numbers
	 *  repeated titles document-wide) */
	before: string[];
	/** local-only blocks cut out of this section (notes about the maintainers' own computer) */
	omitted: number;
}

export interface Guide {
	title: string;
	intro: string;
	sections: GuideSection[];
	/** heading id -> the slug of the section that holds it (null: the intro or the contents) */
	anchors: Map<string, string | null>;
	/** ids of the headings inside local-only blocks: a link to one renders as plain text */
	hidden: Set<string>;
}

const FENCE = /^\s*(```+|~~~+)/;
const H = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const GENERATED = /<!-- atlas:generated (\w+) -->\n[\s\S]*?<!-- \/atlas:generated -->/g;

/** sha256 (12 hex) of a section's public text: without its local-only blocks (local.ts), its generated tables (they
 *  follow the code on their own), runs of blank lines and trailing spaces. */
export function sectionHash(md: string): string {
	const text = publicText(md)
		.replace(GENERATED, (_, name: string) => `<!-- atlas:generated ${name} -->`)
		.split('\n')
		.map((l) => l.trimEnd())
		.join('\n')
		.trim();
	return createHash('sha256').update(text).digest('hex').slice(0, 12);
}

export function parseGuide(text: string): Guide {
	const pub = stripLocalOnly(text);
	const lines = pub.text.split('\n');
	const slugger = new Slugger();
	const anchors = new Map<string, string | null>();
	const hidden = new Set<string>();
	const sections: (GuideSection & { lines: string[] })[] = [];
	const seen: string[] = [];
	let title = '';
	const intro: string[] = [];
	let cur: (GuideSection & { lines: string[] }) | null = null;
	let skipping = false; // inside an unnumbered ## part ("Contents": the site has its own list)
	let fence: string | null = null;
	for (let li = 0; li < lines.length; li++) {
		const line = lines[li];
		// local-only blocks cut out just before this line count for the section they were in
		if (cur && !skipping) cur.omitted += pub.cuts.filter((c) => c === li).length;
		const f = FENCE.exec(line);
		let h: RegExpExecArray | null = null;
		if (f) fence = fence === null ? f[1] : line.trim().startsWith(fence) ? null : fence;
		else if (fence === null) h = H.exec(line);
		const hiddenH = !f && fence === null ? HIDDEN_HEADING.exec(line.trim()) : null;
		if (hiddenH) {
			// a heading inside a local-only block keeps its place in GitHub's id numbering; nothing links to it
			seen.push(hiddenH[1]);
			hidden.add(slugger.slug(hiddenH[1]));
		}
		if (h) {
			seen.push(h[2]);
			const id = slugger.slug(h[2]);
			if (h[1].length === 1 && !title) {
				title = plain(h[2]);
				anchors.set(id, null);
				continue;
			}
			if (h[1].length === 2) {
				const m = /^(\d+)\.\s+(.*)$/.exec(plain(h[2]));
				anchors.set(id, null);
				if (!m) {
					skipping = true;
					cur = null;
					continue;
				}
				skipping = false;
				cur = { n: Number(m[1]), title: m[2], slug: new Slugger().slug(m[2]), id, md: '', hash: '', subs: [], before: seen.slice(), omitted: 0, lines: [] };
				sections.push(cur);
				anchors.set(id, cur.slug);
				continue;
			}
			anchors.set(id, skipping ? null : (cur?.slug ?? null));
			if (cur && !skipping && h[1].length === 3) cur.subs.push({ id, text: plain(h[2]) });
		}
		if (skipping) continue;
		if (cur) cur.lines.push(line);
		else if (title) intro.push(line);
	}
	if (cur && !skipping) cur.omitted += pub.cuts.filter((c) => c >= lines.length).length;
	const trimEnd = (xs: string[]) => {
		while (xs.length && /^\s*(---)?\s*$/.test(xs[xs.length - 1])) xs.pop();
		return xs;
	};
	const out: GuideSection[] = sections.map(({ lines: ls, ...s }) => {
		const md = trimEnd(ls).join('\n');
		return { ...s, md, hash: sectionHash(md) };
	});
	return { title, intro: trimEnd(intro).join('\n'), sections: out, anchors, hidden };
}

/** A guide page's URL (BASE-prefixed; the page swaps BASE for $app/paths base). */
export const guideUrl = (slug?: string | null, anchor?: string) => `${BASE}/devkit/guide/${slug ? `${slug}/` : ''}${anchor ? `#${anchor}` : ''}`;

/** Links in the guide: its own #anchors go to the section page that has them; DEVKIT.md itself to the guide; web links
 *  stay links; other repository paths become plain text (the repository is not public yet). Unknown anchors are
 *  collected in `broken`. */
export function guideLinks(g: Guide, here: string | null, broken?: string[]): NonNullable<RenderOpts['link']> {
	return (href) => {
		if (/^https?:\/\//.test(href)) return { href, external: true };
		const m = /^(?:(?:site\/)?DEVKIT\.md)?#(.+)$/.exec(href);
		if (m) {
			const id = m[1].toLowerCase();
			if (g.hidden.has(id)) return null; // a heading inside a local-only block: not on the website
			if (!g.anchors.has(id)) {
				broken?.push(href);
				return null;
			}
			const sec = g.anchors.get(id) ?? null;
			return { href: here !== null && sec === here ? `#${id}` : guideUrl(sec, id) };
		}
		if (/^(?:site\/)?DEVKIT\.md$/.test(href)) return { href: guideUrl() };
		return null;
	};
}

export function renderSection(g: Guide, s: GuideSection, broken?: string[]): { html: string; headings: Heading[] } {
	const slugger = new Slugger();
	for (const t of s.before) slugger.slug(t);
	const headings: Heading[] = [];
	const html = render(s.md, { link: guideLinks(g, s.slug, broken), headings, slugger, copy: true, shift: -1 });
	return { html, headings };
}

export function renderIntro(g: Guide): string {
	return render(g.intro, { link: guideLinks(g, null), copy: true });
}

/** The ### block of a section whose heading is `heading` ("4.1 Scaffold" or "AK201"), as Markdown (null if missing). */
export function subsection(s: GuideSection, heading: string): string | null {
	const lines = s.md.split('\n');
	let fence: string | null = null;
	let from = -1;
	for (let i = 0; i < lines.length; i++) {
		const f = FENCE.exec(lines[i]);
		if (f) {
			fence = fence === null ? f[1] : lines[i].trim().startsWith(fence) ? null : fence;
			continue;
		}
		if (fence !== null) continue;
		const h = /^###\s+(.*)$/.exec(lines[i]);
		if (!h) continue;
		if (from >= 0) return lines.slice(from, i).join('\n').trim();
		if (plain(h[1]) === heading || plain(h[1]).startsWith(heading + ' ')) from = i + 1;
	}
	return from >= 0 ? lines.slice(from).join('\n').trim() : null;
}

/** Fenced code blocks of a Markdown text: { lang, code }. */
export function codeBlocks(md: string): { lang: string; code: string }[] {
	const out: { lang: string; code: string }[] = [];
	const re = /^(\s*)```([\w-]*)\s*\n([\s\S]*?)\n\s*```\s*$/gm;
	for (const m of md.matchAll(re)) {
		const ind = m[1].length;
		out.push({ lang: m[2], code: m[3].split('\n').map((l) => l.slice(Math.min(ind, /^\s*/.exec(l)![0].length))).join('\n') });
	}
	return out;
}

/** The first pipe table of a Markdown text (null if none). */
export function firstTable(md: string): string | null {
	const lines = md.split('\n');
	const i = lines.findIndex((l, k) => /^\s*\|/.test(l) && /^\s*\|?\s*:?-{2,}/.test(lines[k + 1] ?? ''));
	if (i < 0) return null;
	let j = i;
	while (j < lines.length && /^\s*\|/.test(lines[j])) j++;
	return lines.slice(i, j).join('\n');
}

/** Rows of a generated table: the block between <!-- atlas:generated name --> and <!-- /atlas:generated -->. */
export function generatedTable(md: string, name: string): string[][] {
	const m = new RegExp(`<!-- atlas:generated ${name} -->\\n([\\s\\S]*?)<!-- /atlas:generated -->`).exec(md);
	if (!m) return [];
	const rows = m[1].split('\n').filter((l) => /^\s*\|/.test(l));
	return rows.slice(2).map((r) =>
		r
			.trim()
			.replace(/^\||\|$/g, '')
			.split(/(?<!\\)\|/)
			.map((c) => c.trim().replace(/\\\|/g, '|'))
	);
}

/** Every page of the DevKit section, as paths under /devkit/ ('' is /devkit/ itself): the route's prerender entries and
 *  what check-build.ts expects in the build. */
export function devkitPagePaths(g: Guide): string[] {
	return ['', 'guide', ...g.sections.map((x) => `guide/${x.slug}`), 'reference', 'reference/ops', 'reference/views', 'tutorial', 'contribute'];
}
