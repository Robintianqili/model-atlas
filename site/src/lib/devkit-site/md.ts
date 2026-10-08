// A small Markdown renderer for the DevKit pages (/devkit/), run at BUILD TIME only (server load and Node tests): the
// pages render site/DEVKIT.md, CONTRIBUTING.md and parts of the design file from the files themselves, so the website
// never holds a copy of their text. No dependency, no client code.
//
// The subset is what those files use: ATX headings (GitHub-style ids, so DEVKIT.md's own #anchors resolve), paragraphs,
// nested bullet and numbered lists (with code blocks inside items), fenced code blocks (highlighted here, see
// highlight()), GFM pipe tables (with \| escapes and <br> in cells), block quotes, horizontal rules and HTML comments
// (dropped). Inline: `code`, **bold**, [links](href). Everything else is text and is HTML-escaped; the only raw HTML
// kept is <br> in table cells.
//
// Blocks between <!-- local-only:start --> and <!-- local-only:end --> (notes about the maintainers' own computer) are
// never rendered: render() drops them first (local.ts).
//
// Links are rewritten by a caller-supplied function (relative repo paths and #anchors become site URLs). Every site URL
// starts with the BASE token; the Svelte page replaces the token with $app/paths `base`.

import { BASE } from './base.ts';
import { HIDDEN_HEADING, stripLocalOnly } from './local.ts';

export { BASE };

export interface Heading {
	level: number;
	id: string;
	text: string;
}

export interface RenderOpts {
	/** map an href to a site URL (or an external one); return null to render the link text without a link */
	link?: (href: string) => { href: string; external?: boolean } | null;
	/** shift heading levels (the page has its own h1): -1 turns ### into h2; levels stay within h2..h6 */
	shift?: number;
	/** collects the headings */
	headings?: Heading[];
	/** one slugger per page, so ids stay unique (GitHub numbering) */
	slugger?: Slugger;
	/** a copy button on code blocks */
	copy?: boolean;
}

// ------------------------------------------------------------------ escaping, slugs
export const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** GitHub's heading ids: lowercase, punctuation dropped (letters, digits, '-', '_' and spaces kept), spaces to '-',
 *  repeats get -1, -2, … */
export class Slugger {
	private seen = new Map<string, number>();
	slug(text: string): string {
		const base = plain(text)
			.toLowerCase()
			.replace(/[^\p{L}\p{N}\s_-]/gu, '')
			.replace(/\s/g, '-');
		const n = this.seen.get(base);
		this.seen.set(base, (n ?? 0) + 1);
		return n === undefined ? base : `${base}-${n}`;
	}
}

/** Inline Markdown to plain text (slugs, titles): code spans keep their text, links their label. */
export function plain(s: string): string {
	return s
		.replace(/`([^`]*)`/g, '$1')
		.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
		.replace(/\*\*([^*]*)\*\*/g, '$1')
		.replace(/<br>/g, ' ')
		.trim();
}

// ------------------------------------------------------------------ inline
function findClose(s: string, from: number): number {
	let depth = 0;
	for (let i = from; i < s.length; i++) {
		const c = s[i];
		if (c === '`') {
			const j = s.indexOf('`', i + 1);
			if (j < 0) return -1;
			i = j;
		} else if (c === '[') depth++;
		else if (c === ']' && --depth === 0) return i;
	}
	return -1;
}

export function inline(s: string, o: RenderOpts = {}): string {
	let out = '';
	for (let i = 0; i < s.length; ) {
		const c = s[i];
		if (c === '\\' && i + 1 < s.length && /[\\`*_[\]()|#<>-]/.test(s[i + 1])) {
			out += esc(s[i + 1]);
			i += 2;
			continue;
		}
		if (c === '`') {
			const j = s.indexOf('`', i + 1);
			if (j > i) {
				out += `<code>${esc(s.slice(i + 1, j))}</code>`;
				i = j + 1;
				continue;
			}
		}
		if (c === '[') {
			const k = findClose(s, i);
			if (k > i && s[k + 1] === '(') {
				const m = s.indexOf(')', k + 2);
				if (m > k) {
					const href = s.slice(k + 2, m).trim();
					const r = o.link ? o.link(href) : { href };
					const body = inline(s.slice(i + 1, k), o);
					out += r ? `<a href="${esc(r.href)}"${r.external ? ' rel="noopener"' : ''}>${body}</a>` : body;
					i = m + 1;
					continue;
				}
			}
		}
		if (s.startsWith('**', i)) {
			const j = s.indexOf('**', i + 2);
			if (j > i + 2) {
				out += `<strong>${inline(s.slice(i + 2, j), o)}</strong>`;
				i = j + 2;
				continue;
			}
		}
		if (s.startsWith('<br>', i)) {
			out += '<br>';
			i += 4;
			continue;
		}
		out += esc(c);
		i++;
	}
	return out;
}

// ------------------------------------------------------------------ code highlighting
type Rule = [RegExp, string];
const NUM: Rule = [/^\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?\b/i, 'n'];
const KW_TS = /^(?:import|export|from|const|let|var|type|interface|function|return|new|yield|for|of|in|if|else|as|async|await|default|extends|satisfies|class|throw|try|catch|typeof|keyof|true|false|null|undefined|void)\b/;
const KW_PY = /^(?:def|class|import|from|return|for|in|if|elif|else|with|as|None|True|False|lambda|yield|try|except|raise|not|and|or|is|while|pass|assert)\b/;
const RULES: Record<string, Rule[]> = {
	ts: [
		[/^\/\/[^\n]*/, 'c'],
		[/^\/\*[\s\S]*?\*\//, 'c'],
		[/^'(?:[^'\\\n]|\\.)*'|^"(?:[^"\\\n]|\\.)*"|^`(?:[^`\\]|\\.)*`/, 's'],
		[KW_TS, 'k'],
		NUM
	],
	py: [
		[/^#[^\n]*/, 'c'],
		[/^"""[\s\S]*?"""|^'''[\s\S]*?'''/, 's'],
		[/^[rbfu]{0,2}'(?:[^'\\\n]|\\.)*'|^[rbfu]{0,2}"(?:[^"\\\n]|\\.)*"/i, 's'],
		[/^@[\w.]+/, 'k'],
		[KW_PY, 'k'],
		NUM
	],
	json: [
		[/^"(?:[^"\\\n]|\\.)*"(?=\s*:)/, 'p'],
		[/^"(?:[^"\\\n]|\\.)*"/, 's'],
		[/^(?:true|false|null)\b/, 'k'],
		[/^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/i, 'n']
	],
	sh: [
		[/^#[^\n]*/, 'c'],
		[/^'[^'\n]*'|^"(?:[^"\\\n]|\\.)*"/, 's'],
		[/^(?:npm run atlas --|npm run|npm|node|git|cd)(?=\s|$)/, 'k']
	]
};
const LANG: Record<string, string> = { ts: 'ts', typescript: 'ts', js: 'ts', javascript: 'ts', svelte: 'ts', py: 'py', python: 'py', json: 'json', sh: 'sh', bash: 'sh', shell: 'sh', '': 'sh' };

/** Token spans for a code block: classes tc (comment), ts (string), tk (keyword), tn (number), tp (JSON key). */
export function highlight(code: string, lang: string): string {
	const l = LANG[lang.toLowerCase()];
	const rules = RULES[l ?? ''];
	if (!rules) return esc(code);
	let out = '';
	let run = '';
	for (let i = 0; i < code.length; ) {
		const rest = code.slice(i);
		// inside an identifier nothing starts a keyword or a number; in a shell line a # inside a word (a URL fragment,
		// "house#2") does not start a comment either: only a # at the start or after a space does
		const midWord = i > 0 && /[\w$.]/.test(code[i - 1]);
		const shellHash = l === 'sh' && rest[0] === '#' && i > 0 && !/\s/.test(code[i - 1]);
		let hit: [string, string] | null = null;
		if ((!midWord || /^['"`/#]/.test(rest)) && !shellHash) {
			for (const [re, cls] of rules) {
				const m = re.exec(rest);
				if (m && m[0].length) {
					hit = [m[0], cls];
					break;
				}
			}
		}
		if (hit) {
			out += esc(run) + `<span class="t${hit[1]}">${esc(hit[0])}</span>`;
			run = '';
			i += hit[0].length;
		} else {
			run += code[i++];
		}
	}
	return out + esc(run);
}

/** A highlighted code block; `caption` names the file, `copy` adds a copy button (one delegated handler on the page). */
export function codeBlock(code: string, lang: string, o: { copy?: boolean; caption?: string; start?: number } = {}): string {
	const l = lang.toLowerCase();
	const label = o.caption ?? (l === '' || l === 'sh' ? '' : l);
	const head = label || o.copy ? `<div class="code-head"><span class="code-lang">${esc(label)}</span>${o.copy ? '<button type="button" class="code-copy" data-copy>Copy</button>' : ''}</div>` : '';
	const start = o.start && o.start > 1 ? ` data-start="${o.start}"` : '';
	return `<div class="code">${head}<pre data-lang="${esc(l || 'text')}"${start}><code>${highlight(code.replace(/\n+$/, ''), l)}</code></pre></div>`;
}

// ------------------------------------------------------------------ blocks
const FENCE = /^(\s*)(```+|~~~+)\s*([\w-]*)\s*$/;
const ITEM = /^(\s*)([-*+]|\d+[.)])(\s+)(.*)$/;
const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
const HR = /^\s*(---|\*\*\*)\s*$/;
const indentOf = (l: string) => /^\s*/.exec(l)![0].replace(/\t/g, '    ').length;
const blank = (l: string) => /^\s*$/.test(l);
const startsBlock = (l: string) => FENCE.test(l) || HEADING.test(l) || /^\s*>/.test(l) || HR.test(l) || /^\s*\|/.test(l) || /^\s*<!--/.test(l);

function splitRow(row: string): string[] {
	let r = row.trim();
	if (r.startsWith('|')) r = r.slice(1);
	if (r.endsWith('|') && !r.endsWith('\\|')) r = r.slice(0, -1);
	const cells: string[] = [];
	let cur = '';
	let code = false;
	for (let i = 0; i < r.length; i++) {
		const c = r[i];
		if (c === '\\' && r[i + 1] === '|') {
			cur += '|';
			i++;
			continue;
		}
		if (c === '`') code = !code;
		if (c === '|' && !code) {
			cells.push(cur.trim());
			cur = '';
		} else cur += c;
	}
	cells.push(cur.trim());
	return cells;
}

function table(rows: string[], o: RenderOpts): string {
	const th = splitRow(rows[0])
		.map((c) => `<th scope="col">${inline(c, o)}</th>`)
		.join('');
	const tr = rows
		.slice(2)
		.map((r) => `<tr>${splitRow(r).map((c) => `<td>${inline(c, o)}</td>`).join('')}</tr>`)
		.join('');
	return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

const dedent = (lines: string[], n: number) => lines.map((l) => (blank(l) ? '' : l.slice(Math.min(n, indentOf(l)))));

/** Render Markdown lines to HTML. */
export function blocks(lines: string[], o: RenderOpts = {}): string {
	const out: string[] = [];
	const slugger = (o.slugger ??= new Slugger());
	let i = 0;
	const n = lines.length;
	while (i < n) {
		const line = lines[i];
		if (blank(line)) {
			i++;
			continue;
		}
		if (/^\s*<!--/.test(line)) {
			// a heading of a local-only block renders nothing but keeps its place in the id numbering
			const hidden = HIDDEN_HEADING.exec(line.trim());
			if (hidden) slugger.slug(hidden[1]);
			while (i < n && !/-->/.test(lines[i])) i++;
			i++;
			continue;
		}
		let m = FENCE.exec(line);
		if (m) {
			const ind = m[1].length;
			const fence = m[2];
			const body: string[] = [];
			i++;
			while (i < n && !(lines[i].trim().startsWith(fence) && lines[i].trim().replace(/[`~]/g, '') === '')) {
				body.push(lines[i].slice(Math.min(ind, indentOf(lines[i]))));
				i++;
			}
			i++;
			out.push(codeBlock(body.join('\n'), m[3] ?? '', { copy: o.copy }));
			continue;
		}
		m = HEADING.exec(line);
		if (m) {
			const level = Math.max(2, Math.min(6, m[1].length + (o.shift ?? 0)));
			const id = slugger.slug(m[2]);
			o.headings?.push({ level: m[1].length, id, text: plain(m[2]) });
			out.push(`<h${level} id="${esc(id)}">${inline(m[2], o)}<a class="anchor" href="#${esc(id)}" aria-label="Link to this heading">#</a></h${level}>`);
			i++;
			continue;
		}
		if (HR.test(line)) {
			out.push('<hr>');
			i++;
			continue;
		}
		if (/^\s*\|/.test(line) && i + 1 < n && TABLE_SEP.test(lines[i + 1])) {
			const rows: string[] = [];
			while (i < n && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
			out.push(table(rows, o));
			continue;
		}
		if (/^\s*>/.test(line)) {
			const q: string[] = [];
			while (i < n && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ''));
			out.push(`<blockquote>${blocks(q, o)}</blockquote>`);
			continue;
		}
		if (ITEM.test(line)) {
			const r = list(lines, i, o);
			out.push(r.html);
			i = r.next;
			continue;
		}
		const para: string[] = [line.trim()];
		i++;
		while (i < n && !blank(lines[i]) && !startsBlock(lines[i]) && !(ITEM.test(lines[i]) && indentOf(lines[i]) <= indentOf(line))) para.push(lines[i++].trim());
		out.push(`<p>${inline(para.join(' '), o)}</p>`);
	}
	return out.join('\n');
}

function list(lines: string[], start: number, o: RenderOpts): { html: string; next: number } {
	const first = ITEM.exec(lines[start])!;
	const base = indentOf(lines[start]);
	const ordered = /\d/.test(first[2]);
	const items: string[] = [];
	let loose = false;
	let i = start;
	const n = lines.length;
	while (i < n) {
		const m = ITEM.exec(lines[i]);
		if (!m || indentOf(lines[i]) !== base || /\d/.test(m[2]) !== ordered) break;
		const ci = m[1].length + m[2].length + m[3].length;
		const body: string[] = [' '.repeat(ci) + m[4]];
		i++;
		while (i < n) {
			const l = lines[i];
			if (blank(l)) {
				let j = i + 1;
				while (j < n && blank(lines[j])) j++;
				if (j < n && indentOf(lines[j]) >= ci) {
					for (; i < j; i++) body.push('');
					continue;
				}
				break;
			}
			const ind = indentOf(l);
			if (ind >= ci) body.push(l);
			// a nested list indented less than the item's text ("  - " under "1. "), or a lazy continuation line
			else if (ind > base && (ITEM.test(l) || !startsBlock(l))) body.push(' '.repeat(ci) + l.trimStart());
			else break;
			i++;
		}
		const inner = dedent(body, ci);
		const lastText = inner.map(blank).lastIndexOf(false);
		if (inner.slice(0, lastText).some(blank)) loose = true;
		items.push(blocks(inner, o));
		if (i < n && blank(lines[i])) {
			let j = i;
			while (j < n && blank(lines[j])) j++;
			const nx = j < n ? ITEM.exec(lines[j]) : null;
			if (nx && indentOf(lines[j]) === base && /\d/.test(nx[2]) === ordered) {
				loose = true;
				i = j;
			}
		}
	}
	const tag = ordered ? 'ol' : 'ul';
	const startNo = ordered ? Number(/\d+/.exec(first[2])![0]) : 1;
	const lis = items.map((h) => {
		// a tight item: its first paragraph renders without <p>
		const t = !loose && h.startsWith('<p>') ? h.replace(/^<p>([\s\S]*?)<\/p>/, '$1') : h;
		return `<li>${t}</li>`;
	});
	return { html: `<${tag}${startNo !== 1 ? ` start="${startNo}"` : ''}>${lis.join('')}</${tag}>`, next: i };
}

/** Render Markdown to HTML; local-only blocks are left out (local.ts). */
export function render(md: string, o: RenderOpts = {}): string {
	return blocks(stripLocalOnly(md).text.split('\n'), o);
}
