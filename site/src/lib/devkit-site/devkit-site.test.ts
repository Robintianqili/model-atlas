// Tests of the DevKit pages (/devkit/): the staleness check passes on this tree and catches each kind of change it is
// there for, every anchor DEVKIT.md links to exists, the Markdown renderer handles what the sources use, and notes about
// the maintainers' own computer (local-only blocks, local.ts) never reach a page.
//   node --test src/lib/devkit-site/devkit-site.test.ts
// (The build runs the same checks in the route's server load and fails on a stale page or a local-machine string.)
// Every path, folder and name in the samples and the fixture is made up. The tests of the real DEVKIT.md's markers run
// only where it has them (the maintainers' tree, with its git-ignored machine.local.json); a fresh clone of the public
// repository has neither, and its DEVKIT.md is the public text.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { readSources, type Sources } from './sources.ts';
import { staleness } from './check.ts';
import { guideLinks, parseGuide, renderIntro, renderSection, sectionHash, subsection } from './guide.ts';
import { BASE, codeBlock, highlight, render } from './md.ts';
import { TUTORIAL, excerpt } from './tutorial.ts';
import { afternoonResult, contributingStart, quickStart } from './home.ts';
import { relBase, withBase } from './base.ts';
import { CONTRIBUTING_STATUS_ZH, SECTION_ZH } from './summaries.ts';
import { findLocalStrings, LOCAL_END, LOCAL_START, LOCAL_STRINGS, LocalOnlyError, publicText, stripLocalOnly } from './local.ts';
import { BLOCKS, isMarked, mark, unmark, type Block } from './mark-local.ts';
import { parseMachine } from './machine.ts';

const SITE = fileURLToPath(new URL('../../../', import.meta.url));
const fresh = (): Sources => readSources(SITE);
const FIXTURE = readFileSync(fileURLToPath(new URL('./fixtures/local-only.md', import.meta.url)), 'utf8');
/** the maintainers' tree: DEVKIT.md has local-only markers (a fresh clone of the public repository has none) */
const MARKED = fresh().devkit.includes(LOCAL_START);
const NO_MARKERS = !MARKED && 'DEVKIT.md has no local-only markers (the public text, as in a fresh clone)';
const NO_BLOCKS = (!MARKED || !BLOCKS.length) && 'no local-only markers in DEVKIT.md, or no block list (the git-ignored machine.local.json) on this computer';

test('the DevKit pages are up to date with DEVKIT.md, ops.json, the error list, CONTRIBUTING.md and hello-xor', () => {
	assert.deepEqual(staleness(fresh()), []);
});

test('every #anchor that DEVKIT.md links to exists', () => {
	const s = fresh();
	const g = parseGuide(s.devkit);
	const broken: string[] = [];
	for (const sec of g.sections) renderSection(g, sec, broken);
	assert.deepEqual(broken, []);
	assert.ok(g.sections.length >= 10, `${g.sections.length} sections`);
	assert.equal(g.anchors.get('ak201'), g.sections.find((x) => x.md.includes('<!-- atlas:generated codes -->'))!.slug);
});

test('a changed DEVKIT.md section fails the check until its Chinese summary follows', () => {
	const s = fresh();
	const g0 = parseGuide(s.devkit);
	// a section without local-only blocks: its public text is a slice of the file
	const first = g0.sections.find((x) => x.omitted === 0 && s.devkit.includes(x.md))!;
	const changed = s.devkit.replace(first.md, `${first.md}\n\nOne more sentence.`);
	const problems = staleness({ ...s, devkit: changed });
	assert.equal(problems.length, 1, problems.join('\n'));
	assert.match(problems[0], new RegExp(`section ${first.n} .*set its hash to '${sectionHash(`${first.md}\n\nOne more sentence.`)}'`));
	// a change inside a generated table does not (the table follows the code by itself)
	const codesTable = /<!-- atlas:generated codes -->\n/.exec(s.devkit)!;
	const tableEdit = s.devkit.slice(0, codesTable.index + codesTable[0].length) + '| x | y |\n' + s.devkit.slice(codesTable.index + codesTable[0].length);
	assert.deepEqual(staleness({ ...s, devkit: tableEdit }), []);
	// a new section needs a summary
	const added = s.devkit.replace(/\n## (\d+)\. Troubleshooting/, '\n## 99. A new part\n\nText.\n\n## $1. Troubleshooting');
	assert.ok(staleness({ ...s, devkit: added }).some((p) => /A new part" has no Chinese summary/.test(p)));
});

test('a new or changed error code fails the check', () => {
	const s = fresh();
	assert.ok(staleness({ ...s, codes: { ...s.codes, AK999: 'something new' } }).some((p) => /AK999 .*no Chinese meaning/.test(p)));
	assert.ok(staleness({ ...s, codes: { ...s.codes, AK201: 'a new meaning' } }).some((p) => /meaning of AK201 changed/.test(p)));
	const { AK704: _, ...fewer } = s.codes;
	assert.ok(staleness({ ...s, codes: fewer }).some((p) => /translates AK704/.test(p)));
});

test('a new op in ops.json fails the check until it has a demo', () => {
	const s = fresh();
	const extra = { ...s.ops[0], op: 'gru', zoom: 'gate' };
	assert.ok(staleness({ ...s, ops: [...s.ops, extra] }).some((p) => /op 'gru' .*has no demo/.test(p)));
	assert.ok(staleness({ ...s, ops: s.ops.filter((o) => o.op !== 'lstm') }).some((p) => /demo for 'lstm'/.test(p)));
});

test('a hello-xor change that breaks a tutorial excerpt fails the check', () => {
	const s = fresh();
	const step = TUTORIAL.find((x) => x.file)!;
	const files = { ...s.files, [step.file!.path]: (s.files[step.file!.path] ?? '').replace(step.file!.from, 'renamed') };
	assert.ok(staleness({ ...s, files }).some((p) => p.includes(`tutorial step '${step.id}'`)));
	for (const st of TUTORIAL) if (st.file) assert.ok(excerpt(s.files[st.file.path]!, st.file), st.id);
});

test('the afternoon test result is read from its public record (durations only, no clock times)', () => {
	const text = fresh().afternoon;
	const r = afternoonResult(text);
	assert.ok(r && r.seconds > 0 && r.gaps > 0 && r.built.length > 0, JSON.stringify(r));
	assert.doesNotMatch(text!, /\b\d{1,2}:\d{2}:\d{2}\b/, 'no clock times in the public record');
	assert.deepEqual(findLocalStrings(text!), []);
	assert.deepEqual(afternoonResult('# Afternoon test: demo-x (newcomer)\n\n- From the start to draft gates green: **3 min 05 s** wall clock.'), { seconds: 185, gaps: 0, built: 'demo-x' });
	assert.equal(afternoonResult('Start 10:00:00, draft gates green 10:14:00'), null);
});

test('the Markdown renderer: GitHub ids, tables with escaped pipes, nested lists with code, escaping', () => {
	const html = render(
		[
			'## 4.1 Scaffold',
			'## AK201',
			'',
			'| Op | Attrs |',
			'|---|---|',
			'| `a \\| b` | x<br>y |',
			'',
			'1. First item',
			'   continued.',
			'2. Second:',
			'',
			'   ```python',
			'   def f(x):  # a comment',
			'       return "s"',
			'   ```',
			'',
			'   - nested `[C, H, W]` and [a link](#ak201)',
			'',
			'Text with <tags> & **bold**.'
		].join('\n'),
		{ link: (h) => ({ href: `/x${h}` }) }
	);
	assert.match(html, /<h2 id="41-scaffold">/);
	assert.match(html, /<h2 id="ak201">/);
	assert.match(html, /<td><code>a \| b<\/code><\/td><td>x<br>y<\/td>/);
	assert.match(html, /<ol><li><p>First item continued\.<\/p><\/li><li><p>Second:<\/p>\n?<div class="code">/);
	assert.match(html, /<span class="tk">def<\/span> f\(x\):  <span class="tc"># a comment<\/span>/);
	assert.match(html, /<ul><li>nested <code>\[C, H, W\]<\/code> and <a href="\/x#ak201">a link<\/a><\/li><\/ul>/);
	assert.match(html, /Text with &lt;tags&gt; &amp; <strong>bold<\/strong>\./);
	assert.match(codeBlock('npm run atlas -- check x  # gates', ''), /<span class="tk">npm run atlas --<\/span> check x  <span class="tc"># gates<\/span>/);
	// a # inside a word is no shell comment (DEVKIT.md section 13's error example: "house#2", "site/DEVKIT.md#ak201")
	assert.equal(highlight('  docs    site/DEVKIT.md#ak201', ''), '  docs    site/DEVKIT.md#ak201');
	assert.equal(highlight('case "house#2"', 'sh'), 'case <span class="ts">&quot;house#2&quot;</span>');
});

test('the landing page: the quick start has every step of section 4, the start comes from CONTRIBUTING.md', () => {
	const s = fresh();
	const g = parseGuide(s.devkit);
	const qs = quickStart(g);
	const all = qs.flatMap((q) => q.lines);
	assert.ok(qs.length >= 5, JSON.stringify(qs.map((q) => q.title)));
	// kind A trains before it exports (an inline command of 4.2), and the order of the text is kept
	const a = qs.find((q) => q.title.startsWith('4.2 '))!;
	assert.ok(a.lines.findIndex((l) => l.includes(' py ')) >= 0 && a.lines.findIndex((l) => l.includes(' py ')) < a.lines.findIndex((l) => l.includes(' export ')), JSON.stringify(a.lines));
	// alternatives (later code blocks: new model --like) and lane commands are left out
	assert.ok(!all.some((l) => l.includes('--like') || l.includes(' worktree ')), JSON.stringify(all));
	const st = contributingStart(s.contributing);
	assert.ok(st.setup && st.setup.includes('doctor'), String(st.setup));
	assert.equal(st.status === null, CONTRIBUTING_STATUS_ZH === '');
	assert.deepEqual(contributingStart('# T\n\nText.\n\n## Setup\n\n1. a\n2. b\n\n## Next\n'), { status: null, setup: '1. a\n2. b' });
});

test('generated HTML gets the same relative site root on the server and in the browser', () => {
	assert.equal(relBase('/devkit/'), '..');
	assert.equal(relBase('/devkit/guide/set-up/'), '../../..');
	assert.equal(relBase('/model-atlas/devkit/tutorial/'), '../..');
	assert.equal(withBase(`<a href="${BASE}/devkit/guide/">`, relBase('/devkit/reference/ops/')), '<a href="../../../devkit/guide/">');
});

// ------------------------------------------------------------------ local-only blocks (local.ts): the fixture
test('local-only blocks: the renderer leaves them out, whole lines and inside a line (fixture)', () => {
	const pub = stripLocalOnly(FIXTURE);
	assert.equal(pub.cuts.length, 3);
	assert.deepEqual(findLocalStrings(pub.text), []);
	assert.ok(findLocalStrings(FIXTURE).length >= 6, 'the fixture has local-machine strings inside its blocks');
	// a line cut in the middle keeps its public part; a block of whole lines leaves only its heading's placeholder
	assert.match(pub.text, /^- Git\.$/m);
	assert.match(pub.text, /computer in local-only blocks\.$/m);
	assert.match(pub.text, /^<!-- local-only heading: Notes -->$/m);
	assert.ok(!pub.text.includes('Writing files') && !pub.text.includes('exFAT'));
	// markers inside a fenced code block are text
	assert.match(pub.text, /```\n<!-- local-only:start -->\n```/);

	const g = parseGuide(FIXTURE);
	assert.deepEqual(findLocalStrings(g.intro), []);
	const s1 = g.sections.find((x) => x.n === 1)!;
	assert.equal(s1.omitted, 2);
	assert.equal(g.sections.find((x) => x.n === 2)!.omitted, 0);
	// the hidden "### Notes" keeps its place in GitHub's numbering: the public one is #notes-1, as in the file
	assert.deepEqual(s1.subs, [{ id: 'notes-1', text: 'Notes' }]);
	assert.ok(g.hidden.has('notes') && !g.anchors.has('notes'));
	const broken: string[] = [];
	const { html, headings } = renderSection(g, s1, broken);
	assert.deepEqual(broken, [], 'a link to a hidden heading is plain text, not a broken link');
	assert.deepEqual(headings.map((h) => h.id), ['notes-1']);
	assert.match(html, /<h2 id="notes-1">/);
	assert.match(html, /the hidden notes, the <a href="#notes-1">public notes<\/a>/);
	assert.match(html, /<li>Git\.<\/li>/);
	assert.match(html, /&lt;!-- local-only:start --&gt;/);
	assert.deepEqual(findLocalStrings(html), []);
	assert.deepEqual(findLocalStrings(renderIntro(g)), []);
	assert.equal(guideLinks(g, s1.slug)('#notes'), null);
	// render() of any source drops the blocks too (CONTRIBUTING.md, the design's checklist)
	assert.deepEqual(findLocalStrings(render(FIXTURE)), []);
});

test("local-only blocks: a section's hash ignores what is inside them", () => {
	const s1 = (md: string) => parseGuide(md).sections.find((x) => x.n === 1)!.hash;
	const base = s1(FIXTURE);
	// an edit inside a block: the same hash (a machine note never stales a summary)
	assert.equal(s1(FIXTURE.replace('Write into the scratchpad first', 'Write it somewhere else first')), base);
	assert.equal(s1(FIXTURE.replace('(the drive is\n  exFAT;', '(the drive is\n  NTFS;')), base);
	// the block and its markers removed altogether: the same hash
	const without = FIXTURE.replace(/\n<!-- local-only:start -->\n### Notes[\s\S]*?<!-- local-only:end -->\n/, '\n');
	assert.notEqual(without, FIXTURE);
	assert.ok(!without.includes('Writing files'));
	// (the hidden heading's placeholder is gone too, and with it the numbering: the public heading is #notes now)
	assert.equal(parseGuide(without).sections.find((x) => x.n === 1)!.subs[0].id, 'notes');
	assert.equal(s1(without), base);
	// an edit of the public text changes it
	assert.notEqual(s1(FIXTURE.replace('Public notes.', 'Public notes, edited.')), base);
	// so does publishing a note (taking its markers away)
	assert.notEqual(s1(FIXTURE.replace(`Git.${LOCAL_START}`, 'Git.').replace(`below).${LOCAL_END}`, 'below).')), base);
	assert.ok(publicText(FIXTURE).length > 0);
});

test('local-only blocks: a missing or extra marker fails loudly, naming the line', () => {
	assert.throws(() => stripLocalOnly(`a\n${LOCAL_START}\nb\n`), (e: Error) => e instanceof LocalOnlyError && /line 2: .* has no /.test(e.message));
	assert.throws(() => stripLocalOnly(`a\nb${LOCAL_END}\n`), (e: Error) => e instanceof LocalOnlyError && /line 2: .* without a /.test(e.message));
	assert.throws(
		() => stripLocalOnly(`${LOCAL_START}\na ${LOCAL_START}\n${LOCAL_END}\n`),
		(e: Error) => e instanceof LocalOnlyError && /line 2: .* inside the local-only block opened at line 1/.test(e.message)
	);
	assert.throws(() => stripLocalOnly('a <!-- local-only:begin --> b'), /unknown marker/);
	assert.throws(() => parseGuide(`# T\n\n## 1. A\n\n${LOCAL_START}\nx\n`), LocalOnlyError);
	// two blocks on one line, and a block that is the whole line
	assert.equal(stripLocalOnly(`a${LOCAL_START}x${LOCAL_END}b${LOCAL_START}y${LOCAL_END}c\n${LOCAL_START}z${LOCAL_END}\nd`).text, 'abc\nd');
});

test('the local-machine guard knows each kind of string (generic patterns)', () => {
	const samples = [
		'D:\\work\\atlas-demo',
		'see D:\\ drive',
		'git -c "safe.directory=/x"',
		'On this machine',
		'C:/Users/someone/x',
		'C:\\Users\\someone\\x',
		'%LOCALAPPDATA% AppData',
		'cd /home/someone/atlas',
		'open /Users/someone/atlas',
		'the scratchpad'
	];
	for (const x of samples) assert.ok(findLocalStrings(x, LOCAL_STRINGS).length > 0, x);
	assert.equal(LOCAL_STRINGS.length, 7);
	assert.ok(findLocalStrings('| [AK004](#ak004) | git safe.directory |', LOCAL_STRINGS).length > 0, 'the git setting without a path');
	// generic text the public pages need stays allowed
	for (const x of ['| [AK004](#ak004) | git refuses the repo |', 'machine notes', 'Windows', 'exFAT', 'https://example.org/a/b', 'ratio 3:4', 'x ? a : /re/', '/home/', 'site/src/lib/']) assert.deepEqual(findLocalStrings(x, LOCAL_STRINGS), [], x);
	// JSON page data escapes backslashes: still caught
	assert.ok(findLocalStrings(JSON.stringify({ p: 'D:\\work\\x' }), LOCAL_STRINGS).length > 0);
});

test("machine.local.json: one computer's own strings and block list come from the git-ignored file", () => {
	const m = parseMachine(
		JSON.stringify({
			strings: [{ re: 'Q:[\\\\/]+projects', flags: 'i', what: 'a made-up project folder' }, { re: '\\bsomeone-else\\b', what: 'a made-up user name' }],
			blocks: [{ id: 'b', where: 'a made-up block', from: 'x', fromMode: 'inline', to: 'y', toMode: 'line' }]
		}),
		'made-up.json'
	);
	assert.equal(m.strings.length, 2);
	assert.deepEqual(m.blocks.map((b) => b.id), ['b']);
	const all = [...LOCAL_STRINGS, ...m.strings];
	for (const x of ['see q:/projects/x', 'see Q:\\projects', 'by someone-else']) {
		assert.ok(findLocalStrings(x, all).length > 0, x);
		if (!x.includes('\\')) assert.deepEqual(findLocalStrings(x, LOCAL_STRINGS), [], `${x}: only this computer's list knows it`);
	}
	// a missing file is the same as {} (a fresh clone); a broken one fails loudly, naming the file
	assert.deepEqual(parseMachine('{}', 'empty.json'), { file: 'empty.json', strings: [], blocks: [] });
	assert.throws(() => parseMachine('{ "strings": [{ "re": "(", "what": "x" }] }', 'bad.json'), /bad\.json: strings\[0\] is not a valid regular expression/);
	assert.throws(() => parseMachine('{ "blocks": [{ "id": "x" }] }', 'bad.json'), /bad\.json: blocks\[0\] needs/);
	assert.throws(() => parseMachine('[', 'bad.json'), /bad\.json: not valid JSON/);
});

test('mark-local: a block list marks a text, and removing the markers gives it back (made-up notes)', () => {
	const plain = ['# Guide', '', '## 1. Set up', '', '- Git. On this machine, run `git -c "safe.directory=D:/work/demo" ...`.', '', '### Notes', '', '- Write into the scratchpad first.', '', '## 2. Next', '', 'Text.'].join('\n');
	const blocks: Block[] = [
		{ id: 'git', where: 'the Git item', from: ' On this machine, run', fromMode: 'inline', to: '...`.', toMode: 'inline' },
		{ id: 'notes', where: 'the notes', from: '### Notes', fromMode: 'line', to: 'scratchpad first.', toMode: 'line' }
	];
	const r = mark(plain, blocks);
	assert.equal(r.placed.length, 2);
	assert.equal(unmark(r.text), plain);
	assert.ok(isMarked(r.text, blocks));
	assert.ok(!isMarked(plain, blocks));
	assert.match(stripLocalOnly(r.text).text, /^- Git\.$/m);
	assert.deepEqual(findLocalStrings(publicText(r.text), LOCAL_STRINGS), []);
	assert.ok(findLocalStrings(plain, LOCAL_STRINGS).length >= 3);
	assert.equal(parseGuide(r.text).sections.find((x) => x.n === 1)!.omitted, 2);
	assert.throws(() => mark(r.text, blocks), /already has local-only markers/);
	assert.throws(() => mark(plain.replace('### Notes', '### Other'), blocks), /block 'notes'/);
	assert.throws(() => mark(`${plain}\n### Notes`, blocks), /block 'notes': its start .* occurs more than once/);
});

// ------------------------------------------------------------------ local-only blocks: the real DEVKIT.md
test('DEVKIT.md: no page text rendered from the sources has a local-machine string', () => {
	const s = fresh();
	const g = parseGuide(s.devkit);
	assert.deepEqual(findLocalStrings(renderIntro(g)), []);
	for (const sec of g.sections) {
		assert.deepEqual(findLocalStrings(renderSection(g, sec).html), [], `section ${sec.n}`);
		assert.deepEqual(findLocalStrings(SECTION_ZH[sec.slug]?.summary ?? ''), [], `summary of section ${sec.n}`);
	}
	const errSec = g.sections.find((x) => x.md.includes('<!-- atlas:generated codes -->'))!;
	for (const code of Object.keys(s.codes)) assert.deepEqual(findLocalStrings(subsection(errSec, code) ?? ''), [], code);
	assert.ok((subsection(errSec, 'AK004') ?? '').startsWith('Git refuses the repo'), 'AK004 keeps its public first sentence');
	assert.deepEqual(findLocalStrings(render(s.contributing ?? '')), []);
	// the guide never names a hidden heading
	assert.ok(!g.sections.some((x) => x.subs.some((h) => /machine notes/i.test(h.text))));
});

test('DEVKIT.md: the sections with machine notes say so on their page', { skip: NO_MARKERS }, () => {
	const g = parseGuide(fresh().devkit);
	assert.deepEqual(g.sections.filter((x) => x.omitted).map((x) => x.n), [1, 11, 13, 17]);
});

test('the public text of DEVKIT.md (no markers, as in a fresh clone) passes the check and renders every anchor', () => {
	const s = fresh();
	const pub = publicText(s.devkit);
	assert.ok(!pub.includes('local-only'));
	assert.deepEqual(staleness({ ...s, devkit: pub }), []);
	const g = parseGuide(pub);
	const broken: string[] = [];
	for (const sec of g.sections) renderSection(g, sec, broken);
	assert.deepEqual(broken, []);
	assert.ok(g.sections.every((x) => x.omitted === 0));
	assert.deepEqual(findLocalStrings(pub), []);
});

test('DEVKIT.md is marked exactly as mark-local.ts marks it with this computer\'s block list', { skip: NO_BLOCKS }, () => {
	const s = fresh();
	assert.ok(isMarked(s.devkit), 'site/DEVKIT.md is not what mark-local.ts makes of it');
	const plain = unmark(s.devkit);
	assert.ok(!plain.includes('local-only'));
	const r = mark(plain);
	assert.equal(r.text, s.devkit);
	assert.equal(r.placed.length, BLOCKS.length);
	assert.equal(unmark(r.text), plain);
	// the unmarked file (main before the merge) fails the check, and says first what to run
	const problems = staleness({ ...s, devkit: plain });
	assert.match(problems[0], /no local-only markers yet: .*mark-local\.ts DEVKIT\.md/);
	assert.ok(problems.some((p) => /section 1 "Set up" has notes about the maintainers' own computer/.test(p)));
	// a block whose words changed is refused, not guessed
	assert.throws(() => mark(plain.replace('### Machine notes (read these before you start)', '### Machine notes')), /block 'machine-notes'/);
});
