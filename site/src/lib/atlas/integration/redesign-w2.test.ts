// Redesign wave W2 (redesign spec §5, §11 R1): the lesson app's contracts that can be tested without a
// browser: hash routing of the lesson views (every legacy anchor), resume, chapter groups, layout mode and density, the
// reading-time formula, the R1 lesson fields (chapters, recap, why, refs) and the new lint warnings AK413-AK418.
// Run from site/:  node --test src/lib/atlas/integration/redesign-w2.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chapterGroups, densityOf, hashOf, layoutMode, lessonMinutes, parseHash, LEGACY } from '../../kit/lesson/views.ts';
import { clearSaved, readSaved, writeSaved } from '../../kit/lesson/resume.ts';
import { compileLesson, typedIndices } from '../lesson/compile.ts';
import { placeholderNames, placeholders, render, splitRef } from '../lesson/narrate.ts';
import { seqManifest } from '../lesson/fixtures/seq.ts';
import { fakeRuntime } from '../runtime/fake.ts';
import type { LessonDef } from '../types/lesson.ts';

const MODEL = ['lesson', 'explore', 'article', 'code', 'training'] as const;
const CONCEPT = ['lesson', 'explore', 'pitfalls', 'sources'] as const;

test('parseHash: the lesson owns #lesson and #lesson-<step>; no hash is the lesson (cover or saved step)', () => {
	assert.deepEqual(parseHash('', { kind: 'model' }), { view: 'lesson' });
	assert.deepEqual(parseHash('#lesson', { kind: 'model' }), { view: 'lesson', top: true });
	assert.deepEqual(parseHash('#lesson-conv', { kind: 'model' }), { view: 'lesson', step: 'conv' });
	assert.deepEqual(parseHash('#tour-a', { kind: 'model', lessonId: 'tour' }), { view: 'lesson', step: 'a' });
	assert.deepEqual(parseHash('#goals', { kind: 'model' }), { view: 'lesson', cover: true });
	assert.deepEqual(parseHash('#input', { kind: 'concept' }), { view: 'lesson', input: true });
	assert.deepEqual(parseHash('#lesson-%E6%AD%A5', { kind: 'model' }), { view: 'lesson', step: '步' });
});

test('parseHash: views and their sections; a view the page lacks falls back to the lesson', () => {
	assert.deepEqual(parseHash('#explore', { kind: 'model', has: MODEL }), { view: 'explore' });
	assert.deepEqual(parseHash('#explore-zoom', { kind: 'model', has: MODEL }), { view: 'explore', section: 'zoom' });
	assert.deepEqual(parseHash('#article', { kind: 'model', has: MODEL }), { view: 'article' });
	assert.deepEqual(parseHash('#code', { kind: 'model', has: ['lesson', 'explore'] }), { view: 'lesson' });
	assert.deepEqual(parseHash('#pitfalls', { kind: 'concept', has: CONCEPT }), { view: 'pitfalls' });
	assert.deepEqual(parseHash('#sources', { kind: 'concept', has: CONCEPT }), { view: 'sources' });
	// a concept page has no Code view: #code is not a view there
	assert.equal(parseHash('#code', { kind: 'concept', has: CONCEPT }), null);
	// an anchor inside a view (not a view or a legacy section) leaves the view alone
	assert.equal(parseHash('#fig-2', { kind: 'model', has: MODEL }), null);
	assert.equal(hashOf({ view: 'explore', section: 'unit' }), '#explore-unit');
	assert.equal(hashOf({ view: 'lesson' }), '#lesson');
});

test('parseHash: every old section anchor opens the view that holds it now (spec §5.9)', () => {
	const m = (h: string) => parseHash(h, { kind: 'model', has: MODEL });
	for (const s of ['overview', 'inspector', 'unit', 'output', 'tryThis', 'checkpoint']) assert.deepEqual(m(`#${s}`), { view: 'explore', section: s }, s);
	assert.deepEqual(m('#credits'), { view: 'training', section: 'credits' });
	assert.deepEqual(m('#links'), { view: 'explore', section: 'links' });
	assert.deepEqual(m('#try-flip'), { view: 'explore', section: 'try-flip' });
	assert.deepEqual(m('#facts'), { view: 'lesson' });
	const c = (h: string) => parseHash(h, { kind: 'concept', has: CONCEPT });
	for (const s of ['toy', 'real', 'compare']) assert.deepEqual(c(`#${s}`), { view: 'explore', section: s }, s);
	assert.deepEqual(c('#post'), { view: 'pitfalls', section: 'post' });
	assert.deepEqual(c('#checkpoint'), { view: 'pitfalls', section: 'checkpoint' });
	assert.deepEqual(c('#links'), { view: 'sources' });
	assert.deepEqual(c('#further'), { view: 'sources' });
	assert.deepEqual(c('#hook'), { view: 'lesson' });
	// every legacy entry points at a view of its kind
	for (const [kind, table] of Object.entries(LEGACY)) for (const [, [v]] of Object.entries(table)) assert.ok((kind === 'model' ? MODEL : CONCEPT).includes(v as never), `${kind}: ${v}`);
});

test('resume: atlas.lesson.<slug> remembers the step; blocked storage never throws', () => {
	const store = new Map<string, string>();
	const g = globalThis as { localStorage?: unknown };
	const old = g.localStorage;
	g.localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v), removeItem: (k: string) => void store.delete(k) };
	try {
		assert.equal(readSaved('lora'), null);
		writeSaved('lora', 'predict');
		assert.deepEqual(readSaved('lora'), { step: 'predict', started: true });
		assert.equal(readSaved('tinyvit'), null);
		clearSaved('lora');
		assert.equal(readSaved('lora'), null);
		store.set('atlas.lesson.bad', '{not json');
		assert.equal(readSaved('bad'), null);
		g.localStorage = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); }, removeItem: () => { throw new Error('blocked'); } };
		assert.equal(readSaved('lora'), null);
		writeSaved('lora', 'x');
		clearSaved('lora');
	} finally {
		g.localStorage = old;
	}
});

test('chapter groups: explicit chapters, consecutive step chapters, and steps outside every chapter', () => {
	assert.deepEqual(chapterGroups(5, { chapters: [{ title: 'A', count: 2 }, { title: 'B', count: 2 }] }), [
		{ title: 'A', from: 0, to: 1 },
		{ title: 'B', from: 2, to: 3 },
		{ from: 4, to: 4 }
	]);
	const ch = ['x', 'x', undefined, 'y', 'y'];
	assert.deepEqual(chapterGroups<string>(5, { stepChapter: (i) => ch[i] }), [
		{ title: 'x', from: 0, to: 1 },
		{ from: 2, to: 2 },
		{ title: 'y', from: 3, to: 4 }
	]);
	assert.deepEqual(chapterGroups(3, {}), [{ from: 0, to: 2 }]);
	// I18n chapters compare by value
	assert.equal(chapterGroups<{ en: string }>(2, { stepChapter: () => ({ en: 'Same' }) }).length, 1);
});

test('layout mode and density (spec §5.1, §5.4); the minutes formula (spec §6.2)', () => {
	assert.equal(layoutMode(1280, 800), 'app');
	assert.equal(layoutMode(1280, 680), 'app');
	assert.equal(layoutMode(1024, 768), 'app');
	assert.equal(layoutMode(959, 900), 'flow');
	assert.equal(layoutMode(1280, 599), 'flow');
	assert.equal(layoutMode(390, 844), 'flow');
	assert.equal(densityOf(800, 700), 'roomy');
	assert.equal(densityOf(800, 600), 'regular');
	assert.equal(densityOf(560, 700), 'compact');
	assert.equal(densityOf(800, 470), 'compact');
	assert.equal(densityOf(900, 900, true), 'compact');
	assert.equal(lessonMinutes({ words: 540, steps: 10, beats: 2 }), 7);
	assert.equal(lessonMinutes({ words: 0, steps: 1, beats: 0 }), 1);
});

const step = (id: string, extra: Record<string, unknown> = {}) => ({ id, title: { en: id, zh: id }, narration: { en: 'One. Two.', zh: '一。二。' }, source: 'fwd', view: 'tape.table', provenance: 'live' as const, ...extra });
const lessonOf = (steps: unknown[], more: Record<string, unknown> = {}): LessonDef => ({ title: { en: 'T', zh: 'T' }, summary: { en: 's', zh: 's' }, steps, ...more }) as LessonDef;
const compile = (l: LessonDef, manifest = seqManifest) => compileLesson(l, { spec: null, ops: [], manifest, plan: [] });

test('R1 chapters: compiled in lesson order; AK413 warns on counts, unknown, missing, doubled and out-of-order steps', () => {
	const ok = compile(lessonOf([step('a'), step('b'), step('c'), step('d')], { chapters: [{ id: 'one', title: { en: 'One', zh: '一' }, steps: ['a', 'b'] }, { id: 'two', title: { en: 'Two', zh: '二' }, steps: ['c', 'd'] }] }));
	assert.deepEqual(ok.chapters?.map((c) => c.steps), [['a', 'b'], ['c', 'd']]);
	assert.ok(!ok.issues.some((i) => i.code === 'AK413'), JSON.stringify(ok.issues.filter((i) => i.code === 'AK413')));
	const bad = compile(lessonOf([step('a'), step('b'), step('c')], { chapters: [{ id: 'one', title: { en: 'One', zh: '' }, steps: ['b', 'zz'] }, { id: 'two', title: { en: 'Two', zh: '二' }, steps: ['a', 'b'] }] }));
	const msgs = bad.issues.filter((i) => i.code === 'AK413').map((i) => i.message).join('\n');
	for (const want of [/no zh title/, /names step 'zz'/, /in no chapter: c/, /in two chapters: b/, /out of lesson order/]) assert.match(msgs, want);
	assert.ok(bad.issues.filter((i) => i.code === 'AK413').every((i) => i.level === 'warn'));
	assert.equal(compile(lessonOf([step('a')])).chapters, undefined);
});

test('R1 why, recap and refs compile; AK414-AK418 warn (never fail) on the content caps', () => {
	const long = Array.from({ length: 70 }, (_, i) => `w${i + 1}`).join(' ');
	const c = compile(
		lessonOf(
			[
				step('a', { why: { q: { en: 'Why is the map smaller?', zh: '为什么图变小了？' }, a: { en: 'Because {n} cells.', zh: '因为 {n} 个格子。', vars: () => ({ n: 26 }) } }, refs: () => ({ cell: { field: 'out', index: [1, 2] } }) }),
				step('b', { narration: { en: `${long}. End.`, zh: '长。完。' }, look: { en: Array.from({ length: 30 }, () => 'x').join(' '), zh: '看' } }),
				step('c', { title: { en: 'W0 is frozen', zh: 'W0 冻结' }, why: { q: { en: 'Not a question', zh: '不是问题' }, a: { en: 'ok', zh: '好' } } })
			],
			{ recap: [{ en: 'Seen {n}.', zh: '看到 {n}。', vars: () => ({ n: 3 }) }] }
		),
		{ ...seqManifest, page: { ...seqManifest.page, poster: { step: 'nope' } } } as never
	);
	const rt = fakeRuntime();
	assert.equal(c.steps[0].why?.a(rt, { index: 0 }).en, 'Because 26 cells.');
	assert.deepEqual(c.steps[0].refs?.(rt, { index: 0 }), { cell: { field: 'out', index: [1, 2] } });
	assert.equal(c.recap?.[0](rt, { index: 0 }).en, 'Seen 3.');
	const codes = (k: string) => c.issues.filter((i) => i.code === k);
	assert.equal(codes('AK415').length, 1, 'narration over 60 words');
	assert.equal(codes('AK414').length, 1, 'look over 25 words');
	assert.ok(codes('AK416').some((i) => /W0/.test(i.message)), 'W0 typed as a word');
	assert.ok(codes('AK417').some((i) => /question/.test(i.message)), 'a Why summary that is no question');
	assert.ok(codes('AK418').some((i) => /nope/.test(i.message)), 'poster names no step');
	for (const k of ['AK413', 'AK414', 'AK415', 'AK416', 'AK417', 'AK418']) assert.ok(codes(k).every((i) => i.level === 'warn'), k);
	assert.deepEqual(typedIndices('W0 and x0 but not W₀ or {W0} or B0b'), ['W0', 'x0']);
});

test('R1 number chips: {name@ref} prints as {name} and keeps the ref', () => {
	assert.deepEqual(splitRef('y@cell'), { name: 'y', ref: 'cell' });
	assert.deepEqual(splitRef('fact:a@b'), { name: 'fact:a@b' });
	assert.deepEqual(splitRef('y'), { name: 'y' });
	assert.deepEqual(placeholders('{y@cell|n3} and {z}'), [{ name: 'y', fmt: 'n3', ref: 'cell' }, { name: 'z', fmt: undefined }]);
	assert.deepEqual(placeholderNames('{y@cell} {y}'), ['y']);
	const rt = fakeRuntime();
	const r = render({ en: 'y = {y@cell}.', zh: 'y = {y@cell}。', vars: () => ({ y: 1.5 }) }, rt, { index: 0 }, 'en');
	assert.equal(r.text, 'y = 1.50.');
	assert.deepEqual(r.missing, []);
});
