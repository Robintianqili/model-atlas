// atlas review <slug> [--fixtures a,b,c]   (design §4.12, §6.4)
// Writes notes/review-<slug>.md, the teaching reviewer's packet:
//   - the 10-item checklist of design §6.4 (ticks and the "Reviewer notes" section survive a re-run)
//   - the gate summary from the last `atlas check` (.atlas/<slug>/check.json), or a fresh `check --fast`; K4: it says
//     which mode that check ran in and which gates it did not run (a --quick or --fast check skips site and e2e)
//   - K4: the catalog patch `promote --dry-run` prints (the entry's fields from manifest.catalog, the tech-zh and glossary
//     rows, the checks by hand): the reviewer edits manifest.catalog, not the packet, and `promote --verify` checks it
//   - the e2e contact sheets, lesson print and axe summary (shotsDir: notes/shots/<slug>/, or the lane folder's shots/ for
//     a lane outside the repo, K4), then the screenshot list
//   - every step's narration resolved on 3 inputs (presets first, then lesson and edge fixtures), in en and zh
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';
import type { Narration } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import { posix, readJson, writeText } from '../lib/io.ts';
import { openPackage, today } from '../lib/pkg.ts';
import { roots, shotsDir as shotsFolder } from '../root.ts';
import type { Gate } from './check.ts';

export const CHECKLIST = [
	'Measurable goals: each goal says what the learner can do after the page.',
	'One idea per step, each with a focused "look at" hint.',
	'No term is used before it is defined (glossary terms on first use).',
	'Narration numbers match the screen on 3 presets (the tables below are what the page renders).',
	'Every factual sentence is live or covered by a claim test (claims.test.ts).',
	'The first occurrence of an idea is worked out fully; repeats are summarized.',
	'A misconception or limits step (kindOf: limits | breakIt) tells the honest limit.',
	'Try-this experiments ask for a prediction first.',
	'Natural Chinese with glossary terms (no machine-translation tone, no English left over).',
	'Credits are correct and complete (static/licenses.json rows).'
];

const cell = (s: string) => s.replace(/\|/g, '\\|').replace(/\n/g, ' ');

function keepSection(old: string, title: string): string | null {
	const m = new RegExp(`^## ${title}\\s*$([\\s\\S]*?)(?=^## |(?![\\s\\S]))`, 'm').exec(old);
	return m ? m[1].trim() : null;
}

export async function main(args: string[]): Promise<number> {
	const pos = args.filter((a) => !a.startsWith('--'));
	const fi = args.indexOf('--fixtures');
	const want = fi >= 0 ? (args[fi + 1] ?? '').split(',').filter(Boolean) : null;
	if (fi >= 0) pos.splice(pos.indexOf(args[fi + 1]), 1);
	if (pos.length !== 1) throw new AtlasError('AK301', 'usage: atlas review <slug> [--fixtures a,b,c]');
	const slug = pos[0];
	const r = roots();
	const o = await openPackage(slug);
	const file = join(r.root, 'notes', `review-${slug}.md`);
	const old = existsSync(file) ? readFileSync(file, 'utf8') : '';

	// gate summary
	let gates: Gate[] | null = null;
	let checked = '';
	const cj = join(r.site, '.atlas', slug, 'check.json');
	if (existsSync(cj)) {
		const j = readJson<{ gates: Gate[]; checked?: string; mode?: string }>(cj);
		gates = j.gates;
		const skipped = j.gates.filter((g) => g.status === 'skip').map((g) => g.name);
		checked = `the ${j.mode ? `${j.mode} ` : ''}check of ${(j.checked ?? '').slice(0, 16).replace('T', ' ')}${skipped.length ? ` (not run: ${skipped.join(', ')}; the full check runs every gate)` : ''}`;
	} else {
		const { checkPackage } = await import('./check.ts');
		const res = await checkPackage(slug, { fast: true });
		gates = res.gates;
		checked = `${new Date().toISOString()} (check --fast, run by review)`;
	}

	// narration tables on 3 inputs
	const { compiled } = await import('./lesson.ts');
	const { c } = await compiled(o);
	const all = await o.fixtures();
	const pick = want ? all.filter((f) => want.includes(f.id)) : [...all.filter((f) => f.kind === 'preset'), ...all.filter((f) => f.kind === 'lesson'), ...all.filter((f) => f.kind === 'edge')].slice(0, 3);
	const tables: string[] = [];
	for (const fx of pick.length ? pick : [{ id: 'default', label: { en: 'the default input', zh: '默认输入' }, raw: undefined, kind: 'default' }]) {
		const rt = await o.runtime(fx.raw);
		tables.push(`### ${fx.id}: ${fx.label.en}${fx.label.zh ? ` / ${fx.label.zh}` : ''}`, '', '| # | step | en | zh |', '|---|---|---|---|');
		for (const [i, s] of c.steps.entries()) {
			const pos2 = await o.kit.sources.stepAt(rt, s);
			if (pos2.skipped) {
				tables.push(`| ${i + 1} | \`${s.id}\` | (empty input: the adapter's note) | （空输入：显示适配器的提示） |`);
				continue;
			}
			const en = o.kit.narrate.render(s.src.narration as Narration<never>, rt as never, pos2.at, 'en').text;
			const zh = o.kit.narrate.render(s.src.narration as Narration<never>, rt as never, pos2.at, 'zh').text;
			tables.push(`| ${i + 1} | \`${s.id}\` ${cell(s.title.en)} | ${cell(en)} | ${cell(zh)} |`);
		}
		tables.push('');
		rt.close?.();
	}

	const shotsDir = shotsFolder(slug, r);
	const shotsRel = posix(relative(r.root, shotsDir));
	// K4: the catalog patch, as promote --dry-run prints it (nothing is written)
	let catalog = '';
	try {
		const { planPromote } = await import('./promote.ts');
		const plan = await planPromote(slug);
		catalog = [
			`What \`npm run atlas -- promote ${slug} --dry-run\` prints today. The entry's fields come from \`manifest.catalog\` in explainer.ts (name, family, inputs, mode, hook, shows, runtime, params, size; a concept: title, interaction, engine, mode, links): change them there, then \`promote ${slug} --verify\` applies the patch to a copy of the catalog and runs check:catalog.`,
			'',
			'```ts',
			plan.patch.trimEnd(),
			'```',
			...(plan.notes.length ? ['', 'Check by hand:', ...plan.notes.map((n) => `- ${cell(n)}`)] : [])
		].join('\n');
	} catch (e) {
		catalog = `No patch: ${(e as Error).message.split('\n')[0]}`;
	}
	const shots = existsSync(shotsDir) ? readdirSync(shotsDir).filter((f) => /\.(png|jpe?g|webp)$/i.test(f)).sort() : [];
	// K2: the e2e contact sheets, the lesson print and the axe summary come first in the packet
	const contactDir = join(shotsDir, 'contact');
	const contact = existsSync(contactDir) ? readdirSync(contactDir).filter((f) => /\.(png|txt)$/i.test(f)).sort((a, b) => Number(b.endsWith('.png')) - Number(a.endsWith('.png')) || a.localeCompare(b)) : [];
	const axeSummary = existsSync(join(contactDir, 'axe.txt')) ? readFileSync(join(contactDir, 'axe.txt'), 'utf8').split('\n').filter((l) => /^\S/.test(l) && l.includes(': ')).slice(1) : [];
	const checklist = keepSection(old, 'Checklist') ?? CHECKLIST.map((t, i) => `- [ ] ${i + 1}. ${t}`).join('\n');
	const notes = keepSection(old, 'Reviewer notes') ?? 'Verdict: (ship / fix first). Findings, each with the step id and the fix made inside the package:';
	const m = o.l.pkg.manifest;
	const text = [
		`# Review: ${slug}`,
		'',
		`Generated by \`npm run atlas -- review ${slug}\` on ${today()} (quality: ${m.quality}${m.devOnly ? ', devOnly' : ''}). Re-running it refreshes every section except the checklist and the reviewer notes.`,
		'The reviewer uses this file and the Lesson Lab, edits only this package, and sets `quality: \'reviewed\'` in explainer.ts when the verdict is "ship".',
		'',
		'## Checklist',
		'',
		checklist,
		'',
		'## Reviewer notes',
		'',
		notes,
		'',
		'## Gates',
		'',
		`From ${checked || 'the last check'}:`,
		'',
		'| gate | status | summary |',
		'|---|---|---|',
		...(gates ?? []).map((g) => `| ${g.name} | ${g.status} | ${cell(g.summary)} |`),
		'',
		...(gates ?? []).flatMap((g) => g.issues.map((i) => `- ${i.code} (${i.level}) [${g.name}] ${cell(i.message)}`)),
		'',
		'## Catalog patch',
		'',
		catalog,
		'',
		'## Contact sheets',
		'',
		contact.length
			? [
					"From the last `atlas e2e` (or full check): one sheet per pass with every step, the lesson print (each step's title, narration and numbers per language) and the accessibility scans. Read these first; then click only where they raise a question.",
					'',
					...contact.map((s) => `- ${shotsRel}/contact/${s}`),
					...(axeSummary.length ? ['', 'axe (contact/axe.txt):', ...axeSummary.map((l) => `- ${cell(l)}`)] : [])
				].join('\n')
			: `None yet: run \`npm run atlas -- e2e ${slug}\` (it writes ${shotsRel}/contact/).`,
		'',
		'## Screenshots',
		'',
		shots.length ? shots.map((s) => `- ${shotsRel}/${s}`).join('\n') : `None yet: run \`npm run atlas -- e2e ${slug}\`.`,
		'',
		`## Narration on ${Math.min(3, pick.length || 1)} inputs`,
		'',
		`${c.steps.length} steps. The same tables in one language: \`npm run atlas -- lesson ${slug} print --lang zh\`.`,
		'',
		...tables
	].join('\n');
	writeText(file, text + '\n');
	console.log(`atlas review ${slug}: wrote ${posix(file)} (${c.steps.length} steps x ${pick.length || 1} inputs, ${shots.length} screenshots, ${contact.length} contact files, ${(gates ?? []).length} gates)`);
	return 0;
}
