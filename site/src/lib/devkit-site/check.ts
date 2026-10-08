// The staleness check of the DevKit pages (/devkit/). The pages are rendered from the kit's own sources at build time,
// so they follow DEVKIT.md, ops.json, the error list and hello-xor by themselves. What cannot follow by itself is the
// little the pages add by hand: the Chinese section summaries, the Chinese meanings of the error codes, a demo for each
// op, the tutorial's excerpt anchors and commands, the landing page's examples. When a source changes under one of them,
// this check fails, and with it the build (the route's server load throws) and devkit-site.test.ts, naming the fix.
// It also fails when a note about the maintainers' own computer (a local path, "this machine", ...; local.ts) sits in
// DEVKIT.md or CONTRIBUTING.md outside local-only markers: such notes never reach the public pages.
//
//   node src/lib/devkit-site/check.ts            run the check on this site folder (exit 1 on a problem)
//   node src/lib/devkit-site/check.ts --hashes   print each guide section's current hash (for summaries.ts)
import { firstTable, parseGuide, sectionHash, subsection, type Guide } from './guide.ts';
import { CONTRIBUTING_STATUS_ZH, CONTRIBUTING_ZH, SECTION_ZH } from './summaries.ts';
import { CODE_GROUPS, CODE_ZH, groupOf } from './codes.ts';
import { OP_DEMOS, opLayer, opStage, zoomKeyOf } from './opdemos.ts';
import { HELLO, TUTORIAL, excerpt, stepCommand } from './tutorial.ts';
import { WORKFLOW, afternoonResult, commandTable, contributingStart, portTable, quickStart } from './home.ts';
import type { Sources } from './sources.ts';
import { findLocalStrings, LOCAL_END, LOCAL_START, publicText } from './local.ts';
import { machine, MACHINE_FILE } from './machine.ts';

export function staleness(s: Sources, g: Guide = parseGuide(s.devkit)): string[] {
	const out: string[] = [];
	const p = (m: string) => out.push(m);

	// ---- notes about the maintainers' own computer: only inside local-only markers (local.ts)
	const markersYet = s.devkit.includes(LOCAL_START);
	// this computer's block list (the git-ignored machine.local.json): mark-local.ts can wrap the notes it knows
	const blockList = machine().blocks.length > 0;
	const howToWrap = `wrap them in ${LOCAL_START} ... ${LOCAL_END}${blockList ? ` (the blocks listed in ${MACHINE_FILE}: from site/, node src/lib/devkit-site/mark-local.ts DEVKIT.md)` : ''}`;
	const leaking = new Set<string>();
	const introHits = findLocalStrings(g.intro);
	if (introHits.length) p(`DEVKIT.md's introduction has notes about the maintainers' own computer outside local-only markers (${introHits[0]}): ${howToWrap}`);
	for (const sec of g.sections) {
		const hits = findLocalStrings(sec.md);
		if (!hits.length) continue;
		leaking.add(sec.slug);
		p(`DEVKIT.md section ${sec.n} "${sec.title}" has notes about the maintainers' own computer outside local-only markers (${hits.length} hit(s), first ${hits[0]}): ${howToWrap}`);
	}
	const contribHits = s.contributing ? findLocalStrings(publicText(s.contributing)) : [];
	if (contribHits.length) p(`CONTRIBUTING.md has notes about the maintainers' own computer outside local-only markers (${contribHits[0]}): ${howToWrap}`);

	// ---- the guide's Chinese summaries (summaries.ts)
	const slugs = new Set(g.sections.map((x) => x.slug));
	for (const sec of g.sections) {
		const zh = SECTION_ZH[sec.slug];
		if (leaking.has(sec.slug)) continue; // its hash is the one of its public text: wrap the notes first
		if (!zh) p(`DEVKIT.md section ${sec.n} "${sec.title}" has no Chinese summary: add SECTION_ZH['${sec.slug}'] to src/lib/devkit-site/summaries.ts with hash '${sec.hash}'`);
		else if (zh.hash !== sec.hash)
			p(`DEVKIT.md section ${sec.n} "${sec.title}" changed since its Chinese summary was written: re-read it, update SECTION_ZH['${sec.slug}'] in src/lib/devkit-site/summaries.ts if needed, and set its hash to '${sec.hash}'`);
	}
	for (const k of Object.keys(SECTION_ZH)) if (!slugs.has(k)) p(`summaries.ts has a summary for '${k}', which is no section of DEVKIT.md any more: remove it (or rename it to the section's new slug)`);

	// ---- error codes (codes.ts) and their explanations in DEVKIT.md section 13
	const errSec = g.sections.find((x) => x.md.includes('<!-- atlas:generated codes -->'));
	if (!errSec) p('DEVKIT.md has no generated error-code table (<!-- atlas:generated codes -->)');
	for (const [code, en] of Object.entries(s.codes)) {
		const zh = CODE_ZH[code];
		if (!zh) p(`error code ${code} ("${en}") has no Chinese meaning: add it to CODE_ZH in src/lib/devkit-site/codes.ts`);
		else if (zh.en !== en) p(`the meaning of ${code} changed to "${en}" (was "${zh.en}"): update CODE_ZH['${code}'] in src/lib/devkit-site/codes.ts`);
		if (!CODE_GROUPS[groupOf(code)]) p(`error code ${code} starts a new group (${groupOf(code)}xx): add it to CODE_GROUPS in src/lib/devkit-site/codes.ts`);
		if (errSec && subsection(errSec, code) === null) p(`DEVKIT.md section ${errSec.n} has no "### ${code}" explanation`);
	}
	for (const code of Object.keys(CODE_ZH)) if (!(code in s.codes)) p(`codes.ts translates ${code}, which scripts/atlas/errors.ts no longer has: remove it`);

	// ---- ops (schema/ops.json): every op has a demo that runs, drawn by a kit view
	for (const o of s.ops) {
		if (!OP_DEMOS[o.op]) {
			p(`op '${o.op}' (schema/ops.json) has no demo: add a demo layer for it to OP_DEMOS in src/lib/devkit-site/opdemos.ts`);
			continue;
		}
		if (!s.kitViews.includes(zoomKeyOf(o.zoom))) p(`op '${o.op}' asks for view '${zoomKeyOf(o.zoom)}', which is no kit view`);
		try {
			const l = opLayer(o.op);
			if (!l) p(`the demo layer of op '${o.op}' (${OP_DEMOS[o.op].model}/${OP_DEMOS[o.op].layer}) does not exist`);
			else if (l.layer.op !== o.op) p(`the demo layer of op '${o.op}' is a '${l.layer.op}' layer`);
			else if (!opStage(o.op) || opStage(o.op)!.at.index < 0) p(`the demo of op '${o.op}' finds no frame`);
		} catch (e) {
			p(`the demo of op '${o.op}' fails: ${(e as Error).message}`);
		}
	}
	for (const op of Object.keys(OP_DEMOS)) if (!s.ops.some((o) => o.op === op)) p(`opdemos.ts has a demo for '${op}', which schema/ops.json no longer has: remove it`);

	// ---- the tutorial: excerpts of the real hello-xor files, commands of DEVKIT.md section 4
	const s4 = g.sections.find((x) => x.n === 4);
	if (!s4) p('DEVKIT.md has no section 4 (the walk-through the tutorial follows)');
	for (const st of TUTORIAL) {
		if (st.file) {
			const text = s.files[st.file.path];
			if (text == null) p(`tutorial step '${st.id}': ${st.file.path} is missing`);
			else if (!excerpt(text, st.file)) p(`tutorial step '${st.id}': ${st.file.path} no longer has the lines between "${st.file.from}" and "${st.file.to ?? st.file.before}": move the anchors in src/lib/devkit-site/tutorial.ts`);
		}
		if (st.command && s4 && !stepCommand(subsection(s4, st.command.section4), st.command))
			p(`tutorial step '${st.id}': DEVKIT.md ${st.command.section4} has no command matching /${st.command.re}/ any more: update src/lib/devkit-site/tutorial.ts`);
	}
	const ports = portTable(g);
	if (!(HELLO in ports)) p(`the port table of DEVKIT.md has no '${HELLO}' row (the tutorial's Lesson Lab step reads it)`);

	// ---- the landing page
	const cmds = commandTable(g);
	if (!Object.keys(cmds).length) p('DEVKIT.md has no generated command table (<!-- atlas:generated commands -->)');
	for (const w of WORKFLOW) for (const c of w.cmds) if (!cmds[c]) p(`the workflow step '${w.id}' names 'atlas ${c}', which the command table of DEVKIT.md does not have`);
	if (!quickStart(g).length) p('DEVKIT.md section 4 has no command blocks for the quick start');
	if (!afternoonResult(s.afternoon)) p('research/devkit/afternoon-test.md is missing or has no "… to draft gates green: **<m> min <s> s**" summary line');
	for (const [slug, status] of Object.entries(s.examples)) if (status !== 'live') p(`the landing page's example lesson '${slug}' is ${status ?? 'not in the catalog'}, not live: pick another example in src/lib/devkit-site/home.ts`);

	// ---- the contribute page
	if (!s.contributing) p('CONTRIBUTING.md is missing');
	else if (sectionHash(s.contributing) !== CONTRIBUTING_ZH.hash)
		p(`CONTRIBUTING.md changed since its Chinese summary was written: re-read it, update CONTRIBUTING_ZH (and CONTRIBUTING_STATUS_ZH) in src/lib/devkit-site/summaries.ts if needed, and set its hash to '${sectionHash(s.contributing)}'`);
	if (s.contributing) {
		const st = contributingStart(s.contributing);
		if (st.status && !CONTRIBUTING_STATUS_ZH) p('CONTRIBUTING.md has a status note but CONTRIBUTING_STATUS_ZH in src/lib/devkit-site/summaries.ts is empty: translate it');
		if (!st.status && CONTRIBUTING_STATUS_ZH) p("CONTRIBUTING.md has no status note any more: set CONTRIBUTING_STATUS_ZH in src/lib/devkit-site/summaries.ts to ''");
		if (!st.setup) p('CONTRIBUTING.md has no "## Setup" list (the start of /devkit/ shows it)');
	}
	if (!s.design || !/^### 6\.2 /m.test(s.design)) p('research/devkit/DEVKIT_DESIGN.md has no "### 6.2" checklist');
	const gates = g.sections.find((x) => /^checks/i.test(x.title));
	if (!gates || !firstTable(gates.md)) p('DEVKIT.md has no "Checks and gates" section with its gate table');
	// a DEVKIT.md without any marker on a computer that has a block list (its notes are not wrapped yet): say first what
	// to do. A fresh clone has neither markers nor a block list: its DEVKIT.md is the public text.
	if (out.length && !markersYet && blockList)
		out.unshift("DEVKIT.md has no local-only markers yet: from site/, run node src/lib/devkit-site/mark-local.ts DEVKIT.md first (the Chinese summaries' hashes are those of the marked file, without the maintainers' machine notes)");
	return out;
}

if (process.argv[1] && /check\.ts$/.test(process.argv[1])) {
	const { readSources } = await import('./sources.ts');
	const s = readSources(process.cwd());
	const g = parseGuide(s.devkit);
	if (process.argv.includes('--hashes')) {
		for (const x of g.sections) console.log(`${x.slug.padEnd(46)} ${x.hash}${SECTION_ZH[x.slug]?.hash === x.hash ? '' : '   (summaries.ts differs)'}`);
	} else {
		const problems = staleness(s, g);
		for (const m of problems) console.log(`STALE ${m}`);
		console.log(problems.length ? `devkit-site: ${problems.length} stale item(s)` : 'devkit-site: the DevKit pages are up to date with their sources');
		process.exitCode = problems.length ? 1 : 0;
	}
}
