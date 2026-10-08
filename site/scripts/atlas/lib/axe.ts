// Accessibility scans in `atlas e2e` (K2 speed pack): axe-core through @axe-core/playwright (MPL-2.0, a dev dependency).
//
// Every lesson step of every pass is scanned against the WCAG 2.1 A and AA rules (axe tags wcag2a, wcag2aa, wcag21a,
// wcag21aa): the whole page on the first step of a pass, the lesson section (section.wt) on the others (only the lesson
// changes between steps). Serious and critical violations fail the e2e gate (AK604); moderate and minor ones warn.
// Results: contact/axe-<pass>.json (every violation with its nodes) and contact/axe.txt (a per-step summary).
// Long tasks during a scan are axe's own work, not the page's: e2e leaves them out of AK603 (the scan windows).
import type { Page } from 'playwright-core';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { AtlasError, type AkCode } from '../errors.ts';
import type { Issues } from './pkg.ts';

/** The accessibility code (K2; errors.ts CODES, DEVKIT.md section 13). */
export const AK_A11Y: AkCode = 'AK604';

export const AXE_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];
export type Impact = 'critical' | 'serious' | 'moderate' | 'minor';
export const BLOCKING: Impact[] = ['critical', 'serious'];

export interface AxeNode {
	target: string[];
	html: string;
	summary: string;
}
export interface AxeViolation {
	id: string;
	impact: Impact;
	help: string;
	helpUrl: string;
	nodes: AxeNode[];
}
export interface StepScan {
	pass: string;
	step: number;
	steps: number;
	title: string;
	scope: 'page' | 'lesson';
	ms: number;
	violations: AxeViolation[];
	/** the scan itself failed (axe threw) */
	error?: string;
}

type AxeBuilderCtor = new (o: { page: Page; axeSource?: string }) => {
	withTags(t: string[]): unknown;
	include(sel: string): unknown;
	analyze(): Promise<{ violations: { id: string; impact?: string | null; help: string; helpUrl: string; nodes: { target: unknown[]; html: string; failureSummary?: string }[] }[] }>;
};

/** @axe-core/playwright's AxeBuilder, as the site resolves it; AK001 with the fix when it is missing. */
export function loadAxe(site: string): AxeBuilderCtor {
	const req = createRequire(join(site, 'package.json'));
	try {
		const m = req('@axe-core/playwright') as { default?: AxeBuilderCtor; AxeBuilder?: AxeBuilderCtor };
		const C = m.default ?? m.AxeBuilder;
		if (!C) throw new Error('no AxeBuilder export');
		return C;
	} catch (e) {
		throw new AtlasError('AK001', `@axe-core/playwright is not installed in the site (${(e as Error).message.split('\n')[0]})`, { fix: 'npm install (it is a devDependency of site/package.json); or skip the scans while you iterate with --axe off' });
	}
}

/** One scan: the whole page, or only `include`. */
export async function scan(Axe: AxeBuilderCtor, page: Page, include?: string): Promise<{ violations: AxeViolation[]; ms: number }> {
	const t0 = performance.now();
	// axe-core's source (~0.5 MB) is evaluated in the page twice per scan; once the page has window.axe (the lesson steps
	// never navigate), a stub source keeps the injected copy
	const injected = await page.evaluate(() => !!(window as unknown as { axe?: { run?: unknown } }).axe?.run);
	const b = new Axe(injected ? { page, axeSource: 'void 0' } : { page }) as unknown as { withTags(t: string[]): typeof b; include(s: string): typeof b; setLegacyMode?(on: boolean): typeof b; analyze(): ReturnType<InstanceType<AxeBuilderCtor>['analyze']> };
	// legacy mode runs axe.run in the page itself; the default mode opens a blank page in the pass's browser context for
	// every scan (finishRun), which costs time and is one more page next to the lesson
	let q = b.withTags(AXE_TAGS);
	if (q.setLegacyMode) q = q.setLegacyMode(true);
	if (include) q = q.include(include);
	const res = await q.analyze();
	const violations = res.violations.map((v) => ({
		id: v.id,
		impact: ((v.impact ?? 'minor') as Impact),
		help: v.help,
		helpUrl: v.helpUrl,
		nodes: v.nodes.map((n) => ({ target: n.target.map((t) => (Array.isArray(t) ? t.join(' >>> ') : String(t))), html: n.html.slice(0, 240), summary: (n.failureSummary ?? '').split('\n').slice(0, 3).join(' ').slice(0, 300) }))
	}));
	return { violations, ms: performance.now() - t0 };
}

const steps = (xs: number[]) => {
	const s = [...new Set(xs)].sort((a, b) => a - b);
	const out: string[] = [];
	for (let i = 0; i < s.length; i++) {
		let j = i;
		while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
		out.push(j > i + 1 ? `${s[i]}-${s[j]}` : j === i + 1 ? `${s[i]}, ${s[j]}` : `${s[i]}`);
		i = j;
	}
	return out.join(', ');
};

/** Gate issues: one per rule (AK604 error for serious/critical, warning for moderate/minor), with the steps per pass. */
export function axeIssues(scans: StepScan[], is: Issues, slug?: string): { blocking: number; warnings: number; nodes: number } {
	const byRule = new Map<string, { v: AxeViolation; nodes: number; where: Map<string, number[]>; targets: string[] }>();
	for (const s of scans) {
		if (s.error) is.warn(AK_A11Y, `${slug ? `${slug}: ` : ''}${s.pass}, step ${s.step}/${s.steps}: the axe scan failed: ${s.error}`);
		for (const v of s.violations) {
			const k = `${v.id}|${v.impact}`;
			const e = byRule.get(k) ?? { v, nodes: 0, where: new Map<string, number[]>(), targets: [] as string[] };
			e.nodes += v.nodes.length;
			e.where.set(s.pass, [...(e.where.get(s.pass) ?? []), s.step]);
			for (const n of v.nodes) if (e.targets.length < 3 && !e.targets.includes(n.target.join(' '))) e.targets.push(n.target.join(' '));
			byRule.set(k, e);
		}
	}
	let blocking = 0;
	let warnings = 0;
	let nodes = 0;
	const order = (i: Impact) => ['critical', 'serious', 'moderate', 'minor'].indexOf(i);
	for (const e of [...byRule.values()].sort((a, b) => order(a.v.impact) - order(b.v.impact))) {
		const block = BLOCKING.includes(e.v.impact);
		const where = [...e.where].map(([p, st]) => `${p} step${st.length > 1 ? 's' : ''} ${steps(st)}`).join('; ');
		const msg = `${slug ? `${slug}: ` : ''}axe ${e.v.id} (${e.v.impact}): ${e.v.help}; ${e.nodes} node(s) on ${where}`;
		const info = { where: e.targets.join(' | '), fix: `${e.v.helpUrl} ; per step: notes/shots/<slug>/contact/axe.txt` };
		if (block) {
			blocking++;
			is.error(AK_A11Y, msg, info);
		} else {
			warnings++;
			is.warn(AK_A11Y, msg, info);
		}
		nodes += e.nodes;
	}
	return { blocking, warnings, nodes };
}

/** contact/axe.txt: every step of every pass, its violations by impact (or "clean"). */
export function axeText(slug: string, scans: StepScan[]): string {
	const out = [`axe accessibility scan of ${slug}: WCAG 2.1 A + AA (${AXE_TAGS.join(', ')}); serious and critical fail the gate (AK604), moderate and minor warn`, ''];
	let pass = '';
	for (const s of scans) {
		if (s.pass !== pass) {
			pass = s.pass;
			const ps = scans.filter((x) => x.pass === pass);
			const n = ps.reduce((a, x) => a + x.violations.filter((v) => BLOCKING.includes(v.impact)).length, 0);
			out.push(`${pass}: ${ps.length} step(s), ${n} blocking violation(s); ${Math.round(ps.reduce((a, x) => a + x.ms, 0))} ms of scanning`);
		}
		const head = `  step ${String(s.step).padStart(2)}/${s.steps} (${s.scope === 'page' ? 'whole page' : 'lesson'})${s.title ? ` ${s.title}` : ''}:`;
		if (s.error) {
			out.push(`${head} scan failed: ${s.error}`);
			continue;
		}
		if (!s.violations.length) {
			out.push(`${head} clean`);
			continue;
		}
		out.push(head);
		for (const v of s.violations) out.push(`      ${BLOCKING.includes(v.impact) ? 'FAIL' : 'warn'} ${v.id} (${v.impact}) x${v.nodes.length}: ${v.help}; ${v.nodes.slice(0, 3).map((n) => n.target.join(' ')).join(' | ')}`);
	}
	return out.join('\n') + '\n';
}
