// The files the DevKit pages are generated from, read at build time (Node fs; the route's server load and
// devkit-site.test.ts both use it, so the test checks exactly what the build renders).
//   site/DEVKIT.md                         the guide (its generated tables follow the code: docs.ts --check)
//   CONTRIBUTING.md                        the contributor path
//   research/devkit/DEVKIT_DESIGN.md       §6.2, the checklist
//   research/devkit/afternoon-test.md      the afternoon test (its public record)
//   site/src/lib/atlas/schema/ops.json     the ops (generated from CORE_OPS)
//   site/scripts/atlas/errors.ts           the error codes
//   site/src/lib/atlas/ui/views/<key>/     the kit views
//   the hello-xor files the tutorial cuts its excerpts from
// The repository root is found as the kit finds it (scripts/atlas/root.ts): the nearest ancestor of the site folder with
// CONTRACTS.md and pipeline/, so a lane copy (site/.work/<name>/site) reads the same root files as the main tree.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { CODES } from '../../../scripts/atlas/errors.ts';
import { findRoot } from '../../../scripts/atlas/root.ts';
import { MODEL_BY_SLUG } from '../catalog/models.ts';
import { KINDS } from './home.ts';
import { tutorialFiles } from './tutorial.ts';

export interface OpEntry {
	op: string;
	version: number;
	attrs: Record<string, { type: string; default?: unknown; required?: boolean; doc?: string }>;
	aliases: Record<string, Record<string, unknown>>;
	weights: { slot: string; optional?: boolean }[];
	subs: (string | { name: string })[];
	category: string;
	zoom: string;
	label: { en: string; zh?: string };
	parityBar: number;
}

export interface Sources {
	site: string;
	root: string;
	devkit: string;
	contributing: string | null;
	design: string | null;
	afternoon: string | null;
	codes: Record<string, string>;
	ops: OpEntry[];
	/** kit view keys (folders with a meta.ts) */
	kitViews: string[];
	/** tutorial files by repository path (null: missing) */
	files: Record<string, string | null>;
	/** example slug -> catalog status */
	examples: Record<string, string | null>;
}

const read = (p: string): string | null => (existsSync(p) ? readFileSync(p, 'utf8').replace(/^﻿/, '') : null);

/** Read every source of the DevKit pages for the site folder `site` (process.cwd() during a build). */
export function readSources(site: string): Sources {
	const root = findRoot(site) ?? join(site, '..');
	const devkit = read(join(site, 'DEVKIT.md'));
	if (devkit === null) throw new Error(`devkit-site: ${join(site, 'DEVKIT.md')} is missing`);
	const viewsDir = join(site, 'src', 'lib', 'atlas', 'ui', 'views');
	const files: Record<string, string | null> = {};
	for (const f of tutorialFiles()) files[f] = read(join(root, ...f.split('/')));
	return {
		site,
		root,
		devkit,
		contributing: read(join(root, 'CONTRIBUTING.md')),
		design: read(join(root, 'research', 'devkit', 'DEVKIT_DESIGN.md')),
		afternoon: read(join(root, 'research', 'devkit', 'afternoon-test.md')),
		codes: { ...CODES },
		ops: JSON.parse(read(join(site, 'src', 'lib', 'atlas', 'schema', 'ops.json')) ?? '[]') as OpEntry[],
		kitViews: existsSync(viewsDir) ? readdirSync(viewsDir).filter((k) => existsSync(join(viewsDir, k, 'meta.ts'))).sort() : [],
		files,
		examples: Object.fromEntries(KINDS.map((k) => [k.example, MODEL_BY_SLUG[k.example]?.status ?? null]))
	};
}

/** A file of the repository, by repository path (null: missing). */
export const repoFile = (s: Sources, path: string): string | null => read(join(s.root, ...path.split('/')));
