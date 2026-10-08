// Shared helpers for the CLI tests:  node --test scripts/atlas/tests/*.test.ts   (from site/ or a lane's site/)
import { spawnSync, type SpawnSyncReturns } from 'node:child_process';
import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pythonChoice, pythonCommand } from '../config.ts';
import { cliSite, findRoot } from '../root.ts';

export const SITE = cliSite(); // the site these tests run in (main tree or lane)
export const ROOT = findRoot(SITE)!;
export const CLI = join(SITE, 'scripts', 'atlas', 'cli.ts');
export const HERE = dirname(fileURLToPath(import.meta.url));

/** The configured Python (atlas.config.local.json here), or null when it cannot import atlas_kit + torch. */
export const PYTHON: string | null = (() => {
	const cmd = pythonCommand(pythonChoice({}));
	if (!cmd) return null;
	const r = spawnSync(cmd, ['-c', 'import torch, numpy, atlas_kit'], { env: { ...process.env, PYTHONPATH: join(ROOT, 'pipeline') }, encoding: 'utf8', timeout: 120_000 });
	return r.status === 0 ? cmd : null;
})();

/** A throwaway repo: <tmp>/repo/{CONTRACTS.md, pipeline/, notes/, site/} where site/ has the frozen atlas types, the
 *  i18n types and the catalog (copied from this site), so generated packages load exactly as in the real tree.
 *  `kit: true` copies the whole src/lib/atlas instead of only its types, so commands use the real recorder and tape
 *  writer (W1-C) rather than the CLI's fallback (integration 1a). */
export function tempRepo(o: { kit?: boolean } = {}): { repo: string; site: string; cleanup: () => void } {
	const base = mkdtempSync(join(tmpdir(), 'atlas-cli-'));
	const repo = join(base, 'repo');
	const site = join(repo, 'site');
	mkdirSync(join(repo, 'pipeline'), { recursive: true });
	mkdirSync(join(repo, 'notes'), { recursive: true });
	writeFileSync(join(repo, 'CONTRACTS.md'), '# temp\n');
	mkdirSync(join(site, 'src', 'lib'), { recursive: true });
	writeFileSync(join(site, 'svelte.config.js'), 'export default {};\n');
	writeFileSync(join(site, 'package.json'), '{ "name": "temp-site", "type": "module" }\n');
	if (o.kit) {
		cpSync(join(SITE, 'src', 'lib', 'atlas'), join(site, 'src', 'lib', 'atlas'), { recursive: true });
		// the kit adapters import their model's own preprocessing (drawing28 -> tinycnn.ts, text-bpe -> bpe.ts)
		for (const f of ['tinycnn.ts', 'bpe.ts']) cpSync(join(SITE, 'src', 'lib', 'models', f), join(site, 'src', 'lib', 'models', f));
	}
	else cpSync(join(SITE, 'src', 'lib', 'atlas', 'types'), join(site, 'src', 'lib', 'atlas', 'types'), { recursive: true });
	cpSync(join(SITE, 'src', 'lib', 'i18n', 'types.ts'), join(site, 'src', 'lib', 'i18n', 'types.ts'));
	cpSync(join(SITE, 'src', 'lib', 'catalog'), join(site, 'src', 'lib', 'catalog'), { recursive: true });
	return { repo, site, cleanup: () => rmSync(base, { recursive: true, force: true }) };
}

export function cli(args: string[], env: NodeJS.ProcessEnv = {}): SpawnSyncReturns<string> {
	return spawnSync(process.execPath, [CLI, ...args], { env: { ...process.env, ...env }, encoding: 'utf8', timeout: 300_000 });
}

export function python(code: string, env: NodeJS.ProcessEnv = {}): SpawnSyncReturns<string> {
	return spawnSync(PYTHON!, ['-c', code], {
		env: { ...process.env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1', ...env },
		encoding: 'utf8',
		timeout: 300_000
	});
}
