// atlas lanes [--json]                where lanes live, whether their packages are installed, and every lane there
// atlas lanes --deps                  install the node_modules that lanes outside the main site share
// atlas lanes --rm <name> [--force]   delete a lane (its folder: site/, lane.json, its builds, caches and e2e shots)
// atlas lanes --clean <name|main>     K4: delete a lane's (or the main site's) build output and caches, keep the lane
//
// The lanes folder: ATLAS_LANES_DIR, else "lanesDir" in the main site's atlas.config.local.json or atlas.config.json,
// else site/.work (config.ts lanesChoice). A lane made in site/.work before a lanesDir was set is still listed and found.
// --deps: a lane outside the main site cannot reach the main site's node_modules by walking up, so the lanes share one
//   folder of packages next to them: the main site's package.json (without its scripts) and package-lock.json are copied
//   into the lanes folder, and `npm ci` installs <lanes>/node_modules once. Node, npm (node_modules/.bin), TypeScript and
//   Vite find it from <lanes>/<name>/site by walking up; the lane's vite.config.ts allows it (server.fs.allow). Run it
//   again when the main site's package-lock.json changes (atlas doctor and atlas worktree say so). Stop the lanes' dev
//   servers first: npm ci replaces the folder.
// --rm: stops a dev server the CLI kept for the lane, then deletes the lane's folder. Refused (AK702) while the lane holds
//   owned changes that main does not have (merge --dry-run would copy or remove a file, or one changed in both);
//   --force deletes anyway. Root-level owned paths (pipeline/<slug>/, notes/) live in the main tree and stay; K4: the main
//   tree's notes/shots/<name>/ (where e2e wrote before K4, and still writes for lanes inside the repo) goes with the lane.
// --clean (K4): stops the site's kept dev server, then deletes what a build, svelte-kit and Vite leave: build/,
//   .svelte-kit/, .atlas/build*/ and node_modules/.vite/ (Vite's dependency cache: 67 files of at least 1 MiB each on
//   the exFAT drive). The lane's own files, lane.json and its .atlas gate caches stay. 'main' cleans the main site.
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { lanesChoice } from '../config.ts';
import { AtlasError } from '../errors.ts';
import { runCmd } from '../lib/child.ts';
import { readRecord, stopKept } from '../lib/devserver.ts';
import { posix, readJson, writeJson } from '../lib/io.ts';
import { depsState, type LaneFile } from '../lib/lanes.ts';
import { roots } from '../root.ts';
import { lanePaths, listLanes, planMerge } from './worktree.ts';

export const USAGE = 'atlas lanes [--json] | lanes --deps | lanes --rm <name> [--force] | lanes --clean <name|main>';

/** K4: the build output and caches of a site folder that --clean deletes (site-relative). */
export function cacheDirs(site: string): string[] {
	const out = ['build', '.svelte-kit', join('node_modules', '.vite')].filter((d) => existsSync(join(site, d)));
	const atlas = join(site, '.atlas');
	if (existsSync(atlas)) for (const d of readdirSync(atlas)) if (/^build/.test(d) && statSync(join(atlas, d)).isDirectory()) out.push(join('.atlas', d));
	return out;
}

/** K4: delete a site's build output and caches (after stopping its kept dev server). */
export async function cleanSite(site: string): Promise<{ dirs: string[]; files: number; bytes: number; note: string }> {
	const note = readRecord(site) ? (await stopKept(site)).note : '';
	const dirs = cacheDirs(site);
	let files = 0;
	let bytes = 0;
	for (const d of dirs) {
		const size = folderSize(join(site, d));
		files += size.files;
		bytes += size.bytes;
		rmSync(join(site, d), { recursive: true, force: true, maxRetries: 3, retryDelay: 500 });
	}
	return { dirs: dirs.map(posix), files, bytes, note };
}

/** Files and bytes under a folder. */
export function folderSize(dir: string): { files: number; bytes: number } {
	let files = 0;
	let bytes = 0;
	if (!existsSync(dir)) return { files, bytes };
	// K4: another lane may rewrite its files meanwhile (ENOENT): count what is there
	let ents: import('node:fs').Dirent[] = [];
	try {
		ents = readdirSync(dir, { withFileTypes: true, recursive: true });
	} catch {
		return { files, bytes };
	}
	for (const e of ents) {
		if (!e.isFile()) continue;
		files++;
		try {
			bytes += statSync(join(e.parentPath, e.name)).size;
		} catch {
			/* gone meanwhile */
		}
	}
	return { files, bytes };
}

const mb = (b: number) => `${(b / 1e6).toFixed(1)} MB`;

export interface LaneRow {
	name: string;
	dir: string;
	outside: boolean;
	created?: string;
	synced?: string;
	merged?: string;
	rev?: string;
	files: number;
	bytes: number;
}

/** Every lane of the main site, with what its lane.json says and its size. */
export function laneRows(main = roots().main): LaneRow[] {
	return listLanes(main).map(({ name, paths: p }) => {
		const lj = existsSync(p.laneJson) ? readJson<LaneFile & { rev?: string }>(p.laneJson) : null;
		const size = folderSize(p.laneDir);
		return {
			name,
			dir: posix(p.laneDir),
			outside: p.outside,
			...(lj?.created ? { created: lj.created } : {}),
			...(lj?.synced ? { synced: lj.synced } : {}),
			...(lj?.merged ? { merged: lj.merged } : {}),
			...(lj?.rev ? { rev: lj.rev } : {}),
			...size
		};
	});
}

/** npm ci of the main site's packages into <lanes>/node_modules (lanes outside the main site). */
export async function installDeps(main = roots().main, o: { echo?: boolean } = {}): Promise<number> {
	const lanes = lanesChoice(main);
	if (!lanes.outside) {
		console.log(`atlas lanes --deps: the lanes live inside the main site (${posix(lanes.dir)}) and resolve ${posix(lanes.deps)}: nothing to install`);
		return 0;
	}
	const lock = join(main, 'package-lock.json');
	if (!existsSync(lock)) throw new AtlasError('AK001', `the main site has no package-lock.json (${posix(lock)})`, { fix: 'run npm install in the main site first' });
	const t0 = performance.now();
	mkdirSync(lanes.dir, { recursive: true });
	const pkg = readJson<Record<string, unknown>>(join(main, 'package.json'));
	delete pkg.scripts; // no prepare or other lifecycle script of the site runs in the lanes folder
	writeJson(join(lanes.dir, 'package.json'), { '//': 'Written by `npm run atlas -- lanes --deps` (the main site package.json without its scripts): the packages the lanes in this folder share. Do not edit; run that command again instead.', ...pkg }, 1);
	copyFileSync(lock, join(lanes.dir, 'package-lock.json'));
	console.log(`atlas lanes --deps: npm ci in ${posix(lanes.dir)} (the main site's package-lock.json)`);
	const win = process.platform === 'win32';
	const res = await runCmd(win ? 'npm.cmd' : 'npm', ['ci', '--no-audit', '--no-fund'], { cwd: lanes.dir, echo: o.echo === false ? false : '  [npm ci] ', timeoutMs: 540_000, shell: win });
	if (res.code !== 0)
		throw new AtlasError('AK001', `npm ci in ${posix(lanes.dir)} failed (exit ${res.code})`, {
			likely: (res.stderr || res.stdout).trim().split('\n').slice(-3).join(' | '),
			fix: 'stop the dev servers of the lanes in that folder (they lock files), then run npm run atlas -- lanes --deps again'
		});
	const st = depsState(lanes, main);
	const size = folderSize(lanes.deps);
	console.log(`atlas lanes --deps: ${posix(lanes.deps)}: ${size.files} files, ${mb(size.bytes)}; ${st.why} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
	return st.ok ? 0 : 1;
}

/** Delete a lane's folder (after stopping its kept dev server). Refused while it holds owned work main does not have. */
export async function removeLane(name: string, o: { force?: boolean } = {}): Promise<{ dir: string; files: number; bytes: number; note: string }> {
	const p = lanePaths(name);
	if (!existsSync(p.laneDir)) throw new AtlasError('AK301', `no lane '${name}' (${posix(p.laneDir)})`, { fix: 'npm run atlas -- lanes lists the lanes' });
	const here = roots();
	if (here.laneDir && resolve(here.laneDir) === resolve(p.laneDir)) throw new AtlasError('AK301', `lanes --rm ${name} runs from the main site, not from inside the lane`);
	let note = '';
	if (existsSync(p.lane)) {
		if (!o.force) {
			const { plan } = planMerge(name);
			const pending = [...plan.copy, ...plan.remove, ...plan.both];
			if (pending.length)
				throw new AtlasError('AK702', `lane '${name}' holds ${pending.length} owned change(s) that main does not have`, {
					where: pending.slice(0, 8).join(', ') + (pending.length > 8 ? ', …' : ''),
					fix: `merge it first (npm run atlas -- merge ${name}), or delete it anyway: npm run atlas -- lanes --rm ${name} --force`
				});
		}
		if (readRecord(p.lane)) note = (await stopKept(p.lane)).note;
	}
	const size = folderSize(p.laneDir);
	rmSync(p.laneDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 500 });
	if (existsSync(p.laneDir))
		throw new AtlasError('AK301', `could not delete all of ${posix(p.laneDir)}`, { likely: 'a dev server or an editor still holds files of the lane', fix: 'stop it, then run lanes --rm again' });
	// K4: e2e shots of the lane's package that sit in the main tree (lanes inside the repo, and lanes made before K4)
	const shots = join(p.root, 'notes', 'shots', name);
	if (existsSync(shots)) {
		const s2 = folderSize(shots);
		size.files += s2.files;
		size.bytes += s2.bytes;
		rmSync(shots, { recursive: true, force: true, maxRetries: 3, retryDelay: 500 });
	}
	return { dir: posix(p.laneDir), ...size, note };
}

export async function main(args: string[]): Promise<number> {
	let json = false;
	let deps = false;
	let force = false;
	let rm: string | undefined;
	let clean: string | undefined;
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--json') json = true;
		else if (a === '--deps') deps = true;
		else if (a === '--force') force = true;
		else if (a === '--rm') rm = args[++i] ?? '';
		else if (a === '--clean') clean = args[++i] ?? '';
		else throw new AtlasError('AK301', `unknown argument ${a}`, { fix: USAGE });
	}
	if ([deps, rm !== undefined, clean !== undefined].filter(Boolean).length > 1 || rm === '' || clean === '' || (force && rm === undefined)) throw new AtlasError('AK301', `usage: ${USAGE}`);
	const r = roots();
	if (deps) return installDeps(r.main);
	if (clean !== undefined) {
		const site = clean === 'main' ? r.main : lanePaths(clean).lane;
		if (!existsSync(site)) throw new AtlasError('AK301', `no lane '${clean}' (${posix(site)})`, { fix: 'npm run atlas -- lanes lists the lanes; main cleans the main site' });
		const res = await cleanSite(site);
		console.log(`atlas lanes --clean ${clean}: ${res.dirs.length ? `deleted ${res.dirs.join(', ')} in ${posix(site)} (${res.files} files, ${mb(res.bytes)})` : `nothing to delete in ${posix(site)}`}${res.note ? `; ${res.note}` : ''}`);
		return 0;
	}
	if (rm !== undefined) {
		const res = await removeLane(rm, { force });
		console.log(`atlas lanes --rm ${rm}: deleted ${res.dir} (${res.files} files, ${mb(res.bytes)})${res.note ? `; ${res.note}` : ''}`);
		return 0;
	}
	const lanes = lanesChoice(r.main);
	const st = depsState(lanes, r.main);
	const rows = laneRows(r.main);
	if (json) {
		console.log(JSON.stringify({ dir: posix(lanes.dir), from: lanes.from, fallback: posix(lanes.fallback), outside: lanes.outside, deps: { ...st, dir: posix(st.dir) }, lanes: rows }, null, 1));
		return 0;
	}
	console.log(`atlas lanes: ${posix(lanes.dir)} (${lanes.from})${lanes.outside ? ', outside the main site' : ''}`);
	console.log(`  packages  ${posix(st.dir)}: ${st.why}${st.ok ? '' : st.shared ? ' (npm run atlas -- lanes --deps)' : ''}`);
	if (!rows.length) console.log('  no lanes');
	for (const x of rows)
		console.log(
			`  ${x.name.padEnd(24)} ${String(x.files).padStart(6)} files ${mb(x.bytes).padStart(9)}  ${x.rev ? `from ${x.rev.slice(0, 7)}` : 'from main'}${x.merged ? `, merged ${x.merged}` : ''}${x.dir.startsWith(posix(lanes.dir)) ? '' : `  (${x.dir})`}`
		);
	return 0;
}
