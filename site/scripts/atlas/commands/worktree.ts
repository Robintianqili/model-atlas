// atlas worktree <name> [--owns <glob,...>] [--from <main site>] [--from-git <rev>]   make <lanes>/<name>/site
// atlas worktree <name> --sync [--from-git <rev>]                                   refresh every non-owned file (AK703)
// atlas merge <name> [--dry-run]                                                    copy the lane's owned files back (AK702)
// (design §4.12, §6.4; lanes and lane.json: lib/lanes.ts)
//
// The lanes folder <lanes> is site/.work by default; ATLAS_LANES_DIR or "lanesDir" in the main site's
//   atlas.config(.local).json moves it, for example to another drive (config.ts lanesChoice). A lane already in
//   site/.work stays there and is still found by --sync, merge and `lanes --rm`. git always runs on the main repo.
// worktree: copies the main site without node_modules, .svelte-kit, build, .work, .atlas and package-lock.json (no
//   symlinks), patches the copy's vite.config.ts (server.fs.allow: the main site for a lane inside it, the shared
//   <lanes>/node_modules for a lane outside it; and no .work pattern in server.watch.ignored, so the lane's own dev server
//   sees edits; lane setup, never merged back), and records the owned paths, a sha256 snapshot, the repo root and the main
//   site in <lanes>/<name>/lane.json (a lane outside the repo finds the repo through them: root.ts). Owned paths: --owns,
//   else the design's table for kit lanes (w1-*, k0), else the model ownership of §8.3 (pipeline/<name>/**,
//   site/src/lib/explainers/<name>/**, site/static/models/<name>/**, notes/<name>.md, notes/review-<name>.md (K4),
//   notes/shots/<name>/**). --sync adds notes/review-<name>.md to an older lane.json that owns notes/<name>.md.
// --from-git <rev> (K2): the lane's files come from a committed revision (HEAD, a branch, a commit) instead of the main
//   working tree, which another workflow may be editing: `git archive <rev> site` read in memory (no tar program), the
//   same skips, the snapshot holds the revision's hashes and lane.json records the commit (`rev`). The machine's
//   git-ignored CLI config (scripts/atlas/atlas.config.local.json) is copied from main. git runs with
//   -c safe.directory=<repo> for that one command (never the global config).
// --sync: every non-owned file the lane did not change since the snapshot is refreshed from main (added, replaced or
//   removed), or from a revision with --from-git <rev> (K2); refuses with AK703, changing nothing, when the lane changed a
//   non-owned file. Owned files are never touched.
// merge: the lane's changes (against the snapshot; without a lane.json, against main) inside the owned paths are copied
//   into the main site; any change outside them refuses the whole merge with AK702, as does an owned file that main also
//   changed since the snapshot ("changed in both": sync first). vite.config.ts and the local CLI config are never merged.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, relative } from 'node:path';
import { isInside, lanesChoice, type LanesChoice } from '../config.ts';
import { AtlasError } from '../errors.ts';
import { posix, readJson, writeJson, writeText } from '../lib/io.ts';
import { ALLOW_OUTSIDE, KNOWN_LANES, LANE_FORMAT, LANE_LOCAL, LANE_SETUP, SKIP_DIRS, SKIP_FILES, depsState, matcher, modelOwns, patchViteConfig, sha256, siteFiles, withReview, type LaneFile } from '../lib/lanes.ts';
import { suggestPort } from '../ports.ts';
import { roots } from '../root.ts';

const NAME = /^[a-z0-9]+([-_][a-z0-9]+)*$/;
const now = () => new Date().toISOString().slice(0, 19).replace('T', ' ');
/** K2: machine-local files a lane keeps as they are: never synced, compared or merged (git-ignored; lib/lanes.ts) */
export { LANE_LOCAL };
const notCompared = (rel: string) => LANE_SETUP.includes(rel) || LANE_LOCAL.includes(rel);

/** lane.json with the K2 field: the commit a --from-git lane was made (or last synced) from */
type Lane = LaneFile & { rev?: string };

export interface LanePaths {
	root: string;
	main: string;
	laneDir: string;
	lane: string;
	laneJson: string;
	/** where lanes live (config.ts lanesChoice) */
	lanes: LanesChoice;
	/** the lane is outside the main site: it resolves the shared <lanes>/node_modules (atlas lanes --deps) */
	outside: boolean;
	/** the lane's server.fs.allow (vite.config.ts, lane setup) */
	allow: string[];
}

/** The lane's folders. A lane is made in the configured lanes folder; an existing lane is looked up there and then in
 *  the default site/.work (lanes made before a lanesDir was set). */
export function lanePaths(name: string, root = roots().root, main?: string, o: { create?: boolean } = {}): LanePaths {
	if (!NAME.test(name)) throw new AtlasError('AK301', `lane name '${name}' must be lowercase letters, digits, '-' or '_'`);
	const m = main ?? join(root, 'site');
	const lanes = lanesChoice(m);
	const inFallback = !o.create && !existsSync(join(lanes.dir, name, 'site')) && existsSync(join(lanes.fallback, name, 'site'));
	const laneDir = join(inFallback ? lanes.fallback : lanes.dir, name);
	const lane = join(laneDir, 'site');
	const outside = !isInside(laneDir, m);
	const allow = outside ? ALLOW_OUTSIDE : ['.', posix(relative(lane, m))];
	return { root, main: m, laneDir, lane, laneJson: join(laneDir, 'lane.json'), lanes, outside, allow };
}

/** The lanes this main site has: <lanes>/<name>/site in the configured folder and in the default site/.work. */
export function listLanes(main = roots().main): { name: string; paths: LanePaths }[] {
	const lanes = lanesChoice(main);
	const out = new Map<string, LanePaths>();
	for (const dir of [lanes.dir, lanes.fallback]) {
		if (!existsSync(dir)) continue;
		for (const d of readdirSync(dir, { withFileTypes: true })) {
			if (!d.isDirectory() || d.name === 'node_modules' || !NAME.test(d.name) || out.has(d.name) || !existsSync(join(dir, d.name, 'site'))) continue;
			out.set(d.name, lanePaths(d.name, roots().root, main));
		}
	}
	return [...out].sort(([a], [b]) => (a < b ? -1 : 1)).map(([name, paths]) => ({ name, paths }));
}

export function defaultOwns(name: string): { owns: string[]; from: string } {
	if (KNOWN_LANES[name]) return { owns: KNOWN_LANES[name], from: `the design's table for lane '${name}'` };
	return { owns: modelOwns(name), from: `the model ownership of design §8.3 (slug '${name}')` };
}

function readLane(p: LanePaths): Lane | null {
	return existsSync(p.laneJson) ? readJson<Lane>(p.laneJson) : null;
}

const fileOf = (site: string, rel: string) => join(site, ...rel.split('/'));
const hashOf = (site: string, rel: string) => {
	const f = fileOf(site, rel);
	return existsSync(f) ? sha256(f) : null;
};

// ------------------------------------------------------------------ where a lane's files come from (K2)
/** A site's files: the main working tree, or a committed revision read with git archive. */
export interface SiteTree {
	/** site-relative POSIX paths, sorted, with the lane skips applied (lib/lanes.ts siteFiles) */
	files: string[];
	hash(rel: string): string | null;
	/** copy one file to `dst` (false when the tree has no such file) */
	copyTo(rel: string, dst: string): boolean;
	/** what the messages call it */
	label: string;
	/** the commit, for a revision */
	rev?: string;
}

export function dirTree(site: string): SiteTree {
	return {
		files: siteFiles(site),
		hash: (rel) => hashOf(site, rel),
		copyTo(rel, dst) {
			const f = fileOf(site, rel);
			if (!existsSync(f)) return false;
			mkdirSync(dirname(dst), { recursive: true });
			copyFileSync(f, dst);
			return true;
		},
		label: 'main'
	};
}

/** Run git on the repo with -c safe.directory=<repo> for this one command (an exFAT drive has no owner; AK004). */
export function git(root: string, args: string[], o: { buffer?: boolean } = {}): { code: number; out: Buffer; err: string } {
	const r = spawnSync('git', ['-c', `safe.directory=${posix(root)}`, '-C', root, ...args], { encoding: 'buffer', maxBuffer: 2 ** 31 - 1, windowsHide: true, timeout: 300_000 });
	if ((r.error as NodeJS.ErrnoException | undefined)?.code === 'ENOENT') throw new AtlasError('AK004', 'git not found on PATH', { fix: 'install git' });
	return { code: r.status ?? 1, out: r.stdout ?? Buffer.alloc(0), err: (r.stderr ?? Buffer.alloc(0)).toString('utf8') };
}

/** The full commit id of `rev` (a branch, tag, HEAD~2, a short id). */
export function resolveRev(root: string, rev: string): string {
	if (!rev || rev.startsWith('-')) throw new AtlasError('AK301', `--from-git needs a revision (HEAD, a branch, a commit), got '${rev}'`);
	const r = git(root, ['rev-parse', '--verify', '--quiet', `${rev}^{commit}`]);
	if (r.code !== 0) {
		const top = git(root, ['rev-parse', '--show-toplevel']);
		if (top.code !== 0) throw new AtlasError('AK004', `git cannot read the repo at ${posix(root)}: ${top.err.trim().split(/\r?\n/)[0]}`, { fix: 'run npm run atlas -- doctor' });
		throw new AtlasError('AK301', `--from-git: no commit '${rev}' in ${posix(root)}`, { fix: 'give a revision git knows: HEAD, a branch, a tag or a commit id (git log --oneline)' });
	}
	return r.out.toString('utf8').trim();
}

/** The entries of a tar stream (ustar with pax 'x' path records and GNU 'L' long names, as git archive writes). */
export function untar(buf: Buffer): { path: string; type: string; data: Buffer }[] {
	const out: { path: string; type: string; data: Buffer }[] = [];
	let off = 0;
	let next: string | null = null;
	while (off + 512 <= buf.length) {
		const h = buf.subarray(off, off + 512);
		if (h.every((b) => b === 0)) break;
		const str = (a: number, n: number) => {
			const s = h.subarray(a, a + n);
			const z = s.indexOf(0);
			return s.subarray(0, z < 0 ? n : z).toString('utf8');
		};
		const size = parseInt(str(124, 12).trim() || '0', 8);
		const type = h[156] === 0 ? '0' : String.fromCharCode(h[156]);
		const prefix = str(257, 5) === 'ustar' ? str(345, 155) : '';
		const data = buf.subarray(off + 512, off + 512 + size);
		off += 512 + Math.ceil(size / 512) * 512;
		if (type === 'x') {
			const p = paxRecords(data).path;
			if (p !== undefined) next = p;
			continue;
		}
		if (type === 'g') continue;
		if (type === 'L') {
			next = data.toString('utf8').replace(/\0+$/, '');
			continue;
		}
		const name = str(0, 100);
		out.push({ path: next ?? (prefix ? `${prefix}/${name}` : name), type, data });
		next = null;
	}
	return out;
}

function paxRecords(b: Buffer): Record<string, string> {
	const out: Record<string, string> = {};
	let i = 0;
	while (i < b.length) {
		const sp = b.indexOf(0x20, i);
		if (sp < 0) break;
		const len = parseInt(b.subarray(i, sp).toString('ascii'), 10);
		if (!(len > 0)) break;
		const rec = b.subarray(sp + 1, i + len - 1).toString('utf8');
		const eq = rec.indexOf('=');
		if (eq > 0) out[rec.slice(0, eq)] = rec.slice(eq + 1);
		i += len;
	}
	return out;
}

/** A committed revision's site files (K2): git archive <rev> -- <site>, read in memory, with the lane skips. */
export function gitTree(root: string, main: string, rev: string): SiteTree {
	const sha = resolveRev(root, rev);
	const rel = posix(relative(root, main));
	if (!rel || rel.startsWith('..') || isAbsolute(rel)) throw new AtlasError('AK301', `--from-git: the site ${posix(main)} is not inside the repo ${posix(root)}`);
	const r = git(root, ['archive', '--format=tar', sha, '--', rel]);
	if (r.code !== 0) throw new AtlasError('AK301', `git archive ${sha.slice(0, 7)} -- ${rel} failed: ${r.err.trim().split(/\r?\n/)[0]}`);
	const files = new Map<string, Buffer>();
	for (const e of untar(r.out)) {
		if (e.type !== '0' && e.type !== '7') continue; // files only: no folders, symlinks (exFAT has none) or submodules
		if (!e.path.startsWith(`${rel}/`)) continue;
		const sr = e.path.slice(rel.length + 1);
		const parts = sr.split('/');
		if (parts.slice(0, -1).some((d) => SKIP_DIRS.includes(d)) || (parts.length === 1 && SKIP_FILES.includes(sr)) || sr.endsWith('.tmp')) continue;
		files.set(sr, e.data);
	}
	const hashes = new Map<string, string>();
	const hash = (f: string) => {
		const b = files.get(f);
		if (!b) return null;
		let h = hashes.get(f);
		if (!h) hashes.set(f, (h = createHash('sha256').update(b).digest('hex')));
		return h;
	};
	return {
		files: [...files.keys()].sort(),
		hash,
		copyTo(f, dst) {
			const b = files.get(f);
			if (!b) return false;
			mkdirSync(dirname(dst), { recursive: true });
			writeFileSync(dst, b);
			return true;
		},
		label: `${rev === sha ? sha.slice(0, 7) : `${rev} (${sha.slice(0, 7)})`}`,
		rev: sha
	};
}

/** Create the lane. Returns what was written. K2: `fromGit` takes the files from a committed revision. */
export function createLane(
	name: string,
	o: { owns?: string[]; root?: string; main?: string; fromGit?: string } = {}
): { paths: LanePaths; files: number; bytes: number; owns: string[]; ownsFrom: string; vitePatched: boolean; rev?: string; local: string[]; warnings: string[] } {
	const p = lanePaths(name, o.root, o.main, { create: true });
	for (const existing of [p.lane, join(p.lanes.fallback, name, 'site')])
		if (existsSync(existing)) throw new AtlasError('AK301', `lane '${name}' already exists: ${posix(existing)}`, { fix: `refresh it with: npm run atlas -- worktree ${name} --sync` });
	if (!existsSync(join(p.main, 'package.json'))) throw new AtlasError('AK301', `no main site at ${posix(p.main)}`);
	if (p.outside && isInside(p.laneDir, p.lanes.deps)) throw new AtlasError('AK301', `lane name '${name}' is the shared node_modules folder`);
	const d = o.owns?.length ? { owns: o.owns, from: '--owns' } : defaultOwns(name);
	const tree = o.fromGit !== undefined ? gitTree(p.root, p.main, o.fromGit) : dirTree(p.main);
	const snapshot: Record<string, string> = {};
	let bytes = 0;
	for (const rel of tree.files) {
		const dst = fileOf(p.lane, rel);
		tree.copyTo(rel, dst);
		snapshot[rel] = tree.hash(rel)!;
	}
	for (const rel of tree.files) bytes += readFileSync(fileOf(p.lane, rel)).byteLength;
	// a revision has no machine-local files: take main's (the configured Python)
	const local: string[] = [];
	for (const rel of LANE_LOCAL)
		if (!existsSync(fileOf(p.lane, rel)) && existsSync(fileOf(p.main, rel))) {
			mkdirSync(dirname(fileOf(p.lane, rel)), { recursive: true });
			copyFileSync(fileOf(p.main, rel), fileOf(p.lane, rel));
			local.push(rel);
		}
	let vitePatched = false;
	const vite = join(p.lane, 'vite.config.ts');
	if (existsSync(vite)) {
		const r = patchViteConfig(readFileSync(vite, 'utf8'), p.allow);
		if (r.patched) writeText(vite, r.text);
		vitePatched = r.patched;
	}
	const lane: Lane = { format: LANE_FORMAT, name, created: now(), root: posix(p.root), main: posix(p.main), ...(tree.rev ? { rev: tree.rev } : {}), owns: d.owns, snapshot };
	writeJson(p.laneJson, lane, 1);
	return { paths: p, files: tree.files.length, bytes, owns: d.owns, ownsFrom: d.from, vitePatched, local, warnings: outsideWarnings(name, p), ...(tree.rev ? { rev: tree.rev } : {}) };
}

/** What a lane outside the main site still needs: a CLI that reads lane.json (a revision older than lanes outside the
 *  repo has none) and the shared node_modules. */
function outsideWarnings(name: string, p: LanePaths): string[] {
	if (!p.outside) return [];
	const out: string[] = [];
	const rootTs = fileOf(p.lane, 'scripts/atlas/root.ts');
	if (existsSync(rootTs) && !readFileSync(rootTs, 'utf8').includes('lane.json'))
		out.push(
			`the lane's CLI predates lanes outside the repo: inside the lane it cannot find the repo by itself. Take main's CLI with npm run atlas -- worktree ${name} --sync, or set ATLAS_ROOT=${posix(p.root)} for commands in the lane`
		);
	const deps = depsState(p.lanes, p.main);
	if (!deps.ok) out.push(`the shared node_modules (${posix(p.lanes.deps)}) is ${deps.why}: run npm run atlas -- lanes --deps in the main site before building or testing in the lane`);
	return out;
}

export interface SyncPlan {
	update: string[];
	add: string[];
	remove: string[];
	conflicts: string[];
	ownedSkipped: number;
	noSnapshot: boolean;
	/** K2: what the files came from ('main', or the revision) */
	from?: string;
	rev?: string;
}

/** What --sync would do (nothing is written). K2: `fromGit` compares with a committed revision instead of main. */
export function planSync(name: string, o: { root?: string; main?: string; fromGit?: string; tree?: SiteTree } = {}): { paths: LanePaths; lane: Lane | null; plan: SyncPlan; tree: SiteTree } {
	const p = lanePaths(name, o.root, o.main);
	if (!existsSync(p.lane)) throw new AtlasError('AK301', `no lane '${name}' (${posix(p.lane)})`, { fix: `create it with: npm run atlas -- worktree ${name}` });
	const lane = readLane(p);
	const owns = matcher(lane?.owns ?? defaultOwns(name).owns);
	const tree = o.tree ?? (o.fromGit !== undefined ? gitTree(p.root, p.main, o.fromGit) : dirTree(p.main));
	const plan: SyncPlan = { update: [], add: [], remove: [], conflicts: [], ownedSkipped: 0, noSnapshot: !lane, from: tree.label, ...(tree.rev ? { rev: tree.rev } : {}) };
	const all = new Set([...tree.files, ...siteFiles(p.lane)]);
	for (const rel of [...all].sort()) {
		if (notCompared(rel)) continue;
		if (owns(`site/${rel}`)) {
			plan.ownedSkipped++;
			continue;
		}
		const m = tree.hash(rel);
		const l = hashOf(p.lane, rel);
		if (m === l) continue;
		const snap = lane ? (lane.snapshot[rel] ?? null) : m; // without a snapshot every difference counts as a lane change
		const laneChanged = l !== snap;
		if (laneChanged) {
			plan.conflicts.push(rel);
			continue;
		}
		if (m === null) plan.remove.push(rel);
		else if (l === null) plan.add.push(rel);
		else plan.update.push(rel);
	}
	return { paths: p, lane, plan, tree };
}

export function applySync(name: string, o: { root?: string; main?: string; fromGit?: string } = {}): SyncPlan {
	const { paths: p, lane, plan, tree } = planSync(name, o);
	if (plan.conflicts.length)
		throw new AtlasError('AK703', `sync refused: lane '${name}' changed ${plan.conflicts.length} file(s) outside its owned paths${plan.noSnapshot ? ' (no lane.json: every difference from main counts)' : ''}`, {
			where: plan.conflicts.slice(0, 12).join(', ') + (plan.conflicts.length > 12 ? ', …' : ''),
			likely: 'a kit or shared file was edited inside the lane (lanes never edit the kit)',
			fix: 'undo those edits in the lane (copy the main version back), or move the change into the package (local op, view, section); then sync again'
		});
	const snapshot = { ...(lane?.snapshot ?? {}) };
	for (const rel of [...plan.update, ...plan.add]) {
		tree.copyTo(rel, fileOf(p.lane, rel));
		snapshot[rel] = tree.hash(rel)!;
	}
	for (const rel of plan.remove) {
		rmSync(fileOf(p.lane, rel), { force: true });
		delete snapshot[rel];
	}
	// lane setup: take the source's vite.config.ts again when it changed, re-patched
	const lv = join(p.lane, 'vite.config.ts');
	const tmp = join(p.laneDir, 'vite.config.ts.source.tmp');
	if (tree.copyTo('vite.config.ts', tmp)) {
		const want = patchViteConfig(readFileSync(tmp, 'utf8'), p.allow).text;
		if (!existsSync(lv) || readFileSync(lv, 'utf8') !== want) {
			writeText(lv, want);
			plan.update.push('vite.config.ts (lane setup, re-patched)');
		}
		snapshot['vite.config.ts'] = tree.hash('vite.config.ts')!;
		rmSync(tmp, { force: true });
	}
	const base: Lane = lane ?? { format: LANE_FORMAT, name, created: now(), owns: defaultOwns(name).owns, snapshot };
	// an older lane learns where the repo and the main site are (root.ts reads them)
	const next: Lane = { ...base, root: base.root ?? posix(p.root), main: base.main ?? posix(p.main), owns: withReview(base.owns, name), synced: now(), snapshot };
	// K2: a lane synced from a revision records it; a later sync from main drops it (main is newer than any revision)
	if (tree.rev) next.rev = tree.rev;
	else delete next.rev;
	writeJson(p.laneJson, next, 1);
	return plan;
}

export interface MergePlan {
	copy: string[];
	remove: string[];
	outside: string[];
	both: string[];
	inPlace: string[];
	owns: string[];
	ownsFrom: string;
	noSnapshot: boolean;
}

export function planMerge(name: string, o: { root?: string; main?: string; owns?: string[] } = {}): { paths: LanePaths; lane: Lane | null; plan: MergePlan } {
	const p = lanePaths(name, o.root, o.main);
	if (!existsSync(p.lane)) throw new AtlasError('AK301', `no lane '${name}' (${posix(p.lane)})`);
	const lane = readLane(p);
	const d = o.owns?.length ? { owns: o.owns, from: '--owns' } : lane ? { owns: withReview(lane.owns, name), from: 'lane.json' } : defaultOwns(name);
	const owns = matcher(d.owns);
	const plan: MergePlan = { copy: [], remove: [], outside: [], both: [], inPlace: [], owns: d.owns, ownsFrom: d.from, noSnapshot: !lane };
	const all = new Set([...siteFiles(p.main), ...siteFiles(p.lane)]);
	for (const rel of [...all].sort()) {
		if (notCompared(rel)) continue;
		const m = hashOf(p.main, rel);
		const l = hashOf(p.lane, rel);
		const snap = lane ? (lane.snapshot[rel] ?? null) : m;
		if (l === snap) continue; // the lane did not change it
		if (l === m) continue; // same content on both sides (nothing to merge)
		if (!owns(`site/${rel}`)) {
			plan.outside.push(rel);
			continue;
		}
		if (lane && m !== snap) {
			plan.both.push(rel);
			continue;
		}
		if (l === null) plan.remove.push(rel);
		else plan.copy.push(rel);
	}
	plan.inPlace = d.owns.filter((g) => !g.startsWith('site/'));
	return { paths: p, lane, plan };
}

export function applyMerge(name: string, o: { root?: string; main?: string; owns?: string[] } = {}): MergePlan {
	const { paths: p, lane, plan } = planMerge(name, o);
	refuseIfDirty(name, plan);
	const snapshot = { ...(lane?.snapshot ?? {}) };
	for (const rel of plan.copy) {
		const dst = fileOf(p.main, rel);
		mkdirSync(dirname(dst), { recursive: true });
		copyFileSync(fileOf(p.lane, rel), dst);
		snapshot[rel] = sha256(dst);
	}
	for (const rel of plan.remove) {
		rmSync(fileOf(p.main, rel), { force: true });
		delete snapshot[rel];
	}
	if (lane) writeJson(p.laneJson, { ...lane, merged: now(), snapshot }, 1);
	return plan;
}

function refuseIfDirty(name: string, plan: MergePlan): void {
	if (plan.outside.length)
		throw new AtlasError('AK702', `merge refused: lane '${name}' changed ${plan.outside.length} file(s) outside its owned paths${plan.noSnapshot ? ' (no lane.json: every difference from main counts as a lane change)' : ''}`, {
			where: plan.outside.slice(0, 12).join(', ') + (plan.outside.length > 12 ? ', …' : ''),
			likely: 'an edit to a shared or kit file, or a stale lane copy (main moved on)',
			fix: `revert those files in the lane (or, for a stale copy, npm run atlas -- worktree ${name} --sync), then merge again`
		});
	if (plan.both.length)
		throw new AtlasError('AK702', `merge refused: ${plan.both.length} owned file(s) changed both in lane '${name}' and in main since the lane was made`, {
			where: plan.both.slice(0, 12).join(', '),
			fix: 'merge the two versions by hand inside the lane, then merge again'
		});
}

function parseArgs(args: string[]): { name: string; flags: Record<string, string | boolean> } {
	const pos: string[] = [];
	const flags: Record<string, string | boolean> = {};
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--owns' || a === '--from' || a === '--from-git') flags[a.slice(2)] = args[++i] ?? '';
		else if (a === '--sync' || a === '--dry-run' || a === '--json') flags[a.slice(2)] = true;
		else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`);
		else pos.push(a);
	}
	if (pos.length !== 1) throw new AtlasError('AK301', 'usage: atlas worktree <name> [--owns <glob,...>] [--from-git <rev>] | worktree <name> --sync [--from-git <rev>] | merge <name> [--dry-run]');
	return { name: pos[0], flags };
}

const list = (xs: string[], max = 20) => xs.slice(0, max).map((x) => `    ${x}`).join('\n') + (xs.length > max ? `\n    … ${xs.length - max} more` : '');

export async function main(args: string[]): Promise<number> {
	const { name, flags } = parseArgs(args);
	const main = typeof flags.from === 'string' && flags.from ? flags.from : undefined;
	// K2: --from-git <rev>: a committed revision instead of the main working tree
	const fromGit = typeof flags['from-git'] === 'string' ? (flags['from-git'] as string) : undefined;
	if (fromGit === '') throw new AtlasError('AK301', '--from-git needs a revision: HEAD, a branch, a tag or a commit id');
	if (flags.sync) {
		if (flags['dry-run']) {
			const { plan } = planSync(name, { main, fromGit });
			console.log(`atlas worktree ${name} --sync --dry-run (from ${plan.from}): update ${plan.update.length}, add ${plan.add.length}, remove ${plan.remove.length}, conflicts ${plan.conflicts.length} (owned files skipped: ${plan.ownedSkipped})`);
			if (plan.conflicts.length) console.log(`  conflicts (lane changed a non-owned file; AK703):\n${list(plan.conflicts)}`);
			return plan.conflicts.length ? 1 : 0;
		}
		const plan = applySync(name, { main, fromGit });
		console.log(`atlas worktree ${name} --sync (from ${plan.from}): updated ${plan.update.length}, added ${plan.add.length}, removed ${plan.remove.length} non-owned file(s); ${plan.ownedSkipped} owned file(s) left as they are`);
		for (const [k, xs] of [
			['updated', plan.update],
			['added', plan.add],
			['removed', plan.remove]
		] as const)
			if (xs.length) console.log(`  ${k}:\n${list(xs)}`);
		return 0;
	}
	const owns = typeof flags.owns === 'string' && flags.owns ? flags.owns.split(',').map((s) => s.trim()).filter(Boolean) : undefined;
	const r = createLane(name, { owns, main, fromGit });
	const p = r.paths;
	console.log(`atlas worktree ${name}: ${r.files} files (${(r.bytes / 1e6).toFixed(1)} MB) -> ${posix(p.lane)}${r.rev ? ` from commit ${r.rev.slice(0, 7)} (${fromGit})` : ''}`);
	console.log(`  lanes folder    ${posix(p.lanes.dir)} (${p.lanes.from})${p.outside ? `; outside the main site: packages from ${posix(p.lanes.deps)} (atlas lanes --deps)` : "; packages from the main site's node_modules"}`);
	if (r.local.length) console.log(`  local config    ${r.local.join(', ')} copied from main (git-ignored; never synced or merged)`);
	const allow = '[' + p.allow.map((a) => `'${a}'`).join(', ') + ']';
	console.log(`  vite.config.ts  ${r.vitePatched ? `server.fs.allow: ${allow} added and the .work watch pattern dropped, so edits reload (lane setup; never merged)` : 'NOT patched (no server.fs block found): add allow by hand'}`);
	console.log(`  owned paths     (${r.ownsFrom})\n${list(r.owns)}`);
	console.log(`  snapshot        ${posix(p.laneJson)} (also records the repo root and the main site)`);
	for (const w of r.warnings) console.log(`  WARN            ${w}`);
	console.log(`\nWork inside the lane: cd "${posix(p.lane)}"; npm run dev -- --port ${suggestPort(name, roots().root)}`);
	console.log(`Root-level owned paths (pipeline/, notes/) are edited in place in the main tree. Merge with: npm run atlas -- merge ${name}`);
	return 0;
}

export async function mergeMain(args: string[]): Promise<number> {
	const { name, flags } = parseArgs(args);
	const main = typeof flags.from === 'string' && flags.from ? flags.from : undefined;
	const owns = typeof flags.owns === 'string' && flags.owns ? flags.owns.split(',').map((s) => s.trim()).filter(Boolean) : undefined;
	if (flags['dry-run']) {
		const { plan } = planMerge(name, { main, owns });
		if (flags.json) {
			console.log(JSON.stringify(plan));
			return plan.outside.length || plan.both.length ? 1 : 0;
		}
		console.log(`atlas merge ${name} --dry-run: would copy ${plan.copy.length} and remove ${plan.remove.length} owned file(s) in the main site`);
		console.log(`  owned paths (${plan.ownsFrom}${plan.noSnapshot ? '; no lane.json, so lane files are compared with main' : ''}):\n${list(plan.owns)}`);
		if (plan.copy.length) console.log(`  copy:\n${list(plan.copy, 60)}`);
		if (plan.remove.length) console.log(`  remove:\n${list(plan.remove)}`);
		if (plan.inPlace.length) console.log(`  edited in place in the main tree (nothing to copy):\n${list(plan.inPlace)}`);
		try {
			refuseIfDirty(name, plan);
		} catch (e) {
			console.error((e as AtlasError).format());
			return 1;
		}
		console.log('  ownership OK: nothing outside the owned paths changed');
		return 0;
	}
	const plan = applyMerge(name, { main, owns });
	console.log(`atlas merge ${name}: copied ${plan.copy.length}, removed ${plan.remove.length} owned file(s) into the main site`);
	if (plan.copy.length) console.log(list(plan.copy, 60));
	return 0;
}
