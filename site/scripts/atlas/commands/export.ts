// atlas export <slug> [--no-parity] [--no-draft] [-- <args for export.py>]   (design §4.12, §6.1 step 4)
// Runs pipeline/<slug>/export.py with the configured Python (atlas.config.local.json, else atlas.config.json, else
// `python`; ATLAS_PYTHON overrides) and PYTHONPATH=<root>/pipeline, ATLAS_SITE, ATLAS_ROOT, so the assets land in this
// site (a lane's copy inside a lane). A Python AtlasError (AK1xx) is shown as printed and the command exits 1.
// Then `parity <slug> --write` (spec.parity is written only here and by `atlas parity`) and `lesson <slug> draft`.
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { AtlasError, codeInText } from '../errors.ts';
import { runPython } from '../lib/child.ts';
import { posix } from '../lib/io.ts';
import { roots } from '../root.ts';

export async function main(args: string[]): Promise<number> {
	const dash = args.indexOf('--');
	const own = dash >= 0 ? args.slice(0, dash) : args;
	const pass = dash >= 0 ? args.slice(dash + 1) : [];
	const pos = own.filter((a) => !a.startsWith('--'));
	const flags = new Set(own.filter((a) => a.startsWith('--')));
	for (const f of flags) if (f !== '--no-parity' && f !== '--no-draft') throw new AtlasError('AK301', `unknown flag ${f}`, { fix: 'atlas export <slug> [--no-parity] [--no-draft] [-- <export.py args>]' });
	if (pos.length !== 1) throw new AtlasError('AK301', 'usage: atlas export <slug> [--no-parity] [--no-draft] [-- <export.py args>]');
	const slug = pos[0];
	const r = roots();
	const script = join(r.root, 'pipeline', slug, 'export.py');
	if (!existsSync(script))
		throw new AtlasError('AK301', `no ${posix(script)}`, {
			fix: `scaffold it with npm run atlas -- new model ${slug} --kind trace --input <adapter> (kinds B and C need no export)`
		});
	console.log(`atlas export ${slug}: ${posix(script)} -> ${posix(join(r.site, 'static', 'models', slug))}${r.lane ? ` (lane ${r.lane})` : ''}`);
	const py = await runPython([script, ...pass], { cwd: join(r.root, 'pipeline', slug), echo: true, timeoutMs: 540_000 });
	if (py.code !== 0) {
		const code = codeInText(py.stderr + py.stdout);
		if (code) return 1; // the Python AtlasError block is already on the screen
		throw new AtlasError('AK301', `export.py failed (exit ${py.code})`, { where: posix(script), likely: py.stderr.trim().split('\n').slice(-3).join(' | ') });
	}
	let status = 0;
	if (!flags.has('--no-parity')) {
		console.log(`\n--- atlas parity ${slug} --write`);
		status = await (await import('./parity.ts')).main([slug, '--write']);
	}
	if (!flags.has('--no-draft') && existsSync(join(r.site, 'src', 'lib', 'explainers', slug, 'explainer.ts'))) {
		console.log(`\n--- atlas lesson ${slug} draft`);
		await (await import('./lesson.ts')).main([slug, 'draft']);
	}
	return status;
}
