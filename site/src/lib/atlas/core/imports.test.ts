// W1-A acceptance 6 (Node part) and the dependency rule (design §3.1, AK503) for core/, ops/, graph/ and schema/:
//   - Node imports every non-test .ts file under type stripping (erasable syntax only);
//   - static imports stay inside src/lib/atlas/** (not ui/) or src/lib/i18n/types.ts, with explicit .ts extensions
//     (node: builtins only through hidden dynamic imports in core/io.ts and ops/manifest.ts);
//   - no Math.random (core/rng.ts is the only random source).
// Run from site/: node --test src/lib/atlas/core/imports.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ATLAS = fileURLToPath(new URL('../', import.meta.url));
const LIB = resolve(ATLAS, '..');
const DIRS = ['core', 'ops', 'graph', 'schema'];

function files(): string[] {
	const out: string[] = [];
	for (const d of DIRS) {
		const walk = (p: string) => {
			for (const e of readdirSync(p, { withFileTypes: true })) {
				const f = join(p, e.name);
				if (e.isDirectory()) walk(f);
				else if (e.name.endsWith('.ts') && !e.name.endsWith('.test.ts')) out.push(f);
			}
		};
		walk(join(ATLAS, d));
	}
	return out.sort();
}

test('Node imports every pure file of core/, ops/, graph/ and schema/', async (t) => {
	const list = files();
	for (const f of list) await import(pathToFileURL(f).href);
	t.diagnostic(`${list.length} files imported under type stripping`);
	assert.ok(list.length >= 30);
});

test('dependency rule: imports stay inside atlas/ (or i18n/types.ts); no Math.random', () => {
	const IMPORT = /(?:^|\n)\s*(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]|(?:^|\n)\s*import\s+['"]([^'"]+)['"]/g;
	for (const f of files()) {
		const src = readFileSync(f, 'utf8');
		for (const m of src.matchAll(IMPORT)) {
			const spec = m[1] ?? m[2];
			const where = relative(LIB, f);
			assert.ok(spec.startsWith('.'), `${where}: '${spec}' (only relative imports; node: builtins via hidden dynamic import)`);
			assert.ok(spec.endsWith('.ts'), `${where}: '${spec}' needs an explicit .ts extension`);
			const target = resolve(dirname(f), spec);
			const rel = relative(LIB, target).split('\\').join('/');
			assert.ok((rel.startsWith('atlas/') && !rel.startsWith('atlas/ui/')) || rel === 'i18n/types.ts', `${where}: imports ${rel}`);
		}
		const code = src.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
		assert.ok(!/Math\.random/.test(code) || f.endsWith(join('core', 'rng.ts')), `${relative(LIB, f)}: Math.random is banned outside core/rng.ts (AK503)`);
		assert.ok(!/\benum\s+\w+|\bnamespace\s+\w+|constructor\s*\(\s*(?:public|private|protected|readonly)\s/.test(code), `${relative(LIB, f)}: non-erasable syntax`);
	}
});
