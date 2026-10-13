// Lean gates (2026-10-13): check --all runs its one e2e sequence on the default 2 passes, not the five `all` passes;
// the integrator's run names no packages (no e2e); a long task (AK603) only warns and never fails a check.
//   node --test scripts/atlas/tests/lean-gates.test.ts   (from site/)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { allE2ePlan, parseCheckArgs } from '../commands/check.ts';
import { resolvePasses } from '../lib/passes.ts';
import { SITE } from './helpers.ts';

test('check --all: the packages given run on the default 2 passes; --passes still asks for more', () => {
	const { o } = parseCheckArgs(['--all', 'gcn', 'char-rnn,masked-lm']);
	assert.equal(o.all, true);
	assert.equal(o.passes, undefined);
	const plan = allE2ePlan(o, ['gcn', 'char-rnn', 'masked-lm', 'pinn']);
	assert.deepEqual(plan.slugs, ['gcn', 'char-rnn', 'masked-lm']);
	assert.equal(plan.passes, 'default');
	assert.deepEqual(resolvePasses(plan.passes).map((p) => p.name), ['desktop-en-light', 'phone-zh-dark']);
	// --e2e all: every package, still the default passes
	assert.deepEqual(allE2ePlan(parseCheckArgs(['--all', '--e2e', 'all']).o, ['a', 'b']), { slugs: ['a', 'b'], passes: 'default' });
	// --e2e and positional slugs join, once each
	assert.deepEqual(allE2ePlan(parseCheckArgs(['--all', '--e2e', 'gcn,pinn', 'gcn']).o, []).slugs, ['gcn', 'pinn']);
	// the integrator's run: no packages, no e2e
	assert.deepEqual(allE2ePlan(parseCheckArgs(['--all']).o, ['a', 'b']).slugs, []);
	// asked for: the five passes
	const more = allE2ePlan(parseCheckArgs(['--all', 'gcn', '--passes', 'all']).o, ['gcn']);
	assert.equal(resolvePasses(more.passes).length, 5);
	assert.throws(() => parseCheckArgs(['--all', '--quick']), /no --quick/);
	assert.throws(() => parseCheckArgs(['--all', '--passes', 'huge-en-light']), /unknown e2e pass/);
});

test('check <slug>: flags parse as before (several slugs, --quick, --only; bad combinations refused)', () => {
	const { o, pos } = parseCheckArgs(['tinyvit,gcn', '--quick', '--only', 'lesson']);
	assert.deepEqual(pos, ['tinyvit', 'gcn']);
	assert.equal(o.quick, true);
	assert.deepEqual(o.only, ['lesson']);
	assert.equal(o.passes, undefined);
	assert.equal(o.e2e, undefined, 'without --all, slugs are packages to check, not an e2e list');
	assert.throws(() => parseCheckArgs(['x', '--quick', '--fast']), /pick one/);
	assert.throws(() => parseCheckArgs(['x', '--only', 'speed']), /unknown gate/);
	assert.throws(() => parseCheckArgs(['x', '--bogus']), /unknown flag/);
});

test('AK603 (a long task) only warns: e2e never turns it into an error', () => {
	const src = readFileSync(join(SITE, 'scripts', 'atlas', 'commands', 'e2e.ts'), 'utf8');
	assert.doesNotMatch(src, /is\.error\('AK603'/);
	assert.match(src, /is\.warn\('AK603'/);
});
